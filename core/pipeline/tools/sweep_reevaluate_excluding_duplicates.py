#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (  # noqa: E402
    RSNA_CLASSES,
    RsnaIchSlice25DDataset,
    RsnaSliceRecord,
    preprocessed_db_existing_keys,
    read_rsna_preprocessed_meta,
    iter_rsna_stage2_records_from_csv,
)
from src.models.input_adapters import adapt_first_conv  # noqa: E402
from src.models.mc_dropout import enable_dropout_only, inject_stage_and_head_dropout  # noqa: E402
from src.training.train_rsna_cnn2d_classifier import _group_split_records  # noqa: E402


_SQLITE_MAGIC = b"RSNP"
_SQLITE_VERSION = 1
_DTYPE_U8 = 1
_DTYPE_F16 = 2
_DTYPE_F32 = 3


def _device() -> torch.device:
    import os

    dev = os.environ.get("TORCH_DEVICE", "cpu").strip().lower()
    if dev in {"cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _deserialize_chw(blob: bytes) -> np.ndarray:
    if len(blob) < 12:
        raise ValueError("invalid blob")
    magic, ver, dtype_code, comp, c, h, w = struct.unpack("<4sBBBBHH", blob[:12])
    if magic != _SQLITE_MAGIC or int(ver) != int(_SQLITE_VERSION):
        raise ValueError("invalid blob header")
    payload = blob[12:]
    if int(comp) == 1:
        payload = zlib.decompress(payload)

    if int(dtype_code) == _DTYPE_U8:
        x = np.frombuffer(payload, dtype=np.uint8).reshape((int(c), int(h), int(w))).astype(np.float32) / 255.0
    elif int(dtype_code) == _DTYPE_F16:
        x = np.frombuffer(payload, dtype=np.float16).reshape((int(c), int(h), int(w))).astype(np.float32)
    elif int(dtype_code) == _DTYPE_F32:
        x = np.frombuffer(payload, dtype=np.float32).reshape((int(c), int(h), int(w))).astype(np.float32)
    else:
        raise ValueError("unknown dtype")
    return x


def _tensor_sha256(conn: sqlite3.Connection, image_id: str) -> str:
    row = conn.execute("SELECT blob FROM tensors WHERE key=? LIMIT 1;", (str(image_id),)).fetchone()
    if row is None:
        raise KeyError(image_id)
    blob = row[0]
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    arr = _deserialize_chw(bytes(blob))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x64))).astype(np.float64)


def _weighted_logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    per_elem = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    weights = np.array([2.0 if c == "any" else 1.0 for c in RSNA_CLASSES], dtype=np.float64)
    per_row = (per_elem * weights.reshape(1, -1)).sum(axis=1) / float(np.sum(weights))
    return float(np.mean(per_row))


def _weighted_brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    sq = (p - y) ** 2
    weights = np.array([2.0 if c == "any" else 1.0 for c in RSNA_CLASSES], dtype=np.float64)
    per_row = (sq * weights.reshape(1, -1)).sum(axis=1) / float(np.sum(weights))
    return float(np.mean(per_row))


def _ece_binary(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, 1.0)
    n = int(y.size)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (p >= lo) & (p <= hi) if i == int(n_bins) - 1 else (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        yy = y[mask]
        pp = p[mask]
        acc = float(np.mean(yy))
        conf = float(np.mean(pp))
        ece += (float(pp.size) / float(n)) * abs(acc - conf)
    return float(ece)


def _auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < s_sorted.size:
        j = i + 1
        while j < s_sorted.size and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / float(n_pos * n_neg))


def _aurc(y: np.ndarray, p_mean: np.ndarray, u: np.ndarray) -> float:
    order = np.argsort(u)
    y_s = y[order]
    p_s = p_mean[order]
    risks = []
    for k in range(1, y_s.shape[0] + 1):
        risks.append(_weighted_logloss(y_s[:k], p_s[:k]))
    return float(np.mean(np.asarray(risks, dtype=np.float64)))


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor, max_iter: int = 50) -> float:
    log_t = torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)

    def loss_fn() -> torch.Tensor:
        t = torch.exp(log_t).clamp(0.5, 10.0)
        scaled = logits / t
        per_elem = torch.nn.functional.binary_cross_entropy_with_logits(scaled, targets, reduction="none")
        w = class_weights.to(device=logits.device, dtype=per_elem.dtype).view(1, -1)
        return (per_elem * w).sum(dim=1).mean() / w.sum().clamp_min(1e-12)

    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.5, 10.0).detach().cpu().item())


def _build_model(arch: str, in_channels: int, pretrained: bool, first_conv_init: str) -> torch.nn.Module:
    arch_s = str(arch).strip().lower()
    import torchvision  # type: ignore

    if arch_s == "convnext_tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.convnext_tiny(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, len(RSNA_CLASSES))
        return m
    if arch_s == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.resnet18(weights=weights)
        old = m.conv1
        m.conv1 = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.fc = torch.nn.Linear(m.fc.in_features, len(RSNA_CLASSES))
        return m
    if arch_s == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(RSNA_CLASSES))
        return m
    raise ValueError(f"unsupported arch: {arch}")


def _evaluate_on_records(
    *,
    ckpt: Path,
    arch: str,
    in_channels: int,
    val_records: list[RsnaSliceRecord],
    pre_db: Path,
    image_size: int,
    windows: str,
    preprocess: str,
    stack_slices: int,
    batch_size: int,
    num_workers: int,
    mc_samples: int,
    mc_seed: int,
    dropout_stage_p: float,
    dropout_head_p: float,
    fit_temperature: bool,
    fixed_temperature: float,
    coverage: float,
    ece_bins: int,
) -> dict[str, float]:
    dev = _device()
    model = _build_model(arch=arch, in_channels=in_channels, pretrained=False, first_conv_init="mean")
    model = inject_stage_and_head_dropout(
        model,
        arch=str(arch),
        p_stage=float(dropout_stage_p),
        p_head=float(dropout_head_p),
    )
    sd = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(dev)
    model.eval()

    ds = RsnaIchSlice25DDataset(
        val_records,
        out_size=int(image_size),
        windows=str(windows),
        stack_slices=int(stack_slices),
        preprocess=str(preprocess),
        preprocessed_db=pre_db,
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))

    logits_det: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(dev)
            yb = batch["y"].to(dev)
            logits_det.append(model(xb))
            targets_all.append(yb)
    logits_det_t = torch.cat(logits_det, dim=0)
    targets_t = torch.cat(targets_all, dim=0)

    cw = torch.tensor([2.0 if c == "any" else 1.0 for c in RSNA_CLASSES], device=dev, dtype=torch.float32)
    if bool(fit_temperature):
        temp = _fit_temperature(logits_det_t.float(), targets_t.float(), cw.float())
    else:
        temp = float(fixed_temperature)

    model.eval()
    enable_dropout_only(model)
    torch.manual_seed(int(mc_seed))
    np.random.seed(int(mc_seed))

    probs_mean: list[np.ndarray] = []
    probs_std: list[np.ndarray] = []
    ys_np: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(dev)
            yb = batch["y"].detach().cpu().numpy().astype(np.float64)
            samples = []
            for _ in range(int(mc_samples)):
                logits = model(xb).detach().cpu().float().numpy()
                samples.append(_sigmoid(logits / float(temp)))
            samp = np.stack(samples, axis=0)
            probs_mean.append(samp.mean(axis=0))
            probs_std.append(samp.std(axis=0))
            ys_np.append(yb)

    y_all = np.concatenate(ys_np, axis=0)
    p_all = np.concatenate(probs_mean, axis=0)
    pstd_all = np.concatenate(probs_std, axis=0)

    any_i = int(RSNA_CLASSES.index("any"))
    y_any = y_all[:, any_i]
    p_any = p_all[:, any_i]
    u = pstd_all[:, any_i]

    cov = float(np.clip(float(coverage), 0.0, 1.0))
    order = np.argsort(u)
    k = int(max(1, round(cov * len(order))))
    keep = order[:k]

    acc_full = float(np.mean((p_any >= 0.5) == (y_any >= 0.5)))
    acc_cov = float(np.mean((p_any[keep] >= 0.5) == (y_any[keep] >= 0.5)))

    incorrect_any = ((p_any >= 0.5) != (y_any >= 0.5)).astype(np.int64)

    out = {
        "temperature": float(temp),
        "ece_any": float(_ece_binary(y_any, p_any, n_bins=int(ece_bins))),
        "brier_any": float(np.mean((p_any - y_any) ** 2)),
        "brier_weighted": float(_weighted_brier(y_all, p_all)),
        "nll_weighted_logloss": float(_weighted_logloss(y_all, p_all)),
        "auroc_uncertainty_detect_error_any": float(_auroc_binary(incorrect_any, u)),
        "aurc_weighted_logloss": float(_aurc(y_all, p_all, u)),
        "accuracy_any_full": float(acc_full),
        "accuracy_any_at_coverage": float(acc_cov),
        "accuracy_any_improve_pp": float((acc_cov - acc_full) * 100.0),
        "n_val": int(y_all.shape[0]),
    }
    return out


def _nan_stats(xs: list[float]) -> dict[str, float]:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Re-evaluate uncertainty metrics after excluding val samples whose decoded tensor is identical to any train sample."
        )
    )
    p.add_argument("--rsna-root", required=True, type=str)
    p.add_argument("--preprocessed-root", required=True, type=str)
    p.add_argument("--baseline-ckpt", required=True, type=str)
    p.add_argument("--retrain-root", required=True, type=str)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=str)

    p.add_argument("--arch", default="convnext_tiny", type=str)
    p.add_argument("--image-size", default=384, type=int)
    p.add_argument("--windows", default="40,80;80,200;600,2800", type=str)
    p.add_argument("--preprocess", default="gpt52", type=str)
    p.add_argument("--stack-slices", default=3, type=int)
    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)

    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--mc-samples", default=30, type=int)
    p.add_argument("--dropout-stage-p", default=0.2, type=float)
    p.add_argument("--dropout-head-p", default=0.2, type=float)
    p.add_argument("--fit-temperature", action="store_true")
    p.add_argument("--temperature", default=1.0, type=float)
    p.add_argument("--coverage", default=0.8, type=float)
    p.add_argument("--ece-bins", default=15, type=int)
    p.add_argument("--out-json", default="/tmp/rsna_exclusion_reeval_seed0to9.json", type=str)
    ns = p.parse_args(argv)

    rsna_root = Path(str(ns.rsna_root)).expanduser().resolve()
    pre_root = Path(str(ns.preprocessed_root)).expanduser().resolve()
    pre_db = pre_root / "train.sqlite"
    baseline_ckpt = Path(str(ns.baseline_ckpt)).expanduser().resolve()
    retrain_root = Path(str(ns.retrain_root)).expanduser().resolve()

    seeds = [int(x.strip()) for x in str(ns.seeds).split(",") if x.strip()]
    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    conn = sqlite3.connect(str(pre_db), check_same_thread=False)

    any_idx = int(RSNA_CLASSES.index("any"))

    def _pos_group_selector(records: list[RsnaIchSliceRecord]) -> bool:
        return any(float(r.y[any_idx]) > 0.5 for r in records)

    n_win = len([w for w in str(ns.windows).split(";") if w.strip()]) or 1
    in_channels = int(n_win) * int(ns.stack_slices)

    metric_keys = [
        "ece_any",
        "brier_any",
        "nll_weighted_logloss",
        "auroc_uncertainty_detect_error_any",
        "aurc_weighted_logloss",
        "accuracy_any_improve_pp",
    ]

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        records = iter_rsna_stage2_records_from_csv(
            csv_path=csv_path,
            dicom_dir=dcm_dir,
            limit_images=int(ns.limit_images) if int(ns.limit_images) > 0 else None,
            seed=int(seed),
        )
        exist = preprocessed_db_existing_keys([str(r.image_id) for r in records], db_path=pre_db)
        records = [r for r in records if str(r.image_id) in exist]

        meta: dict[str, tuple[str | None, str | None]] = {}
        group_ids: list[str] = []
        for r in records:
            img = str(r.image_id)
            if img not in meta:
                study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
                meta[img] = (study_uid, series_uid)
            st, se = meta[img]
            group_ids.append(st or se or img)

        train_records, val_records = _group_split_records(
            records=records,
            group_ids=group_ids,
            val_frac=float(ns.val_frac),
            seed=int(seed),
            pos_group_selector=_pos_group_selector,
        )[:2]

        tr_hashes = {_tensor_sha256(conn, str(r.image_id)) for r in train_records}
        val_keep: list[RsnaSliceRecord] = []
        n_dropped = 0
        for r in val_records:
            h = _tensor_sha256(conn, str(r.image_id))
            if h in tr_hashes:
                n_dropped += 1
            else:
                val_keep.append(r)

        compare_json = retrain_root / f"seed{seed}" / "compare.json"
        if not compare_json.exists():
            raise FileNotFoundError(f"missing compare.json: {compare_json}")
        full = json.loads(compare_json.read_text(encoding="utf-8"))

        retr_ckpt = retrain_root / f"seed{seed}" / "best.pt"
        if not retr_ckpt.exists():
            raise FileNotFoundError(f"missing retrained ckpt: {retr_ckpt}")

        baseline_ex = _evaluate_on_records(
            ckpt=baseline_ckpt,
            arch=str(ns.arch),
            in_channels=in_channels,
            val_records=val_keep,
            pre_db=pre_db,
            image_size=int(ns.image_size),
            windows=str(ns.windows),
            preprocess=str(ns.preprocess),
            stack_slices=int(ns.stack_slices),
            batch_size=int(ns.batch_size),
            num_workers=int(ns.num_workers),
            mc_samples=int(ns.mc_samples),
            mc_seed=int(seed),
            dropout_stage_p=float(ns.dropout_stage_p),
            dropout_head_p=float(ns.dropout_head_p),
            fit_temperature=bool(ns.fit_temperature),
            fixed_temperature=float(ns.temperature),
            coverage=float(ns.coverage),
            ece_bins=int(ns.ece_bins),
        )
        retr_ex = _evaluate_on_records(
            ckpt=retr_ckpt,
            arch=str(ns.arch),
            in_channels=in_channels,
            val_records=val_keep,
            pre_db=pre_db,
            image_size=int(ns.image_size),
            windows=str(ns.windows),
            preprocess=str(ns.preprocess),
            stack_slices=int(ns.stack_slices),
            batch_size=int(ns.batch_size),
            num_workers=int(ns.num_workers),
            mc_samples=int(ns.mc_samples),
            mc_seed=int(seed),
            dropout_stage_p=float(ns.dropout_stage_p),
            dropout_head_p=float(ns.dropout_head_p),
            fit_temperature=bool(ns.fit_temperature),
            fixed_temperature=float(ns.temperature),
            coverage=float(ns.coverage),
            ece_bins=int(ns.ece_bins),
        )

        baseline_full = full["baseline"]
        retr_full = full["retrained_with_dropout"]

        baseline_diff = {k: float(baseline_ex[k]) - float(baseline_full[k]) for k in metric_keys}
        retr_diff = {k: float(retr_ex[k]) - float(retr_full[k]) for k in metric_keys}

        rows.append(
            {
                "seed": int(seed),
                "n_val_full": int(len(val_records)),
                "n_val_excluded": int(len(val_keep)),
                "n_val_dropped_exact_duplicates": int(n_dropped),
                "baseline_full": {k: float(baseline_full[k]) for k in metric_keys},
                "baseline_excluded": {k: float(baseline_ex[k]) for k in metric_keys},
                "baseline_excluded_minus_full": baseline_diff,
                "retrained_full": {k: float(retr_full[k]) for k in metric_keys},
                "retrained_excluded": {k: float(retr_ex[k]) for k in metric_keys},
                "retrained_excluded_minus_full": retr_diff,
            }
        )

    summary: dict[str, Any] = {
        "seeds": list(seeds),
        "rows": rows,
        "summary": {
            "n_val_dropped_exact_duplicates": _nan_stats([float(r["n_val_dropped_exact_duplicates"]) for r in rows]),
            "baseline_excluded_minus_full": {},
            "retrained_excluded_minus_full": {},
        },
    }

    for k in metric_keys:
        summary["summary"]["baseline_excluded_minus_full"][k] = _nan_stats(
            [float(r["baseline_excluded_minus_full"][k]) for r in rows]
        )
        summary["summary"]["retrained_excluded_minus_full"][k] = _nan_stats(
            [float(r["retrained_excluded_minus_full"][k]) for r in rows]
        )

    out_path = Path(str(ns.out_json)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
