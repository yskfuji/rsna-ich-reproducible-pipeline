#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make `src` importable regardless of current working directory.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (  # noqa: E402
    RSNA_CLASSES,
    RsnaIchSlice25DDataset,
    RsnaIchSlice2DDataset,
    RsnaSliceRecord,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    preprocessed_db_has_meta,
    read_rsna_preprocessed_meta,
)
from src.models.mc_dropout import enable_dropout_only, inject_stage_and_head_dropout  # noqa: E402


def _device() -> torch.device:
    import os

    dev = os.environ.get("TORCH_DEVICE", "cpu").strip().lower()
    if dev in {"cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x64))).astype(np.float64)


def _weighted_logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-7) -> float:
    """Kaggle-style weighted multi-label logloss (any=2, others=1)."""

    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)

    # (N,C)
    per_elem = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    weights = np.array([2.0 if c == "any" else 1.0 for c in RSNA_CLASSES], dtype=np.float64)
    wsum = float(np.sum(weights))
    per_row = (per_elem * weights.reshape(1, -1)).sum(axis=1) / max(1e-12, wsum)
    return float(np.mean(per_row))


def _aurc_from_uncertainty(y: np.ndarray, p_mean: np.ndarray, u: np.ndarray) -> float:
    """Compute AURC (area under risk-coverage curve) using weighted logloss as risk.

    - Sort by uncertainty ascending (most confident first)
    - For k=1..N, risk(k) = weighted_logloss on first k
    - AURC = mean_k risk(k)
    """

    y = np.asarray(y, dtype=np.float64)
    p_mean = np.asarray(p_mean, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y must be (N,C)")
    if p_mean.shape != y.shape:
        raise ValueError("p_mean shape mismatch")
    if u.shape != (y.shape[0],):
        raise ValueError("u must be (N,)")

    order = np.argsort(u)  # low uncertainty first
    y_s = y[order]
    p_s = p_mean[order]

    # Prefix risks
    risks = []
    for k in range(1, y_s.shape[0] + 1):
        risks.append(_weighted_logloss(y_s[:k], p_s[:k]))
    return float(np.mean(np.asarray(risks, dtype=np.float64)))


def _resolve_path(p: str | None) -> Path | None:
    if p is None:
        return None
    v = str(p).strip()
    if not v:
        return None
    return Path(v).expanduser().resolve()


def _build_group_ids(*, records: list[RsnaSliceRecord], split_by: str, pre_db: Path | None) -> list[str]:
    split_by_s = str(split_by).strip().lower()
    if split_by_s not in {"study", "series"}:
        raise ValueError("split_by must be 'study' or 'series'")

    group_ids: list[str] = []
    if pre_db is None:
        # Fallback: cannot compute study/series without DICOM; use image_id (degrades to slice split).
        return [str(r.image_id) for r in records]

    if not preprocessed_db_has_meta(pre_db):
        raise SystemExit("preprocessed DB has no meta table. Rebuild/backfill via tools/precompute_rsna_preprocessed_sqlite.py")

    cache: dict[str, tuple[str | None, str | None]] = {}
    for r in records:
        img = str(getattr(r, "image_id"))
        if img not in cache:
            study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
            cache[img] = (study_uid, series_uid)
        study_uid, series_uid = cache[img]
        if split_by_s == "study":
            group_ids.append(study_uid or series_uid or img)
        else:
            group_ids.append(series_uid or study_uid or img)

    return group_ids


def _group_split_indices(
    *,
    records: list[RsnaSliceRecord],
    group_ids: list[str],
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if len(records) != len(group_ids):
        raise ValueError("records and group_ids length mismatch")

    groups: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(str(g), []).append(i)

    rng = np.random.default_rng(int(seed))
    all_groups = list(groups.keys())

    any_idx = int(RSNA_CLASSES.index("any"))

    def _is_pos_group(idxs: list[int]) -> bool:
        return any(float(records[i].y[any_idx]) > 0.5 for i in idxs)

    pos_groups: list[str] = []
    neg_groups: list[str] = []
    for g in all_groups:
        (pos_groups if _is_pos_group(groups[g]) else neg_groups).append(g)

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    n_groups = len(all_groups)
    n_val_groups = int(max(1, round(float(val_frac) * n_groups)))
    n_val_groups = int(min(n_val_groups, n_groups - 1)) if n_groups >= 2 else 1

    pos_target = int(round(n_val_groups * (len(pos_groups) / max(1, n_groups))))
    pos_target = int(min(max(0, pos_target), len(pos_groups)))
    neg_target = int(n_val_groups - pos_target)
    neg_target = int(min(max(0, neg_target), len(neg_groups)))
    while pos_target + neg_target < n_val_groups:
        if len(pos_groups) - pos_target > len(neg_groups) - neg_target:
            if pos_target < len(pos_groups):
                pos_target += 1
                continue
        if neg_target < len(neg_groups):
            neg_target += 1
            continue
        break

    val_group_set = set(pos_groups[:pos_target] + neg_groups[:neg_target])
    if not val_group_set and all_groups:
        val_group_set.add(all_groups[0])

    tr_idx: list[int] = []
    va_idx: list[int] = []
    for g, idxs in groups.items():
        (va_idx if g in val_group_set else tr_idx).extend(idxs)

    return tr_idx, va_idx


def _build_model(arch: str, in_channels: int, pretrained: bool, first_conv_init: str) -> torch.nn.Module:
    arch_s = str(arch).strip().lower()

    if arch_s == "resnet18":
        import torchvision  # type: ignore

        from src.models.input_adapters import adapt_first_conv

        weights = torchvision.models.ResNet18_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.resnet18(weights=weights)
        old = m.conv1
        m.conv1 = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.fc = torch.nn.Linear(m.fc.in_features, len(RSNA_CLASSES))
        return m

    if arch_s == "efficientnet_b0":
        import torchvision  # type: ignore

        from src.models.input_adapters import adapt_first_conv

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(RSNA_CLASSES))
        return m

    if arch_s == "convnext_tiny":
        import torchvision  # type: ignore

        from src.models.input_adapters import adapt_first_conv

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.convnext_tiny(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, len(RSNA_CLASSES))
        return m

    raise ValueError(f"Unsupported arch: {arch}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate MC-Dropout uncertainty on a holdout validation split using a single metric: "
            "coverage-risk (AURC) with Kaggle weighted logloss as risk." 
        )
    )
    p.add_argument("--rsna-root", required=True, type=str, help="RSNA root containing stage_2_train.csv and stage_2_train/")
    p.add_argument("--preprocessed-root", default=None, type=str, help="If set, uses <preprocessed_root>/train.sqlite")
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint .pt (state_dict)")
    p.add_argument("--arch", default="convnext_tiny", type=str, help="resnet18 | efficientnet_b0 | convnext_tiny")
    p.add_argument("--pretrained", action="store_true", help="Only affects model construction if ckpt missing keys")
    p.add_argument("--first-conv-init", default="repeat", type=str, help="repeat | mean")

    p.add_argument("--image-size", default=384, type=int)
    p.add_argument("--windows", default="40,80;80,200;600,2800", type=str)
    p.add_argument("--preprocess", default="gpt52", type=str)
    p.add_argument("--stack-slices", default=3, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--num-workers", default=0, type=int)

    p.add_argument("--limit-images", default=8000, type=int)
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--split-by", default="study", type=str, help="study | series")
    p.add_argument("--seed", default=0, type=int)

    p.add_argument("--mc-samples", default=30, type=int, help="MC passes T (default 30)")
    p.add_argument("--mc-seed", default=0, type=int)
    p.add_argument("--dropout-stage-p", default=0.2, type=float)
    p.add_argument("--dropout-head-p", default=0.2, type=float)

    ns = p.parse_args(argv)

    rsna_root = _resolve_path(ns.rsna_root)
    if rsna_root is None:
        raise SystemExit("--rsna-root is required")

    pre_root = _resolve_path(ns.preprocessed_root)
    pre_db: Path | None = None
    if pre_root is not None:
        pre_db = pre_root / "train.sqlite"
        if not pre_db.exists():
            raise SystemExit(f"Missing preprocessed DB: {pre_db}")

    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    records: list[RsnaSliceRecord] = iter_rsna_stage2_records_from_csv(
        csv_path=csv_path,
        dicom_dir=dcm_dir,
        limit_images=int(ns.limit_images) if int(ns.limit_images) > 0 else None,
        seed=int(ns.seed),
    )

    if pre_db is not None:
        exist = preprocessed_db_existing_keys([str(r.image_id) for r in records], db_path=pre_db)
        before = len(records)
        records = [r for r in records if str(r.image_id) in exist]
        dropped = before - len(records)
        if dropped:
            print(f"[preprocessed] filtered missing keys: kept={len(records)} dropped={dropped}", file=sys.stderr, flush=True)

    group_ids = _build_group_ids(records=records, split_by=str(ns.split_by), pre_db=pre_db)
    tr_idx, va_idx = _group_split_indices(
        records=records,
        group_ids=group_ids,
        val_frac=float(ns.val_frac),
        seed=int(ns.seed),
    )
    val_records = [records[i] for i in va_idx]

    n_win = len([w for w in str(ns.windows).split(";") if w.strip()]) or 1
    stack = int(ns.stack_slices)
    if stack != 1 and (stack < 3 or stack % 2 != 1):
        raise SystemExit("stack_slices must be 1 or an odd number >=3")
    in_channels = int(n_win) * int(stack)

    ds = (
        RsnaIchSlice2DDataset(
            val_records,
            out_size=int(ns.image_size),
            windows=str(ns.windows),
            preprocess=str(ns.preprocess),
            preprocessed_db=pre_db,
        )
        if stack == 1
        else RsnaIchSlice25DDataset(
            val_records,
            out_size=int(ns.image_size),
            windows=str(ns.windows),
            stack_slices=int(stack),
            preprocess=str(ns.preprocess),
            preprocessed_db=pre_db,
        )
    )

    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=int(ns.batch_size), shuffle=False, num_workers=int(ns.num_workers))

    dev = _device()
    model = _build_model(
        arch=str(ns.arch),
        in_channels=int(in_channels),
        pretrained=bool(ns.pretrained),
        first_conv_init=str(ns.first_conv_init),
    )

    model = inject_stage_and_head_dropout(
        model,
        arch=str(ns.arch),
        p_stage=float(ns.dropout_stage_p),
        p_head=float(ns.dropout_head_p),
    )

    sd = torch.load(str(Path(ns.ckpt).expanduser().resolve()), map_location="cpu")
    if not isinstance(sd, dict):
        raise SystemExit("ckpt must be a state_dict-like dict")
    model.load_state_dict(sd, strict=True)

    model.to(dev)
    model.eval()
    enable_dropout_only(model)

    t = int(ns.mc_samples)
    if t < 2:
        raise SystemExit("mc_samples must be >=2 for MC-Dropout")

    torch.manual_seed(int(ns.mc_seed))
    np.random.seed(int(ns.mc_seed))

    ys: list[np.ndarray] = []
    p_means: list[np.ndarray] = []
    u_scores: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(dev)
            yb = batch["y"].detach().cpu().numpy().astype(np.float64)

            samples = []
            for _ in range(t):
                logits = model(xb).detach().cpu().float().numpy()
                samples.append(_sigmoid(logits))
            samp = np.stack(samples, axis=0)  # (T,B,C)
            p_mean = samp.mean(axis=0)
            p_std = samp.std(axis=0)

            # Uncertainty score per sample: mean probability std across classes.
            u = p_std.mean(axis=1)

            ys.append(yb)
            p_means.append(p_mean)
            u_scores.append(u)

    y_all = np.concatenate(ys, axis=0)
    p_all = np.concatenate(p_means, axis=0)
    u_all = np.concatenate(u_scores, axis=0)

    aurc = _aurc_from_uncertainty(y_all, p_all, u_all)

    # Output: single-metric JSON (stdout is machine-friendly)
    print(json.dumps({"aurc_weighted_logloss": float(aurc)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
