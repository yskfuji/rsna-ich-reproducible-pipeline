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
    compute_subset_fingerprint_sha256,
    deduplicate_records_by_preprocessed_tensor_hash,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    preprocessed_db_has_meta,
    read_rsna_preprocessed_meta,
)
from src.models.mc_dropout import enable_dropout_only, inject_stage_and_head_dropout  # noqa: E402
from src.training.train_rsna_cnn2d_classifier import _group_split_records  # noqa: E402


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
    per_elem = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    weights = np.array([2.0 if c == "any" else 1.0 for c in RSNA_CLASSES], dtype=np.float64)
    per_row = (per_elem * weights.reshape(1, -1)).sum(axis=1) / float(np.sum(weights))
    return float(np.mean(per_row))


def _weighted_brier(y: np.ndarray, p: np.ndarray) -> float:
    """Weighted multi-label Brier score (any=2, others=1).

    Returns mean over samples of weighted mean squared error across classes.
    """

    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 2 or p.shape != y.shape:
        raise ValueError("y/p must be (N,C)")
    p = np.clip(p, 0.0, 1.0)

    sq = (p - y) ** 2
    weights = np.array([2.0 if c == "any" else 1.0 for c in RSNA_CLASSES], dtype=np.float64)
    per_row = (sq * weights.reshape(1, -1)).sum(axis=1) / float(np.sum(weights))
    return float(np.mean(per_row))


def _ece_binary(y: np.ndarray, p: np.ndarray, *, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary probabilities.

    y: (N,) in {0,1}
    p: (N,) in [0,1]
    """

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, 1.0)
    n = int(y.size)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        yy = y[mask]
        pp = p[mask]
        # Reliability diagram definition:
        #   acc(bin) = empirical positive rate, conf(bin) = mean predicted prob.
        acc = float(np.mean(yy))
        conf = float(np.mean(pp))
        w = float(pp.size) / float(n)
        ece += w * abs(acc - conf)
    return float(ece)


def _auroc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC for binary labels using rank statistics (no sklearn).

    y_true: (N,) values in {0,1}
    y_score: (N,) higher means more likely positive
    """

    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError("y_true/y_score size mismatch")

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank scores with average ties.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)

    # Handle ties: set rank to average within each tie group.
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
    # Mann–Whitney U -> AUROC
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return float(u / float(n_pos * n_neg))


def _aurc(y: np.ndarray, p_mean: np.ndarray, u: np.ndarray) -> float:
    """AURC (area under risk-coverage curve) with weighted logloss as risk.

    - Sort by uncertainty ascending (most confident first)
    - risk(k) computed on prefix k
    - AURC = mean_k risk(k)
    """

    y = np.asarray(y, dtype=np.float64)
    p_mean = np.asarray(p_mean, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if y.ndim != 2 or p_mean.shape != y.shape:
        raise ValueError("y/p_mean must be (N,C)")
    if u.shape != (y.shape[0],):
        raise ValueError("u must be (N,)")

    order = np.argsort(u)
    y_s = y[order]
    p_s = p_mean[order]

    risks = []
    for k in range(1, y_s.shape[0] + 1):
        risks.append(_weighted_logloss(y_s[:k], p_s[:k]))
    return float(np.mean(np.asarray(risks, dtype=np.float64)))


def _plot_coverage_risk(
    *,
    out_png: Path,
    y: np.ndarray,
    p_mean: np.ndarray,
    u: np.ndarray,
    coverages: list[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    y = np.asarray(y, dtype=np.float64)
    p_mean = np.asarray(p_mean, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    order = np.argsort(u)
    y_s = y[order]
    p_s = p_mean[order]

    xs = []
    ys_risk = []
    n = int(y_s.shape[0])
    for c in coverages:
        k = int(max(1, round(float(c) * n)))
        xs.append(float(k) / float(n))
        ys_risk.append(_weighted_logloss(y_s[:k], p_s[:k]))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys_risk, marker="o", linewidth=1.5)
    plt.xlabel("coverage")
    plt.ylabel("risk (weighted logloss)")
    plt.title("Coverage–Risk curve (risk=weighted logloss)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()


def _plot_reliability_diagram(
    *,
    out_png: Path,
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    xs: list[float] = []
    ys: list[float] = []
    ns: list[int] = []
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        pp = p[mask]
        yy = y[mask]
        xs.append(float(np.mean(pp)))
        ys.append(float(np.mean(yy)))
        ns.append(int(pp.size))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", linewidth=1.0)
    if xs:
        plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel("mean predicted probability")
    plt.ylabel("empirical positive rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()


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

    if pre_db is None:
        # Without preprocessed meta (or DICOM), we cannot reliably get Study/Series here.
        # Degrade to per-slice ids.
        return [str(r.image_id) for r in records]

    if not preprocessed_db_has_meta(pre_db):
        raise SystemExit("preprocessed DB has no meta table. Rebuild/backfill via tools/precompute_rsna_preprocessed_sqlite.py")

    group_ids: list[str] = []
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


def _holdout_split_records(
    *,
    records: list[RsnaSliceRecord],
    group_ids: list[str],
    val_frac: float,
    seed: int,
) -> tuple[list[RsnaSliceRecord], list[RsnaSliceRecord]]:
    """Holdout split using the exact same implementation as training."""

    any_idx = int(RSNA_CLASSES.index("any"))

    def _pos_group_selector(rs: list[RsnaSliceRecord]) -> bool:
        return any(float(x.y[any_idx]) > 0.5 for x in rs)

    train_records, val_records, _stats = _group_split_records(
        records=records,
        group_ids=group_ids,
        val_frac=float(val_frac),
        seed=int(seed),
        pos_group_selector=_pos_group_selector,
    )
    return train_records, val_records


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


def _fit_temperature(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    max_iter: int = 50,
) -> float:
    """Fit a scalar temperature T>0 by minimizing weighted BCE.

    Uses LBFGS on log_T to keep positivity.
    """

    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits/targets must be (N,C)")

    device = logits.device
    dtype = logits.dtype

    log_t = torch.zeros((), device=device, dtype=dtype, requires_grad=True)

    def loss_fn() -> torch.Tensor:
        t = torch.exp(log_t).clamp(0.5, 10.0)
        scaled = logits / t
        per_elem = torch.nn.functional.binary_cross_entropy_with_logits(scaled, targets, reduction="none")
        w = class_weights.to(device=device, dtype=per_elem.dtype).view(1, -1)
        return (per_elem * w).sum(dim=1).mean() / w.sum().clamp_min(1e-12)

    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=int(max_iter), line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).clamp(0.5, 10.0).detach().cpu().item())
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "RSNA ICH uncertainty + calibration evaluation using MC-Dropout and temperature scaling. "
            "Outputs: ECE (any only), a coverage–risk curve (PNG), and a README-friendly statement for coverage@80%."
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
    p.add_argument(
        "--dedup-before-split",
        action="store_true",
        default=True,
        help="If preprocessed DB is used, deduplicate exact tensor duplicates before split.",
    )
    p.add_argument("--no-dedup-before-split", action="store_false", dest="dedup_before_split")
    p.add_argument("--val-frac", default=0.05, type=float)
    p.add_argument("--split-by", default="study", type=str, help="study | series")
    p.add_argument("--seed", default=0, type=int)

    p.add_argument("--mc-samples", default=30, type=int, help="MC passes T (default 30)")
    p.add_argument("--mc-seed", default=0, type=int)
    p.add_argument("--dropout-stage-p", default=0.2, type=float)
    p.add_argument("--dropout-head-p", default=0.2, type=float)

    p.add_argument("--fit-temperature", action="store_true", help="Fit temperature on the same holdout val split")
    p.add_argument("--temperature", default=1.0, type=float, help="Use fixed temperature instead of fitting")

    p.add_argument("--ece-bins", default=15, type=int)
    p.add_argument(
        "--out-reliability-png",
        default="results/uncertainty/reliability_any.png",
        type=str,
        help="Output path for reliability diagram PNG (any class)",
    )
    p.add_argument("--coverage", default=0.8, type=float, help="Coverage point for README statement (default 0.8)")
    p.add_argument("--curve-step", default=0.05, type=float, help="Coverage step for curve plot")
    p.add_argument("--out-curve-png", default="results/uncertainty/coverage_risk.png", type=str)

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
        if bool(ns.dedup_before_split):
            records, dedup_stats = deduplicate_records_by_preprocessed_tensor_hash(records, db_path=pre_db)
            if int(dedup_stats.get("dedup_dropped_duplicates", 0)) > 0:
                print(
                    "[dedup] before={dedup_records_before} after={dedup_records_after} dropped={dedup_dropped_duplicates}".format(
                        **dedup_stats
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    group_ids = _build_group_ids(records=records, split_by=str(ns.split_by), pre_db=pre_db)
    _train_records, val_records = _holdout_split_records(
        records=records,
        group_ids=group_ids,
        val_frac=float(ns.val_frac),
        seed=int(ns.seed),
    )

    adopted_subset_fingerprint_sha256 = compute_subset_fingerprint_sha256([r.image_id for r in records])
    val_subset_fingerprint_sha256 = compute_subset_fingerprint_sha256([r.image_id for r in val_records])

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

    # Collect deterministic logits for optional temperature fitting.
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

    if bool(ns.fit_temperature):
        temp = _fit_temperature(logits=logits_det_t.float(), targets=targets_t.float(), class_weights=cw.float())
    else:
        temp = float(ns.temperature)
        if temp <= 0.0:
            raise SystemExit("temperature must be > 0")

    # MC-Dropout sampling: probability mean + std
    t_mc = int(ns.mc_samples)
    if t_mc < 2:
        raise SystemExit("mc_samples must be >=2")

    model.eval()
    enable_dropout_only(model)

    torch.manual_seed(int(ns.mc_seed))
    np.random.seed(int(ns.mc_seed))

    probs_mean: list[np.ndarray] = []
    probs_std: list[np.ndarray] = []
    ys_np: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            xb = batch["x"].to(dev)
            yb = batch["y"].detach().cpu().numpy().astype(np.float64)

            samples = []
            for _ in range(t_mc):
                logits = model(xb).detach().cpu().float().numpy()
                samples.append(_sigmoid(logits / float(temp)))
            samp = np.stack(samples, axis=0)  # (T,B,C)
            probs_mean.append(samp.mean(axis=0))
            probs_std.append(samp.std(axis=0))
            ys_np.append(yb)

    y_all = np.concatenate(ys_np, axis=0)
    p_all = np.concatenate(probs_mean, axis=0)
    pstd_all = np.concatenate(probs_std, axis=0)

    any_i = int(RSNA_CLASSES.index("any"))
    y_any = y_all[:, any_i]
    p_any = p_all[:, any_i]

    # Uncertainty score: std of p(any) across MC samples
    u = pstd_all[:, any_i]

    ece_any = _ece_binary(y_any, p_any, n_bins=int(ns.ece_bins))

    # Brier scores
    brier_any = float(np.mean((p_any.astype(np.float64) - y_any.astype(np.float64)) ** 2))
    brier_weighted = _weighted_brier(y_all, p_all)

    # NLL (negative log likelihood) for Bernoulli = (weighted) logloss
    nll_weighted = _weighted_logloss(y_all, p_all)

    # Coverage@c: keep most confident c fraction
    cov = float(ns.coverage)
    cov = float(np.clip(cov, 0.0, 1.0))
    order = np.argsort(u)
    n = int(order.size)
    k = int(max(1, round(cov * n)))
    keep = order[:k]

    acc_full = float(np.mean((p_any >= 0.5) == (y_any >= 0.5)))
    acc_cov = float(np.mean((p_any[keep] >= 0.5) == (y_any[keep] >= 0.5)))
    acc_improve_pp = (acc_cov - acc_full) * 100.0

    # Uncertainty AUROC for error detection: label=1 if incorrect, score=uncertainty (higher => more likely incorrect)
    incorrect_any = ((p_any >= 0.5) != (y_any >= 0.5)).astype(np.int64)
    auroc_u_detect_error_any = _auroc_binary(incorrect_any, u)

    aurc = _aurc(y_all, p_all, u)

    step = float(ns.curve_step)
    if step <= 0.0 or step > 1.0:
        raise SystemExit("curve_step must be in (0,1]")
    coverages = [float(c) for c in np.arange(step, 1.0 + 1e-9, step)]

    out_png = Path(str(ns.out_curve_png)).expanduser().resolve()
    _plot_coverage_risk(out_png=out_png, y=y_all, p_mean=p_all, u=u, coverages=coverages)

    rel_png = Path(str(ns.out_reliability_png)).expanduser().resolve()
    _plot_reliability_diagram(
        out_png=rel_png,
        y=y_any,
        p=p_any,
        n_bins=int(ns.ece_bins),
        title=f"Reliability diagram (any) | bins={int(ns.ece_bins)}",
    )

    out: dict[str, Any] = {
        "temperature": float(temp),
        "adopted_subset_fingerprint_sha256": str(adopted_subset_fingerprint_sha256),
        "val_subset_fingerprint_sha256": str(val_subset_fingerprint_sha256),
        "ece_any": float(ece_any),
        "brier_any": float(brier_any),
        "brier_weighted": float(brier_weighted),
        "nll_weighted_logloss": float(nll_weighted),
        "auroc_uncertainty_detect_error_any": float(auroc_u_detect_error_any),
        "aurc_weighted_logloss": float(aurc),
        "coverage_point": float(cov),
        "accuracy_any_full": float(acc_full),
        "accuracy_any_at_coverage": float(acc_cov),
        "accuracy_any_improve_pp": float(acc_improve_pp),
        "curve_png": str(out_png),
        "reliability_png": str(rel_png),
        "n_val": int(y_all.shape[0]),
        "mc_samples": int(t_mc),
        "dropout_stage_p": float(ns.dropout_stage_p),
        "dropout_head_p": float(ns.dropout_head_p),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
