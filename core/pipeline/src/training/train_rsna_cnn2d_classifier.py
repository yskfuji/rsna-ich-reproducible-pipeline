from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from collections.abc import Callable, Sequence
import time
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from ..datasets.rsna_ich_dataset import (
    RSNA_CLASSES,
    RsnaIchSlice25DDataset,
    RsnaIchSlice2DDataset,
    RsnaSliceRecord,
    compute_subset_fingerprint_sha256,
    deduplicate_records_by_preprocessed_tensor_hash,
    preprocessed_db_has_meta,
    preprocessed_db_existing_keys,
    read_rsna_preprocessed_meta,
    iter_rsna_stage2_records_from_csv,
)
from ..models.cnn2d_classifier import CNN2DClassifier
from ..models.mc_dropout import inject_stage_and_head_dropout

app = typer.Typer(add_completion=False)


class InputNormalizeMode(str, Enum):
    auto = "auto"
    imagenet = "imagenet"
    none = "none"


def _parse_input_normalize_mode(x: InputNormalizeMode | str) -> InputNormalizeMode:
    if isinstance(x, InputNormalizeMode):
        return x
    s = str(x).strip()
    if s.startswith("InputNormalizeMode."):
        s = s.split(".", 1)[1]
    return InputNormalizeMode(s)


def _needs_imagenet_norm(arch: str, pretrained: bool) -> bool:
    if not bool(pretrained):
        return False
    a = str(arch).strip().lower()
    return a in {"resnet18", "efficientnet_b0", "convnext_tiny"}


def _imagenet_norm_stats(channels: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) shaped (1,C,1,1) for ImageNet-pretrained torchvision models."""
    mean3 = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
    std3 = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
    c = int(channels) if int(channels) > 0 else 3
    rep = int((c + 2) // 3)
    mean = mean3.repeat(rep)[:c]
    std = std3.repeat(rep)[:c]
    return mean.view(1, c, 1, 1), std.view(1, c, 1, 1)


def _save_train_state(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: Any,
    best_val: float,
    best_auc: float,
    best_wlogloss: float,
) -> None:
    path = path.expanduser().resolve()
    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "best_val": float(best_val),
        "best_auc": float(best_auc),
        "best_wlogloss": float(best_wlogloss),
    }
    torch.save(state, path)


def _try_resume(
    state_path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched: Any,
) -> tuple[int, float, float, float]:
    """Resume from a training-state checkpoint.

    Returns:
        start_epoch, best_val, best_auc, best_wlogloss
    """
    p = state_path.expanduser().resolve()
    state = torch.load(str(p), map_location="cpu")
    if not isinstance(state, dict) or "model" not in state:
        raise ValueError(f"Invalid train state: {p}")

    model.load_state_dict(state["model"], strict=True)
    if "opt" in state and isinstance(state["opt"], dict):
        try:
            opt.load_state_dict(state["opt"])  # type: ignore[arg-type]
        except Exception:
            # Optimizer state mismatch is non-fatal; continue with fresh optimizer.
            pass
    if sched is not None and "sched" in state and isinstance(state.get("sched"), dict):
        try:
            sched.load_state_dict(state["sched"])  # type: ignore[arg-type]
        except Exception:
            pass

    last_epoch = int(state.get("epoch", 0))
    best_val = float(state.get("best_val", float("inf")))
    best_auc = float(state.get("best_auc", float("-inf")))
    best_wlogloss = float(state.get("best_wlogloss", float("inf")))
    return last_epoch + 1, best_val, best_auc, best_wlogloss


RSNA_LOGLOSS_CLASS_WEIGHTS: dict[str, float] = {
    "epidural": 1.0,
    "intraparenchymal": 1.0,
    "intraventricular": 1.0,
    "subarachnoid": 1.0,
    "subdural": 1.0,
    "any": 2.0,
}


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Weighted multi-label BCEWithLogits.

    Args:
        logits: (B, C)
        targets: (B, C)
        class_weights: (C,)
        pos_weight: (C,) or None
    """
    loss_per_elem = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    w = class_weights.to(loss_per_elem.device, dtype=loss_per_elem.dtype).view(1, -1)
    # Weighted average over classes, then mean over batch.
    return (loss_per_elem * w).sum(dim=1).mean() / w.sum().clamp_min(1e-12)


def _sigmoid_np(x: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)


def _roc_auc_binary(y_true: NDArray[np.floating[Any]], y_score: NDArray[np.floating[Any]]) -> float:
    """ROC AUC for binary labels without sklearn.

    Returns NaN if y_true has <2 unique values.
    """
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.size == 0:
        return float("nan")
    # Need both classes present
    if np.min(y_true) == np.max(y_true):
        return float("nan")

    # Rank-based AUC (equivalent to Mann–Whitney U)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_true.size + 1, dtype=np.float64)

    pos = (y_true > 0.5)
    n_pos = float(np.sum(pos))
    n_neg = float(y_true.size - n_pos)
    sum_ranks_pos = float(np.sum(ranks[pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _read_study_and_series_uid(dcm_path: Path) -> tuple[str | None, str | None]:
    import pydicom  # lazy import
    import warnings

    warnings.filterwarnings("ignore", message=r"Invalid value for VR UI:.*", module=r"pydicom\..*")
    try:
        ds = pydicom.dcmread(
            str(dcm_path),
            stop_before_pixels=True,
            specific_tags=["StudyInstanceUID", "SeriesInstanceUID"],
            force=True,
        )
    except Exception:
        return None, None

    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip() or None
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip() or None
    return study_uid, series_uid


def _group_split_records(
    records: Sequence[RsnaSliceRecord],
    group_ids: Sequence[str],
    val_frac: float,
    seed: int,
    pos_group_selector: Callable[[Sequence[RsnaSliceRecord]], bool],
) -> tuple[list[RsnaSliceRecord], list[RsnaSliceRecord], dict[str, Any]]:
    if len(records) != len(group_ids):
        raise ValueError("records and group_ids length mismatch")

    # group -> list indices
    groups: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(str(g), []).append(i)

    rng = np.random.default_rng(int(seed))
    all_groups = list(groups.keys())

    pos_groups: list[str] = []
    neg_groups: list[str] = []
    for g in all_groups:
        idxs = groups[g]
        if bool(pos_group_selector([records[i] for i in idxs])):
            pos_groups.append(g)
        else:
            neg_groups.append(g)

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    n_groups = len(all_groups)
    n_val_groups = int(max(1, round(float(val_frac) * n_groups)))
    n_val_groups = int(min(n_val_groups, n_groups - 1)) if n_groups >= 2 else 1

    # allocate val groups roughly preserving pos/neg proportion
    pos_target = int(round(n_val_groups * (len(pos_groups) / max(1, n_groups))))
    pos_target = int(min(max(0, pos_target), len(pos_groups)))
    neg_target = int(n_val_groups - pos_target)
    neg_target = int(min(max(0, neg_target), len(neg_groups)))
    # if short due to clamping, fill from the other side
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

    train_records: list[RsnaSliceRecord] = [records[i] for i in tr_idx]
    val_records: list[RsnaSliceRecord] = [records[i] for i in va_idx]

    stats = {
        "split_groups_total": n_groups,
        "split_groups_val": len(val_group_set),
        "split_groups_train": len(all_groups) - len(val_group_set),
        "split_groups_pos_total": len(pos_groups),
        "split_groups_neg_total": len(neg_groups),
        "split_groups_pos_val": sum(1 for g in val_group_set if g in set(pos_groups)),
        "split_groups_neg_val": sum(1 for g in val_group_set if g in set(neg_groups)),
    }
    return train_records, val_records, stats


def _stratified_group_kfold_records(
    records: Sequence[RsnaSliceRecord],
    group_ids: Sequence[str],
    n_splits: int,
    fold_index: int,
    seed: int,
    pos_group_selector: Callable[[Sequence[RsnaSliceRecord]], bool],
) -> tuple[list[RsnaSliceRecord], list[RsnaSliceRecord], dict[str, Any]]:
    """A simple stratified group-kfold splitter.

    Distributes positive/negative groups across folds in round-robin order after shuffling.
    This is not identical to sklearn's GroupKFold, but meets the intent: each fold holds out
    a disjoint set of groups while roughly balancing positives.
    """

    if len(records) != len(group_ids):
        raise ValueError("records and group_ids length mismatch")

    k = int(n_splits)
    if k < 2:
        raise ValueError("n_splits must be >= 2")
    fi = int(fold_index)
    if fi < 0 or fi >= k:
        raise ValueError("fold_index must be in [0, n_splits)")

    groups: dict[str, list[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(str(g), []).append(i)

    rng = np.random.default_rng(int(seed))
    all_groups = list(groups.keys())

    pos_groups: list[str] = []
    neg_groups: list[str] = []
    for g in all_groups:
        idxs = groups[g]
        if bool(pos_group_selector([records[i] for i in idxs])):
            pos_groups.append(g)
        else:
            neg_groups.append(g)

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    fold_groups: list[list[str]] = [[] for _ in range(k)]
    for i, g in enumerate(pos_groups):
        fold_groups[i % k].append(g)
    for i, g in enumerate(neg_groups):
        fold_groups[i % k].append(g)

    val_group_set = set(fold_groups[fi])
    tr_idx: list[int] = []
    va_idx: list[int] = []
    for g, idxs in groups.items():
        (va_idx if g in val_group_set else tr_idx).extend(idxs)

    train_records: list[RsnaSliceRecord] = [records[i] for i in tr_idx]
    val_records: list[RsnaSliceRecord] = [records[i] for i in va_idx]

    stats = {
        "split_mode": "group_kfold",
        "cv_folds": k,
        "cv_fold_index": fi,
        "split_groups_total": len(all_groups),
        "split_groups_val": len(val_group_set),
        "split_groups_train": len(all_groups) - len(val_group_set),
        "split_groups_pos_total": len(pos_groups),
        "split_groups_neg_total": len(neg_groups),
        "split_groups_pos_val": sum(1 for g in val_group_set if g in set(pos_groups)),
        "split_groups_neg_val": sum(1 for g in val_group_set if g in set(neg_groups)),
        "val_frac_effective": float(len(val_records) / max(1, len(records))),
    }
    return train_records, val_records, stats


def _binary_logloss(y_true: NDArray[np.floating[Any]], p: NDArray[np.floating[Any]], eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, float(eps), 1.0 - float(eps))
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def _weighted_multilabel_logloss(
    y_true: NDArray[np.floating[Any]],
    p: NDArray[np.floating[Any]],
    class_names: list[str],
    class_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    per_class: dict[str, float] = {}
    w_sum = 0.0
    num = 0.0
    for i, name in enumerate(class_names):
        ll = _binary_logloss(y_true[:, i], p[:, i])
        per_class[name] = float(ll)
        w = float(class_weights.get(name, 1.0))
        w_sum += w
        num += w * ll
    return float(num / max(1e-12, w_sum)), per_class


def _compute_pos_weight_from_records(records: Iterable[object]) -> torch.Tensor:
    ys = np.stack([np.asarray(getattr(r, "y"), dtype=np.float32) for r in records], axis=0)  # (N,C)
    pos = ys.sum(axis=0)
    n = float(ys.shape[0])
    neg = n - pos
    # pos_weight = neg/pos ; if pos==0 -> 1
    pw = np.ones_like(pos, dtype=np.float32)
    m = pos > 0
    pw[m] = (neg[m] / pos[m]).astype(np.float32)
    pw = np.clip(pw, 1.0, 50.0)
    return torch.tensor(pw, dtype=torch.float32)


def _augment_batch(x: torch.Tensor, p_flip: float = 0.5) -> torch.Tensor:
    """Light augmentations on (B,C,H,W) in [0,1]."""
    if p_flip > 0:
        do = torch.rand((x.shape[0],), device=x.device) < float(p_flip)
        if bool(do.any()):
            x = x.clone()
            x[do] = torch.flip(x[do], dims=[-1])

    # mild intensity jitter
    # scale in [0.9,1.1], bias in [-0.02,0.02]
    scale = (0.9 + 0.2 * torch.rand((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype))
    bias = (-0.02 + 0.04 * torch.rand((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype))
    x = x * scale + bias
    return torch.clamp(x, 0.0, 1.0)


def _device() -> torch.device:
    import os

    dev_req = os.environ.get("TORCH_DEVICE", "cpu").strip().lower()

    if dev_req in {"cuda", "cuda:0"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print(f"[device] TORCH_DEVICE={dev_req} requested but CUDA not available -> cpu", flush=True)
        return torch.device("cpu")

    if dev_req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] TORCH_DEVICE=mps requested but MPS not available -> cpu", flush=True)
        return torch.device("cpu")

    return torch.device("cpu")


@app.command()
def diagnose() -> None:
    """Print runtime environment diagnostics (device availability, torch build, etc)."""

    dev = _device()
    print(f"torch_version={torch.__version__}")
    print(f"device_selected={dev}")
    print(f"cuda_available={torch.cuda.is_available()}")
    try:
        mps_avail = torch.backends.mps.is_available()
    except Exception:
        mps_avail = False
    print(f"mps_available={mps_avail}")
    mps_built = None
    try:
        # older builds may not have is_built
        mps_built = bool(torch.backends.mps.is_built())  # type: ignore[attr-defined]
    except Exception:
        pass
    if mps_built is not None:
        print(f"mps_built={mps_built}")


@app.command()
def train(
    rsna_root: Path = typer.Option(..., help="RSNA dataset root containing stage_2_train.csv and stage_2_train/"),
    out_dir: Path = typer.Option(Path("results/rsna_ich_2d"), help="Output directory"),
    preprocessed_root: Path | None = typer.Option(
        None,
        help="If set, read preprocessed tensors from <preprocessed_root>/train.sqlite and do not read DICOM pixels. "
        "rsna_root is still used for stage_2_train.csv.",
    ),
    cache_dir: Path | None = typer.Option(
        None,
        help="Optional on-disk cache for decoded+windowed+resized tensors (.pt). Speeds up later epochs.",
    ),
    limit_images: int = typer.Option(4000, help="Number of unique slices (image_ids) to use (quick mode)"),
    dedup_before_split: bool = typer.Option(
        True,
        help="If --preprocessed-root is set, deduplicate exact tensor duplicates before split to avoid cross-split duplicate leakage.",
    ),
    val_frac: float = typer.Option(0.1, help="Validation fraction (0=use all data for training)"),
    split_by: str = typer.Option("study", help="Validation split unit: slice | series | study (avoid leakage)"),
    cv_folds: int = typer.Option(0, help="If >=2, use group k-fold split on split_by (study/series)."),
    cv_fold_index: int = typer.Option(0, help="Fold index (0..cv_folds-1) when cv_folds>=2"),
    enforce_any_max: bool = typer.Option(
        True,
        help="During validation metrics, set p(any)=max(p(any), max(p(subtypes))). (bool flags: use --enforce-any-max/--no-enforce-any-max)",
    ),
    seed: int = typer.Option(0, help="Random seed"),
    base_ch: int = typer.Option(32, help="CNN base channels"),
    image_size: int = typer.Option(256, help="Resize to (image_size,image_size)"),
    windows: str = typer.Option("40,80;80,200;600,2800", help="CT windows as 'L,W;L,W;...'"),
    preprocess: str = typer.Option(
        "legacy",
        help="RSNA DICOM preprocessing mode: legacy | gpt52. gpt52 adds MONOCHROME1/12-bit fixes + simple skull-strip/crop.",
    ),
    stack_slices: int = typer.Option(1, help="2.5D: odd number of slices to stack as channels (1=2D)"),
    batch_size: int = typer.Option(16, help="Batch size"),
    num_workers: int = typer.Option(0, help="DataLoader workers (macOS/MPSは0推奨。CPU実行は2-8推奨)"),
    epochs: int = typer.Option(3, help="Epochs (quick baseline)"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    arch: str = typer.Option("cnn", help="Model arch: cnn | resnet18 | efficientnet_b0 | convnext_tiny"),
    pretrained: bool = typer.Option(True, help="Use pretrained weights. (bool flags: use --pretrained/--no-pretrained)"),
    first_conv_init: str = typer.Option(
        "repeat",
        help="First conv init for multi-channel pretrained backbones: repeat | mean",
    ),
    dropout_stage_p: float = typer.Option(
        0.0,
        help="(MC-Dropout) Dropout2d probability applied right after the last feature stage (e.g., stage4). 0=disable.",
    ),
    dropout_head_p: float = typer.Option(
        0.0,
        help="(MC-Dropout) Dropout probability applied before the classifier head (Linear). 0=disable.",
    ),
    use_pos_weight: bool = typer.Option(
        True,
        help="Use per-class pos_weight computed from train split. (bool flags: use --use-pos-weight/--no-use-pos-weight)",
    ),
    use_sampler: bool = typer.Option(
        True,
        help="Use WeightedRandomSampler to upsample any-positive slices. (bool flags: use --use-sampler/--no-use-sampler)",
    ),
    sampler_pos_factor: float = typer.Option(3.0, help="Sampling weight multiplier for y['any']==1 when use_sampler"),
    aug: bool = typer.Option(True, help="Use light augmentations (flip + intensity jitter). (bool flags: --aug/--no-aug)"),
    input_normalize: InputNormalizeMode = typer.Option(
        InputNormalizeMode.auto,
        help="Input normalization: auto|imagenet|none. auto=ImageNet mean/std only when using pretrained torchvision backbones.",
    ),
    scheduler: bool = typer.Option(True, help="Use ReduceLROnPlateau on val_loss. (bool flags: --scheduler/--no-scheduler)"),
    loss_any_weight: float = typer.Option(
        1.0,
        help="Training loss class-weight for 'any' label (1.0 disables). Use 2.0 to match Kaggle weighted logloss.",
    ),
    resume: bool = typer.Option(False, help="Resume training from out_dir/last_state.pt if present. (bool flags: --resume/--no-resume)"),
    init_from: Path | None = typer.Option(
        None,
        help="Initialize model weights from a .pt file (state_dict). Useful for fine-tuning from best.pt. ",
    ),
    optimize_plain_loss: bool = typer.Option(
        False,
        help="Backprop using plain BCEWithLogitsLoss (unweighted). Useful when targeting val_loss_plain. ",
    ),
    log_every_steps: int = typer.Option(200, help="Print train progress every N steps (0=disable)."),
):
    """Train a quick RSNA ICH slice-level multi-label classifier using a small 2D CNN.

    - stack_slices=1: 2D input (C,H,W)
    - stack_slices=3/5/...: 2.5D input (C*stack_slices,H,W)
    """

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    rsna_root = rsna_root.expanduser().resolve()
    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    pre_db: Path | None = None
    if preprocessed_root is not None:
        pr = Path(preprocessed_root).expanduser().resolve()
        pre_db = pr / "train.sqlite"
        if not pre_db.exists():
            raise FileNotFoundError(f"Missing preprocessed DB: {pre_db}")

    cache_dir_r: Path | None = None
    if cache_dir is not None:
        cache_dir_r = Path(cache_dir).expanduser().resolve()

    records = iter_rsna_stage2_records_from_csv(
        csv_path=csv_path,
        dicom_dir=dcm_dir,
        limit_images=int(limit_images) if int(limit_images) > 0 else None,
        seed=int(seed),
    )

    if pre_db is None:
        records = [r for r in records if r.dcm_path.exists()]
        if not records:
            raise FileNotFoundError(f"No DICOM files found under: {dcm_dir}")
        dedup_stats: dict[str, int] = {
            "dedup_records_before": int(len(records)),
            "dedup_records_after": int(len(records)),
            "dedup_dropped_duplicates": 0,
            "dedup_missing_blob": 0,
        }
    else:
        # Preprocessed mode does not require DICOM presence.
        # But the DB might be incomplete; filter to existing keys to avoid runtime failures.
        exist = preprocessed_db_existing_keys([r.image_id for r in records], db_path=pre_db)
        before = len(records)
        records = [r for r in records if r.image_id in exist]
        dropped = before - len(records)
        if dropped:
            print(f"[preprocessed] filtered missing keys: kept={len(records)} dropped={dropped}", flush=True)
        if not records:
            raise FileNotFoundError(f"No matching keys found in preprocessed DB: {pre_db}")
        if bool(dedup_before_split):
            records, dedup_stats = deduplicate_records_by_preprocessed_tensor_hash(records, db_path=pre_db)
            if int(dedup_stats.get("dedup_dropped_duplicates", 0)) > 0:
                print(
                    "[dedup] before={dedup_records_before} after={dedup_records_after} dropped={dedup_dropped_duplicates}".format(
                        **dedup_stats
                    ),
                    flush=True,
                )
        else:
            dedup_stats = {
                "dedup_records_before": int(len(records)),
                "dedup_records_after": int(len(records)),
                "dedup_dropped_duplicates": 0,
                "dedup_missing_blob": 0,
            }
        if (str(split_by).strip().lower() != "slice" or int(stack_slices) != 1) and not preprocessed_db_has_meta(pre_db):
            raise ValueError(
                "When using --preprocessed-root with split_by!=slice or stack_slices!=1, the DB must include meta table. "
                "Rebuild/backfill via tools/precompute_rsna_preprocessed_sqlite.py (newer version)."
            )

    val_frac_f = float(val_frac)
    if val_frac_f < 0.0 or val_frac_f >= 1.0:
        raise ValueError("val_frac must be in [0, 1)")

    split_by_s = str(split_by).strip().lower()
    if split_by_s not in {"slice", "series", "study"}:
        raise ValueError("split_by must be one of: slice | series | study")

    cv_folds_i = int(cv_folds)
    cv_fold_i = int(cv_fold_index)
    use_cv = cv_folds_i >= 2
    if use_cv and split_by_s == "slice":
        raise ValueError("cv_folds requires split_by=study or split_by=series")
    if use_cv and (cv_fold_i < 0 or cv_fold_i >= cv_folds_i):
        raise ValueError("cv_fold_index must be in [0, cv_folds)")

    split_stats: dict[str, Any] = {"split_by": split_by_s}
    if use_cv:
        group_ids: list[str] = []
        if pre_db is None:
            # Group split by reading DICOM headers (stop_before_pixels)
            cache: dict[str, tuple[str | None, str | None]] = {}
            for r in records:
                key = str(r.dcm_path)
                if key not in cache:
                    cache[key] = _read_study_and_series_uid(r.dcm_path)
                study_uid, series_uid = cache[key]
                if split_by_s == "study":
                    group_ids.append(study_uid or series_uid or r.image_id)
                else:
                    group_ids.append(series_uid or study_uid or r.image_id)
        else:
            # Group split using SQLite meta table (no DICOM needed).
            cache2: dict[str, tuple[str | None, str | None]] = {}
            for r in records:
                img = str(r.image_id)
                if img not in cache2:
                    study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
                    cache2[img] = (study_uid, series_uid)
                study_uid, series_uid = cache2[img]
                if split_by_s == "study":
                    group_ids.append(study_uid or series_uid or r.image_id)
                else:
                    group_ids.append(series_uid or study_uid or r.image_id)

        any_idx = int(RSNA_CLASSES.index("any"))

        def _pos_group_selector(rs: Sequence[RsnaSliceRecord]) -> bool:
            return any(float(x.y[any_idx]) > 0.5 for x in rs)

        train_records, val_records, cv_stats = _stratified_group_kfold_records(
            records=records,
            group_ids=group_ids,
            n_splits=int(cv_folds_i),
            fold_index=int(cv_fold_i),
            seed=int(seed),
            pos_group_selector=_pos_group_selector,
        )
        split_stats.update(cv_stats)
    elif val_frac_f == 0.0:
        train_records = records
        val_records: list[RsnaSliceRecord] = []
        split_stats.update({"split_mode": "no_val", "split_groups_total": None})
    elif split_by_s == "slice":
        n = len(records)
        n_val = int(max(1, round(float(val_frac_f) * n)))
        n_tr = int(max(1, n - n_val))
        train_records = records[:n_tr]
        val_records = records[n_tr:]
    else:
        group_ids: list[str] = []
        if pre_db is None:
            # Group split by reading DICOM headers (stop_before_pixels)
            cache: dict[str, tuple[str | None, str | None]] = {}
            for r in records:
                key = str(r.dcm_path)
                if key not in cache:
                    cache[key] = _read_study_and_series_uid(r.dcm_path)
                study_uid, series_uid = cache[key]
                if split_by_s == "study":
                    group_ids.append(study_uid or series_uid or r.image_id)
                else:
                    group_ids.append(series_uid or study_uid or r.image_id)
        else:
            # Group split using SQLite meta table (no DICOM needed).
            cache2: dict[str, tuple[str | None, str | None]] = {}
            for r in records:
                img = str(r.image_id)
                if img not in cache2:
                    study_uid, series_uid, _inst, _series_idx = read_rsna_preprocessed_meta(image_id=img, db_path=pre_db)
                    cache2[img] = (study_uid, series_uid)
                study_uid, series_uid = cache2[img]
                if split_by_s == "study":
                    group_ids.append(study_uid or series_uid or r.image_id)
                else:
                    group_ids.append(series_uid or study_uid or r.image_id)

        any_idx = int(RSNA_CLASSES.index("any"))

        def _pos_group_selector(rs: Sequence[RsnaSliceRecord]) -> bool:
            return any(float(x.y[any_idx]) > 0.5 for x in rs)

        train_records, val_records, split_stats = _group_split_records(
            records=records,
            group_ids=group_ids,
            val_frac=float(val_frac_f),
            seed=int(seed),
            pos_group_selector=_pos_group_selector,
        )

    split_stats.setdefault("val_frac_effective", float(len(val_records) / max(1, len(records))))

    if int(stack_slices) == 1:
        train_ds = RsnaIchSlice2DDataset(
            train_records,
            out_size=int(image_size),
            windows=str(windows),
            preprocess=str(preprocess),
            cache_dir=cache_dir_r,
            preprocessed_db=pre_db,
        )
        val_ds = RsnaIchSlice2DDataset(
            val_records,
            out_size=int(image_size),
            windows=str(windows),
            preprocess=str(preprocess),
            cache_dir=cache_dir_r,
            preprocessed_db=pre_db,
        )
    else:
        train_ds = RsnaIchSlice25DDataset(
            train_records,
            out_size=int(image_size),
            windows=str(windows),
            preprocess=str(preprocess),
            stack_slices=int(stack_slices),
            cache_dir=cache_dir_r,
            preprocessed_db=pre_db,
        )
        val_ds = RsnaIchSlice25DDataset(
            val_records,
            out_size=int(image_size),
            windows=str(windows),
            preprocess=str(preprocess),
            stack_slices=int(stack_slices),
            cache_dir=cache_dir_r,
            preprocessed_db=pre_db,
        )

    nw = int(num_workers)
    sampler = None
    if bool(use_sampler):
        # Upsample positive "any" to make batches less dominated by negatives
        w = np.ones((len(train_records),), dtype=np.float64)
        any_idx = int(RSNA_CLASSES.index("any"))
        for i, r in enumerate(train_records):
            if float(r.y[any_idx]) > 0.5:
                w[i] = float(sampler_pos_factor)
        sampler = WeightedRandomSampler(
            weights=w.tolist(),
            num_samples=len(train_records),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        persistent_workers=(nw > 0),
    )
    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=nw,
            persistent_workers=(nw > 0),
        )

    dev = _device()

    # 3 window channels by default
    n_win = int(len(windows.split(";"))) if str(windows).strip() else 1
    in_channels = n_win * int(stack_slices)

    arch_s = str(arch).strip().lower()
    if arch_s == "cnn":
        model = CNN2DClassifier(
            in_channels=in_channels,
            num_classes=len(RSNA_CLASSES),
            base_ch=int(base_ch),
            norm="instance" if dev.type == "mps" else "batch",
            dropout=0.1,
        ).to(dev)
    elif arch_s == "resnet18":
        import torchvision  # type: ignore[import-not-found]
        from ..models.input_adapters import adapt_first_conv

        weights = torchvision.models.ResNet18_Weights.DEFAULT if bool(pretrained) else None
        rn = torchvision.models.resnet18(weights=weights)
        old = rn.conv1
        rn.conv1 = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        rn.fc = torch.nn.Linear(rn.fc.in_features, len(RSNA_CLASSES))
        model = rn.to(dev)
    elif arch_s == "efficientnet_b0":
        import torchvision  # type: ignore[import-not-found]
        from ..models.input_adapters import adapt_first_conv

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(RSNA_CLASSES))
        model = m.to(dev)
    elif arch_s == "convnext_tiny":
        import torchvision  # type: ignore[import-not-found]
        from ..models.input_adapters import adapt_first_conv

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.convnext_tiny(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode=str(first_conv_init))
        m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, len(RSNA_CLASSES))
        model = m.to(dev)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # Optional MC-Dropout injection for torchvision backbones.
    # This preserves checkpoint compatibility (no parameter key changes).
    if arch_s in {"resnet18", "efficientnet_b0", "convnext_tiny"}:
        model = inject_stage_and_head_dropout(
            model,
            arch=arch_s,
            p_stage=float(dropout_stage_p),
            p_head=float(dropout_head_p),
        )

    mean_t: torch.Tensor | None = None
    std_t: torch.Tensor | None = None
    norm_mode = _parse_input_normalize_mode(input_normalize)
    if norm_mode == InputNormalizeMode.imagenet:
        mean_t, std_t = _imagenet_norm_stats(int(in_channels), device=dev, dtype=torch.float32)
    elif norm_mode == InputNormalizeMode.auto:
        if _needs_imagenet_norm(arch_s, pretrained=pretrained):
            mean_t, std_t = _imagenet_norm_stats(int(in_channels), device=dev, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos_weight = None
    if bool(use_pos_weight):
        pos_weight = _compute_pos_weight_from_records(train_records).to(dev)

    # Optionally upweight 'any' during training to align with Kaggle weighted logloss.
    cw = [float(RSNA_LOGLOSS_CLASS_WEIGHTS[c]) for c in RSNA_CLASSES]
    class_weights = torch.tensor(cw, dtype=torch.float32, device=dev)
    if float(loss_any_weight) != 1.0:
        any_i = int(RSNA_CLASSES.index("any"))
        class_weights[any_i] = float(loss_any_weight)

    loss_fn_plain = torch.nn.BCEWithLogitsLoss()
    sched = None
    if bool(scheduler):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if init_from is not None:
        init_from_p = Path(init_from).expanduser().resolve()
        if not init_from_p.exists():
            raise FileNotFoundError(f"init_from not found: {init_from_p}")
        sd = torch.load(init_from_p, map_location="cpu")
        if not isinstance(sd, dict):
            raise ValueError(f"init_from must be a state_dict-like dict: {init_from_p}")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[init_from] loaded with missing={len(missing)} unexpected={len(unexpected)} from {init_from_p}", flush=True)
        else:
            print(f"[init_from] loaded from {init_from_p}", flush=True)

    meta_val_frac_effective = float(split_stats.get("val_frac_effective", float(val_frac)))

    subset_fingerprint_sha256 = compute_subset_fingerprint_sha256(
        [r.image_id for r in (list(train_records) + list(val_records))]
    )

    meta = {
        "rsna_root": str(rsna_root),
        "limit_images": int(limit_images),
        "subset_fingerprint_sha256": str(subset_fingerprint_sha256),
        "dedup_before_split": bool(dedup_before_split),
        "dedup_stats": dedup_stats,
        "cache_dir": str(cache_dir_r) if cache_dir_r is not None else None,
        "val_frac_requested": float(val_frac),
        "val_frac": meta_val_frac_effective,
        "split_by": split_stats.get("split_by", split_by_s),
        "split_stats": split_stats,
        "cv_folds": int(cv_folds_i),
        "cv_fold_index": int(cv_fold_i) if use_cv else None,
        "enforce_any_max": bool(enforce_any_max),
        "seed": int(seed),
        "base_ch": int(base_ch),
        "image_size": int(image_size),
        "windows": str(windows),
        "preprocess": str(preprocess),
        "stack_slices": int(stack_slices),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "arch": str(arch_s),
        "pretrained": bool(pretrained),
        "first_conv_init": str(first_conv_init),
        "dropout_stage_p": float(dropout_stage_p),
        "dropout_head_p": float(dropout_head_p),
        "use_pos_weight": bool(use_pos_weight),
        "pos_weight": pos_weight.detach().cpu().numpy().tolist() if pos_weight is not None else None,
        "use_sampler": bool(use_sampler),
        "sampler_pos_factor": float(sampler_pos_factor),
        "aug": bool(aug),
        "input_normalize": str(norm_mode.value),
        "scheduler": bool(scheduler),
        "loss_any_weight": float(loss_any_weight),
        "resume": bool(resume),
        "init_from": str(Path(init_from).expanduser().resolve()) if init_from is not None else None,
        "optimize_plain_loss": bool(optimize_plain_loss),
        "log_every_steps": int(log_every_steps),
        "classes": list(RSNA_CLASSES),
        "device": str(dev),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    best_val = float("inf")
    best_auc = float("-inf")
    best_wlogloss = float("inf")

    start_epoch = 1
    state_path = out_dir / "last_state.pt"
    if bool(resume) and state_path.exists():
        try:
            start_epoch, best_val, best_auc, best_wlogloss = _try_resume(
                state_path,
                model=model,
                opt=opt,
                sched=sched,
            )
            print(f"[resume] from {state_path} (start_epoch={start_epoch})", flush=True)
        except Exception as e:
            print(f"[resume] failed: {e} (starting fresh)", flush=True)
            start_epoch = 1

    for epoch in range(int(start_epoch), int(epochs) + 1):
        model.train()
        tr_losses: list[float] = []
        tr_losses_plain: list[float] = []

        last_t = time.time()
        n_steps = 0
        for batch in train_loader:
            x = batch["x"].to(dev, non_blocking=True)
            y = batch["y"].to(dev, non_blocking=True)
            if bool(aug):
                x = _augment_batch(x)
            if mean_t is not None and std_t is not None:
                x = (x - mean_t) / std_t
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = _weighted_bce_with_logits(
                logits,
                y,
                class_weights=class_weights,
                pos_weight=pos_weight,
            )
            loss_plain = loss_fn_plain(logits, y)
            loss_to_backprop = loss_plain if bool(optimize_plain_loss) else loss
            loss_to_backprop.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
            tr_losses_plain.append(float(loss_plain.item()))

            n_steps += 1
            le = int(log_every_steps)
            if le > 0 and (n_steps % le == 0):
                now = time.time()
                dt = now - last_t
                last_t = now
                ex_per_s = (le * int(batch_size)) / max(dt, 1e-9)
                print(
                    f"[epoch {epoch}] step {n_steps} train_loss={float(np.mean(tr_losses)):.6f} ({ex_per_s:.1f} ex/s)",
                    flush=True,
                )

        model.eval()
        va_losses: list[float] = []
        va_losses_plain: list[float] = []
        all_logits: list[NDArray[np.float32]] = []
        all_y: list[NDArray[np.float32]] = []
        if val_loader is not None:
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(dev, non_blocking=True)
                    y = batch["y"].to(dev, non_blocking=True)
                    if mean_t is not None and std_t is not None:
                        x = (x - mean_t) / std_t
                    logits = model(x)
                    loss = _weighted_bce_with_logits(
                        logits,
                        y,
                        class_weights=class_weights,
                        pos_weight=pos_weight,
                    )
                    loss_plain = loss_fn_plain(logits, y)
                    va_losses.append(float(loss.item()))
                    va_losses_plain.append(float(loss_plain.item()))
                    all_logits.append(logits.detach().cpu().float().numpy())
                    all_y.append(y.detach().cpu().float().numpy())

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        tr_plain = float(np.mean(tr_losses_plain)) if tr_losses_plain else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        va_plain = float(np.mean(va_losses_plain)) if va_losses_plain else float("nan")

        # If validation is disabled (val_frac=0), keep the pipeline working by
        # using train loss as a proxy so that best.pt can still be selected.
        if val_loader is None:
            va = tr
            va_plain = tr_plain

        val_auc_mean = float("nan")
        val_auc_per_class = None
        if all_logits and all_y:
            lg = np.concatenate(all_logits, axis=0)
            yy = np.concatenate(all_y, axis=0)
            prob = _sigmoid_np(lg)
            if bool(enforce_any_max):
                any_i = int(RSNA_CLASSES.index("any"))
                sub_max = prob[:, :any_i].max(axis=1)
                prob[:, any_i] = np.maximum(prob[:, any_i], sub_max)
            aucs = [_roc_auc_binary(yy[:, i], prob[:, i]) for i in range(yy.shape[1])]
            val_auc_per_class = {RSNA_CLASSES[i]: float(aucs[i]) for i in range(len(RSNA_CLASSES))}
            val_auc_mean = float(np.nanmean(np.asarray(aucs, dtype=np.float64)))

            val_wlogloss, val_logloss_per_class = _weighted_multilabel_logloss(
                y_true=yy,
                p=prob,
                class_names=list(RSNA_CLASSES),
                class_weights=RSNA_LOGLOSS_CLASS_WEIGHTS,
            )
        else:
            val_wlogloss = float("nan")
            val_logloss_per_class = None

        lr_now = float(opt.param_groups[0]["lr"])
        log = {
            "epoch": epoch,
            "train_loss": tr,
            "train_loss_plain": tr_plain,
            "val_loss": va,
            "val_loss_plain": va_plain,
            "val_auc_mean": val_auc_mean,
            "val_auc": val_auc_per_class,
            "val_logloss_weighted": val_wlogloss,
            "val_logloss_per_class": val_logloss_per_class,
            "lr": lr_now,
        }
        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        print(
            f"[epoch {epoch}] train_loss={tr:.6f} val_loss={va:.6f} val_loss_plain={va_plain:.6f} val_auc_mean={val_auc_mean:.4f} lr={lr_now:.2e}",
            flush=True,
        )

        torch.save(model.state_dict(), out_dir / "last.pt")
        _save_train_state(
            out_dir / "last_state.pt",
            epoch=epoch,
            model=model,
            opt=opt,
            sched=sched,
            best_val=best_val,
            best_auc=best_auc,
            best_wlogloss=best_wlogloss,
        )
        if np.isfinite(val_auc_mean) and val_auc_mean > best_auc:
            best_auc = float(val_auc_mean)
            torch.save(model.state_dict(), out_dir / "best_auc.pt")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), out_dir / "best.pt")
        if np.isfinite(val_wlogloss) and val_wlogloss < best_wlogloss:
            best_wlogloss = float(val_wlogloss)
            torch.save(model.state_dict(), out_dir / "best_wlogloss.pt")

        if sched is not None:
            metric = val_wlogloss if np.isfinite(val_wlogloss) else va
            if np.isfinite(metric):
                sched.step(metric)


if __name__ == "__main__":
    app()
