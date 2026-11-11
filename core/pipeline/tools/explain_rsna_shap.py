#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.datasets.rsna_ich_dataset import (  # noqa: E402
    RSNA_CLASSES,
    RsnaSliceRecord,
    iter_rsna_stage2_records_from_csv,
    preprocessed_db_existing_keys,
    read_rsna_preprocessed_to_tensor2d,
    read_rsna_preprocessed_to_tensor25d,
)


DEFAULT_ALL_TARGETS = tuple(str(x) for x in RSNA_CLASSES)


class DeviceMode(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class InputNormalizeMode(str, Enum):
    auto = "auto"
    imagenet = "imagenet"
    none = "none"


class OutputSpace(str, Enum):
    logit = "logit"
    probability = "probability"


class BackgroundMode(str, Enum):
    mixed = "mixed"
    negatives = "negatives"
    positives = "positives"


@dataclass(frozen=True)
class ExplainConfig:
    rsna_root: Path
    preprocessed_root: Path
    preprocessed_db: Path
    arch: str
    pretrained: bool
    first_conv_init: str
    image_size: int
    windows: str
    preprocess: str
    stack_slices: int
    input_normalize: InputNormalizeMode
    classes: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _resolve_device(mode: DeviceMode) -> torch.device:
    if mode == DeviceMode.cpu:
        return torch.device("cpu")
    if mode == DeviceMode.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    if mode == DeviceMode.mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_windows(windows: str) -> list[str]:
    return [part.strip() for part in str(windows).split(";") if part.strip()]


def _channel_labels(windows: str, stack_slices: int) -> list[str]:
    win_labels = _parse_windows(windows)
    if not win_labels:
        win_labels = ["default"]
    if int(stack_slices) <= 1:
        slice_offsets = [0]
    else:
        half = int(stack_slices) // 2
        slice_offsets = list(range(-half, half + 1))
    labels: list[str] = []
    for offset in slice_offsets:
        slice_tag = f"z{offset:+d}"
        for wi, window in enumerate(win_labels):
            labels.append(f"{slice_tag}|w{wi + 1}:{window}")
    return labels


def _needs_imagenet_norm(arch: str, pretrained: bool) -> bool:
    if not bool(pretrained):
        return False
    return str(arch).strip().lower() in {"resnet18", "efficientnet_b0", "convnext_tiny"}


def _imagenet_norm_stats(channels: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean3 = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
    std3 = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
    rep = int((int(channels) + 2) // 3)
    mean = mean3.repeat(rep)[: int(channels)]
    std = std3.repeat(rep)[: int(channels)]
    return mean.view(1, int(channels), 1, 1), std.view(1, int(channels), 1, 1)


def _parse_input_normalize_mode(value: str | InputNormalizeMode) -> InputNormalizeMode:
    if isinstance(value, InputNormalizeMode):
        return value
    raw = str(value).strip()
    if raw.startswith("InputNormalizeMode."):
        raw = raw.split(".", 1)[1]
    return InputNormalizeMode(raw)


def _build_model(arch: str, in_channels: int, pretrained: bool, first_conv_init: str) -> torch.nn.Module:
    arch_s = str(arch).strip().lower()
    import torchvision  # type: ignore[import-not-found]

    from src.models.input_adapters import adapt_first_conv

    if arch_s == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if bool(pretrained) else None
        model = torchvision.models.resnet18(weights=weights)
        model.conv1 = adapt_first_conv(model.conv1, int(in_channels), init_mode=str(first_conv_init))
        model.fc = torch.nn.Linear(model.fc.in_features, len(RSNA_CLASSES))
        return model

    if arch_s == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if bool(pretrained) else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        model.features[0][0] = adapt_first_conv(model.features[0][0], int(in_channels), init_mode=str(first_conv_init))
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(RSNA_CLASSES))
        return model

    if arch_s == "convnext_tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if bool(pretrained) else None
        model = torchvision.models.convnext_tiny(weights=weights)
        model.features[0][0] = adapt_first_conv(model.features[0][0], int(in_channels), init_mode=str(first_conv_init))
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(RSNA_CLASSES))
        return model

    raise ValueError(f"Unsupported arch: {arch}")


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) for k in obj.keys()):
            tensor_values = sum(isinstance(v, torch.Tensor) for v in obj.values())
            if tensor_values > 0:
                return {str(k): v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        for key in ("state_dict", "model_state_dict", "model"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                return _unwrap_state_dict(inner)
    raise ValueError("Checkpoint does not contain a state_dict")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _resolve_paths(meta: dict[str, Any], rsna_root: Path | None, preprocessed_root: Path | None) -> tuple[Path, Path]:
    meta_root_raw = str(meta.get("rsna_root") or "").strip()
    meta_root = Path(meta_root_raw).expanduser().resolve() if meta_root_raw else None

    resolved_rsna_root = rsna_root.expanduser().resolve() if rsna_root is not None else None
    if resolved_rsna_root is None and meta_root is not None:
        if (meta_root / "stage_2_train.csv").exists():
            resolved_rsna_root = meta_root
        elif (meta_root / "rsna_meta" / "stage_2_train.csv").exists():
            resolved_rsna_root = meta_root / "rsna_meta"
        elif meta_root.name == "rsna_meta":
            resolved_rsna_root = meta_root

    resolved_pre_root = preprocessed_root.expanduser().resolve() if preprocessed_root is not None else None
    if resolved_pre_root is None and meta_root is not None:
        if (meta_root / "train.sqlite").exists():
            resolved_pre_root = meta_root
        elif (meta_root.parent / "train.sqlite").exists():
            resolved_pre_root = meta_root.parent

    if resolved_rsna_root is None:
        raise FileNotFoundError("Could not infer rsna_root. Pass --rsna-root explicitly.")
    if resolved_pre_root is None:
        raise FileNotFoundError("Could not infer preprocessed_root. Pass --preprocessed-root explicitly.")
    return resolved_rsna_root, resolved_pre_root


def _load_config(meta_path: Path, rsna_root: Path | None, preprocessed_root: Path | None) -> ExplainConfig:
    meta = _load_json(meta_path)
    resolved_rsna_root, resolved_pre_root = _resolve_paths(meta, rsna_root, preprocessed_root)
    pre_db = resolved_pre_root / "train.sqlite"
    if not pre_db.exists():
        raise FileNotFoundError(f"Missing preprocessed DB: {pre_db}")
    return ExplainConfig(
        rsna_root=resolved_rsna_root,
        preprocessed_root=resolved_pre_root,
        preprocessed_db=pre_db,
        arch=str(meta.get("arch") or "convnext_tiny"),
        pretrained=bool(meta.get("pretrained", True)),
        first_conv_init=str(meta.get("first_conv_init") or "repeat"),
        image_size=int(meta.get("image_size") or 384),
        windows=str(meta.get("windows") or "40,80;80,200;600,2800"),
        preprocess=str(meta.get("preprocess") or "gpt52"),
        stack_slices=int(meta.get("stack_slices") or 1),
        input_normalize=_parse_input_normalize_mode(meta.get("input_normalize") or InputNormalizeMode.auto),
        classes=tuple(str(x) for x in meta.get("classes") or RSNA_CLASSES),
    )


def _lookup_labels(csv_path: Path, image_ids: list[str]) -> dict[str, np.ndarray]:
    import pandas as pd

    wanted = {str(x) for x in image_ids if str(x).strip()}
    labels: dict[str, np.ndarray] = {}
    if not wanted:
        return labels
    for chunk in pd.read_csv(csv_path, usecols=("ID", "Label"), chunksize=200_000):  # type: ignore[call-overload]
        ids = chunk["ID"].astype(str)
        vals = chunk["Label"].astype(np.float32).to_numpy()
        image_col = ids.str.rsplit("_", n=1).str[0]
        class_col = ids.str.rsplit("_", n=1).str[1]
        for image_id, cls_name, val in zip(image_col.to_numpy(), class_col.to_numpy(), vals, strict=False):
            image_id_s = str(image_id)
            if image_id_s not in wanted:
                continue
            try:
                class_idx = RSNA_CLASSES.index(str(cls_name))
            except ValueError:
                continue
            labels.setdefault(image_id_s, np.zeros((len(RSNA_CLASSES),), dtype=np.float32))[class_idx] = float(val)
        if len(labels) == len(wanted):
            break
    return labels


def _load_pool_records(config: ExplainConfig, limit_images: int, seed: int) -> list[RsnaSliceRecord]:
    records = iter_rsna_stage2_records_from_csv(
        csv_path=config.rsna_root / "stage_2_train.csv",
        dicom_dir=config.rsna_root / "stage_2_train",
        limit_images=int(limit_images) if int(limit_images) > 0 else None,
        seed=int(seed),
    )
    existing = preprocessed_db_existing_keys([r.image_id for r in records], db_path=config.preprocessed_db)
    return [r for r in records if r.image_id in existing]


def _filter_records_by_label(records: list[RsnaSliceRecord], class_name: str, positive_only: bool) -> list[RsnaSliceRecord]:
    class_idx = RSNA_CLASSES.index(str(class_name))
    if positive_only:
        return [r for r in records if float(r.y[class_idx]) > 0.5]
    return list(records)


def _select_background_records(records: list[RsnaSliceRecord], class_name: str, mode: BackgroundMode) -> list[RsnaSliceRecord]:
    if mode == BackgroundMode.mixed:
        return list(records)
    class_idx = RSNA_CLASSES.index(str(class_name))
    threshold_cmp = (lambda y: float(y[class_idx]) <= 0.5) if mode == BackgroundMode.negatives else (lambda y: float(y[class_idx]) > 0.5)
    return [r for r in records if threshold_cmp(r.y)]


def _load_tensor(config: ExplainConfig, image_id: str) -> torch.Tensor:
    if int(config.stack_slices) > 1:
        x = read_rsna_preprocessed_to_tensor25d(
            image_id=str(image_id),
            db_path=config.preprocessed_db,
            stack_slices=int(config.stack_slices),
        )
    else:
        x = read_rsna_preprocessed_to_tensor2d(image_id=str(image_id), db_path=config.preprocessed_db)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")
    if int(x.shape[-1]) != int(config.image_size) or int(x.shape[-2]) != int(config.image_size):
        raise ValueError(
            f"Image size mismatch for {image_id}: expected {config.image_size}, got {tuple(int(v) for v in x.shape[-2:])}"
        )
    return x.float()


def _stack_tensors(config: ExplainConfig, image_ids: list[str]) -> torch.Tensor:
    xs = [_load_tensor(config, image_id) for image_id in image_ids]
    return torch.stack(xs, dim=0)


def _apply_input_normalization(x: torch.Tensor, *, arch: str, pretrained: bool, mode: InputNormalizeMode) -> torch.Tensor:
    resolved = mode
    if resolved == InputNormalizeMode.auto:
        resolved = InputNormalizeMode.imagenet if _needs_imagenet_norm(arch, pretrained) else InputNormalizeMode.none
    if resolved == InputNormalizeMode.none:
        return x
    mean, std = _imagenet_norm_stats(int(x.shape[1]), x.device, x.dtype)
    return (x - mean) / std


class _ExplainWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        arch: str,
        pretrained: bool,
        input_normalize: InputNormalizeMode,
        class_idx: int,
        output_space: OutputSpace,
    ) -> None:
        super().__init__()
        self.model = model
        self.arch = str(arch)
        self.pretrained = bool(pretrained)
        self.input_normalize = input_normalize
        self.class_idx = int(class_idx)
        self.output_space = output_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = _apply_input_normalization(
            x,
            arch=self.arch,
            pretrained=self.pretrained,
            mode=self.input_normalize,
        )
        logits = self.model(x_n)
        selected = logits[:, self.class_idx]
        if self.output_space == OutputSpace.probability:
            selected = torch.sigmoid(selected)
        return selected.unsqueeze(1)


def _predict_logits_and_probs(model: torch.nn.Module, x: torch.Tensor, *, arch: str, pretrained: bool, mode: InputNormalizeMode) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(_apply_input_normalization(x, arch=arch, pretrained=pretrained, mode=mode))
        probs = torch.sigmoid(logits)
    return logits.detach().cpu().numpy(), probs.detach().cpu().numpy()


def _prepare_axes_grid(count: int, *, figsize_unit: float = 3.0) -> tuple[plt.Figure, np.ndarray]:
    cols = min(3, max(1, int(count)))
    rows = int(np.ceil(float(count) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_unit * cols, figsize_unit * rows))
    axes_arr = np.array(axes, dtype=object).reshape(-1)
    return fig, axes_arr


def _save_channel_grid(data: np.ndarray, labels: list[str], out_path: Path, *, cmap: str, title: str) -> None:
    fig, axes = _prepare_axes_grid(int(data.shape[0]))
    vmax = None
    if cmap != "gray":
        vmax = float(np.percentile(np.abs(data), 99.0)) if np.any(data) else 1.0
        vmax = max(vmax, 1e-6)
    for idx, ax in enumerate(axes):
        if idx >= int(data.shape[0]):
            ax.axis("off")
            continue
        img = data[idx]
        if cmap == "gray":
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(labels[idx], fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_overlay(base_image: np.ndarray, heatmap: np.ndarray, out_path: Path, *, title: str) -> None:
    vmax = float(np.percentile(np.abs(heatmap), 99.0)) if np.any(heatmap) else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base_image, cmap="gray")
    ax.imshow(heatmap, cmap="coolwarm", alpha=0.5, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_bar(values: np.ndarray, labels: list[str], out_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.4), 3.5))
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _safe_slug(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(out).strip("_") or "item"


def _center_slice_base_image(x: np.ndarray, n_windows: int, stack_slices: int) -> np.ndarray:
    if int(stack_slices) <= 1:
        center = x[: int(n_windows)]
    else:
        half = int(stack_slices) // 2
        start = half * int(n_windows)
        end = start + int(n_windows)
        center = x[start:end]
    return np.mean(center, axis=0)


def _channel_group_summaries(abs_shap: np.ndarray, n_windows: int, stack_slices: int) -> tuple[np.ndarray, np.ndarray]:
    reshaped = abs_shap.reshape(int(stack_slices), int(n_windows), abs_shap.shape[-2], abs_shap.shape[-1])
    slice_scores = reshaped.sum(axis=(1, 2, 3))
    window_scores = reshaped.sum(axis=(0, 2, 3))
    return slice_scores.astype(np.float64), window_scores.astype(np.float64)


def _candidate_records_from_explicit_ids(config: ExplainConfig, image_ids: list[str]) -> list[RsnaSliceRecord]:
    labels = _lookup_labels(config.rsna_root / "stage_2_train.csv", image_ids)
    out: list[RsnaSliceRecord] = []
    for image_id in image_ids:
        if image_id not in labels:
            raise FileNotFoundError(f"Could not find labels for image_id: {image_id}")
        out.append(
            RsnaSliceRecord(
                image_id=str(image_id),
                dcm_path=config.rsna_root / "stage_2_train" / f"{image_id}.dcm",
                y=labels[image_id],
            )
        )
    return out


def _sqlite_key_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COUNT(*) FROM tensors;").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain RSNA ICH checkpoints with SHAP GradientExplainer")
    parser.add_argument("--ckpt", required=True, type=str, help="Model checkpoint (.pt or .pth)")
    parser.add_argument("--meta", default=None, type=str, help="Run meta.json. Default: <ckpt_dir>/meta.json")
    parser.add_argument("--rsna-root", default=None, type=str, help="Directory containing stage_2_train.csv")
    parser.add_argument("--preprocessed-root", default=None, type=str, help="Directory containing train.sqlite")
    parser.add_argument("--out-dir", default=None, type=str, help="Output directory for SHAP artifacts")
    parser.add_argument("--image-id", action="append", default=[], help="Repeat --image-id to explain specific slices")
    parser.add_argument("--target", action="append", default=[], help="Repeat --target to explain multiple classes. Default: any")
    parser.add_argument("--all-targets", action="store_true", help="Explain all RSNA classes in one run")
    parser.add_argument("--output-space", default=OutputSpace.logit.value, choices=[x.value for x in OutputSpace])
    parser.add_argument("--device", default=DeviceMode.auto.value, choices=[x.value for x in DeviceMode])
    parser.add_argument("--max-samples", default=2, type=int)
    parser.add_argument("--candidate-pool", default=512, type=int)
    parser.add_argument("--background-size", default=4, type=int)
    parser.add_argument("--background-mode", default=BackgroundMode.negatives.value, choices=[x.value for x in BackgroundMode])
    parser.add_argument("--select-class", default="any", type=str)
    parser.add_argument("--select-positive-only", dest="select_positive_only", action="store_true")
    parser.add_argument("--select-all", dest="select_positive_only", action="store_false")
    parser.set_defaults(select_positive_only=True)
    parser.add_argument("--shap-nsamples", default=32, type=int)
    parser.add_argument("--seed", default=0, type=int)
    ns = parser.parse_args(argv)

    try:
        import shap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'shap'. Install it in the active Python environment first.") from exc

    torch.manual_seed(int(ns.seed))
    np.random.seed(int(ns.seed))

    ckpt_path = Path(ns.ckpt).expanduser().resolve()
    meta_path = Path(ns.meta).expanduser().resolve() if ns.meta is not None else ckpt_path.parent / "meta.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json: {meta_path}")

    rsna_root = Path(ns.rsna_root).expanduser().resolve() if ns.rsna_root is not None else None
    preprocessed_root = Path(ns.preprocessed_root).expanduser().resolve() if ns.preprocessed_root is not None else None
    out_dir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir is not None else None
    output_space = OutputSpace(str(ns.output_space))
    device = DeviceMode(str(ns.device))
    background_mode = BackgroundMode(str(ns.background_mode))

    config = _load_config(meta_path, rsna_root=rsna_root, preprocessed_root=preprocessed_root)
    run_device = _resolve_device(device)
    if bool(ns.all_targets):
        targets = list(DEFAULT_ALL_TARGETS)
    else:
        targets = [str(t).strip() for t in (ns.target or ["any"]) if str(t).strip()]
    invalid_targets = [t for t in targets if t not in RSNA_CLASSES]
    if invalid_targets:
        raise ValueError(f"Unsupported targets: {invalid_targets}")
    if str(ns.select_class) not in RSNA_CLASSES:
        raise ValueError(f"Unsupported select_class: {ns.select_class}")

    if out_dir is None:
        out_root = _ROOT / "results" / "rsna_shap" / ckpt_path.parent.name
    else:
        out_root = out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if ns.image_id:
        explain_records = _candidate_records_from_explicit_ids(config, list(ns.image_id))
        pool_records = _load_pool_records(config, limit_images=max(int(ns.candidate_pool), int(ns.background_size) * 4), seed=int(ns.seed))
    else:
        pool_records = _load_pool_records(config, limit_images=int(ns.candidate_pool), seed=int(ns.seed))
        selected_pool = _filter_records_by_label(pool_records, class_name=str(ns.select_class), positive_only=bool(ns.select_positive_only))
        if not selected_pool:
            raise RuntimeError("No candidate records matched the current auto-selection settings")
        explain_records = selected_pool[: max(1, int(ns.max_samples))]

    background_candidates = _select_background_records(pool_records, class_name=str(ns.select_class), mode=background_mode)
    explain_ids = {r.image_id for r in explain_records}
    background_candidates = [r for r in background_candidates if r.image_id not in explain_ids]
    if len(background_candidates) < int(ns.background_size):
        fallback = [r for r in pool_records if r.image_id not in explain_ids]
        background_candidates = fallback
    if len(background_candidates) < int(ns.background_size):
        raise RuntimeError("Not enough background candidates for SHAP. Increase --candidate-pool.")

    background_records = background_candidates[: int(ns.background_size)]
    explain_ids_list = [r.image_id for r in explain_records]
    background_ids_list = [r.image_id for r in background_records]

    n_windows = len(_parse_windows(config.windows)) or 1
    expected_channels = int(n_windows) * int(config.stack_slices)
    model = _build_model(
        arch=config.arch,
        in_channels=expected_channels,
        pretrained=config.pretrained,
        first_conv_init=config.first_conv_init,
    )
    state_obj = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = _strip_module_prefix(_unwrap_state_dict(state_obj))
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(
            f"Checkpoint does not match arch={config.arch}: {len(load_result.missing_keys)} missing keys, "
            f"e.g. {list(load_result.missing_keys)[:3]}. Explaining a partly random model would be misleading."
        )
    model.eval()
    model.to(run_device)

    background_x = _stack_tensors(config, background_ids_list).to(run_device)
    explain_x = _stack_tensors(config, explain_ids_list).to(run_device)
    if int(explain_x.shape[1]) != expected_channels:
        raise ValueError(f"Input channel mismatch: expected {expected_channels}, got {int(explain_x.shape[1])}")

    logits, probs = _predict_logits_and_probs(
        model,
        explain_x,
        arch=config.arch,
        pretrained=config.pretrained,
        mode=config.input_normalize,
    )

    channel_labels = _channel_labels(config.windows, config.stack_slices)
    if len(channel_labels) != expected_channels:
        raise ValueError(f"Channel label mismatch: expected {expected_channels}, got {len(channel_labels)}")

    manifest: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "meta": str(meta_path),
        "config": {
            "rsna_root": str(config.rsna_root),
            "preprocessed_root": str(config.preprocessed_root),
            "preprocessed_db": str(config.preprocessed_db),
            "sqlite_keys": _sqlite_key_count(config.preprocessed_db),
            "arch": config.arch,
            "pretrained": config.pretrained,
            "first_conv_init": config.first_conv_init,
            "image_size": config.image_size,
            "windows": config.windows,
            "preprocess": config.preprocess,
            "stack_slices": config.stack_slices,
            "input_normalize": config.input_normalize.value,
            "device": str(run_device),
        },
        "targets": targets,
        "background": {
            "mode": background_mode.value,
            "image_ids": background_ids_list,
        },
        "state_dict": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "samples": [],
    }

    explain_x_cpu = explain_x.detach().cpu().numpy()
    base_images = [_center_slice_base_image(explain_x_cpu[i], n_windows=n_windows, stack_slices=config.stack_slices) for i in range(len(explain_records))]
    slice_names = [f"z{offset:+d}" for offset in (range(-(config.stack_slices // 2), config.stack_slices // 2 + 1) if config.stack_slices > 1 else [0])]
    window_names = [f"w{i + 1}" for i in range(n_windows)]

    for target_name in targets:
        class_idx = RSNA_CLASSES.index(str(target_name))
        wrapped = _ExplainWrapper(
            model,
            arch=config.arch,
            pretrained=config.pretrained,
            input_normalize=config.input_normalize,
            class_idx=class_idx,
            output_space=output_space,
        )
        wrapped.eval()
        explainer = shap.GradientExplainer(wrapped, background_x)
        shap_result = explainer.shap_values(explain_x, nsamples=int(ns.shap_nsamples))
        if isinstance(shap_result, list):
            if len(shap_result) != 1:
                raise ValueError(f"Unexpected SHAP output length: {len(shap_result)}")
            shap_array = shap_result[0]
        else:
            shap_array = shap_result
        shap_array = np.asarray(shap_array)
        if shap_array.ndim == 5 and shap_array.shape[-1] == 1:
            shap_array = shap_array[..., 0]
        if shap_array.shape != explain_x_cpu.shape:
            raise ValueError(f"Unexpected SHAP shape: expected {explain_x_cpu.shape}, got {shap_array.shape}")

        target_root = out_root / _safe_slug(target_name)
        target_root.mkdir(parents=True, exist_ok=True)

        for idx, record in enumerate(explain_records):
            sample_root = target_root / _safe_slug(record.image_id)
            sample_root.mkdir(parents=True, exist_ok=True)
            sample_input = explain_x_cpu[idx]
            sample_shap = shap_array[idx]
            sample_abs = np.abs(sample_shap)
            overlay_signed = np.sum(sample_shap, axis=0)
            overlay_abs = np.sum(sample_abs, axis=0)
            slice_scores, window_scores = _channel_group_summaries(sample_abs, n_windows=n_windows, stack_slices=config.stack_slices)

            np.savez_compressed(
                sample_root / "arrays.npz",
                input=sample_input.astype(np.float32),
                shap=sample_shap.astype(np.float32),
                shap_abs=sample_abs.astype(np.float32),
                logits=logits[idx].astype(np.float32),
                probs=probs[idx].astype(np.float32),
                label=np.asarray(record.y, dtype=np.float32),
            )

            _save_channel_grid(sample_input, channel_labels, sample_root / "input_channels.png", cmap="gray", title=f"Input channels: {record.image_id}")
            _save_channel_grid(sample_abs, channel_labels, sample_root / "shap_abs_channels.png", cmap="hot", title=f"|SHAP| channels: {record.image_id}")
            _save_overlay(base_images[idx], overlay_signed, sample_root / "overlay_signed.png", title=f"Signed SHAP: {record.image_id} / {target_name}")
            _save_overlay(base_images[idx], overlay_abs, sample_root / "overlay_abs.png", title=f"Absolute SHAP: {record.image_id} / {target_name}")
            _save_bar(slice_scores, slice_names, sample_root / "slice_contrib.png", title="Absolute SHAP by slice offset")
            _save_bar(window_scores, window_names, sample_root / "window_contrib.png", title="Absolute SHAP by CT window")

            sample_summary = {
                "image_id": record.image_id,
                "target": target_name,
                "output_space": output_space.value,
                "label": [float(x) for x in record.y.tolist()],
                "logits": [float(x) for x in logits[idx].tolist()],
                "probs": [float(x) for x in probs[idx].tolist()],
                "target_logit": float(logits[idx][class_idx]),
                "target_prob": float(probs[idx][class_idx]),
                "slice_abs_scores": [float(x) for x in slice_scores.tolist()],
                "window_abs_scores": [float(x) for x in window_scores.tolist()],
                "files": {
                    "arrays": str(sample_root / "arrays.npz"),
                    "input_channels": str(sample_root / "input_channels.png"),
                    "shap_abs_channels": str(sample_root / "shap_abs_channels.png"),
                    "overlay_signed": str(sample_root / "overlay_signed.png"),
                    "overlay_abs": str(sample_root / "overlay_abs.png"),
                    "slice_contrib": str(sample_root / "slice_contrib.png"),
                    "window_contrib": str(sample_root / "window_contrib.png"),
                },
            }
            (sample_root / "summary.json").write_text(json.dumps(sample_summary, indent=2))
            manifest["samples"].append(sample_summary)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "out_dir": str(out_root),
        "samples": len(explain_records),
        "targets": targets,
        "device": str(run_device),
        "state_dict_missing_keys": list(load_result.missing_keys),
        "state_dict_unexpected_keys": list(load_result.unexpected_keys),
        "image_ids": explain_ids_list,
        "background_ids": background_ids_list,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())