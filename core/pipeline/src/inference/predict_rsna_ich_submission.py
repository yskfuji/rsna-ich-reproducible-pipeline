from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import torch
import typer

from ..datasets.rsna_ich_dataset import (
    RSNA_CLASSES,
    read_rsna_dicom_to_tensor2d,
    read_rsna_preprocessed_to_tensor2d,
    read_rsna_preprocessed_to_tensor25d,
    preprocessed_db_has_meta,
)
from ..models.mc_dropout import enable_dropout_only, inject_stage_and_head_dropout

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
    mean3 = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)
    std3 = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)
    c = int(channels) if int(channels) > 0 else 3
    rep = int((c + 2) // 3)
    mean = mean3.repeat(rep)[:c]
    std = std3.repeat(rep)[:c]
    return mean.view(1, c, 1, 1), std.view(1, c, 1, 1)


def _device() -> torch.device:
    import os

    dev = os.environ.get("TORCH_DEVICE", "cpu").strip().lower()
    if dev in {"cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model(arch: str, in_channels: int, pretrained: bool) -> torch.nn.Module:
    arch_s = str(arch).strip().lower()

    if arch_s == "resnet18":
        import torchvision  # type: ignore[import-not-found]
        from ..models.input_adapters import adapt_first_conv

        weights = torchvision.models.ResNet18_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.resnet18(weights=weights)
        old = m.conv1
        m.conv1 = adapt_first_conv(old, int(in_channels), init_mode="repeat")
        m.fc = torch.nn.Linear(m.fc.in_features, len(RSNA_CLASSES))
        return m

    if arch_s == "efficientnet_b0":
        import torchvision  # type: ignore[import-not-found]
        from ..models.input_adapters import adapt_first_conv

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.efficientnet_b0(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode="repeat")
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(RSNA_CLASSES))
        return m

    if arch_s == "convnext_tiny":
        import torchvision  # type: ignore[import-not-found]
        from ..models.input_adapters import adapt_first_conv

        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if bool(pretrained) else None
        m = torchvision.models.convnext_tiny(weights=weights)
        old = m.features[0][0]
        m.features[0][0] = adapt_first_conv(old, int(in_channels), init_mode="repeat")
        m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, len(RSNA_CLASSES))
        return m

    raise ValueError(f"Unsupported arch: {arch}. Use resnet18 | efficientnet_b0 | convnext_tiny")


def _parse_windows(windows: str) -> int:
    s = str(windows).strip()
    if not s:
        return 1
    return len([p for p in s.split(";") if p.strip()])


def _sigmoid(x: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    x64 = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x64))).astype(np.float64)


def _read_series_and_instance(dcm_path: Path) -> tuple[str, int] | None:
    """Read minimal metadata needed for 2.5D neighboring-slice lookup."""
    import pydicom  # lazy import
    import warnings

    warnings.filterwarnings("ignore", message=r"Invalid value for VR UI:.*", module=r"pydicom\..*")

    try:
        ds = pydicom.dcmread(
            str(dcm_path),
            stop_before_pixels=True,
            specific_tags=["SeriesInstanceUID", "InstanceNumber"],
            force=True,
        )
    except Exception:
        return None

    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
    if not series_uid:
        return None
    inst = getattr(ds, "InstanceNumber", None)
    try:
        inst_i = int(inst) if inst is not None else -1
    except Exception:
        inst_i = -1
    return series_uid, inst_i


def _build_test_index(test_dir: Path) -> tuple[dict[str, tuple[str, int] | None], dict[str, dict[int, Path]]]:
    """Build {image_id -> (series_uid, instance)} and {series_uid -> {instance -> path}}."""
    meta: dict[str, tuple[str, int] | None] = {}
    series_index: dict[str, dict[int, Path]] = {}

    # Iterate files once (stop_before_pixels) to keep it reasonably fast.
    for p in test_dir.iterdir():
        if p.suffix.lower() != ".dcm":
            continue
        img_id = p.stem
        m = _read_series_and_instance(p)
        meta[img_id] = m
        if m is None:
            continue
        series_uid, inst = m
        series_index.setdefault(series_uid, {})[int(inst)] = p

    return meta, series_index


@app.command()
def main(
    rsna_root: Path = typer.Option(..., help="RSNA dataset root (contains stage_2_test/ and stage_2_sample_submission.csv)"),
    preprocessed_root: Path | None = typer.Option(
        None,
        help="If set, read preprocessed tensors from <preprocessed_root>/test.sqlite and do not read DICOM pixels. "
        "rsna_root is still used for stage_2_sample_submission.csv.",
    ),
    ckpt: list[Path] = typer.Option(
        None, help="One or more model checkpoints (.pt). Repeat --ckpt to ensemble (single-arch)."
    ),
    model: list[str] = typer.Option(
        None,
        help="Heterogeneous ensemble. Repeat --model as 'ARCH:CKPT_PATH'. Example: --model efficientnet_b0:best.pt --model convnext_tiny:best.pt",
    ),
    out_csv: Path = typer.Option(Path("submission.csv"), help="Output submission.csv"),
    arch: str = typer.Option("resnet18", help="resnet18 | efficientnet_b0 | convnext_tiny"),
    pretrained: bool = typer.Option(False, help="Only affects model construction if ckpt missing keys"),
    image_size: int = typer.Option(256, help="Resize input"),
    windows: str = typer.Option("40,80;80,200;600,2800", help="CT windows"),
    preprocess: str = typer.Option(
        "legacy",
        help="RSNA DICOM preprocessing mode: legacy | gpt52. Must match training.",
    ),
    stack_slices: int = typer.Option(1, help="2.5D: odd number of slices to stack as channels (1=2D)"),
    cache_dir: Path | None = typer.Option(None, help="Optional on-disk cache for decoded+windowed+resized tensors (.pt)."),
    batch_size: int = typer.Option(32, help="Inference batch size"),
    max_test_images: int = typer.Option(
        0,
        help="For smoke: only run model on first N unique image_ids; remaining rows are filled with 0.5. 0=all.",
    ),
    tta_hflip: bool = typer.Option(False, help="Test-time augmentation: horizontal flip"),
    input_normalize: InputNormalizeMode = typer.Option(
        InputNormalizeMode.auto,
        help="Input normalization: auto|imagenet|none. auto=ImageNet mean/std when using torchvision ImageNet backbones.",
    ),
    enforce_any_max: bool = typer.Option(True, help="Set p(any)=max(p(any), max(p(subtypes)))"),
    mc_samples: int = typer.Option(
        30,
        help="(MC-Dropout) Number of stochastic forward passes (T). Activated only when mc_dropout_stage_p>0 or mc_dropout_head_p>0.",
    ),
    mc_seed: int = typer.Option(0, help="(MC-Dropout) Random seed for MC sampling (dropout RNG)."),
    mc_dropout_stage_p: float = typer.Option(
        0.0,
        help="(MC-Dropout) Dropout2d probability applied right after the last feature stage (e.g., stage4). 0=disable.",
    ),
    mc_dropout_head_p: float = typer.Option(
        0.0,
        help="(MC-Dropout) Dropout probability applied before the classifier head (Linear). 0=disable.",
    ),
    out_uncertainty_csv: Path | None = typer.Option(
        None,
        help="If set and MC-Dropout is enabled, write per-row uncertainty as probability std (same ID format as submission).",
    ),
):
    rsna_root = rsna_root.expanduser().resolve()
    test_dir = rsna_root / "stage_2_test"
    sample_path = rsna_root / "stage_2_sample_submission.csv"
    pre_db: Path | None = None
    if preprocessed_root is not None:
        pr = Path(preprocessed_root).expanduser().resolve()
        pre_db = pr / "test.sqlite"
        if not pre_db.exists():
            raise FileNotFoundError(f"Missing preprocessed DB: {pre_db}")
    else:
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing: {test_dir}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing: {sample_path}")

    model_specs = [s for s in (model or []) if str(s).strip()]
    ckpts_in = [p for p in (ckpt or [])]
    if not model_specs and not ckpts_in:
        raise typer.BadParameter("Provide either --ckpt (repeatable) or --model (repeatable).")

    ckpts = [p.expanduser().resolve() for p in ckpts_in]
    out_csv = out_csv.expanduser().resolve()

    stack = int(stack_slices)
    if stack != 1 and (stack < 3 or stack % 2 != 1):
        raise typer.BadParameter("stack_slices must be 1 or an odd number >= 3")
    if pre_db is not None and stack != 1 and not preprocessed_db_has_meta(pre_db):
        raise typer.BadParameter(
            "When using --preprocessed-root with stack_slices!=1, the DB must include meta table. "
            "Rebuild/backfill via tools/precompute_rsna_preprocessed_sqlite.py (newer version)."
        )

    cache_dir_r: Path | None = None
    if cache_dir is not None:
        cache_dir_r = Path(cache_dir).expanduser().resolve()

    n_win = _parse_windows(windows)
    in_channels = int(n_win) * int(stack)

    dev = _device()

    mean_t: torch.Tensor | None = None
    std_t: torch.Tensor | None = None

    norm_mode = _parse_input_normalize_mode(input_normalize)
    if norm_mode == InputNormalizeMode.imagenet:
        mean_t, std_t = _imagenet_norm_stats(int(in_channels), device=dev, dtype=torch.float32)
    elif norm_mode == InputNormalizeMode.auto:
        # Infer from arch names (do not depend on --pretrained to avoid train/infer mismatch).
        if model_specs:
            try:
                spec_arches = [s.split(":", 1)[0] for s in model_specs]
            except Exception:
                spec_arches = []
            do_norm = bool(spec_arches) and all(_needs_imagenet_norm(a, pretrained=True) for a in spec_arches)
        else:
            do_norm = _needs_imagenet_norm(str(arch), pretrained=True)
        if do_norm:
            mean_t, std_t = _imagenet_norm_stats(int(in_channels), device=dev, dtype=torch.float32)
    models: list[torch.nn.Module] = []
    mc_n_req = int(mc_samples)
    if mc_n_req < 0:
        raise typer.BadParameter("mc_samples must be >= 0")
    # Avoid surprise 30x slowdown unless user explicitly enables injected dropout.
    mc_enabled = (float(mc_dropout_stage_p) > 0.0) or (float(mc_dropout_head_p) > 0.0)
    mc_n = int(mc_n_req) if mc_enabled else 0
    if model_specs:
        for spec in model_specs:
            if ":" not in spec:
                raise typer.BadParameter(f"Invalid --model '{spec}'. Use 'ARCH:CKPT_PATH'.")
            a, p_s = spec.split(":", 1)
            p = Path(p_s).expanduser().resolve()
            m = _build_model(arch=a, in_channels=in_channels, pretrained=pretrained)
            # Inject dropout modules (no params) + forward override for MC-Dropout.
            if mc_n >= 2:
                m = inject_stage_and_head_dropout(
                    m,
                    arch=str(a),
                    p_stage=float(mc_dropout_stage_p),
                    p_head=float(mc_dropout_head_p),
                )
            sd = torch.load(str(p), map_location="cpu")
            m.load_state_dict(sd, strict=True)
            m.to(dev)
            m.eval()
            models.append(m)
    else:
        for p in ckpts:
            m = _build_model(arch=arch, in_channels=in_channels, pretrained=pretrained)
            if mc_n >= 2:
                m = inject_stage_and_head_dropout(
                    m,
                    arch=str(arch),
                    p_stage=float(mc_dropout_stage_p),
                    p_head=float(mc_dropout_head_p),
                )
            sd = torch.load(str(p), map_location="cpu")
            m.load_state_dict(sd, strict=True)
            m.to(dev)
            m.eval()
            models.append(m)

    # Read sample submission rows, but predict per image_id once
    rows: list[dict[str, Any]] = []
    image_ids: list[str] = []
    with sample_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            # ID format: "<image_id>_<class>" where image_id itself contains an underscore: "ID_xxx"
            image_ids.append(str(r["ID"]).rsplit("_", 1)[0])

    unique_ids = list(dict.fromkeys(image_ids))

    max_n = int(max_test_images)
    if max_n < 0:
        raise typer.BadParameter("max_test_images must be >= 0")
    run_ids = unique_ids if max_n == 0 else unique_ids[:max_n]

    probs_by_image: dict[str, NDArray[np.float64]] = {}
    std_by_image: dict[str, NDArray[np.float64]] = {}

    test_meta = None
    test_series_index = None
    # When using preprocessed DB, neighboring-slice lookup is done via DB meta JOINs,
    # so we don't need to touch stage_2_test DICOM files.
    if pre_db is None and stack != 1:
        print("[index] building test series index (2.5D)...", flush=True)
        test_meta, test_series_index = _build_test_index(test_dir)
        print(f"[index] done: images={len(test_meta)} series={len(test_series_index)}", flush=True)

    expected_chw: tuple[int, int, int] | None = None
    if pre_db is not None:
        expected_chw = (int(in_channels), int(image_size), int(image_size))

    def _predict_batch(ids: list[str]) -> None:
        xs: list[torch.Tensor] = []
        keep: list[str] = []
        for img_id in ids:
            if pre_db is not None:
                try:
                    if stack == 1:
                        x = read_rsna_preprocessed_to_tensor2d(image_id=img_id, db_path=pre_db)
                    else:
                        x = read_rsna_preprocessed_to_tensor25d(
                            image_id=img_id,
                            db_path=pre_db,
                            stack_slices=int(stack),
                        )
                except Exception:
                    probs_by_image[img_id] = np.full((len(RSNA_CLASSES),), 0.5, dtype=np.float64)
                    continue
                if expected_chw is not None:
                    if tuple(int(v) for v in x.shape) != expected_chw:
                        raise ValueError(
                            f"Preprocessed tensor shape mismatch for {img_id}: got {tuple(int(v) for v in x.shape)} expected {expected_chw}. "
                            "Rebuild preprocessed DB with matching --windows/--image-size/--stack-slices." 
                        )
            else:
                p = test_dir / f"{img_id}.dcm"
                if not p.exists():
                    # Some environments may not have all files; fill 0.5
                    probs_by_image[img_id] = np.full((len(RSNA_CLASSES),), 0.5, dtype=np.float64)
                    continue

                if stack == 1:
                    x = read_rsna_dicom_to_tensor2d(
                        p,
                        out_size=int(image_size),
                        windows=str(windows),
                        preprocess=str(preprocess),
                        cache_dir=cache_dir_r,
                        cache_key=img_id,
                    )
                else:
                    assert test_meta is not None and test_series_index is not None
                    m = test_meta.get(img_id)
                    if m is None:
                        x0 = read_rsna_dicom_to_tensor2d(
                            p,
                            out_size=int(image_size),
                            windows=str(windows),
                            preprocess=str(preprocess),
                            cache_dir=cache_dir_r,
                            cache_key=img_id,
                        )
                        xs2 = [x0] * stack
                    else:
                        series_uid, inst = m
                        series = test_series_index.get(series_uid, {})
                        half = stack // 2
                        xs2 = []
                        for di in range(-half, half + 1):
                            pp = series.get(int(inst) + int(di))
                            if pp is None:
                                pp = p
                            xs2.append(
                                read_rsna_dicom_to_tensor2d(
                                    pp,
                                    out_size=int(image_size),
                                    windows=str(windows),
                                    preprocess=str(preprocess),
                                    cache_dir=cache_dir_r,
                                    cache_key=pp.stem,
                                )
                            )
                    x = torch.cat(xs2, dim=0)
            xs.append(x)
            keep.append(img_id)

        if not xs:
            return
        xb = torch.stack(xs, dim=0).to(dev)
        if mean_t is not None and std_t is not None:
            xb = (xb - mean_t) / std_t
        with torch.no_grad():
            # Collect samples across (models * mc_samples). If mc_samples < 2, this is a single deterministic pass.
            samples: list[NDArray[np.float64]] = []
            if mc_n >= 2:
                torch.manual_seed(int(mc_seed))
                np.random.seed(int(mc_seed))

            for m in models:
                m.eval()
                if mc_n >= 2:
                    enable_dropout_only(m)
                n_pass = mc_n if mc_n >= 2 else 1
                for _ in range(int(n_pass)):
                    logits = m(xb).detach().cpu().float().numpy()
                    p1 = _sigmoid(logits)
                    if bool(tta_hflip):
                        xb_flip = torch.flip(xb, dims=[3])
                        logits_f = m(xb_flip).detach().cpu().float().numpy()
                        p_f = _sigmoid(logits_f)
                        p1 = 0.5 * (p1 + p_f)
                    if bool(enforce_any_max):
                        any_i = int(RSNA_CLASSES.index("any"))
                        p1[:, any_i] = np.maximum(p1[:, any_i], p1[:, :any_i].max(axis=1))
                    samples.append(p1.astype(np.float64))

            samp = np.stack(samples, axis=0)  # (S, B, C)
            p = samp.mean(axis=0)
            p_std = samp.std(axis=0)
        if bool(enforce_any_max):
            any_i = int(RSNA_CLASSES.index("any"))
            p[:, any_i] = np.maximum(p[:, any_i], p[:, :any_i].max(axis=1))
        for img_id, pi in zip(keep, p, strict=False):
            probs_by_image[img_id] = pi.astype(np.float64)
        if mc_n >= 2:
            for img_id, si in zip(keep, p_std, strict=False):
                std_by_image[img_id] = si.astype(np.float64)

    bs = int(batch_size)
    for i in range(0, len(run_ids), bs):
        _predict_batch(run_ids[i : i + bs])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Label"])
        writer.writeheader()
        for r in rows:
            full_id = str(r["ID"])
            img, cls = full_id.rsplit("_", 1)
            try:
                cls_i = int(RSNA_CLASSES.index(cls))
            except ValueError:
                cls_i = 0
            pr = float(probs_by_image.get(img, np.full((len(RSNA_CLASSES),), 0.5))[cls_i])
            writer.writerow({"ID": full_id, "Label": pr})

    if out_uncertainty_csv is not None:
        if mc_n < 2:
            raise typer.BadParameter("--out-uncertainty-csv requires --mc-samples >= 2")
        out_u = Path(out_uncertainty_csv).expanduser().resolve()
        out_u.parent.mkdir(parents=True, exist_ok=True)
        with out_u.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "ProbStd"])
            writer.writeheader()
            for r in rows:
                full_id = str(r["ID"])
                img, cls = full_id.rsplit("_", 1)
                try:
                    cls_i = int(RSNA_CLASSES.index(cls))
                except ValueError:
                    cls_i = 0
                sd = float(std_by_image.get(img, np.zeros((len(RSNA_CLASSES),), dtype=np.float64))[cls_i])
                writer.writerow({"ID": full_id, "ProbStd": sd})
        print(f"Wrote: {out_u}")

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    app()
