"""Minimal evaluation script using sliding window and Dice."""
from pathlib import Path
import json
import gc
import numpy as np
import torch
import typer
from typing import Tuple, Iterable, Optional, cast
from numpy.typing import NDArray
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import zoom as nd_zoom
from ..datasets.isles_dataset import IslesVolumeDataset
from ..models.unet_3d import UNet3D
from ..inference.infer_sliding_window import sliding_window_inference_3d
from .metrics_segmentation import dice_score
from ..training.utils_train import prepare_device

app = typer.Typer(add_completion=False)


_STRUCT_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _safe_zooms_xyz(zooms_xyz: list[float] | tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if zooms_xyz is None:
        return None
    try:
        x, y, z = float(zooms_xyz[0]), float(zooms_xyz[1]), float(zooms_xyz[2])
    except Exception:
        return None
    if (not np.isfinite(x)) or (not np.isfinite(y)) or (not np.isfinite(z)):
        return None
    if x <= 0 or y <= 0 or z <= 0:
        return None
    return (x, y, z)


def _voxel_volume_mm3(zooms_xyz: list[float] | tuple[float, float, float] | None) -> float | None:
    z = _safe_zooms_xyz(zooms_xyz)
    if z is None:
        return None
    return float(z[0] * z[1] * z[2])


def _surface_mask(mask: NDArray[np.uint8]) -> NDArray[np.bool_]:
    m = (mask > 0).astype(bool, copy=False)
    if not bool(m.any()):
        return np.zeros_like(m, dtype=bool)
    er = binary_erosion(m, structure=_STRUCT_26, border_value=0)
    return (m & (~er)).astype(bool, copy=False)


def _surface_distance_metrics_mm(
    pred: NDArray[np.uint8],
    gt: NDArray[np.uint8],
    zooms_xyz: list[float] | tuple[float, float, float] | None,
) -> dict[str, float | None]:
    """Return symmetric boundary distances in mm: ASSD, HD, HD95.

    Notes:
    - If both masks are empty: distances are 0.
    - If exactly one mask is empty: distances are None.
    """
    pred_any = bool((pred > 0).any())
    gt_any = bool((gt > 0).any())
    if (not pred_any) and (not gt_any):
        return {"assd_mm": 0.0, "hd_mm": 0.0, "hd95_mm": 0.0}
    if pred_any != gt_any:
        return {"assd_mm": None, "hd_mm": None, "hd95_mm": None}

    z = _safe_zooms_xyz(zooms_xyz)
    if z is None:
        sampling_zyx = (1.0, 1.0, 1.0)
    else:
        # arrays are Z,Y,X while zooms are X,Y,Z
        sampling_zyx = (float(z[2]), float(z[1]), float(z[0]))

    sp = _surface_mask(pred)
    sg = _surface_mask(gt)
    if (not bool(sp.any())) or (not bool(sg.any())):
        # unexpected if pred_any==gt_any==True, but be safe
        return {"assd_mm": None, "hd_mm": None, "hd95_mm": None}

    dt_to_gt = distance_transform_edt(~sg, sampling=sampling_zyx)
    dt_to_pred = distance_transform_edt(~sp, sampling=sampling_zyx)
    d1 = dt_to_gt[sp]
    d2 = dt_to_pred[sg]
    if d1.size == 0 or d2.size == 0:
        return {"assd_mm": None, "hd_mm": None, "hd95_mm": None}
    d = np.concatenate([d1.astype(np.float32, copy=False), d2.astype(np.float32, copy=False)], axis=0)
    assd = float(np.mean(d))
    hd = float(np.max(d))
    hd95 = float(np.percentile(d, 95.0))
    return {"assd_mm": assd, "hd_mm": hd, "hd95_mm": hd95}


def _lesionwise_stats(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> dict[str, int]:
    """Compute lesion-wise overlap counts using 26-connectivity.

    Returns:
      n_gt, n_pred, tp_gt (GT lesions overlapped), tp_pred (pred lesions overlapped)
    """
    g = (gt > 0).astype(np.uint8, copy=False)
    p = (pred > 0).astype(np.uint8, copy=False)
    lbl_g, n_gt = cc_label(g, structure=_STRUCT_26)
    lbl_p, n_pred = cc_label(p, structure=_STRUCT_26)

    if n_gt == 0 and n_pred == 0:
        return {"n_gt": 0, "n_pred": 0, "tp_gt": 0, "tp_pred": 0}

    tp_pred = 0
    tp_gt = 0
    if n_pred > 0 and int(g.sum()) > 0:
        ids = np.unique(lbl_p[g > 0])
        tp_pred = int(np.sum(ids > 0))
    if n_gt > 0 and int(p.sum()) > 0:
        ids = np.unique(lbl_g[p > 0])
        tp_gt = int(np.sum(ids > 0))

    return {"n_gt": int(n_gt), "n_pred": int(n_pred), "tp_gt": int(tp_gt), "tp_pred": int(tp_pred)}


def _f1(prec: float | None, rec: float | None) -> float | None:
    if prec is None or rec is None:
        return None
    if (prec + rec) <= 0:
        return 0.0
    return float((2.0 * prec * rec) / (prec + rec))


def _fp_component_sizes(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> list[int]:
    """Return FP component sizes.

    FP component is defined as a connected component in `pred` with zero overlap to `gt`.
    """
    lbl = cc_label(pred)[0].astype(np.int64, copy=False)
    if int(lbl.max()) == 0:
        return []
    sizes = np.bincount(lbl.ravel())
    out: list[int] = []
    for comp_id in range(1, int(lbl.max()) + 1):
        comp_sz = int(sizes[comp_id]) if comp_id < len(sizes) else int((lbl == comp_id).sum())
        if comp_sz <= 0:
            continue
        if int((gt[lbl == comp_id] > 0).sum()) == 0:
            out.append(comp_sz)
    return out


def _fp_component_stats(pred: NDArray[np.uint8], gt: NDArray[np.uint8]) -> tuple[int, int, float | None]:
    """Return (num_fp_components, fp_component_voxels, fp_component_size_p90)."""
    sizes = _fp_component_sizes(pred, gt)
    if not sizes:
        return 0, 0, None
    fp_cc = int(len(sizes))
    fp_cc_vox = int(np.sum(np.asarray(sizes, dtype=np.int64)))
    p90 = float(np.percentile(np.asarray(sizes, dtype=np.float32), 90.0))
    return fp_cc, fp_cc_vox, p90


def _parse_gt_size_bins(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("gt_size_bins must be non-empty")
    bins: list[int] = []
    for p in parts:
        v = int(float(p))
        if v <= 0:
            raise ValueError(f"gt_size_bins must be positive ints, got: {s!r}")
        bins.append(v)
    bins = sorted(set(bins))
    return bins


def _gt_size_bucket_name(gt_vox: int, bins: list[int]) -> str:
    if gt_vox <= 0:
        return "neg"
    if len(bins) == 1:
        return "small" if gt_vox < bins[0] else "large"
    if gt_vox < bins[0]:
        return "small"
    if gt_vox < bins[1]:
        return "medium"
    return "large"


def _parse_slice_spacing_bins_mm(s: str) -> list[float]:
    s = str(s).strip().lower()
    if not s or s in {"none", "off", "false"}:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bins: list[float] = []
    for p in parts:
        v = float(p)
        if (not np.isfinite(v)) or v <= 0:
            raise ValueError(f"slice_spacing_bins_mm must be positive numbers, got: {s!r}")
        bins.append(float(v))
    bins = sorted(set(bins))
    return bins


def _slice_spacing_bucket_name(slice_spacing_mm: float | None, bins: list[float]) -> str:
    if slice_spacing_mm is None or (not np.isfinite(float(slice_spacing_mm))):
        return "unknown"
    if not bins:
        return "all"
    v = float(slice_spacing_mm)
    b0 = float(bins[0])
    if v <= b0:
        return f"le_{b0:g}mm"
    for prev, edge in zip(bins[:-1], bins[1:]):
        p = float(prev)
        e = float(edge)
        if p < v <= e:
            return f"gt_{p:g}_le_{e:g}mm"
    blast = float(bins[-1])
    return f"gt_{blast:g}mm"


def _ensure_required_thresholds(thr_list: list[float], required: Iterable[float] = (0.3, 0.8)) -> list[float]:
    out: list[float] = []
    for t in thr_list:
        tf = float(t)
        if any(abs(tf - x) < 1e-9 for x in out):
            continue
        out.append(tf)
    for r in required:
        rf = float(r)
        if not any(abs(rf - x) < 1e-9 for x in out):
            out.append(rf)
    return out


def _keep_top_k_components(pred: NDArray[np.uint8], k: int) -> NDArray[np.uint8]:
    """Keep only the k largest connected components in `pred`.

    Args:
        pred: binary mask (Z,Y,X) in {0,1}
        k: number of components to keep. k<=0 disables.
    """
    kk = int(k)
    if kk <= 0:
        return pred
    if pred.max() == 0:
        return pred
    lbl = cc_label(pred)[0].astype(np.int64, copy=False)
    n = int(lbl.max())
    if n <= 0:
        return pred
    sizes = np.bincount(lbl.ravel())
    if sizes.shape[0] <= 1:
        return pred

    # component ids are 1..n, background is 0
    comp_ids = np.arange(1, sizes.shape[0], dtype=np.int64)
    comp_sizes = sizes[1:]
    if comp_ids.size == 0:
        return pred
    if comp_ids.size <= kk:
        return pred

    keep_ids = comp_ids[np.argsort(comp_sizes)[::-1][:kk]]
    keep_mask = np.isin(lbl, keep_ids)
    return keep_mask.astype(np.uint8)


def _filter_components_by_score(
    pred: NDArray[np.uint8],
    probs: NDArray[np.float32],
    score_mode: str,
    score_thr: float,
) -> NDArray[np.uint8]:
    """Remove connected components with weak probability support.

    Applied after thresholding (and optionally after min_size filtering).

    Args:
        pred: binary mask (Z,Y,X) in {0,1}
        probs: probability map (Z,Y,X) in [0,1]
        score_mode: component score definition
        score_thr: keep component iff score >= score_thr
    """
    score_mode = str(score_mode).strip().lower()  # type: ignore[assignment]
    if score_mode in {"none", "off", "false"}:
        return pred
    if pred.max() == 0:
        return pred
    if probs.shape != pred.shape:
        raise ValueError(f"probs shape {probs.shape} must match pred shape {pred.shape}")

    lbl = cc_label(pred)[0].astype(np.int64, copy=False)
    if lbl.max() == 0:
        return pred

    score_thr_f = float(score_thr)
    out = pred.copy()
    for comp_id in range(1, int(lbl.max()) + 1):
        comp_mask = lbl == comp_id
        if not np.any(comp_mask):
            continue
        vals = probs[comp_mask]
        if vals.size == 0:
            out[comp_mask] = 0
            continue
        if score_mode == "max_prob":
            score = float(vals.max())
        elif score_mode == "mean_prob":
            score = float(vals.mean())
        elif score_mode == "p95_prob":
            score = float(np.percentile(vals, 95.0))
        else:
            raise ValueError(f"Unknown cc score mode: {score_mode!r}")
        if score < score_thr_f:
            out[comp_mask] = 0
    return out


def _resample_to_max_zoom_mm(
    vol_czyx: NDArray[np.float32],
    mask_zyx: NDArray[np.float32],
    zooms_mm_xyz: list[float] | tuple[float, float, float],
    target_mm: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[float], int]:
    """Resample volume+mask along the axis with the maximum voxel size to `target_mm`.

    IMPORTANT: In this codebase, volumes are (C, Z, Y, X) and masks are (Z, Y, X),
    while NIfTI zooms are provided in (X, Y, Z) order.

    This function takes `zooms_mm_xyz` in (X, Y, Z) and applies the corresponding
    zoom factor to the matching spatial axis in (Z, Y, X).

    Uses linear interpolation for image and nearest for mask.
    """
    z = [float(x) for x in zooms_mm_xyz]
    if len(z) != 3:
        raise ValueError(f"Expected zooms_mm_xyz with 3 values, got: {zooms_mm_xyz!r}")
    tgt = float(target_mm)
    if (not np.isfinite(tgt)) or tgt <= 0:
        raise ValueError(f"target_mm must be a positive finite number, got: {target_mm!r}")

    axis_xyz = int(np.argmax(np.asarray(z, dtype=np.float32)))
    old = float(z[axis_xyz])
    if (not np.isfinite(old)) or old <= 0:
        return vol_czyx, mask_zyx, z, axis_xyz
    if abs(old - tgt) < 1e-9:
        return vol_czyx, mask_zyx, z, axis_xyz

    # Build ndimage.zoom factors for arrays in (Z, Y, X) order.
    # axis mapping: xyz -> zyx
    #   X (0) -> X (last)
    #   Y (1) -> Y (middle)
    #   Z (2) -> Z (first)
    factors_zyx = [1.0, 1.0, 1.0]
    scale = old / tgt
    if axis_xyz == 0:
        factors_zyx[2] = scale
    elif axis_xyz == 1:
        factors_zyx[1] = scale
    else:
        factors_zyx[0] = scale

    # resample image per-channel for stability
    chs: list[NDArray[np.float32]] = []
    for c in range(int(vol_czyx.shape[0])):
        ch = nd_zoom(vol_czyx[c], zoom=factors_zyx, order=1)
        chs.append(ch.astype(np.float32, copy=False))
    vol_rs = np.stack(chs, axis=0).astype(np.float32, copy=False)

    mask_rs = nd_zoom(mask_zyx, zoom=factors_zyx, order=0).astype(np.float32, copy=False)

    z_new = list(z)
    z_new[axis_xyz] = tgt
    return vol_rs, mask_rs, z_new, axis_xyz


@app.command()
def main(
    model_path: Optional[str] = typer.Option(
        None,
        help="model checkpoint (required unless --probs-dir is set)",
    ),
    csv_path: str = typer.Option(..., help="split csv"),
    root: str = typer.Option(..., help="processed root"),
    split: str = typer.Option("val", help="val or test"),
    out_dir: str = typer.Option("results", help="output directory"),
    patch_size: str = typer.Option("96,96,64", help="patch size"),
    overlap: float = typer.Option(0.5, help="overlap"),
    base_ch: Optional[int] = typer.Option(None, help="base channels; auto-infer from checkpoint when omitted"),
    deep_supervision: bool = typer.Option(False, help="checkpoint uses deep supervision heads"),
    thresholds: str = typer.Option(
        "0.5",
        help=(
            "comma-separated thresholds for Dice (after sigmoid), or special values: "
            "'from_run_best'/'from_run_last' to load the val threshold saved during training"
        ),
    ),
    min_size: int = typer.Option(0, help="remove connected components smaller than this (0 disables)"),
    cc_score: str = typer.Option(
        "none",
        help=(
            "connected component score filter (applied after threshold and min_size): "
            "none | max_prob | p95_prob | mean_prob"
        ),
    ),
    cc_score_thr: float = typer.Option(
        0.5,
        help="component score threshold; keep component iff score>=thr (used when --cc-score != none)",
    ),
    top_k: int = typer.Option(
        0,
        help="keep only top-k largest connected components after filtering (0 disables)",
    ),
    temperature: str = typer.Option(
        "1.0",
        help=(
            "temperature scaling for logits before sigmoid. "
            "Pass a number (e.g., 1.0) or 'from_run_best'/'from_run_last' to load temperature saved next to checkpoint."
        ),
    ),
    normalize: str = typer.Option(
        "legacy_zscore",
        help=(
            "input normalization: legacy_zscore | nonzero_zscore | fixed_nonzero_zscore | robust_nonzero_zscore | nonzero_minmax01 | none"
        ),
    ),
    allow_missing_label: bool = typer.Option(False, help="treat missing label as all-zero mask (negative case)"),
    tta: str = typer.Option(
        "full",
        help=(
            "test-time augmentation: full (rot90+flip, slow) | flip (flip only) | none (fast). "
            "Default keeps prior behavior."
        ),
    ),
    quiet: bool = typer.Option(False, help="suppress per-case prints"),
    gt_size_bins: str = typer.Option(
        "250,1000",
        help="GT voxel size bins for size-stratified metrics. Default matches ISLES small/medium/large split.",
    ),
    slice_spacing_bins_mm: str = typer.Option(
        "3.0",
        help=(
            "Slice/z spacing bins in mm for stratified metrics (based on max NIfTI zoom). "
            "Default 3.0 splits val z≈2.0 vs 4.8. Set to 'none' to disable."
        ),
    ),
    slice_spacing_source: str = typer.Option(
        "effective",
        help=(
            "Which spacing to use for slice-spacing stratification: "
            "effective (after optional resample) | raw (original)."
        ),
    ),
    resample_max_zoom_mm: float = typer.Option(
        0.0,
        help=(
            "[PR-4] Resample volumes/masks along the axis with the maximum NIfTI zoom to this spacing (mm). "
            "0 disables. Example: 2.0 to resample val 4.8mm -> 2.0mm."
        ),
    ),
    probs_dir: Optional[str] = typer.Option(
        None,
        help=(
            "Use precomputed probability maps instead of running the model. "
            "Directory must contain <case_id>.npz with key 'probs' (Z,Y,X)."
        ),
    ),
    save_probs: bool = typer.Option(
        False,
        help=(
            "Save per-case probability maps (after optional resample and TTA) to out_dir/probs as NPZ. "
            "Useful for ensembling." 
        ),
    ),
    save_probs_dtype: str = typer.Option(
        "float16",
        help="Data type for saved probs: float16 | float32",
    ),
    extra_metrics: bool = typer.Option(
        False,
        help=(
            "Compute ISLES-style extra metrics at the best threshold (by mean Dice) and store them in summary.json: "
            "volume difference, lesion count difference, lesion-wise precision/recall/F1, and boundary distances (ASSD/HD/HD95). "
            "Requires --probs-dir or --save-probs (so probabilities can be reloaded without re-inference)."
        ),
    ),
):
    device = prepare_device()
    if probs_dir is None and not model_path:
        raise ValueError("Either --model-path or --probs-dir must be provided")
    ps_list = [int(x) for x in patch_size.split(",") if x.strip()]
    if len(ps_list) != 3:
        raise ValueError(f"patch_size must have 3 ints (D,H,W), got: {patch_size!r}")
    ps: Tuple[int, int, int] = (ps_list[0], ps_list[1], ps_list[2])
    thresholds_s = thresholds.strip().lower()
    if thresholds_s in {"from_run_best", "from_run_last"}:
        run_dir = Path(model_path).expanduser().resolve().parent
        meta_name = "val_threshold_best.json" if thresholds_s == "from_run_best" else "val_threshold_last.json"
        meta_path = run_dir / meta_name
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_name} not found next to checkpoint. Expected: {meta_path}. "
                "Run training once with the updated script to generate it, or pass --thresholds as numbers."
            )
        meta = json.loads(meta_path.read_text())
        thr_list = [float(meta["val_threshold"])]
    else:
        thr_list = [float(t) for t in thresholds.split(",") if t.strip()]
    if not thr_list:
        raise ValueError("thresholds must be non-empty")

    # PR-0 requirement: always include thr=0.3 and thr=0.8 in outputs.
    thr_list = _ensure_required_thresholds(thr_list, required=(0.3, 0.8))

    temperature_s = str(temperature).strip().lower()
    if temperature_s in {"from_run_best", "from_run_last"}:
        if not model_path:
            raise ValueError("--temperature from_run_* requires --model-path")
        run_dir = Path(model_path).expanduser().resolve().parent
        meta_name = "temperature_best.json" if temperature_s == "from_run_best" else "temperature_last.json"
        meta_path = run_dir / meta_name
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_name} not found next to checkpoint. Expected: {meta_path}. "
                "Run fit_temperature.py to generate it, or pass --temperature as a number."
            )
        meta = json.loads(meta_path.read_text())
        temp = float(meta.get("temperature", 1.0))
    else:
        temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0:
        raise ValueError(f"temperature must be a finite positive number, got: {temp}")
    ds = IslesVolumeDataset(
        csv_path,
        split=split,
        root=root,
        transform=None,
        normalize=normalize,
        allow_missing_label=bool(allow_missing_label),
    )
    # infer input channels from first volume when loading
    first_vol = IslesVolumeDataset(csv_path, split=split, root=root, transform=None, normalize=normalize)[0]["image"]
    in_ch = first_vol.shape[0] if first_vol.ndim == 4 else 1
    model = None
    if probs_dir is None:
        if not model_path:
            raise ValueError("--model-path is required unless --probs-dir is set")
        state = torch.load(model_path, map_location=device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        if base_ch is None:
            w = state_dict.get("enc1.0.weight")
            if w is None:
                raise RuntimeError("Cannot infer base_ch; specify --base-ch explicitly")
            base_ch = int(w.shape[0])

        # Auto-detect deep supervision from checkpoint keys (aux heads).
        # If the user explicitly passes --deep-supervision/--no-deep-supervision, honor it.
        ckpt_has_ds = any(
            (k.startswith("aux2_conv.") or k.startswith("aux3_conv."))
            for k in (state_dict.keys() if hasattr(state_dict, "keys") else [])
        )
        ds_flag = bool(deep_supervision) or bool(ckpt_has_ds)

        model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=base_ch, deep_supervision=ds_flag)
        # If user forces DS=True but the checkpoint has no aux heads, allow missing aux keys.
        strict = not (ds_flag and (not ckpt_has_ds))
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    probs_dir_p: Optional[Path] = None
    if probs_dir is not None:
        probs_dir_p = Path(probs_dir).expanduser().resolve()
        if not probs_dir_p.exists():
            raise FileNotFoundError(f"--probs-dir not found: {probs_dir_p}")

    save_probs_dir_p: Optional[Path] = None
    save_probs_dtype_s = str(save_probs_dtype).strip().lower()
    if save_probs:
        if save_probs_dtype_s not in {"float16", "float32"}:
            raise ValueError("--save-probs-dtype must be float16 or float32")
        save_probs_dir_p = out_dir_p / "probs"
        save_probs_dir_p.mkdir(parents=True, exist_ok=True)

    if extra_metrics and (probs_dir_p is None) and (save_probs_dir_p is None):
        raise ValueError("--extra-metrics requires --probs-dir or --save-probs")

    bins = _parse_gt_size_bins(gt_size_bins)
    ss_bins = _parse_slice_spacing_bins_mm(slice_spacing_bins_mm)
    resample_mm = float(resample_max_zoom_mm)
    if (not np.isfinite(resample_mm)) or resample_mm < 0:
        raise ValueError(f"resample_max_zoom_mm must be >=0, got: {resample_max_zoom_mm}")
    results = []
    # Aggregate stats per threshold.
    gt_pos_flags: list[bool] = []
    dice_by_thr: dict[float, list[float]] = {float(t): [] for t in thr_list}
    det_flags_by_thr: dict[float, list[bool]] = {float(t): [] for t in thr_list}
    alarm_flags_by_thr: dict[float, list[bool]] = {float(t): [] for t in thr_list}
    pred_vox_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}
    tp_vox_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}
    fn_vox_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}
    fp_vox_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}
    fp_cc_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}
    fp_cc_vox_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}
    fp_cc_size_p90_case_by_thr: dict[float, list[float]] = {float(t): [] for t in thr_list}
    fp_cc_sizes_by_thr: dict[float, list[int]] = {float(t): [] for t in thr_list}

    # size-stratified accumulators (by GT volume)
    bucket_names = ["small", "medium", "large", "neg"]
    dice_by_thr_bucket: dict[float, dict[str, list[float]]] = {float(t): {b: [] for b in bucket_names} for t in thr_list}
    tp_by_thr_bucket: dict[float, dict[str, list[int]]] = {float(t): {b: [] for b in bucket_names} for t in thr_list}
    fp_by_thr_bucket: dict[float, dict[str, list[int]]] = {float(t): {b: [] for b in bucket_names} for t in thr_list}
    fn_by_thr_bucket: dict[float, dict[str, list[int]]] = {float(t): {b: [] for b in bucket_names} for t in thr_list}
    det_by_thr_bucket: dict[float, dict[str, list[bool]]] = {float(t): {b: [] for b in bucket_names} for t in thr_list}

    # slice-spacing stratified accumulators (PR-4)
    ss_bucket_names = ["all"] if not ss_bins else [f"le_{float(ss_bins[0]):g}mm"]
    for prev, edge in zip(ss_bins[:-1], ss_bins[1:]):
        ss_bucket_names.append(f"gt_{float(prev):g}_le_{float(edge):g}mm")
    if ss_bins:
        ss_bucket_names.append(f"gt_{float(ss_bins[-1]):g}mm")
    ss_bucket_names.append("unknown")
    dice_by_thr_ss_bucket: dict[float, dict[str, list[float]]] = {float(t): {b: [] for b in ss_bucket_names} for t in thr_list}
    tp_by_thr_ss_bucket: dict[float, dict[str, list[int]]] = {float(t): {b: [] for b in ss_bucket_names} for t in thr_list}
    fp_by_thr_ss_bucket: dict[float, dict[str, list[int]]] = {float(t): {b: [] for b in ss_bucket_names} for t in thr_list}
    fn_by_thr_ss_bucket: dict[float, dict[str, list[int]]] = {float(t): {b: [] for b in ss_bucket_names} for t in thr_list}
    det_by_thr_ss_bucket: dict[float, dict[str, list[bool]]] = {float(t): {b: [] for b in ss_bucket_names} for t in thr_list}

    ss_bucket_counts: dict[str, int] = {b: 0 for b in ss_bucket_names}

    ss_source = str(slice_spacing_source).strip().lower()
    if ss_source not in {"effective", "eff", "raw"}:
        raise ValueError("slice_spacing_source must be one of: effective|raw")

    for sample in ds:
        vol = sample["image"]
        mask_gt = sample["mask"]
        if vol.ndim == 3:
            vol = vol[None, ...]

        case_id = None
        try:
            case_id = str(sample.get("case_id"))
        except Exception:
            case_id = None

        meta = sample.get("meta") or {}
        zooms_raw = None
        if isinstance(meta, dict) and "zooms_mm" in meta:
            try:
                z = meta.get("zooms_mm")
                if isinstance(z, (list, tuple)) and len(z) >= 3:
                    zooms_raw = [float(z[0]), float(z[1]), float(z[2])]
            except Exception:
                zooms_raw = None

        zooms_eff = zooms_raw
        max_zoom_axis = None
        if resample_mm > 0 and zooms_raw is not None and vol.ndim == 4 and mask_gt.ndim == 3:
            try:
                vol, mask_gt, zooms_eff, max_zoom_axis = _resample_to_max_zoom_mm(
                    vol.astype(np.float32, copy=False),
                    mask_gt.astype(np.float32, copy=False),
                    zooms_mm_xyz=zooms_raw,
                    target_mm=resample_mm,
                )
            except Exception:
                # best-effort: if resampling fails for a case, fall back to raw
                zooms_eff = zooms_raw
                max_zoom_axis = None

        if probs_dir_p is not None:
            if not case_id:
                raise ValueError("--probs-dir requires sample['case_id']")
            npz_path = probs_dir_p / f"{case_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing probs for case_id={case_id!r}: {npz_path}")
            with np.load(str(npz_path)) as z:
                probs_zyx = z["probs"]
            probs_zyx = probs_zyx.astype(np.float32, copy=False)
            probs = probs_zyx[None, None, ...]
        else:
            tta_mode = str(tta).strip().lower()
            if tta_mode in {"none", "off", "false", "0"}:
                logits_mean = infer_logits(vol, model, ps, overlap, device)
            elif tta_mode in {"flip"}:
                logits_mean = infer_logits_with_flip_tta(vol, model, ps, overlap, device)
            elif tta_mode in {"full", "on", "true", "1"}:
                # infer_logits_with_tta() returns mean logits over TTA.
                logits_mean = infer_logits_with_tta(vol, model, ps, overlap, device)
            else:
                raise ValueError(f"Unknown --tta mode: {tta!r} (expected: none|flip|full)")
            probs = 1.0 / (1.0 + np.exp(-(logits_mean / float(temp))))

            if save_probs_dir_p is not None and case_id:
                arr = probs[0, 0]
                arr = arr.astype(np.float16 if save_probs_dtype_s == "float16" else np.float32, copy=False)
                np.savez_compressed(
                    str(save_probs_dir_p / f"{case_id}.npz"),
                    probs=arr,
                    zooms_raw=np.array(zooms_raw if zooms_raw is not None else [], dtype=np.float32),
                    zooms_eff=np.array(zooms_eff if zooms_eff is not None else [], dtype=np.float32),
                    resample_max_zoom_mm=np.array([float(resample_mm)], dtype=np.float32),
                )

        gt = (mask_gt > 0.5).astype(np.uint8)
        gt_vox = int(gt.sum())
        gt_pos = gt_vox > 0
        gt_pos_flags.append(gt_pos)
        size_bucket = _gt_size_bucket_name(gt_vox, bins=bins)

        slice_spacing_mm = None
        slice_spacing_mm_raw = None
        if isinstance(meta, dict) and "slice_spacing_mm" in meta:
            try:
                v0 = meta.get("slice_spacing_mm")
                slice_spacing_mm_raw = float(v0) if v0 is not None else None
            except Exception:
                slice_spacing_mm_raw = None

        # Raw spacing: prefer zooms_raw when available (more reliable than stored scalar).
        slice_spacing_mm_raw_from_zooms = slice_spacing_mm_raw
        if zooms_raw is not None:
            try:
                slice_spacing_mm_raw_from_zooms = float(max(zooms_raw))
            except Exception:
                slice_spacing_mm_raw_from_zooms = slice_spacing_mm_raw

        # Effective spacing: after optional resample.
        slice_spacing_mm_eff = slice_spacing_mm_raw_from_zooms
        if zooms_eff is not None:
            try:
                slice_spacing_mm_eff = float(max(zooms_eff))
            except Exception:
                slice_spacing_mm_eff = slice_spacing_mm_raw_from_zooms

        slice_spacing_mm = slice_spacing_mm_eff
        slice_spacing_for_bucket = slice_spacing_mm_eff if ss_source in {"effective", "eff"} else slice_spacing_mm_raw_from_zooms

        ss_bucket = _slice_spacing_bucket_name(slice_spacing_for_bucket, bins=ss_bins)
        if ss_bucket not in ss_bucket_counts:
            ss_bucket_counts["unknown"] = int(ss_bucket_counts.get("unknown", 0)) + 1
            ss_bucket = "unknown"
        else:
            ss_bucket_counts[ss_bucket] = int(ss_bucket_counts.get(ss_bucket, 0)) + 1

        # Compute per-threshold stats.
        dice_map: dict[float, float] = {}
        stats_map: dict[float, dict[str, int | float | bool | None]] = {}
        for thr in thr_list:
            thr = float(thr)
            pred = (probs[0, 0] > thr).astype(np.uint8)
            if min_size and min_size > 0:
                lbl = cc_label(pred)[0]
                lbl = lbl.astype(np.int64, copy=False)
                sizes = np.bincount(lbl.ravel())
                remove = sizes < int(min_size)
                remove[0] = False
                pred[remove[lbl]] = 0

            pred = _filter_components_by_score(
                pred,
                probs[0, 0],
                score_mode=str(cc_score).strip().lower(),
                score_thr=float(cc_score_thr),
            )

            pred = _keep_top_k_components(pred, k=int(top_k))

            pred_vox = int(pred.sum())
            tp_vox = int((pred & gt).sum())
            fp_vox = int((pred & (1 - gt)).sum())
            fn_vox = int(((1 - pred) & gt).sum())
            fp_cc, fp_cc_vox, fp_cc_p90 = _fp_component_stats(pred, gt)
            detected = bool(tp_vox > 0) if gt_pos else None
            alarm = bool(pred_vox > 0) if (not gt_pos) else None

            dsc = float(dice_score(pred, mask_gt))
            dice_map[thr] = dsc
            stats_map[thr] = {
                "pred_vox": pred_vox,
                "tp_vox": tp_vox,
                "fp_vox": fp_vox,
                "fn_vox": fn_vox,
                "fp_cc": int(fp_cc),
                "fp_cc_vox": int(fp_cc_vox),
                "fp_cc_size_p90": fp_cc_p90,
                "detected": bool(detected) if gt_pos else None,
                "alarm": bool(alarm) if (not gt_pos) else None,
            }

            dice_by_thr[thr].append(dsc)
            det_flags_by_thr[thr].append(bool(detected) if gt_pos else True)
            alarm_flags_by_thr[thr].append(bool(alarm) if (not gt_pos) else False)
            pred_vox_by_thr[thr].append(pred_vox)
            tp_vox_by_thr[thr].append(tp_vox)
            fn_vox_by_thr[thr].append(fn_vox)
            fp_vox_by_thr[thr].append(fp_vox)
            fp_cc_by_thr[thr].append(int(fp_cc))
            fp_cc_vox_by_thr[thr].append(int(fp_cc_vox))
            if fp_cc_p90 is not None:
                fp_cc_size_p90_case_by_thr[thr].append(float(fp_cc_p90))
            fp_cc_sizes_by_thr[thr].extend(_fp_component_sizes(pred, gt))

            # size-stratified
            dice_by_thr_bucket[thr][size_bucket].append(float(dsc))
            tp_by_thr_bucket[thr][size_bucket].append(int(tp_vox))
            fp_by_thr_bucket[thr][size_bucket].append(int(fp_vox))
            fn_by_thr_bucket[thr][size_bucket].append(int(fn_vox))
            det_by_thr_bucket[thr][size_bucket].append(bool(tp_vox > 0) if gt_pos else bool(pred_vox == 0))

            # slice-spacing stratified
            dice_by_thr_ss_bucket[thr][ss_bucket].append(float(dsc))
            tp_by_thr_ss_bucket[thr][ss_bucket].append(int(tp_vox))
            fp_by_thr_ss_bucket[thr][ss_bucket].append(int(fp_vox))
            fn_by_thr_ss_bucket[thr][ss_bucket].append(int(fn_vox))
            det_by_thr_ss_bucket[thr][ss_bucket].append(bool(tp_vox > 0) if gt_pos else bool(pred_vox == 0))

        thr0 = float(thr_list[0])
        s0 = stats_map[thr0]
        s0_pred_vox = cast(int, s0["pred_vox"])
        s0_tp_vox = cast(int, s0["tp_vox"])
        s0_fp_vox = cast(int, s0["fp_vox"])
        s0_fn_vox = cast(int, s0["fn_vox"])
        s0_fp_cc = cast(int, s0["fp_cc"])
        s0_fp_cc_vox = cast(int, s0["fp_cc_vox"])
        case_res = {
            "case_id": sample["case_id"],
            "dice": float(dice_map[thr0]),
            "gt_vox": gt_vox,
            "gt_size_bucket": size_bucket,
            "zooms_mm_raw": zooms_raw,
            "zooms_mm": zooms_eff,
            "max_zoom_axis": max_zoom_axis,
            "slice_spacing_mm_raw": slice_spacing_mm_raw,
            "slice_spacing_mm": slice_spacing_mm,
            "slice_spacing_bucket": ss_bucket,
            "pred_vox": s0_pred_vox,
            "tp_vox": s0_tp_vox,
            "fp_vox": s0_fp_vox,
            "fn_vox": s0_fn_vox,
            "detected": s0["detected"],
            "alarm": s0.get("alarm"),
            "fp_cc": s0_fp_cc,
            "fp_cc_vox": s0_fp_cc_vox,
            "fp_cc_size_p90": s0.get("fp_cc_size_p90"),
        }
        for thr in thr_list[1:]:
            thr_f = float(thr)
            case_res[f"dice@{thr_f}"] = float(dice_map[thr_f])
        results.append(case_res)

        if not quiet:
            print(sample["case_id"], case_res)

        # Best-effort memory hygiene for long MPS runs.
        try:
            del logits_mean
        except NameError:
            pass
        try:
            del probs
        except NameError:
            pass
        try:
            del dice_map
        except NameError:
            pass
        try:
            del stats_map
        except NameError:
            pass
        gc.collect()
        if device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    (out_dir_p / "metrics.json").write_text(json.dumps(results, indent=2))
    with (out_dir_p / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_gt_pos = int(sum(1 for x in gt_pos_flags if x))
    n_gt_neg = int(sum(1 for x in gt_pos_flags if not x))

    def _mean(xs: list[float] | list[int]) -> float | None:
        return float(np.mean(xs)) if xs else None

    def _median(xs: list[float] | list[int]) -> float | None:
        return float(np.median(xs)) if xs else None

    def _sum(xs: list[int]) -> int:
        return int(np.sum(xs)) if xs else 0

    def _p90_int(xs: list[int]) -> float | None:
        if not xs:
            return None
        return float(np.percentile(np.asarray(xs, dtype=np.float32), 90.0))

    def _by_bucket(thr: float) -> dict[str, dict[str, float | int | None]]:
        out: dict[str, dict[str, float | int | None]] = {}
        for b in bucket_names:
            if not dice_by_thr_bucket[thr][b]:
                continue
            tp_sum_b = _sum(tp_by_thr_bucket[thr][b])
            fp_sum_b = _sum(fp_by_thr_bucket[thr][b])
            fn_sum_b = _sum(fn_by_thr_bucket[thr][b])
            prec_b = float(tp_sum_b / float(tp_sum_b + fp_sum_b)) if (tp_sum_b + fp_sum_b) > 0 else None
            rec_b = float(tp_sum_b / float(tp_sum_b + fn_sum_b)) if (tp_sum_b + fn_sum_b) > 0 else None
            det_b = float(np.mean(det_by_thr_bucket[thr][b])) if det_by_thr_bucket[thr][b] else None
            out[b] = {
                "n": int(len(dice_by_thr_bucket[thr][b])),
                "mean_dice": _mean(dice_by_thr_bucket[thr][b]),
                "median_dice": _median(dice_by_thr_bucket[thr][b]),
                "voxel_precision": prec_b,
                "voxel_recall": rec_b,
                "detection_rate_case": det_b,
            }
        return out

    def _by_slice_spacing(thr: float) -> dict[str, dict[str, float | int | None]]:
        out: dict[str, dict[str, float | int | None]] = {}
        for b in ss_bucket_names:
            if not dice_by_thr_ss_bucket[thr][b]:
                continue
            tp_sum_b = _sum(tp_by_thr_ss_bucket[thr][b])
            fp_sum_b = _sum(fp_by_thr_ss_bucket[thr][b])
            fn_sum_b = _sum(fn_by_thr_ss_bucket[thr][b])
            prec_b = float(tp_sum_b / float(tp_sum_b + fp_sum_b)) if (tp_sum_b + fp_sum_b) > 0 else None
            rec_b = float(tp_sum_b / float(tp_sum_b + fn_sum_b)) if (tp_sum_b + fn_sum_b) > 0 else None
            det_b = float(np.mean(det_by_thr_ss_bucket[thr][b])) if det_by_thr_ss_bucket[thr][b] else None
            out[b] = {
                "n": int(len(dice_by_thr_ss_bucket[thr][b])),
                "mean_dice": _mean(dice_by_thr_ss_bucket[thr][b]),
                "median_dice": _median(dice_by_thr_ss_bucket[thr][b]),
                "voxel_precision": prec_b,
                "voxel_recall": rec_b,
                "detection_rate_case": det_b,
            }
        return out

    per_thr = []
    for thr in [float(t) for t in thr_list]:
        tp_sum = _sum(tp_vox_by_thr[thr])
        fp_sum = _sum(fp_vox_by_thr[thr])
        fn_sum = _sum(fn_vox_by_thr[thr])
        prec = None
        rec = None
        if (tp_sum + fp_sum) > 0:
            prec = float(tp_sum / float(tp_sum + fp_sum))
        if (tp_sum + fn_sum) > 0:
            rec = float(tp_sum / float(tp_sum + fn_sum))

        det_rate = None
        if n_gt_pos > 0:
            det_rate = float(np.mean([d for d, pos in zip(det_flags_by_thr[thr], gt_pos_flags) if pos]))

        far = None
        if n_gt_neg > 0:
            far = float(np.mean([a for a, pos in zip(alarm_flags_by_thr[thr], gt_pos_flags) if not pos]))
        per_thr.append(
            {
                "threshold": float(thr),
                "n": int(len(results)),
                "n_gt_pos": int(n_gt_pos),
                "n_gt_neg": int(n_gt_neg),
                "mean_dice": _mean(dice_by_thr[thr]),
                "median_dice": _median(dice_by_thr[thr]),
                "voxel_precision": prec,
                "voxel_recall": rec,
                "mean_pred_vox": _mean(pred_vox_by_thr[thr]),
                "median_pred_vox": _median(pred_vox_by_thr[thr]),
                "detection_rate_case": det_rate,
                "false_alarm_rate_case": far,
                "mean_fp_vox": _mean(fp_vox_by_thr[thr]),
                "mean_fp_cc": _mean(fp_cc_by_thr[thr]),
                "mean_fp_cc_vox": _mean(fp_cc_vox_by_thr[thr]),
                "mean_fp_cc_size_p90_case": _mean(fp_cc_size_p90_case_by_thr[thr]),
                "fp_cc_size_p90": _p90_int(fp_cc_sizes_by_thr[thr]),
                "by_gt_size": _by_bucket(float(thr)),
                "by_slice_spacing": _by_slice_spacing(float(thr)),
            }
        )

    def _best_row_by_mean_dice(rows: list[dict]) -> dict | None:
        best = None
        best_d = None
        for r in rows:
            d = r.get("mean_dice")
            if d is None:
                continue
            try:
                dv = float(d)
            except Exception:
                continue
            if best is None or (best_d is None) or dv > best_d:
                best = r
                best_d = dv
        return best

    best_row = _best_row_by_mean_dice(per_thr)
    best_thr = None if best_row is None else float(best_row["threshold"])

    thr0 = float(thr_list[0])
    primary = next((x for x in per_thr if float(x["threshold"]) == thr0), None)

    extra_best: dict | None = None
    if extra_metrics and best_thr is not None:
        probs_src_dir = probs_dir_p if probs_dir_p is not None else save_probs_dir_p

        vol_diff_ml: list[float] = []
        abs_vol_diff_ml: list[float] = []
        lesion_count_diff: list[int] = []
        abs_lesion_count_diff: list[int] = []

        # micro-averaged lesion-wise stats
        total_gt_lesions = 0
        total_pred_lesions = 0
        total_tp_gt = 0
        total_tp_pred = 0

        assd_mm: list[float] = []
        hd_mm: list[float] = []
        hd95_mm: list[float] = []
        n_dist_valid = 0

        for sample in ds:
            vol = sample["image"]
            mask_gt = sample["mask"]
            if vol.ndim == 3:
                vol = vol[None, ...]

            case_id = None
            try:
                case_id = str(sample.get("case_id"))
            except Exception:
                case_id = None
            if not case_id:
                raise ValueError("--extra-metrics requires sample['case_id']")

            meta = sample.get("meta") or {}
            zooms_raw = None
            if isinstance(meta, dict) and "zooms_mm" in meta:
                try:
                    z = meta.get("zooms_mm")
                    if isinstance(z, (list, tuple)) and len(z) >= 3:
                        zooms_raw = [float(z[0]), float(z[1]), float(z[2])]
                except Exception:
                    zooms_raw = None

            zooms_eff = zooms_raw
            if resample_mm > 0 and zooms_raw is not None and vol.ndim == 4 and mask_gt.ndim == 3:
                try:
                    vol, mask_gt, zooms_eff, _ = _resample_to_max_zoom_mm(
                        vol.astype(np.float32, copy=False),
                        mask_gt.astype(np.float32, copy=False),
                        zooms_mm_xyz=zooms_raw,
                        target_mm=resample_mm,
                    )
                except Exception:
                    zooms_eff = zooms_raw

            npz_path = Path(probs_src_dir) / f"{case_id}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing probs for case_id={case_id!r}: {npz_path}")
            with np.load(str(npz_path)) as z:
                probs_zyx = z["probs"].astype(np.float32, copy=False)

            gt = (mask_gt > 0.5).astype(np.uint8)
            pred = (probs_zyx > float(best_thr)).astype(np.uint8)

            if min_size and min_size > 0:
                lbl = cc_label(pred)[0]
                lbl = lbl.astype(np.int64, copy=False)
                sizes = np.bincount(lbl.ravel())
                remove = sizes < int(min_size)
                remove[0] = False
                pred[remove[lbl]] = 0

            pred = _filter_components_by_score(
                pred,
                probs_zyx,
                score_mode=str(cc_score).strip().lower(),
                score_thr=float(cc_score_thr),
            )
            pred = _keep_top_k_components(pred, k=int(top_k))

            vv = _voxel_volume_mm3(zooms_eff)
            if vv is not None:
                gt_ml = float(int(gt.sum()) * vv / 1000.0)
                pred_ml = float(int(pred.sum()) * vv / 1000.0)
                d = float(pred_ml - gt_ml)
                vol_diff_ml.append(d)
                abs_vol_diff_ml.append(float(abs(d)))

            lw = _lesionwise_stats(pred, gt)
            n_gt = int(lw["n_gt"])
            n_pred = int(lw["n_pred"])
            tp_gt = int(lw["tp_gt"])
            tp_pred = int(lw["tp_pred"])
            lesion_count_diff.append(int(n_pred - n_gt))
            abs_lesion_count_diff.append(int(abs(n_pred - n_gt)))
            total_gt_lesions += n_gt
            total_pred_lesions += n_pred
            total_tp_gt += tp_gt
            total_tp_pred += tp_pred

            dm = _surface_distance_metrics_mm(pred, gt, zooms_eff)
            if dm.get("assd_mm") is not None:
                n_dist_valid += 1
                assd_mm.append(float(dm["assd_mm"]))
                hd_mm.append(float(dm["hd_mm"]))
                hd95_mm.append(float(dm["hd95_mm"]))

        # lesion-wise micro metrics
        if total_pred_lesions > 0:
            lesion_prec = float(total_tp_pred / float(total_pred_lesions))
        else:
            lesion_prec = 1.0 if total_gt_lesions == 0 else 0.0
        if total_gt_lesions > 0:
            lesion_rec = float(total_tp_gt / float(total_gt_lesions))
        else:
            lesion_rec = 1.0 if total_pred_lesions == 0 else 0.0
        lesion_f1 = _f1(lesion_prec, lesion_rec)

        extra_best = {
            "threshold": float(best_thr),
            "volume_diff_ml": {
                "mean": float(np.mean(vol_diff_ml)) if vol_diff_ml else None,
                "mean_abs": float(np.mean(abs_vol_diff_ml)) if abs_vol_diff_ml else None,
            },
            "lesion_count_diff": {
                "mean": float(np.mean(lesion_count_diff)) if lesion_count_diff else None,
                "mean_abs": float(np.mean(abs_lesion_count_diff)) if abs_lesion_count_diff else None,
            },
            "lesionwise": {
                "precision_micro": float(lesion_prec),
                "recall_micro": float(lesion_rec),
                "f1_micro": lesion_f1,
                "total_gt_lesions": int(total_gt_lesions),
                "total_pred_lesions": int(total_pred_lesions),
            },
            "boundary_distance_mm": {
                "n_valid": int(n_dist_valid),
                "assd_mean": float(np.mean(assd_mm)) if assd_mm else None,
                "hd_mean": float(np.mean(hd_mm)) if hd_mm else None,
                "hd95_mean": float(np.mean(hd95_mm)) if hd95_mm else None,
            },
        }

    summary_obj = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "root": str(root),
        "split": str(split),
        "patch_size": list(ps),
        "overlap": float(overlap),
        "normalize": str(normalize),
        "temperature": float(temp),
        "thresholds": [float(t) for t in thr_list],
        "gt_size_bins": [int(x) for x in bins],
        "slice_spacing_bins_mm": [float(x) for x in ss_bins],
        "resample_max_zoom_mm": float(resample_mm),
        "slice_spacing_counts": {k: int(v) for k, v in ss_bucket_counts.items()},
        "min_size": int(min_size),
        "cc_score": str(cc_score),
        "cc_score_thr": float(cc_score_thr),
        "top_k": int(top_k),
        "n": int(len(results)),
        "n_gt_pos": int(n_gt_pos),
        "n_gt_neg": int(n_gt_neg),
        # keep top-level fields for compatibility (primary threshold)
        "detection_rate_case": None if primary is None else primary.get("detection_rate_case"),
        "false_alarm_rate_case": None if primary is None else primary.get("false_alarm_rate_case"),
        "mean_fp_vox": None if primary is None else primary.get("mean_fp_vox"),
        "mean_fp_cc": None if primary is None else primary.get("mean_fp_cc"),
        "mean_fp_cc_vox": None if primary is None else primary.get("mean_fp_cc_vox"),
        "fp_cc_size_p90": None if primary is None else primary.get("fp_cc_size_p90"),
        "voxel_precision": None if primary is None else primary.get("voxel_precision"),
        "voxel_recall": None if primary is None else primary.get("voxel_recall"),
        "mean_pred_vox": None if primary is None else primary.get("mean_pred_vox"),
        "median_pred_vox": None if primary is None else primary.get("median_pred_vox"),
        "per_threshold": per_thr,
        "best_threshold_by_mean_dice": best_thr,
        "extra_metrics_best": extra_best,
    }

    (out_dir_p / "summary.json").write_text(json.dumps(summary_obj, indent=2))


def infer_with_tta(vol: NDArray[np.float32], model: torch.nn.Module, patch_size: Tuple[int, int, int], overlap: float, device: torch.device) -> NDArray[np.float32]:
    """Flip + rot90 TTA over spatial axes (D,H,W).

    Aggregation matches training-time validation: mean(logits) -> sigmoid.
    Returns probability volume in [0, 1].
    """
    flip_axes: Iterable[Tuple[int, ...]] = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    acc: NDArray[np.float32] | None = None
    num = 0
    for rot_k in (0, 1, 2, 3):  # rotate on (H, W)
        v_rot = np.rot90(vol, k=rot_k, axes=(2, 3))
        for axes in flip_axes:
            v = np.flip(v_rot, axis=axes) if axes else v_rot
            out = sliding_window_inference_3d(v, model, patch_size=patch_size, overlap=overlap, device=device, aggregate="logits")
            if axes:
                out = np.flip(out, axis=tuple(a + 1 for a in axes))
            if rot_k:
                out = np.rot90(out, k=-rot_k, axes=(3, 4))
            acc = out if acc is None else acc + out
            num += 1
    if acc is None:
        raise RuntimeError("TTA produced no outputs (unexpected)")
    logits_mean = (acc / float(num)).astype(np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits_mean))
    return probs.astype(np.float32)


def infer_logits_with_tta(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> NDArray[np.float32]:
    """Flip + rot90 TTA over spatial axes (D,H,W), returning mean(logits)."""
    flip_axes: Iterable[Tuple[int, ...]] = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    acc: NDArray[np.float32] | None = None
    num = 0
    for rot_k in (0, 1, 2, 3):  # rotate on (H, W)
        v_rot = np.rot90(vol, k=rot_k, axes=(2, 3))
        for axes in flip_axes:
            v = np.flip(v_rot, axis=axes) if axes else v_rot
            out = sliding_window_inference_3d(v, model, patch_size=patch_size, overlap=overlap, device=device, aggregate="logits")
            if axes:
                out = np.flip(out, axis=tuple(a + 1 for a in axes))
            if rot_k:
                out = np.rot90(out, k=-rot_k, axes=(3, 4))
            acc = out if acc is None else acc + out
            num += 1
    if acc is None:
        raise RuntimeError("TTA produced no outputs (unexpected)")
    return (acc / float(num)).astype(np.float32)


def infer_logits(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> NDArray[np.float32]:
    """No TTA: single sliding-window pass, returning logits."""
    return sliding_window_inference_3d(
        vol,
        model,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
        aggregate="logits",
    ).astype(np.float32)


def infer_logits_with_flip_tta(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> NDArray[np.float32]:
    """Flip-only TTA over spatial axes (D,H,W), returning mean(logits)."""
    flip_axes: Iterable[Tuple[int, ...]] = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    acc: NDArray[np.float32] | None = None
    num = 0
    for axes in flip_axes:
        v = np.flip(vol, axis=axes) if axes else vol
        out = sliding_window_inference_3d(v, model, patch_size=patch_size, overlap=overlap, device=device, aggregate="logits")
        if axes:
            out = np.flip(out, axis=tuple(a + 1 for a in axes))
        acc = out if acc is None else acc + out
        num += 1
    if acc is None:
        raise RuntimeError("Flip TTA produced no outputs (unexpected)")
    return (acc / float(num)).astype(np.float32)


if __name__ == "__main__":
    app()
