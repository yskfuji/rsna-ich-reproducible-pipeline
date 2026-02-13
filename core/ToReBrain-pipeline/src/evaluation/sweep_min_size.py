"""Sweep connected-component min_size at a fixed probability threshold.

Goal:
- Reuse the same inference (probability map) within each case.
- Evaluate how increasing CC removal (min_size) reduces FP while maintaining detection.

This script:
- Runs sliding-window inference with TTA (same as evaluate_isles: mean(logits) -> sigmoid).
- Thresholds once per case.
- Computes per-component stats once per case and re-aggregates for multiple min_size values.
- Writes one evaluation directory per min_size under out_dir_base so it plugs into reports/make_eval_report.py.

Example:
  PYTHONPATH=$PWD python -m src.evaluation.sweep_min_size \
    --model-path runs/3d_unet/.../best.pt \
    --csv-path data/splits/...csv \
    --root data/processed/... \
    --split test \
    --out-dir-base results/3d_unet_medseg/test_e20_thr022_min_size_sweep \
    --threshold 0.22 \
    --min-sizes 0,10,20,30,40,50,75,100,150,200 \
    --patch-size 64,64,48 \
    --overlap 0.5 \
    --normalize nonzero_zscore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import label as cc_label

from ..datasets.isles_dataset import IslesVolumeDataset
from ..inference.infer_sliding_window import sliding_window_inference_3d
from ..models.unet_3d import UNet3D
from ..training.utils_train import prepare_device


def _parse_int_list(s: str) -> list[int]:
    vals: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("min_sizes must be non-empty")
    return vals


def _parse_patch_size(s: str) -> Tuple[int, int, int]:
    ps = [int(x) for x in s.split(",") if x.strip()]
    if len(ps) != 3:
        raise ValueError(f"patch_size must have 3 ints (D,H,W), got: {s!r}")
    return (ps[0], ps[1], ps[2])


def infer_with_tta(
    vol: NDArray[np.float32],
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> NDArray[np.float32]:
    """Flip + rot90 TTA over spatial axes (D,H,W).

    Aggregation: mean(logits) -> sigmoid. Returns probability volume in [0, 1].
    """

    flip_axes: Iterable[Tuple[int, ...]] = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    acc: NDArray[np.float32] | None = None
    num = 0
    for rot_k in (0, 1, 2, 3):
        v_rot = np.rot90(vol, k=rot_k, axes=(2, 3))
        for axes in flip_axes:
            v = np.flip(v_rot, axis=axes) if axes else v_rot
            out = sliding_window_inference_3d(
                v, model, patch_size=patch_size, overlap=overlap, device=device, aggregate="logits"
            )
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


def _infer_base_ch_from_ckpt(state_dict: dict) -> int:
    w = state_dict.get("enc1.0.weight")
    if w is None:
        raise RuntimeError("Cannot infer base_ch from checkpoint; pass --base-ch")
    return int(w.shape[0])


def _component_stats(
    lbl: NDArray[np.int64],
    gt: NDArray[np.uint8],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return (comp_sizes, comp_tp) indexed by component id.

    - comp_sizes[k] = number of voxels where lbl==k
    - comp_tp[k] = number of voxels where (lbl==k) & (gt==1)

    Index 0 is background.
    """

    flat_lbl = lbl.ravel()
    comp_sizes = np.bincount(flat_lbl)

    flat_gt = gt.ravel().astype(np.int64, copy=False)
    comp_tp = np.bincount(flat_lbl, weights=flat_gt)

    # Ensure same length
    if comp_tp.shape[0] < comp_sizes.shape[0]:
        comp_tp = np.pad(comp_tp, (0, comp_sizes.shape[0] - comp_tp.shape[0]))
    elif comp_sizes.shape[0] < comp_tp.shape[0]:
        comp_sizes = np.pad(comp_sizes, (0, comp_tp.shape[0] - comp_sizes.shape[0]))

    return comp_sizes.astype(np.int64, copy=False), comp_tp.astype(np.int64, copy=False)


def _aggregate_for_min_size(
    comp_sizes: NDArray[np.int64],
    comp_tp: NDArray[np.int64],
    gt_vox: int,
    min_size: int,
    gt_pos: bool,
    eps: float = 1e-6,
) -> dict[str, float | int | bool | None]:
    """Aggregate pred/tp/fp/fn/dice/detected and FP component stats for a given min_size."""

    if comp_sizes.shape[0] <= 1:
        pred_vox = 0
        tp_vox = 0
        fp_vox = 0
        fn_vox = gt_vox
        dice = float((eps) / (gt_vox + eps))
        detected = bool(tp_vox > 0) if gt_pos else bool(pred_vox == 0)
        return {
            "pred_vox": int(pred_vox),
            "tp_vox": int(tp_vox),
            "fp_vox": int(fp_vox),
            "fn_vox": int(fn_vox),
            "dice": float(dice),
            "detected": bool(detected) if gt_pos else None,
            "fp_cc": int(0),
            "fp_cc_vox": int(0),
        }

    keep = comp_sizes >= int(min_size)
    keep[0] = False  # background

    pred_vox = int(comp_sizes[keep].sum())
    tp_vox = int(comp_tp[keep].sum())
    fp_vox = int((comp_sizes[keep] - comp_tp[keep]).sum())
    fn_vox = int(gt_vox - tp_vox)

    den = float(pred_vox + gt_vox) + eps
    dice = float((2.0 * float(tp_vox) + eps) / den)

    detected = bool(tp_vox > 0) if gt_pos else bool(pred_vox == 0)

    # FP components: kept components with zero overlap to gt
    fp_comp_mask = keep & (comp_tp == 0) & (comp_sizes > 0)
    fp_cc = int(fp_comp_mask.sum())
    fp_cc_vox = int(comp_sizes[fp_comp_mask].sum())

    return {
        "pred_vox": int(pred_vox),
        "tp_vox": int(tp_vox),
        "fp_vox": int(fp_vox),
        "fn_vox": int(fn_vox),
        "dice": float(dice),
        "detected": bool(detected) if gt_pos else None,
        "fp_cc": int(fp_cc),
        "fp_cc_vox": int(fp_cc_vox),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--csv-path", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir-base", required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--min-sizes", required=True, help="comma-separated ints")
    p.add_argument("--patch-size", default="64,64,48")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--base-ch", type=int, default=None)
    p.add_argument("--normalize", default="nonzero_zscore")

    args = p.parse_args()

    thr = float(args.threshold)
    min_sizes = sorted(set(_parse_int_list(args.min_sizes)))
    ps = _parse_patch_size(args.patch_size)
    overlap = float(args.overlap)

    device = prepare_device()

    ds = IslesVolumeDataset(args.csv_path, split=args.split, root=args.root, transform=None, normalize=args.normalize)
    first = ds[0]["image"]
    in_ch = int(first.shape[0]) if first.ndim == 4 else 1

    state = torch.load(args.model_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    base_ch = args.base_ch
    if base_ch is None:
        if not isinstance(state_dict, dict):
            raise RuntimeError("Checkpoint state_dict is not a dict; pass --base-ch")
        base_ch = _infer_base_ch_from_ckpt(state_dict)

    model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=int(base_ch))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    out_base = Path(args.out_dir_base)
    out_base.mkdir(parents=True, exist_ok=True)

    # Per-min_size accumulators
    per_case_by_ms: dict[int, list[dict]] = {ms: [] for ms in min_sizes}

    for sample in ds:
        vol = sample["image"]
        mask_gt = sample["mask"]
        if vol.ndim == 3:
            vol = vol[None, ...]

        probs = infer_with_tta(vol, model, ps, overlap, device)
        pred_raw = (probs[0, 0] > thr).astype(np.uint8)

        gt = (mask_gt > 0.5).astype(np.uint8)
        gt_vox = int(gt.sum())
        gt_pos = gt_vox > 0

        lbl, _n = cc_label(pred_raw)
        lbl = lbl.astype(np.int64, copy=False)
        comp_sizes, comp_tp = _component_stats(lbl, gt)

        for ms in min_sizes:
            agg = _aggregate_for_min_size(comp_sizes, comp_tp, gt_vox=gt_vox, min_size=ms, gt_pos=gt_pos)
            per_case_by_ms[ms].append(
                {
                    "case_id": sample["case_id"],
                    "dice": float(agg["dice"]),
                    "gt_vox": int(gt_vox),
                    "pred_vox": int(agg["pred_vox"]),
                    "tp_vox": int(agg["tp_vox"]),
                    "fp_vox": int(agg["fp_vox"]),
                    "fn_vox": int(agg["fn_vox"]),
                    "detected": agg["detected"],
                    "fp_cc": int(agg["fp_cc"]),
                    "fp_cc_vox": int(agg["fp_cc_vox"]),
                    "threshold": float(thr),
                    "min_size": int(ms),
                }
            )

    # Write one dir per min_size
    for ms in min_sizes:
        rows = per_case_by_ms[ms]
        out_dir = out_base / f"cc{ms}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))

        gt_pos_flags = [bool(r["gt_vox"] > 0) for r in rows]
        n_gt_pos = int(sum(gt_pos_flags))
        n_gt_neg = int(len(rows) - n_gt_pos)

        det_rate = None
        if n_gt_pos > 0:
            det_vals = [bool(r["detected"]) for r in rows if bool(r["gt_vox"] > 0)]
            det_rate = float(np.mean(det_vals)) if det_vals else None

        mean = lambda xs: float(np.mean(xs)) if xs else None
        median = lambda xs: float(np.median(xs)) if xs else None

        dice_list = [float(r["dice"]) for r in rows]
        fp_vox_list = [int(r["fp_vox"]) for r in rows]
        fp_cc_list = [int(r["fp_cc"]) for r in rows]
        fp_cc_vox_list = [int(r["fp_cc_vox"]) for r in rows]

        summary = {
            "model_path": str(args.model_path),
            "csv_path": str(args.csv_path),
            "root": str(args.root),
            "split": str(args.split),
            "patch_size": list(ps),
            "overlap": float(overlap),
            "normalize": str(args.normalize),
            "thresholds": [float(thr)],
            "min_size": int(ms),
            "n": int(len(rows)),
            "n_gt_pos": int(n_gt_pos),
            "n_gt_neg": int(n_gt_neg),
            "mean_dice": mean(dice_list),
            "median_dice": median(dice_list),
            "detection_rate_case": det_rate,
            "mean_fp_vox": mean(fp_vox_list),
            "mean_fp_cc": mean(fp_cc_list),
            "mean_fp_cc_vox": mean(fp_cc_vox_list),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote min_size sweep under: {out_base}")


if __name__ == "__main__":
    main()
