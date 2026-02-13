"""Sweep connected-component score threshold at fixed prob threshold.

We:
- Run inference once per case to get probability map.
- Binarize with `--threshold` to get candidate components.
- Remove small components with `--min-size`.
- Compute per-component scores from the probability map:
  - max_prob
  - mean_prob
  - p95_prob
- Keep components whose score >= score_threshold.

This is designed to reduce FP components without extra inference.

Outputs one evaluation directory per (score, score_threshold) under out_dir_base so it plugs into
reports/make_eval_report.py.
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


def _parse_float_list(s: str) -> list[float]:
    vals: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("score_thresholds must be non-empty")
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
    """Flip + rot90 TTA over spatial axes (D,H,W), mean(logits)->sigmoid."""

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
    probs: NDArray[np.float32],
    gt: NDArray[np.uint8],
    min_size: int,
) -> dict[str, NDArray[np.float32] | NDArray[np.int64]]:
    """Compute per-component stats.

    Returns arrays indexed by component id (0 is background).

    We compute:
    - comp_size (int64)
    - comp_tp (int64)
    - comp_mean (float32)
    - comp_max (float32)
    - comp_p95 (float32)  (computed only for comps passing min_size; others set to 0)
    """

    flat_lbl = lbl.ravel()
    comp_size = np.bincount(flat_lbl).astype(np.int64, copy=False)

    flat_gt = gt.ravel().astype(np.int64, copy=False)
    comp_tp = np.bincount(flat_lbl, weights=flat_gt).astype(np.int64, copy=False)

    flat_probs = probs.ravel().astype(np.float64, copy=False)
    comp_sum = np.bincount(flat_lbl, weights=flat_probs)

    # align lengths
    n = int(max(comp_size.shape[0], comp_tp.shape[0], comp_sum.shape[0]))
    if comp_size.shape[0] < n:
        comp_size = np.pad(comp_size, (0, n - comp_size.shape[0]))
    if comp_tp.shape[0] < n:
        comp_tp = np.pad(comp_tp, (0, n - comp_tp.shape[0]))
    if comp_sum.shape[0] < n:
        comp_sum = np.pad(comp_sum, (0, n - comp_sum.shape[0]))

    comp_mean = np.zeros((n,), dtype=np.float32)
    nz = comp_size > 0
    comp_mean[nz] = (comp_sum[nz] / comp_size[nz]).astype(np.float32)

    comp_max = np.zeros((n,), dtype=np.float32)
    comp_p95 = np.zeros((n,), dtype=np.float32)

    # compute max/p95 only for candidates (size>=min_size)
    cand_ids = np.where((comp_size >= int(min_size)) & (np.arange(n) > 0))[0]
    for cid in cand_ids.tolist():
        vals = probs[lbl == cid]
        if vals.size == 0:
            continue
        comp_max[cid] = float(vals.max())
        comp_p95[cid] = float(np.percentile(vals, 95))

    return {
        "size": comp_size,
        "tp": comp_tp,
        "mean": comp_mean,
        "max": comp_max,
        "p95": comp_p95,
    }


def _aggregate(
    stats: dict[str, NDArray],
    gt_vox: int,
    gt_pos: bool,
    min_size: int,
    score_name: str,
    score_thr: float,
    eps: float = 1e-6,
) -> dict[str, int | float | bool | None]:
    size = stats["size"].astype(np.int64, copy=False)
    tp = stats["tp"].astype(np.int64, copy=False)

    score = stats[score_name].astype(np.float32, copy=False)

    keep = (np.arange(size.shape[0]) > 0) & (size >= int(min_size)) & (score >= float(score_thr))

    pred_vox = int(size[keep].sum())
    tp_vox = int(tp[keep].sum())
    fp_vox = int((size[keep] - tp[keep]).sum())
    fn_vox = int(gt_vox - tp_vox)

    dice = float((2.0 * float(tp_vox) + eps) / (float(pred_vox + gt_vox) + eps))

    detected = bool(tp_vox > 0) if gt_pos else bool(pred_vox == 0)

    fp_comp_mask = keep & (tp == 0) & (size > 0)
    fp_cc = int(fp_comp_mask.sum())
    fp_cc_vox = int(size[fp_comp_mask].sum())

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

    p.add_argument("--threshold", type=float, required=True, help="prob threshold to binarize")
    p.add_argument("--min-size", type=int, default=20)

    p.add_argument(
        "--scores",
        default="max,p95,mean",
        help="comma-separated score names among: max,p95,mean",
    )
    p.add_argument("--score-thresholds", required=True, help="comma-separated floats")

    p.add_argument("--patch-size", default="64,64,48")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--base-ch", type=int, default=None)
    p.add_argument("--normalize", default="nonzero_zscore")

    args = p.parse_args()

    thr = float(args.threshold)
    min_size = int(args.min_size)
    ps = _parse_patch_size(args.patch_size)
    overlap = float(args.overlap)

    score_names = [s.strip().lower() for s in args.scores.split(",") if s.strip()]
    for s in score_names:
        if s not in {"max", "p95", "mean"}:
            raise ValueError(f"Unknown score: {s}")

    score_thrs = sorted(set(_parse_float_list(args.score_thresholds)))

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

    # store per (score, score_thr) metrics rows
    per_key_rows: dict[tuple[str, float], list[dict]] = {(sn, st): [] for sn in score_names for st in score_thrs}

    for sample in ds:
        vol = sample["image"]
        mask_gt = sample["mask"]
        if vol.ndim == 3:
            vol = vol[None, ...]

        probs = infer_with_tta(vol, model, ps, overlap, device)
        prob_map = probs[0, 0]

        pred_raw = (prob_map > thr).astype(np.uint8)
        lbl, _n = cc_label(pred_raw)
        lbl = lbl.astype(np.int64, copy=False)

        gt = (mask_gt > 0.5).astype(np.uint8)
        gt_vox = int(gt.sum())
        gt_pos = gt_vox > 0

        stats = _component_stats(lbl, prob_map.astype(np.float32, copy=False), gt, min_size=min_size)

        for sn in score_names:
            for st in score_thrs:
                agg = _aggregate(stats, gt_vox=gt_vox, gt_pos=gt_pos, min_size=min_size, score_name=sn, score_thr=st)
                per_key_rows[(sn, st)].append(
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
                        "min_size": int(min_size),
                        "score": sn,
                        "score_threshold": float(st),
                    }
                )

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    def median(xs):
        return float(np.median(xs)) if xs else None

    # write
    for (sn, st), rows in per_key_rows.items():
        # stable folder name
        st_s = f"{st:.3f}".rstrip("0").rstrip(".")
        out_dir = out_base / f"score_{sn}_ge{st_s}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))

        gt_pos_flags = [bool(r["gt_vox"] > 0) for r in rows]
        n_gt_pos = int(sum(gt_pos_flags))
        n_gt_neg = int(len(rows) - n_gt_pos)

        det_rate = None
        if n_gt_pos > 0:
            det_vals = [bool(r["detected"]) for r in rows if bool(r["gt_vox"] > 0)]
            det_rate = float(np.mean(det_vals)) if det_vals else None

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
            "min_size": int(min_size),
            "n": int(len(rows)),
            "n_gt_pos": int(n_gt_pos),
            "n_gt_neg": int(n_gt_neg),
            "mean_dice": mean(dice_list),
            "median_dice": median(dice_list),
            "detection_rate_case": det_rate,
            "mean_fp_vox": mean(fp_vox_list),
            "mean_fp_cc": mean(fp_cc_list),
            "mean_fp_cc_vox": mean(fp_cc_vox_list),
            "score": sn,
            "score_threshold": float(st),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote component-score sweep under: {out_base}")


if __name__ == "__main__":
    main()
