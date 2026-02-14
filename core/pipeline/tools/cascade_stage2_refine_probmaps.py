"""Cascade Stage2: refine probability maps around Stage1 candidate boxes.

This script takes:
- Stage1 saved probability maps (NPZ, key 'probs' with shape (Z,Y,X))
- A candidate JSONL (one record per candidate bbox)
- A Stage2 model checkpoint

It produces:
- Refined probability maps (NPZ, key 'probs' with shape (Z,Y,X))

Intended usage:
1) Generate Stage1 probs with evaluate_isles --save-probs
2) Generate candidates with cascade_stage1_make_candidates.py
3) Run this script to create refined probs
4) Evaluate refined probs via evaluate_isles --probs-dir

Example:
  cd /Users/yusukefujinami/ToReBrain/ToReBrain-pipeline
  PYTHONPATH=$PWD /opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output \
    python tools/cascade_stage2_refine_probmaps.py \
      --stage1-probs-dir results/diag/.../saveprobs_test/probs \
      --candidates-jsonl data/cascade/stage1_candidates_test_thr0p20.jsonl \
      --stage2-model runs/3d_unet/.../best.pt \
      --csv-path data/splits/my_dataset_dwi_adc_flair_train_val_test.csv \
      --root data/processed/my_dataset_dwi_adc_flair \
      --split test \
      --normalize nonzero_zscore \
      --resample-max-zoom-mm 2.0 \
      --patch-size 56,56,24 \
      --tta none \
      --out-probs-dir results/diag/.../refined_probs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.datasets.isles_dataset import IslesVolumeDataset
from src.evaluation.evaluate_isles import _resample_to_max_zoom_mm
from src.inference.infer_sliding_window import sliding_window_inference_3d
from src.models.unet_3d import UNet3D
from src.training.utils_train import prepare_device


def _parse_ints_csv(s: str, n: int) -> tuple[int, ...]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"Expected {n} comma-separated ints, got: {s!r}")
    out: list[int] = []
    for p in parts:
        out.append(int(float(p)))
    return tuple(out)


def _load_candidates(candidates_jsonl: Path, split: str) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    with candidates_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("split", "")) != str(split):
                continue
            case_id = str(rec.get("case_id"))
            bbox = rec.get("bbox_zyxzyx")
            if (not isinstance(bbox, list)) or len(bbox) != 6:
                continue
            out.setdefault(case_id, []).append([int(x) for x in bbox])

    # Stable order: use cand_rank when available, otherwise keep file order.
    # (We don't need rank for correctness.)
    return out


def _resize_czyx_linear(vol_czyx: NDArray[np.float32], target_zyx: tuple[int, int, int]) -> NDArray[np.float32]:
    """Linear resize (per-channel) + crop/pad to match target spatial shape."""
    try:
        from scipy.ndimage import zoom as nd_zoom
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for resizing") from e

    C, Z, Y, X = vol_czyx.shape
    tz, ty, tx = (int(target_zyx[0]), int(target_zyx[1]), int(target_zyx[2]))

    if (Z, Y, X) == (tz, ty, tx):
        return vol_czyx

    zoom_f = (tz / max(1, Z), ty / max(1, Y), tx / max(1, X))
    chs: list[NDArray[np.float32]] = []
    for c in range(C):
        ch = nd_zoom(vol_czyx[c], zoom=zoom_f, order=1).astype(np.float32, copy=False)
        chs.append(ch)
    v2 = np.stack(chs, axis=0).astype(np.float32, copy=False)

    out = np.zeros((C, tz, ty, tx), dtype=np.float32)
    cz = min(tz, int(v2.shape[1]))
    cy = min(ty, int(v2.shape[2]))
    cx = min(tx, int(v2.shape[3]))
    out[:, :cz, :cy, :cx] = v2[:, :cz, :cy, :cx]
    return out


def _crop_pad_patch_czyx(
    vol_czyx: NDArray[np.float32],
    center_zyx: tuple[int, int, int],
    patch_zyx: tuple[int, int, int],
) -> tuple[NDArray[np.float32], tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    """Return (patch, src_slices_zyx, dst_slices_zyx)."""
    C, Z, Y, X = vol_czyx.shape
    pz, py, px = patch_zyx
    cz, cy, cx = center_zyx

    z0 = int(cz - pz // 2)
    y0 = int(cy - py // 2)
    x0 = int(cx - px // 2)
    z1 = z0 + int(pz)
    y1 = y0 + int(py)
    x1 = x0 + int(px)

    src_z0 = max(0, z0)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_z1 = min(Z, z1)
    src_y1 = min(Y, y1)
    src_x1 = min(X, x1)

    dst_z0 = src_z0 - z0
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    patch = np.zeros((C, pz, py, px), dtype=np.float32)
    patch[:, dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = vol_czyx[:, src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]

    src_slices = (slice(src_z0, src_z1), slice(src_y0, src_y1), slice(src_x0, src_x1))
    dst_slices = (slice(dst_z0, dst_z1), slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))
    return patch, src_slices, dst_slices


def _infer_region_probs(
    vol_czyx: NDArray[np.float32],
    model: torch.nn.Module,
    device: torch.device,
    patch_zyx: tuple[int, int, int],
    overlap: float,
    tta: str,
) -> NDArray[np.float32]:
    """Infer probabilities for a cropped region using sliding-window (optionally flip-TTA)."""
    tta_mode = str(tta).strip().lower()
    if tta_mode in {"none", "off", "false", "0"}:
        out = sliding_window_inference_3d(vol_czyx, model, patch_size=patch_zyx, overlap=float(overlap), device=device, aggregate="probs")
        return out[0, 0].astype(np.float32, copy=False)

    if tta_mode in {"flip"}:
        flip_axes = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        acc = None
        num = 0
        for axes in flip_axes:
            v = np.flip(vol_czyx, axis=axes) if axes else vol_czyx
            # Use logits aggregation, then sigmoid after averaging.
            logits = sliding_window_inference_3d(v, model, patch_size=patch_zyx, overlap=float(overlap), device=device, aggregate="logits")
            if axes:
                logits = np.flip(logits, axis=tuple(a + 2 for a in axes))
            acc = logits if acc is None else (acc + logits)
            num += 1
        if acc is None:
            raise RuntimeError("flip TTA produced no outputs")
        logits_mean = (acc / float(num)).astype(np.float32, copy=False)
        probs = 1.0 / (1.0 + np.exp(-logits_mean))
        return probs[0, 0].astype(np.float32, copy=False)

    raise ValueError(f"Unknown --tta: {tta!r} (expected none|flip)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-probs-dir", required=True)
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--stage2-model", required=True)

    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])

    ap.add_argument("--normalize", default="nonzero_zscore")
    ap.add_argument("--resample-max-zoom-mm", type=float, default=0.0)

    ap.add_argument("--patch-size", default="56,56,24")
    ap.add_argument("--overlap", type=float, default=0.5, help="sliding-window overlap for region inference")
    ap.add_argument("--tta", default="none", help="none|flip")

    ap.add_argument(
        "--infer-max-cands-per-case",
        type=int,
        default=8,
        help="Process only the first N candidate boxes per case (for speed).",
    )

    ap.add_argument(
        "--stage1-fallback-weight",
        type=float,
        default=0.0,
        help="If >0, blend in Stage1 probs via refined = max(refined, stage1_probs * weight).",
    )

    ap.add_argument("--out-probs-dir", required=True)
    ap.add_argument("--save-fp16", action="store_true", help="save probs as float16 (default float32)")

    args = ap.parse_args()

    stage1_probs_dir = Path(args.stage1_probs_dir).expanduser().resolve()
    candidates_jsonl = Path(args.candidates_jsonl).expanduser().resolve()
    stage2_model_path = Path(args.stage2_model).expanduser().resolve()
    out_probs_dir = Path(args.out_probs_dir).expanduser().resolve()
    out_probs_dir.mkdir(parents=True, exist_ok=True)

    ps = _parse_ints_csv(args.patch_size, 3)
    patch_zyx = (int(ps[0]), int(ps[1]), int(ps[2]))

    device = prepare_device()

    # Dataset (for image loading + normalization)
    ds = IslesVolumeDataset(
        args.csv_path,
        split=str(args.split),
        root=args.root,
        transform=None,
        normalize=str(args.normalize),
        allow_missing_label=True,
    )
    case_to_idx = {str(ds.df.iloc[i]["case_id"]): int(i) for i in range(len(ds))}

    # Load candidates
    cand = _load_candidates(candidates_jsonl, split=str(args.split))

    # Load Stage2 model
    first_vol = ds[0]["image"]
    in_ch = int(first_vol.shape[0]) if first_vol.ndim == 4 else 1

    state = torch.load(str(stage2_model_path), map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state

    w = state_dict.get("enc1.0.weight")
    if w is None:
        raise RuntimeError("Cannot infer base_ch from checkpoint; missing enc1.0.weight")
    base_ch = int(w.shape[0])

    ckpt_has_ds = any(
        (k.startswith("aux2_conv.") or k.startswith("aux3_conv."))
        for k in (state_dict.keys() if hasattr(state_dict, "keys") else [])
    )
    model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=base_ch, deep_supervision=bool(ckpt_has_ds))
    strict = True
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    resample_mm = float(args.resample_max_zoom_mm)

    n_cases = 0
    n_cases_missing_stage1 = 0
    n_cases_no_cands = 0
    n_total_cands = 0

    for case_id, boxes in cand.items():
        prob_path = stage1_probs_dir / f"{case_id}.npz"
        if not prob_path.exists():
            n_cases_missing_stage1 += 1
            continue

        z = np.load(str(prob_path))
        stage1_probs = z["probs"]
        if stage1_probs.ndim != 3:
            raise ValueError(f"Expected stage1 probs (Z,Y,X), got {stage1_probs.shape} for {case_id}")
        target_zyx = (int(stage1_probs.shape[0]), int(stage1_probs.shape[1]), int(stage1_probs.shape[2]))

        idx = case_to_idx.get(str(case_id))
        if idx is None:
            continue
        sample = ds[int(idx)]
        vol = sample["image"].astype(np.float32, copy=False)
        if vol.ndim == 3:
            vol = vol[None, ...]

        # Match evaluate_isles preprocessing: optional resample to max zoom.
        zooms_mm_xyz = sample.get("meta", {}).get("zooms_mm", [1.0, 1.0, 1.0])
        if resample_mm > 0:
            vol, _, _, _ = _resample_to_max_zoom_mm(vol, np.zeros(vol.shape[1:], dtype=np.float32), zooms_mm_xyz, resample_mm)

        # Ensure vol shape matches stage1 probs shape.
        if tuple(vol.shape[1:]) != tuple(target_zyx):
            vol = _resize_czyx_linear(vol.astype(np.float32, copy=False), target_zyx)

        refined = np.zeros(target_zyx, dtype=np.float32)

        if not boxes:
            n_cases_no_cands += 1
            out_path = out_probs_dir / f"{case_id}.npz"
            np.savez_compressed(str(out_path), probs=refined.astype(np.float16 if args.save_fp16 else np.float32, copy=False))
            n_cases += 1
            continue

        max_cands = int(args.infer_max_cands_per_case)
        use_boxes = boxes[: max(0, max_cands)] if max_cands > 0 else boxes

        # Region inference per candidate bbox, expanded to at least patch size.
        mz, my, mx = (patch_zyx[0] // 2, patch_zyx[1] // 2, patch_zyx[2] // 2)
        for bbox in use_boxes:
            z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox]
            rz0 = max(0, z0 - mz)
            ry0 = max(0, y0 - my)
            rx0 = max(0, x0 - mx)
            rz1 = min(target_zyx[0], z1 + mz)
            ry1 = min(target_zyx[1], y1 + my)
            rx1 = min(target_zyx[2], x1 + mx)

            crop = vol[:, rz0:rz1, ry0:ry1, rx0:rx1]
            crop = np.ascontiguousarray(crop)
            crop_probs = _infer_region_probs(
                crop,
                model,
                device=device,
                patch_zyx=patch_zyx,
                overlap=float(args.overlap),
                tta=str(args.tta),
            )

            refined[rz0:rz1, ry0:ry1, rx0:rx1] = np.maximum(refined[rz0:rz1, ry0:ry1, rx0:rx1], crop_probs)

        w_fb = float(args.stage1_fallback_weight)
        if np.isfinite(w_fb) and w_fb > 0:
            refined = np.maximum(refined, stage1_probs.astype(np.float32, copy=False) * float(w_fb))

        out_path = out_probs_dir / f"{case_id}.npz"
        np.savez_compressed(str(out_path), probs=refined.astype(np.float16 if args.save_fp16 else np.float32, copy=False))

        n_cases += 1
        n_total_cands += int(len(use_boxes))

    summary = {
        "split": str(args.split),
        "stage1_probs_dir": str(stage1_probs_dir),
        "candidates_jsonl": str(candidates_jsonl),
        "stage2_model": str(stage2_model_path),
        "normalize": str(args.normalize),
        "resample_max_zoom_mm": float(resample_mm),
        "patch_size_zyx": [int(x) for x in patch_zyx],
        "tta": str(args.tta),
        "overlap": float(args.overlap),
        "infer_max_cands_per_case": int(args.infer_max_cands_per_case),
        "stage1_fallback_weight": float(args.stage1_fallback_weight),
        "out_probs_dir": str(out_probs_dir),
        "n_cases_written": int(n_cases),
        "n_cases_missing_stage1_probs": int(n_cases_missing_stage1),
        "n_cases_no_candidates": int(n_cases_no_cands),
        "n_total_candidates_used": int(n_total_cands),
    }

    (out_probs_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
