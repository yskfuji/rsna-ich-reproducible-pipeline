"""Cascade Stage1: extract candidate bounding boxes from saved probability maps.

Input:
- probs_dir/<case_id>.npz with key 'probs' (Z,Y,X) float32/float16
- dataset CSV to resolve label paths

Output:
- JSONL with one record per candidate box
- summary.json with basic counts

This is intended to prepare Stage2 (refinement) crop lists.

Example:
  /opt/anaconda3/bin/python tools/cascade_stage1_make_candidates.py \
    --probs-dir results/diag/cascade_stage1_20251225_140000/saveprobs_train/probs \
    --csv-path data/splits/my_dataset_dwi_adc_flair_train_val_test.csv \
    --root data/processed/my_dataset_dwi_adc_flair \
    --split train \
    --threshold 0.20 \
    --max-cands-per-case 32 \
    --margin-zyx 6,12,12 \
    --out-jsonl results/diag/cascade_stage1_20251225_140000/candidates_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom as nd_zoom

from src.preprocess.utils_io import load_nifti


def _parse_ints_csv(s: str, n: int) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"Expected {n} comma-separated ints, got: {s!r}")
    out: list[int] = []
    for p in parts:
        v = int(float(p))
        if v < 0:
            raise ValueError(f"Expected non-negative int, got: {s!r}")
        out.append(v)
    return out


def _bbox_from_mask(mask: NDArray[np.bool_]) -> tuple[int, int, int, int, int, int] | None:
    if not np.any(mask):
        return None
    zz, yy, xx = np.where(mask)
    z0 = int(zz.min())
    z1 = int(zz.max()) + 1
    y0 = int(yy.min())
    y1 = int(yy.max()) + 1
    x0 = int(xx.min())
    x1 = int(xx.max()) + 1
    return z0, z1, y0, y1, x0, x1


def _expand_bbox(
    bbox: tuple[int, int, int, int, int, int],
    shape_zyx: tuple[int, int, int],
    margin_zyx: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    mz, my, mx = margin_zyx
    Z, Y, X = shape_zyx
    z0 = max(0, z0 - mz)
    z1 = min(Z, z1 + mz)
    y0 = max(0, y0 - my)
    y1 = min(Y, y1 + my)
    x0 = max(0, x0 - mx)
    x1 = min(X, x1 + mx)
    return z0, z1, y0, y1, x0, x1


def _resolve_label_path(row: pd.Series, root: Path) -> Path:
    lbl_path_raw = None
    if "label_path" in row.index:
        v = row.get("label_path")
        if isinstance(v, str) and v.strip():
            lbl_path_raw = v.strip()
    if lbl_path_raw is None:
        return root / "labels" / f"{str(row['case_id'])}.nii.gz"
    p = Path(lbl_path_raw)
    return p if p.is_absolute() else (root / p)


def _align_zyx(vol: NDArray[np.float32] | NDArray[np.uint8], target_shape_zyx: tuple[int, int, int]) -> NDArray[Any]:
    """Try to align a loaded NIfTI array to (Z,Y,X) by permuting axes if needed."""
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")
    if tuple(vol.shape) == tuple(target_shape_zyx):
        return vol
    perms = [
        (2, 1, 0),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
    ]
    for p in perms:
        v = np.transpose(vol, p)
        if tuple(v.shape) == tuple(target_shape_zyx):
            return v

    # If shapes still don't match, fall back to nearest-neighbor resize.
    tz, ty, tx = (int(target_shape_zyx[0]), int(target_shape_zyx[1]), int(target_shape_zyx[2]))
    sz, sy, sx = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    zoom_f = (tz / max(1, sz), ty / max(1, sy), tx / max(1, sx))
    v2 = nd_zoom(vol, zoom=zoom_f, order=0)

    # Make shape exact by crop/pad.
    out = np.zeros((tz, ty, tx), dtype=v2.dtype)
    cz = min(tz, int(v2.shape[0])); cy = min(ty, int(v2.shape[1])); cx = min(tx, int(v2.shape[2]))
    out[:cz, :cy, :cx] = v2[:cz, :cy, :cx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", required=True)
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])

    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--max-cands-per-case", type=int, default=32)
    ap.add_argument("--margin-zyx", default="6,12,12", help="bbox margin in voxels (Z,Y,X)")

    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--write-summary", action="store_true")

    args = ap.parse_args()

    probs_dir = Path(args.probs_dir)
    csv_path = Path(args.csv_path)
    root = Path(args.root)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    thr = float(args.threshold)
    min_size = int(args.min_size)
    max_cands = int(args.max_cands_per_case)
    margin_zyx = tuple(_parse_ints_csv(args.margin_zyx, 3))

    df = pd.read_csv(csv_path)
    df = df[df["split"] == str(args.split)].reset_index(drop=True)

    n_cases = 0
    n_cands = 0
    n_pos_cands = 0
    n_missing_probs = 0

    with out_jsonl.open("w") as f:
        for _, row in df.iterrows():
            case_id = str(row["case_id"])
            prob_path = probs_dir / f"{case_id}.npz"
            if not prob_path.exists():
                n_missing_probs += 1
                continue

            z = np.load(str(prob_path))
            probs = z["probs"].astype(np.float32, copy=False)
            if probs.ndim != 3:
                raise ValueError(f"Expected probs (Z,Y,X) for {case_id}, got {probs.shape}")

            pred = (probs >= thr).astype(np.uint8)
            if pred.max() == 0:
                n_cases += 1
                continue

            lbl, n = cc_label(pred)
            n = int(n)
            if n <= 0:
                n_cases += 1
                continue

            # Load GT for overlap stats.
            gt_path = _resolve_label_path(row, root)
            if gt_path.exists():
                gt, _ = load_nifti(str(gt_path))
                gt = _align_zyx(gt.astype(np.float32, copy=False), probs.shape)
                gt_bin = (gt > 0).astype(np.uint8)
            else:
                gt_bin = np.zeros_like(pred, dtype=np.uint8)

            sizes = np.bincount(lbl.ravel())
            comp_ids = np.arange(1, n + 1, dtype=np.int64)
            comp_sizes = sizes[1 : n + 1] if sizes.shape[0] >= (n + 1) else np.array([(lbl == i).sum() for i in comp_ids])

            # filter by min_size
            keep = comp_sizes >= max(0, min_size)
            comp_ids = comp_ids[keep]
            comp_sizes = comp_sizes[keep]

            # sort by size desc
            if comp_ids.size == 0:
                n_cases += 1
                continue
            order = np.argsort(comp_sizes)[::-1]
            comp_ids = comp_ids[order]
            comp_sizes = comp_sizes[order]

            if comp_ids.size > max_cands:
                comp_ids = comp_ids[:max_cands]
                comp_sizes = comp_sizes[:max_cands]

            for rank, (cid, sz) in enumerate(zip(comp_ids.tolist(), comp_sizes.tolist()), start=1):
                comp_mask = lbl == int(cid)
                bbox = _bbox_from_mask(comp_mask)
                if bbox is None:
                    continue
                bbox = _expand_bbox(bbox, probs.shape, margin_zyx)

                z0, z1, y0, y1, x0, x1 = bbox
                comp_probs = probs[comp_mask]
                max_prob = float(comp_probs.max()) if comp_probs.size else 0.0
                mean_prob = float(comp_probs.mean()) if comp_probs.size else 0.0

                gt_overlap = int((gt_bin[comp_mask] > 0).sum())
                is_pos = bool(gt_overlap > 0)

                rec: dict[str, Any] = {
                    "case_id": case_id,
                    "split": str(args.split),
                    "cand_rank": int(rank),
                    "cand_id": int(cid),
                    "size_vox": int(sz),
                    "bbox_zyxzyx": [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                    "threshold": float(thr),
                    "max_prob": float(max_prob),
                    "mean_prob": float(mean_prob),
                    "gt_overlap_vox": int(gt_overlap),
                    "is_pos": bool(is_pos),
                }
                f.write(json.dumps(rec) + "\n")

                n_cands += 1
                if is_pos:
                    n_pos_cands += 1

            n_cases += 1

    summary = {
        "split": str(args.split),
        "probs_dir": str(probs_dir),
        "csv_path": str(csv_path),
        "root": str(root),
        "threshold": float(thr),
        "min_size": int(min_size),
        "max_cands_per_case": int(max_cands),
        "margin_zyx": [int(x) for x in margin_zyx],
        "n_cases": int(n_cases),
        "n_candidates": int(n_cands),
        "n_pos_candidates": int(n_pos_cands),
        "n_missing_probs": int(n_missing_probs),
    }

    if args.write_summary:
        out_summary = out_jsonl.with_suffix(".summary.json")
        out_summary.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
