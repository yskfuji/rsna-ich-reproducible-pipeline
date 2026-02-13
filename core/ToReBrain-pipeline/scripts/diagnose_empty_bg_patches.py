"""Estimate whether empty-background patches exist for a given patch size.

This answers: for patch_size (pD,pH,pW), does each case have *any* window with 0 foreground voxels?
We approximate by random window sampling with a 3D prefix sum for O(1) window queries.

Usage (zsh):
  cd ToReBrain-pipeline
  PYTHONPATH=$PWD /opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output \
    python scripts/diagnose_empty_bg_patches.py \
      --csv data/splits/my_dataset_dwi_adc_flair_train_val_test.csv \
      --root data/processed/my_dataset_dwi_adc_flair \
      --split train \
      --patch 56,56,24 \
      --trials 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.datasets.isles_dataset import IslesVolumeDataset


def parse_triplet(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--patch must be like '56,56,24'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def floor_to_multiple(x: int, m: int) -> int:
    x = int(x)
    m = int(max(1, m))
    return x - (x % m)


def effective_patch(patch: Tuple[int, int, int], shape: Tuple[int, int, int], multiple: int = 8) -> Tuple[int, int, int]:
    pD, pH, pW = (int(patch[0]), int(patch[1]), int(patch[2]))
    D, H, W = (int(shape[0]), int(shape[1]), int(shape[2]))
    effD = max(multiple, min(pD, floor_to_multiple(D, multiple)))
    effH = max(multiple, min(pH, floor_to_multiple(H, multiple)))
    effW = max(multiple, min(pW, floor_to_multiple(W, multiple)))
    return effD, effH, effW


def prefix_sum_3d(bin_vol: np.ndarray) -> np.ndarray:
    v = bin_vol.astype(np.int64, copy=False)
    p = np.pad(v, ((1, 0), (1, 0), (1, 0)), mode="constant", constant_values=0)
    return p.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)


def window_sum(prefix: np.ndarray, z0: int, y0: int, x0: int, z1: int, y1: int, x1: int) -> int:
    return int(
        prefix[z1, y1, x1]
        - prefix[z0, y1, x1]
        - prefix[z1, y0, x1]
        - prefix[z1, y1, x0]
        + prefix[z0, y0, x1]
        + prefix[z0, y1, x0]
        + prefix[z1, y0, x0]
        - prefix[z0, y0, x0]
    )


@dataclass
class CaseStats:
    case_id: str
    shape: Tuple[int, int, int]
    mask_sum: int
    min_fg_vox: int
    found_empty: bool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--patch", required=True, type=parse_triplet)
    ap.add_argument("--trials", type=int, default=4096)
    ap.add_argument("--normalize", default="nonzero_zscore")
    args = ap.parse_args()

    pD, pH, pW = args.patch
    trials = int(max(1, args.trials))

    ds = IslesVolumeDataset(csv_path=args.csv, split=args.split, root=args.root, normalize=args.normalize)

    rng = np.random.default_rng(0)
    stats = []
    for i in range(len(ds)):
        s = ds[i]
        mask = (s["mask"] > 0.5)
        img = s["image"]
        case_id = s["case_id"]
        D, H, W = mask.shape

        pDe, pHe, pWe = effective_patch((pD, pH, pW), (D, H, W), multiple=8)

        fg_total = int(mask.sum())
        if fg_total <= 0:
            stats.append(CaseStats(case_id=case_id, shape=(D, H, W), mask_sum=0, min_fg_vox=0, found_empty=True))
            continue

        brain = (np.abs(img).max(axis=0) > 0.0)
        fg_prefix = prefix_sum_3d(mask)
        brain_prefix = prefix_sum_3d(brain)

        max_z0 = max(D - pDe, 0)
        max_y0 = max(H - pHe, 0)
        max_x0 = max(W - pWe, 0)

        best = None
        found = False
        for _ in range(trials):
            z0 = int(rng.integers(0, max_z0 + 1))
            y0 = int(rng.integers(0, max_y0 + 1))
            x0 = int(rng.integers(0, max_x0 + 1))
            z1, y1, x1 = z0 + pDe, y0 + pHe, x0 + pWe

            if window_sum(brain_prefix, z0, y0, x0, z1, y1, x1) <= 0:
                continue

            fg = window_sum(fg_prefix, z0, y0, x0, z1, y1, x1)
            if best is None or fg < best:
                best = fg
            if fg == 0:
                found = True
                best = 0
                break

        if best is None:
            # no inside-brain window found in trials; treat as not found
            best = fg_total
            found = False

        stats.append(CaseStats(case_id=case_id, shape=(D, H, W), mask_sum=fg_total, min_fg_vox=int(best), found_empty=bool(found)))

    n = len(stats)
    n_pos = sum(1 for s in stats if s.mask_sum > 0)
    n_pos_has_empty = sum(1 for s in stats if s.mask_sum > 0 and s.found_empty)
    min_fg = np.array([s.min_fg_vox for s in stats if s.mask_sum > 0], dtype=np.int64)

    print(f"split={args.split} n={n} patch={args.patch} trials={trials}")
    print(f"pos_cases={n_pos}  pos_cases_with_empty_window≈{n_pos_has_empty} ({(n_pos_has_empty/max(1,n_pos)):.3f})")
    if n_pos > 0:
        print(
            "min_fg_vox (pos cases)  "
            f"min={int(min_fg.min())}  med={int(np.median(min_fg))}  p90={int(np.percentile(min_fg, 90))}  max={int(min_fg.max())}"
        )


if __name__ == "__main__":
    main()
