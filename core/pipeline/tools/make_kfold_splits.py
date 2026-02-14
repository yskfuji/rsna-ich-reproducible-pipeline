"""Create k-fold train/val splits while keeping the existing test split fixed.

- Reads an input CSV with columns: case_id, split (train/val/test)
- Keeps rows with split==test untouched in every output fold
- Re-partitions (train+val) into k folds: fold_i_val is one fold, fold_i_train is the rest
- Stratifies approximately by (gt_pos, slice_spacing_bucket) to reduce fold skew

The slice-spacing proxy matches IslesVolumeDataset/evaluate_isles conventions:
  slice_spacing_mm = max(header.get_zooms()[:3])

Example:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/make_kfold_splits.py \
    --csv-in data/splits/my_dataset_train_val_test.csv \
    --root data/processed/my_dataset \
    --out-dir data/splits/kfold5_my_dataset \
    --k 5 --seed 42 --slice-spacing-thr-mm 3.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _slice_spacing_mm(path: Path) -> float | None:
    import nibabel as nib

    img = nib.load(str(path))
    z = img.header.get_zooms()
    if z is None or len(z) < 3:
        return None
    a = [float(z[0]), float(z[1]), float(z[2])]
    if not all(np.isfinite(a)):
        return None
    return float(max(a))


def _gt_vox(path: Path) -> int:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return int(np.sum(data > 0.5))


@dataclass(frozen=True)
class CaseMeta:
    case_id: str
    gt_pos: bool
    slice_bucket: str


def _make_buckets(case_ids: list[str], *, root: Path, slice_thr_mm: float) -> dict[str, CaseMeta]:
    out: dict[str, CaseMeta] = {}
    thr = float(slice_thr_mm)
    for cid in case_ids:
        img_path = root / "images" / f"{cid}.nii.gz"
        lbl_path = root / "labels" / f"{cid}.nii.gz"
        ss = _slice_spacing_mm(img_path)
        gt = _gt_vox(lbl_path) if lbl_path.exists() else 0
        bucket = "unknown"
        if ss is not None and np.isfinite(ss):
            bucket = f"le_{thr:g}mm" if float(ss) <= thr else f"gt_{thr:g}mm"
        out[cid] = CaseMeta(case_id=cid, gt_pos=bool(gt > 0), slice_bucket=bucket)
    return out


def _round_robin_split(indices: list[int], k: int, rng: np.random.Generator) -> list[list[int]]:
    # Shuffle then assign round-robin to keep fold sizes close.
    idx = indices[:]
    rng.shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, v in enumerate(idx):
        folds[i % k].append(int(v))
    return folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in", required=True)
    ap.add_argument("--root", required=True, help="Processed dataset root (expects images/ and labels/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slice-spacing-thr-mm", type=float, default=3.0)
    args = ap.parse_args()

    k = int(args.k)
    if k < 2:
        raise ValueError("--k must be >=2")

    root = Path(args.root).expanduser()
    csv_in = Path(args.csv_in).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_csv(str(csv_in))
    if "case_id" not in df.columns or "split" not in df.columns:
        raise ValueError("CSV must have columns: case_id, split")

    df = df.copy()
    df["case_id"] = df["case_id"].astype(str)
    df["split"] = df["split"].astype(str)

    df_non_test = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    non_test_case_ids = [str(x) for x in df_non_test["case_id"].tolist()]

    # Build stratification metadata by reading NIfTI headers + labels.
    meta = _make_buckets(non_test_case_ids, root=root, slice_thr_mm=float(args.slice_spacing_thr_mm))

    # Build strata -> list of row indices within df_non_test.
    strata: dict[str, list[int]] = {}
    for i, row in df_non_test.iterrows():
        cid = str(row["case_id"])
        m = meta.get(cid)
        if m is None:
            key = "unknown"
        else:
            key = f"gtpos={int(m.gt_pos)}|slice={m.slice_bucket}"
        strata.setdefault(key, []).append(int(i))

    rng = np.random.default_rng(int(args.seed))

    # For each stratum, split its indices into k folds, then merge across strata.
    folds: list[list[int]] = [[] for _ in range(k)]
    for _, idxs in sorted(strata.items(), key=lambda kv: kv[0]):
        parts = _round_robin_split(idxs, k=k, rng=rng)
        for fi in range(k):
            folds[fi].extend(parts[fi])

    # Write fold CSVs.
    summary: dict[str, object] = {
        "csv_in": str(csv_in),
        "root": str(root),
        "k": int(k),
        "seed": int(args.seed),
        "slice_spacing_thr_mm": float(args.slice_spacing_thr_mm),
        "n_non_test": int(len(df_non_test)),
        "n_test": int(len(df_test)),
        "folds": [],
        "strata": {k: int(len(v)) for k, v in strata.items()},
    }

    for fi in range(k):
        val_idx = set(int(x) for x in folds[fi])
        split_col = []
        for i in range(len(df_non_test)):
            split_col.append("val" if i in val_idx else "train")

        df_fold = df_non_test.copy()
        df_fold["split"] = split_col
        df_out = pd.concat([df_fold, df_test], ignore_index=True)

        out_csv = out_dir / f"fold{fi}.csv"
        df_out.to_csv(str(out_csv), index=False)

        fold_info = {
            "fold": int(fi),
            "train": int(np.sum(df_out["split"] == "train")),
            "val": int(np.sum(df_out["split"] == "val")),
            "test": int(np.sum(df_out["split"] == "test")),
        }
        summary["folds"].append(fold_info)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
