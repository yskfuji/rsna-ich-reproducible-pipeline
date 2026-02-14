"""Generate a simple nnU-Net-like preprocessing plan from the processed dataset.

This is intentionally lightweight and *consistent with this repo's own metadata usage*:
- voxel sizes are read from NIfTI header.get_zooms()[:3]
- slice spacing proxy is max(zooms[:3])

Outputs a JSON plan with per-split spacing stats and a few target-spacing candidates.

Example:
  /opt/anaconda3/envs/medseg_unet/bin/python tools/plan_nnunet_like_preprocess.py \
    --config configs/generated/_recipe_20251227/medseg_3d_unet_recipe_dwi_adc_fp50_dicebce_patch96_e200.yaml \
    --out results/diag/nnunet_plan_my_dataset.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def _percentiles(x: np.ndarray, ps: Iterable[float]) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": None for p in ps}
    vals = np.percentile(x, list(ps)).astype(np.float64)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _stats_1d(x: list[float]) -> dict[str, float | None]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, **_percentiles(a, [10, 25, 50, 75, 90])}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        **_percentiles(a, [10, 25, 50, 75, 90]),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _resolve_paths(cfg: dict[str, Any], repo: Path) -> tuple[Path, Path, list[str]]:
    data_root = Path(cfg["data"]["root"]).expanduser()
    csv_path = Path(cfg["data"]["csv_path"]).expanduser()
    if not data_root.is_absolute():
        data_root = (repo / data_root).resolve()
    if not csv_path.is_absolute():
        csv_path = (repo / csv_path).resolve()
    mods = [str(m) for m in (cfg.get("data", {}) or {}).get("modalities", [])]
    return data_root, csv_path, mods


def _nifti_zooms_mm(path: Path) -> tuple[float, float, float] | None:
    import nibabel as nib

    img = nib.load(str(path))
    z = img.header.get_zooms()
    if z is None or len(z) < 3:
        return None
    return float(z[0]), float(z[1]), float(z[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config containing data.root and data.csv_path")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--max-cases", type=int, default=0, help="Limit cases per split (0=all)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo / cfg_path).resolve()
    cfg = _load_yaml(cfg_path)

    data_root, csv_path, modalities = _resolve_paths(cfg, repo)

    import pandas as pd

    df = pd.read_csv(str(csv_path))
    if "split" not in df.columns or "case_id" not in df.columns:
        raise ValueError(f"CSV must contain columns case_id,split: {csv_path}")

    out: dict[str, Any] = {
        "repo": str(repo),
        "config": str(cfg_path),
        "data_root": str(data_root),
        "csv_path": str(csv_path),
        "modalities": modalities,
        "splits": {},
        "candidates": {},
        "notes": [
            "zooms_mm are read via NIfTI header.get_zooms()[:3], consistent with IslesVolumeDataset meta.",
            "slice_spacing_mm proxy = max(zooms_mm).",
        ],
    }

    zooms_all: list[tuple[float, float, float]] = []

    for split in ["train", "val", "test"]:
        df_s = df[df["split"] == split].reset_index(drop=True)
        if int(args.max_cases) > 0:
            df_s = df_s.iloc[: int(args.max_cases)].reset_index(drop=True)

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        slice_sp: list[float] = []

        for _, row in df_s.iterrows():
            case_id = str(row["case_id"])
            img_path = data_root / "images" / f"{case_id}.nii.gz"
            zmm = _nifti_zooms_mm(img_path)
            if zmm is None:
                continue
            x, y, z = zmm
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
            slice_sp.append(float(max(x, y, z)))
            zooms_all.append((float(x), float(y), float(z)))

        out["splits"][split] = {
            "n": int(len(df_s)),
            "voxel_sizes_mm": {
                "x": _stats_1d(xs),
                "y": _stats_1d(ys),
                "z": _stats_1d(zs),
                "slice_spacing_mm": _stats_1d(slice_sp),
            },
        }

    if zooms_all:
        arr = np.asarray(zooms_all, dtype=np.float64)
        med = np.median(arr, axis=0).astype(np.float64)
        p75 = np.percentile(arr, 75, axis=0).astype(np.float64)
        # candidates are *suggestions*; selection depends on memory + anisotropy.
        out["candidates"] = {
            "median_spacing_mm": [float(med[0]), float(med[1]), float(med[2])],
            "p75_spacing_mm": [float(p75[0]), float(p75[1]), float(p75[2])],
            "isotropic_1p5mm": [1.5, 1.5, 1.5],
            "isotropic_2p0mm": [2.0, 2.0, 2.0],
            "cap_z_at_2p0_using_median_xy": [float(med[0]), float(med[1]), float(min(med[2], 2.0))],
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
