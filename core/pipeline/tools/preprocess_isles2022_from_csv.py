from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import typer

from src.preprocess.utils_io import load_nifti, sha256_file, save_nifti_with_affine

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    ses: str
    dwi_path: Path
    adc_path: Path
    mask_path: Path


def _find_single_session(case_dir: Path) -> str:
    ses_dirs = sorted([p.name for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("ses-")])
    if not ses_dirs:
        raise FileNotFoundError(f"No session dir under: {case_dir}")
    return ses_dirs[0]


def _resolve_isles2022_paths(raw_root: Path, derivatives_root: Path, case_id: str) -> CasePaths:
    case_dir = raw_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case dir not found: {case_dir}")

    ses = _find_single_session(case_dir)

    dwi_dir = case_dir / ses / "dwi"
    if not dwi_dir.exists():
        raise FileNotFoundError(f"dwi dir not found: {dwi_dir}")

    dwi_path = dwi_dir / f"{case_id}_{ses}_dwi.nii.gz"
    adc_path = dwi_dir / f"{case_id}_{ses}_adc.nii.gz"

    mask_path = derivatives_root / case_id / ses / f"{case_id}_{ses}_msk.nii.gz"

    missing = [p for p in [dwi_path, adc_path, mask_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(str(p) for p in missing))

    return CasePaths(case_id=case_id, ses=ses, dwi_path=dwi_path, adc_path=adc_path, mask_path=mask_path)


def _bbox_from_foreground(fg: np.ndarray) -> tuple[slice, slice, slice]:
    # fg: (X,Y,Z) bool
    if fg.dtype != np.bool_:
        fg = fg.astype(bool)
    if not np.any(fg):
        # no foreground; return full
        return slice(0, fg.shape[0]), slice(0, fg.shape[1]), slice(0, fg.shape[2])

    coords = np.where(fg)
    x0, x1 = int(coords[0].min()), int(coords[0].max())
    y0, y1 = int(coords[1].min()), int(coords[1].max())
    z0, z1 = int(coords[2].min()), int(coords[2].max())
    return slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1)


def _expand_slices(
    sl: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    margin: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    out: list[slice] = []
    for i, (s, m) in enumerate(zip(sl, margin)):
        start = 0 if s.start is None else int(s.start)
        stop = shape[i] if s.stop is None else int(s.stop)
        start2 = max(0, start - int(m))
        stop2 = min(shape[i], stop + int(m))
        out.append(slice(start2, stop2))
    return out[0], out[1], out[2]


def _crop_to_foreground(img_cxyz: np.ndarray, mask_xyz: np.ndarray, margin: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    # img_cxyz: (C,X,Y,Z)
    fg = np.any(img_cxyz != 0, axis=0)
    sl = _bbox_from_foreground(fg)
    sl = _expand_slices(sl, fg.shape, margin)

    img_out = img_cxyz[:, sl[0], sl[1], sl[2]]
    mask_out = mask_xyz[sl[0], sl[1], sl[2]]
    return img_out, mask_out


def _resample_to_target_grid(
    *,
    src_img: Any,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    order: int,
) -> Any:
    import nibabel as nib
    from nibabel.processing import resample_from_to

    # resample_from_to accepts (shape, affine) or image as target
    return resample_from_to(src_img, (target_shape, target_affine), order=order)


def _make_target_grid_from_ref(ref_img: Any, target_spacing: tuple[float, float, float]) -> tuple[tuple[int, int, int], np.ndarray]:
    import nibabel as nib
    from nibabel.processing import resample_to_output

    # First, resample reference to desired voxel sizes, letting nibabel pick output shape/affine.
    ref_rs = resample_to_output(ref_img, voxel_sizes=target_spacing, order=1)
    shape_xyz = tuple(int(x) for x in ref_rs.shape[:3])
    affine = np.asarray(ref_rs.affine, dtype=np.float64)
    return shape_xyz, affine


def _iter_case_ids(csv_path: Path) -> Iterable[str]:
    df = pd.read_csv(str(csv_path))
    if "case_id" not in df.columns:
        raise ValueError(f"case_id column not found in {csv_path}")
    return [str(x) for x in df["case_id"].tolist()]


@app.command()
def main(
    csv_path: Path = typer.Option(..., help="Input split CSV (case_id,split). Used only for case list."),
    raw_root: Path = typer.Option(..., help="ISLES-2022 root (contains sub-*)."),
    derivatives_root: Path = typer.Option(..., help="ISLES-2022 derivatives root (contains masks)."),
    out_root: Path = typer.Option(..., help="Output processed root (images/, labels/)."),
    target_spacing: str = typer.Option("2.0,2.0,2.0", help="Target spacing mm as 'x,y,z'."),
    crop_margin: str = typer.Option("8,8,4", help="CropForeground margin in voxels after resampling (x,y,z)."),
    limit: int = typer.Option(0, help="Process only first N cases (0=all)."),
    log_path: Path = typer.Option("", help="Write JSONL log (default: <out_root>/preprocess_log.jsonl)."),
):
    ts = tuple(float(x) for x in target_spacing.split(","))
    if len(ts) != 3 or any((not np.isfinite(v) or v <= 0) for v in ts):
        raise ValueError(f"Invalid target_spacing: {target_spacing!r}")

    margin = tuple(int(x) for x in crop_margin.split(","))
    if len(margin) != 3 or any(m < 0 for m in margin):
        raise ValueError(f"Invalid crop_margin: {crop_margin!r}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    log_path_eff = log_path if str(log_path).strip() else (out_root / "preprocess_log.jsonl")
    log_path_eff.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "raw_root": str(raw_root),
        "derivatives_root": str(derivatives_root),
        "csv_path": str(csv_path),
        "out_root": str(out_root),
        "target_spacing_mm": [float(ts[0]), float(ts[1]), float(ts[2])],
        "crop_margin_vox": [int(margin[0]), int(margin[1]), int(margin[2])],
    }
    (out_root / "preprocess_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    import nibabel as nib

    case_ids = list(_iter_case_ids(csv_path))
    if limit and len(case_ids) > int(limit):
        case_ids = case_ids[: int(limit)]

    ok = 0
    skipped = 0

    with log_path_eff.open("w", encoding="utf-8") as f:
        for i, case_id in enumerate(case_ids):
            rec: dict[str, Any] = {"case_id": case_id}
            try:
                paths = _resolve_isles2022_paths(raw_root, derivatives_root, case_id)
                rec["ses"] = paths.ses

                dwi_img = nib.load(str(paths.dwi_path))
                adc_img = nib.load(str(paths.adc_path))
                msk_img = nib.load(str(paths.mask_path))

                target_shape_xyz, target_affine = _make_target_grid_from_ref(dwi_img, ts)  # xyz

                dwi_rs = _resample_to_target_grid(
                    src_img=dwi_img,
                    target_shape=target_shape_xyz,
                    target_affine=target_affine,
                    order=1,
                )
                adc_rs = _resample_to_target_grid(
                    src_img=adc_img,
                    target_shape=target_shape_xyz,
                    target_affine=target_affine,
                    order=1,
                )
                msk_rs = _resample_to_target_grid(
                    src_img=msk_img,
                    target_shape=target_shape_xyz,
                    target_affine=target_affine,
                    order=0,
                )

                dwi = dwi_rs.get_fdata().astype(np.float32)
                adc = adc_rs.get_fdata().astype(np.float32)
                msk = msk_rs.get_fdata().astype(np.float32)

                # Ensure strictly 3D
                dwi = np.asarray(dwi)[..., 0] if dwi.ndim == 4 and dwi.shape[-1] == 1 else np.asarray(dwi)
                adc = np.asarray(adc)[..., 0] if adc.ndim == 4 and adc.shape[-1] == 1 else np.asarray(adc)
                msk = np.asarray(msk)[..., 0] if msk.ndim == 4 and msk.shape[-1] == 1 else np.asarray(msk)

                if dwi.ndim != 3 or adc.ndim != 3 or msk.ndim != 3:
                    raise RuntimeError(f"Unexpected ndim after resample: dwi={dwi.shape}, adc={adc.shape}, msk={msk.shape}")

                img_cxyz = np.stack([dwi, adc], axis=0)  # (C,X,Y,Z)
                msk_xyz = (msk > 0.5).astype(np.float32)

                img_cxyz, msk_xyz = _crop_to_foreground(img_cxyz, msk_xyz, margin)

                out_img = out_root / "images" / f"{case_id}.nii.gz"
                out_lbl = out_root / "labels" / f"{case_id}.nii.gz"

                save_nifti_with_affine(img_cxyz, target_affine, str(out_img))
                save_nifti_with_affine(msk_xyz, target_affine, str(out_lbl))

                rec.update(
                    {
                        "ok": True,
                        "inputs": {
                            "dwi": str(paths.dwi_path),
                            "adc": str(paths.adc_path),
                            "msk": str(paths.mask_path),
                        },
                        "outputs": {"image": str(out_img), "label": str(out_lbl)},
                        "hash": {
                            "input": {
                                "dwi": sha256_file(str(paths.dwi_path)),
                                "adc": sha256_file(str(paths.adc_path)),
                                "msk": sha256_file(str(paths.mask_path)),
                            },
                            "output": {
                                "image": sha256_file(str(out_img)),
                                "label": sha256_file(str(out_lbl)),
                            },
                        },
                        "shape_cxyz": [int(x) for x in img_cxyz.shape],
                    }
                )
                ok += 1
            except Exception as e:
                rec.update({"ok": False, "error": str(e)})
                skipped += 1

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(case_ids)}] ok={ok} skipped={skipped}", flush=True)

    print(f"done: ok={ok} skipped={skipped} out_root={out_root}")


if __name__ == "__main__":
    app()
