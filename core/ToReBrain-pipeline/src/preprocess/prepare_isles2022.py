"""Prepare ISLES-2022 into the processed layout used by our 3D U-Net.

- Reads raw BIDS-like structure at raw_root (sub-*/ses-0001/dwi/*.nii.gz, anat/FLAIR optional)
- Reads labels from derivatives_root/sub-*/ses-0001/*_msk.nii.gz
- Stacks selected modalities (e.g., DWI, ADC, FLAIR) into channel-first volumes
- Applies spacing resample, intensity scaling, and foreground crop
- Writes processed volumes to out_root/images and labels to out_root/labels
- Produces a split CSV with columns [case_id, split]

Notes:
- Not all ISLES cases contain FLAIR. If you request a modality that's missing for a case,
  that case will be skipped (and logged) to keep channel counts consistent.
"""
from pathlib import Path
from typing import Dict, List, Sequence
import json
import random
import numpy as np
import monai.transforms as mt
import nibabel as nib
import nibabel.processing as nibproc
import typer
from .utils_io import load_nifti, save_nifti, sha256_file

app = typer.Typer(add_completion=False)


def _build_transforms(target_spacing: Sequence[float]):
    return _build_transforms_with_intensity(target_spacing, intensity="percentile_chwise")


def _build_transforms_with_intensity(target_spacing: Sequence[float], intensity: str):
    intensity = (intensity or "none").lower()
    tx = [
        mt.EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
        mt.Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
    ]
    if intensity in {"percentile_chwise", "percentile", "pctl"}:
        # Critical: for multi-channel inputs (DWI/ADC/FLAIR...), always scale per-channel.
        tx.append(
            mt.ScaleIntensityRangePercentilesd(
                keys="image",
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            )
        )
    elif intensity in {"none", "off", "false"}:
        pass
    else:
        raise ValueError(f"Unknown intensity mode: {intensity!r}")
    tx.append(mt.CropForegroundd(keys=["image", "label"], source_key="image"))
    return mt.Compose(tx)


def _find_single(paths: List[str]) -> Path:
    for p in paths:
        candidate = Path(p)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No files found for: {paths}")


def _parse_modalities(modalities: str) -> List[str]:
    mods = [m.strip().upper() for m in (modalities or "").split(",") if m.strip()]
    if not mods:
        raise ValueError("modalities must be non-empty (e.g., 'DWI,ADC' or 'DWI,FLAIR')")
    return mods


def _find_modality_path(case_dir: Path, case_id: str, modality: str) -> Path:
    m = modality.strip().upper()
    if m == "DWI":
        return _find_single([str(case_dir / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz")])
    if m == "ADC":
        return _find_single([str(case_dir / "dwi" / f"{case_id}_ses-0001_adc.nii.gz")])
    if m == "FLAIR":
        anat_dir = case_dir / "anat"
        candidates: List[str] = [
            str(anat_dir / f"{case_id}_ses-0001_FLAIR.nii.gz"),
            str(anat_dir / f"{case_id}_ses-0001_flair.nii.gz"),
        ]
        # Some datasets use slightly different naming; fall back to a glob within anat/.
        if anat_dir.exists():
            glob_hits = sorted([p for p in anat_dir.glob(f"{case_id}_ses-0001_*FLAIR*.nii.gz")])
            candidates.extend([str(p) for p in glob_hits])
        return _find_single(candidates)
    raise ValueError(f"Unknown modality: {modality!r} (supported: DWI, ADC, FLAIR)")


def _resample_img_to_ref(img: nib.Nifti1Image, ref_img: nib.Nifti1Image, order: int) -> nib.Nifti1Image:
    # Resample only when grid differs; keeps stacking safe for multi-modality.
    if img.shape != ref_img.shape or not np.allclose(img.affine, ref_img.affine):
        return nibproc.resample_from_to(img, (ref_img.shape, ref_img.affine), order=order)
    return img


def preprocess_case(case_id: str, raw_root: Path, derivatives_root: Path, out_root: Path, modalities: Sequence[str], tx):
    # Locate inputs
    case_dir = raw_root / case_id / "ses-0001"
    mask_path = _find_single([str(derivatives_root / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz")])

    img_list = []
    input_paths: Dict[str, str] = {}
    ref_img = None
    for m in modalities:
        mp = _find_modality_path(case_dir, case_id, m)
        arr, img = load_nifti(str(mp))
        input_paths[m] = str(mp)
        if ref_img is None:
            ref_img = img
            img_list.append(arr)
        else:
            # Align other modalities to the reference grid before stacking.
            img_res = _resample_img_to_ref(img, ref_img, order=1)
            img_list.append(img_res.get_fdata().astype(np.float32))

    if ref_img is None:
        raise RuntimeError(f"No reference image loaded for case {case_id}")

    mask_arr, mask_img = load_nifti(str(mask_path))
    # Ensure mask matches the reference grid (nearest neighbor).
    if mask_img.shape != ref_img.shape or not np.allclose(mask_img.affine, ref_img.affine):
        mask_img = _resample_img_to_ref(mask_img, ref_img, order=0)
        mask_arr = mask_img.get_fdata().astype(np.float32)

    image = np.stack(img_list, axis=0)
    label = mask_arr[None, ...]

    sample = {"image": image, "label": label}
    sample_tx = tx(sample)
    image_tx = sample_tx["image"]
    label_tx = sample_tx["label"][0]

    out_img_path = out_root / "images" / f"{case_id}.nii.gz"
    out_lbl_path = out_root / "labels" / f"{case_id}.nii.gz"
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

    save_nifti(image_tx, ref_img, str(out_img_path))
    save_nifti(label_tx, ref_img, str(out_lbl_path))

    return {
        "case_id": case_id,
        "modalities": list(modalities),
        "input_paths": {"mask": str(mask_path), **{k.lower(): v for k, v in input_paths.items()}},
        "output_paths": {"image": str(out_img_path), "label": str(out_lbl_path)},
        "hash": {
            "input": {"mask": sha256_file(str(mask_path)), **{k.lower(): sha256_file(v) for k, v in input_paths.items()}},
            "output": {"image": sha256_file(str(out_img_path)), "label": sha256_file(str(out_lbl_path))},
        },
    }


def write_splits(case_ids: List[str], csv_path: Path, train_ratio: float, val_ratio: float, seed: int = 42):
    random.seed(seed)
    shuffled = case_ids[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    lines = ["case_id,split"]
    lines += [f"{cid},train" for cid in train_ids]
    lines += [f"{cid},val" for cid in val_ids]
    lines += [f"{cid},test" for cid in test_ids]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines))
    return {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}


@app.command()
def main(
    raw_root: Path = typer.Option(..., help="Path to ISLES-2022 root (with sub-*)."),
    derivatives_root: Path = typer.Option(..., help="Path to derivatives containing *_msk.nii.gz."),
    out_root: Path = typer.Option("data/processed/my_dataset", help="Output processed root."),
    log_path: Path = typer.Option("logs/preprocess_isles2022.jsonl", help="Where to write processing log."),
    split_csv: Path = typer.Option("data/splits/my_dataset_train_val_test.csv", help="Train/val/test split CSV."),
    target_spacing: str = typer.Option("1.5,1.5,1.5", help="Spacing mm."),
    modalities: str = typer.Option(
        "DWI,ADC",
        help="Comma-separated modalities to stack: DWI,ADC (default) or DWI,FLAIR, etc. Missing modalities will skip the case.",
    ),
    intensity: str = typer.Option(
        "percentile_chwise",
        help="Intensity normalization in preprocessing: percentile_chwise | none. (Multi-channel safety: percentile is applied per-channel)",
    ),
    train_ratio: float = typer.Option(0.8, help="Train split ratio."),
    val_ratio: float = typer.Option(0.1, help="Val split ratio."),
    seed: int = typer.Option(42, help="Shuffle seed."),
):
    spacing = [float(x) for x in target_spacing.split(",")]
    tx = _build_transforms_with_intensity(spacing, intensity=intensity)
    mods = _parse_modalities(modalities)

    case_dirs = sorted([p for p in Path(raw_root).iterdir() if p.name.startswith("sub-")])
    case_ids = [p.name for p in case_dirs]

    processed_case_ids: List[str] = []

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        for cid in case_ids:
            try:
                record = preprocess_case(cid, Path(raw_root), Path(derivatives_root), Path(out_root), mods, tx)
                processed_case_ids.append(cid)
                print(f"processed {cid}", flush=True)
            except Exception as e:
                record = {"case_id": cid, "error": str(e), "modalities": mods}
                print(f"skipped  {cid}: {e}", flush=True)
            lf.write(json.dumps(record) + "\n")

    if not processed_case_ids:
        raise RuntimeError("No cases were processed successfully. Check raw_root/derivatives_root and modalities.")

    split_info = write_splits(processed_case_ids, Path(split_csv), train_ratio, val_ratio, seed)
    print(f"Splits: {split_info}", flush=True)


if __name__ == "__main__":
    app()
