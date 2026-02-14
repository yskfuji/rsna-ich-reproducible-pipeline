"""Preprocessing pipeline for 3D medical segmentation datasets (ISLES-compatible).

- Load raw modalities and mask
- Resample to target spacing
- Intensity normalization
- Crop foreground
- Save processed volumes and log hashes
"""
import json
from pathlib import Path
from typing import List
import numpy as np
import monai.transforms as mt
import typer
from .utils_io import load_nifti, save_nifti, sha256_file

app = typer.Typer(add_completion=False)


def _build_transforms(target_spacing: List[float]):
    return mt.Compose(
        [
        # Arrays are already channel-first; explicitly set channel_dim to avoid metadata requirement
        mt.EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
        mt.Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        mt.ScaleIntensityRangePercentilesd(
            keys="image",
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            channel_wise=True,
        ),
        mt.CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )


@app.command()
def main(
    raw_root: str = typer.Option(..., help="Raw dataset root with train/val/test"),
    out_root: str = typer.Option(..., help="Output processed root"),
    use_modalities: str = typer.Option("DWI,ADC", help="Comma-separated modality names"),
    target_spacing: str = typer.Option("1.5,1.5,1.5", help="Spacing mm"),
    log_path: str = typer.Option(..., help="JSONL log path"),
):
    raw_root_p = Path(raw_root)
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)

    modalities = [m.strip() for m in use_modalities.split(",") if m.strip()]
    spacing = [float(x) for x in target_spacing.split(",")]

    log_f = Path(log_path)
    log_f.parent.mkdir(parents=True, exist_ok=True)
    tx = _build_transforms(spacing)

    cases = sorted([p.name for p in (raw_root_p / "train").iterdir() if p.is_dir()])

    for case_id in cases:
        record = preprocess_single_case(case_id, raw_root_p, out_root_p, modalities, spacing, tx)
        with log_f.open("a") as f:
            f.write(json.dumps(record) + "\n")


def preprocess_single_case(case_id, raw_root, out_root, modalities, spacing, tx):
    img_list = []
    ref_img = None
    input_paths = {}

    for m in modalities:
        path = raw_root / "train" / case_id / f"{m.lower()}.nii.gz"
        data, img = load_nifti(str(path))
        input_paths[m] = str(path)
        if ref_img is None:
            ref_img = img
        img_list.append(data)

    mask_path = raw_root / "train" / case_id / "mask.nii.gz"
    mask_data, _ = load_nifti(str(mask_path))

    image = np.stack(img_list, axis=0)
    label = mask_data[None, ...]

    sample = {"image": image, "label": label}
    sample_tx = tx(sample)
    image_tx = sample_tx["image"]
    label_tx = sample_tx["label"][0]

    out_img_path = out_root / "images" / f"{case_id}.nii.gz"
    out_lbl_path = out_root / "labels" / f"{case_id}.nii.gz"

    save_nifti(image_tx, ref_img, str(out_img_path))
    save_nifti(label_tx, ref_img, str(out_lbl_path))

    record = {
        "case_id": case_id,
        "modalities": modalities,
        "target_spacing": spacing,
        "input_paths": input_paths,
        "output_paths": {"image": str(out_img_path), "label": str(out_lbl_path)},
        "hash": {
            "input": {m: sha256_file(p) for m, p in input_paths.items()},
            "output": {"image": sha256_file(str(out_img_path)), "label": sha256_file(str(out_lbl_path))},
        },
    }
    return record


if __name__ == "__main__":
    app()
