"""Basic input validation helpers."""
from pathlib import Path


def validate_nii_path(path: str):
    if not path.endswith((".nii", ".nii.gz")):
        raise ValueError("Only .nii or .nii.gz allowed")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p
