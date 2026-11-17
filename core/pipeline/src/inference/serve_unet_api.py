"""Minimal FastAPI server wrapping trained UNet models."""
from pathlib import Path
import tempfile
import torch
from fastapi import FastAPI, UploadFile, HTTPException, Depends, status
import nibabel as nib
from ..models.unet_3d import UNet3D
from ..training.utils_train import prepare_device
from ..security.access_control import authorize
from ..inference.infer_sliding_window import sliding_window_inference_3d

app = FastAPI()

device = prepare_device()
model_cache = {}


def load_model(model_path: str):
    if model_path in model_cache:
        return model_cache[model_path]
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    dummy_ch = 1
    model = UNet3D(in_channels=dummy_ch, out_channels=1)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    model_cache[model_path] = model
    return model


@app.post("/infer")
async def infer(file: UploadFile, model_path: str, token: str = Depends(authorize)):
    if file.filename.split(".")[-1] not in {"nii", "nii.gz"}:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only NIfTI allowed")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp.flush()
        img = nib.load(tmp.name)
        volume = img.get_fdata().astype("float32")
    if volume.ndim == 3:
        volume = volume[None, ...]
    model = load_model(model_path)
    probs = sliding_window_inference_3d(volume, model, patch_size=(96, 96, 64), overlap=0.5, device=device)
    mask = (probs[0, 0] > 0.5).astype("uint8")
    return {"shape": mask.shape, "sum": int(mask.sum())}
