"""CLI for single-case inference using a trained model."""
from pathlib import Path
import numpy as np
import torch
import typer
from ..inference.infer_sliding_window import sliding_window_inference_3d
from ..models.unet_3d import UNet3D
from ..preprocess.utils_io import load_nifti, save_nifti
from ..training.utils_train import prepare_device

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model_path: str = typer.Option(..., help="Path to trained model .pt"),
    image_path: str = typer.Option(..., help="Path to processed image NIfTI"),
    out_path: str = typer.Option(..., help="Where to save prediction NIfTI"),
    patch_size: str = typer.Option("96,96,64", help="patch size d,h,w"),
    overlap: float = typer.Option(0.5, help="sliding window overlap"),
    threshold: float = typer.Option(0.5, help="prob threshold"),
):
    device = prepare_device()
    ps = tuple(int(x) for x in patch_size.split(","))
    volume, ref = load_nifti(image_path)
    if volume.ndim == 3:
        volume = volume[None, ...]

    model = UNet3D(in_channels=volume.shape[0], out_channels=1)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)

    probs = sliding_window_inference_3d(volume, model, patch_size=ps, overlap=overlap, device=device)
    pred = (probs[0, 0] > threshold).astype(np.uint8)
    save_nifti(pred, ref, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    app()
