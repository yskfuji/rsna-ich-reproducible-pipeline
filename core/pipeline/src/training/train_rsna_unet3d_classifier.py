from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from ..datasets.rsna_ich_dataset import RSNA_CLASSES, RsnaIchSliceDataset, iter_rsna_stage2_records_from_csv
from ..models.unet3d_encoder_classifier import UNet3DEncoderClassifier, load_isles_unet_weights_into_classifier

app = typer.Typer(add_completion=False)


def _device() -> torch.device:
    # Keep consistent with existing codebase: TORCH_DEVICE env var if set.
    import os

    dev = os.environ.get("TORCH_DEVICE", "cpu").strip().lower()
    if dev in {"cuda", "cuda:0"} and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.command()
def train(
    rsna_root: Path = typer.Option(..., help="RSNA dataset root containing stage_2_train.csv and stage_2_train/"),
    out_dir: Path = typer.Option(Path("results/rsna_ich"), help="Output directory"),
    limit_images: int = typer.Option(4000, help="Number of unique slices (image_ids) to use (quick mode)"),
    val_frac: float = typer.Option(0.1, help="Validation fraction (random split, quick baseline)"),
    seed: int = typer.Option(0, help="Random seed"),
    base_ch: int = typer.Option(16, help="UNet base channels"),
    image_size: int = typer.Option(256, help="Resize to (image_size,image_size)"),
    windows: str = typer.Option("40,80;80,200;600,2800", help="CT windows as 'L,W;L,W;...'"),
    batch_size: int = typer.Option(4, help="Batch size"),
    num_workers: int = typer.Option(0, help="DataLoader workers (macOS/MPSは0推奨。CPU実行は2-8推奨)"),
    epochs: int = typer.Option(3, help="Epochs (quick baseline)"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay"),
    init_from_isles: Path | None = typer.Option(None, help="Optional ISLES UNet3D checkpoint best.pt"),
):
    """Train a quick RSNA ICH slice-level multi-label classifier using UNet3D encoder."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    rsna_root = rsna_root.expanduser().resolve()
    csv_path = rsna_root / "stage_2_train.csv"
    dcm_dir = rsna_root / "stage_2_train"

    records = iter_rsna_stage2_records_from_csv(
        csv_path=csv_path,
        dicom_dir=dcm_dir,
        limit_images=int(limit_images) if int(limit_images) > 0 else None,
        seed=int(seed),
    )

    # Filter missing files (some datasets may be incomplete)
    records = [r for r in records if r.dcm_path.exists()]
    if not records:
        raise FileNotFoundError(f"No DICOM files found under: {dcm_dir}")

    n = len(records)
    n_val = int(max(1, round(float(val_frac) * n)))
    n_tr = int(max(1, n - n_val))
    train_records = records[:n_tr]
    val_records = records[n_tr:]

    train_ds = RsnaIchSliceDataset(train_records, out_size=int(image_size), windows=str(windows))
    val_ds = RsnaIchSliceDataset(val_records, out_size=int(image_size), windows=str(windows))

    nw = int(num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=nw,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
    )

    dev = _device()

    # 3 window channels by default
    in_channels = int(len(windows.split(";"))) if str(windows).strip() else 1

    model = UNet3DEncoderClassifier(
        in_channels=in_channels,
        num_classes=len(RSNA_CLASSES),
        base_ch=int(base_ch),
        norm="instance" if dev.type == "mps" else "batch",
        dropout=0.1,
    ).to(dev)

    init_report = None
    if init_from_isles is not None:
        missing, unexpected = load_isles_unet_weights_into_classifier(
            model,
            ckpt_path=str(init_from_isles.expanduser().resolve()),
            device=dev,
            allow_input_channel_adapt=True,
        )
        init_report = {"missing": len(missing), "unexpected": len(unexpected)}

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "rsna_root": str(rsna_root),
        "limit_images": int(limit_images),
        "val_frac": float(val_frac),
        "seed": int(seed),
        "base_ch": int(base_ch),
        "image_size": int(image_size),
        "windows": str(windows),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "classes": list(RSNA_CLASSES),
        "init_from_isles": str(init_from_isles) if init_from_isles is not None else None,
        "init_report": init_report,
        "device": str(dev),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    best_val = float("inf")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            x = batch["x"].to(dev, non_blocking=True)
            y = batch["y"].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(dev, non_blocking=True)
                y = batch["y"].to(dev, non_blocking=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                va_losses.append(float(loss.item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        log = {"epoch": epoch, "train_loss": tr, "val_loss": va}
        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        print(f"[epoch {epoch}] train_loss={tr:.6f} val_loss={va:.6f}", flush=True)

        torch.save(model.state_dict(), out_dir / "last.pt")
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), out_dir / "best.pt")


if __name__ == "__main__":
    app()
