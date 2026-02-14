"""Minimal 2.5D U-Net training loop (smoke-test friendly)."""
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import typer
from ..datasets.isles_dataset import IslesVolumeDataset, IslesSliceDataset
from ..models.unet_2_5d import UNet2D
from .losses import DiceBCELoss
from .utils_train import set_seed, prepare_device, AverageMeter, dice_from_logits

app = typer.Typer(add_completion=False)


def _to_tensor(sample):
    if not torch.is_tensor(sample["image"]):
        sample["image"] = torch.from_numpy(sample["image"])
    if not torch.is_tensor(sample["mask"]):
        sample["mask"] = torch.from_numpy(sample["mask"])
    sample["image"] = sample["image"].float()
    sample["mask"] = sample["mask"].float()
    return sample


@app.command()
def main(config: str = typer.Option(..., help="Path to YAML config")):
    cfg = yaml.safe_load(Path(config).read_text())
    set_seed(42)
    device = prepare_device()

    # Apply tensor conversion only after slice extraction to avoid double-wrapping tensors
    vol_ds = IslesVolumeDataset(cfg["data"]["csv_path"], split="train", root=cfg["data"]["root"], transform=None)
    slice_ds = IslesSliceDataset(vol_ds, k=cfg["data"]["k_slices"], transform=_to_tensor)
    loader = DataLoader(slice_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    in_ch = len(cfg["data"]["modalities"]) * (2 * cfg["data"]["k_slices"] + 1)
    model = UNet2D(in_channels=in_ch, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and torch.cuda.is_available())

    out_dir = Path(cfg["log"]["out_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        meter = AverageMeter()
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)
                loss = criterion(logits, masks)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            meter.update(loss.item(), imgs.size(0))

        dice = dice_from_logits(logits.detach(), masks.detach())
        ckpt = out_dir / "last.pt"
        torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict()}, ckpt)
        print(f"epoch {epoch} loss {meter.avg:.4f} dice {dice:.4f}")

        if epoch % cfg["log"]["save_interval"] == 0:
            torch.save(model.state_dict(), out_dir / f"epoch{epoch}.pt")
            torch.save(model.state_dict(), out_dir / "best.pt")


if __name__ == "__main__":
    app()
