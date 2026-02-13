from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
import yaml

from src.inference.infer_sliding_window import sliding_window_inference_3d
from src.models.unet_3d import UNet3D
from src.preprocess.utils_io import load_nifti, save_nifti
from src.training.losses import DiceBCELoss
from src.training.train_3d_unet import sample_patch_3d
from src.training.utils_train import prepare_device, set_seed
from tools.meta_store import init_or_load_run
from tools.plotting import save_overlay_3plane
from tools.run_meta import meta_to_dict, sha256_json, write_text

app = typer.Typer(add_completion=False)


def _resolve_paths(cfg: dict[str, Any], repo_root: Path) -> tuple[Path, Path]:
    data_root = Path(cfg["data"]["root"]).expanduser()
    csv_path = Path(cfg["data"]["csv_path"]).expanduser()
    if not data_root.is_absolute():
        data_root = (repo_root / data_root).resolve()
    if not csv_path.is_absolute():
        csv_path = (repo_root / csv_path).resolve()
    return data_root, csv_path


def _find_case_row(csv_path: Path, split: str, case_id: str) -> dict[str, str] | None:
    import pandas as pd

    df = pd.read_csv(str(csv_path))
    df = df[df["split"] == split]
    df = df[df["case_id"].astype(str) == str(case_id)]
    if len(df) == 0:
        return None
    row = df.iloc[0]
    out: dict[str, str] = {"case_id": str(row["case_id"])}
    if "image_path" in df.columns:
        v = row.get("image_path")
        if isinstance(v, str) and v.strip():
            out["image_path"] = v.strip()
    if "label_path" in df.columns:
        v = row.get("label_path")
        if isinstance(v, str) and v.strip():
            out["label_path"] = v.strip()
    return out


def _case_paths(data_root: Path, row: dict[str, str]) -> tuple[Path, Path]:
    case_id = row["case_id"]
    img_rel = row.get("image_path")
    lbl_rel = row.get("label_path")
    img_path = Path(img_rel) if img_rel else (data_root / "images" / f"{case_id}.nii.gz")
    lbl_path = Path(lbl_rel) if lbl_rel else (data_root / "labels" / f"{case_id}.nii.gz")
    if not img_path.is_absolute():
        img_path = (data_root / img_path).resolve()
    if not lbl_path.is_absolute():
        lbl_path = (data_root / lbl_path).resolve()
    return img_path, lbl_path


@app.command()
def main(
    config: str = typer.Option(..., help="Path to YAML config"),
    case_id: str = typer.Option(..., help="Fixed train case_id to overfit"),
    split: str = typer.Option("train", help="Which split to draw the case from"),
    iters: int = typer.Option(300, help="training iterations (200-500 recommended)"),
    lr: float = typer.Option(1e-3, help="learning rate for this overfit test"),
    patch_size: str = typer.Option("56,56,24", help="patch size for training (D,H,W)"),
    infer_patch_size: str = typer.Option("56,56,24", help="patch size for inference (D,H,W)"),
    overlap: float = typer.Option(0.5, help="sliding window overlap"),
    threshold: float = typer.Option(0.5, help="prob threshold for pred_bin"),
    out_root: str = typer.Option("runs", help="Base output dir (runs/<run_id>/...)"),
    run_id: str = typer.Option("", help="Reuse an existing run_id (optional)"),
    seed: int = typer.Option(42, help="seed"),
    dataset_hash_mode: str = typer.Option("stat", help="dataset hash mode: stat|full"),
):
    cfg_path = Path(config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    repo_root = Path(__file__).resolve().parents[1]
    data_root, csv_path = _resolve_paths(cfg, repo_root)

    run_id_opt = run_id.strip() or None
    meta, run_dir = init_or_load_run(
        repo_root=repo_root,
        out_root=Path(out_root),
        run_id=run_id_opt,
        seed=seed,
        config_path=cfg_path,
        config_obj=cfg,
        csv_path=csv_path,
        data_root=data_root,
        dataset_hash_mode=dataset_hash_mode,
    )

    out_dir = run_dir / "overfit_one"
    out_dir.mkdir(parents=True, exist_ok=True)

    row = _find_case_row(csv_path, split=split, case_id=case_id)
    if row is None:
        raise ValueError(f"case_id={case_id!r} not found in split={split!r} csv={csv_path}")
    img_path, lbl_path = _case_paths(data_root, row)

    # Load original NIfTI for correct affine and to avoid any dataset-side surprises.
    img_arr, img_ref = load_nifti(str(img_path))
    lbl_arr, lbl_ref = load_nifti(str(lbl_path)) if lbl_path.exists() else (np.zeros(img_arr.shape[1:], np.float32), img_ref)

    if img_arr.ndim == 3:
        img_arr = img_arr[None, ...]

    # Normalize exactly like dataset.
    norm_mode = str(cfg.get("data", {}).get("normalize", "legacy_zscore")).lower()
    x = img_arr.astype(np.float32)
    if norm_mode in {"none", "off", "false"}:
        pass
    elif norm_mode in {"legacy", "legacy_zscore", "zscore"}:
        for c in range(x.shape[0]):
            ch = x[c]
            x[c] = (ch - float(ch.mean())) / (float(ch.std()) + 1e-8)
    elif norm_mode in {"nonzero", "nonzero_zscore", "nnunet"}:
        for c in range(x.shape[0]):
            ch = x[c]
            nz = ch != 0
            if np.any(nz):
                vals = ch[nz]
                lo, hi = np.percentile(vals, [0.5, 99.5])
                vals = np.clip(vals, lo, hi)
                mean = float(vals.mean())
                std = float(vals.std())
                out = np.zeros_like(ch, dtype=np.float32)
                out[nz] = (np.clip(ch[nz], lo, hi) - mean) / (std + 1e-8)
                x[c] = out
            else:
                x[c] = (ch - float(ch.mean())) / (float(ch.std()) + 1e-8)
    else:
        raise ValueError(f"Unknown normalize mode: {norm_mode!r}")

    y = (lbl_arr > 0.5).astype(np.float32)

    set_seed(int(seed))
    device = prepare_device()

    ps = tuple(int(v) for v in patch_size.split(","))
    ips = tuple(int(v) for v in infer_patch_size.split(","))

    base_ch = int(cfg.get("train", {}).get("base_ch", 16))
    deep_supervision = bool(cfg.get("train", {}).get("deep_supervision", False))

    model = UNet3D(in_channels=int(x.shape[0]), out_channels=1, base_ch=base_ch, deep_supervision=deep_supervision).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.0)
    criterion = DiceBCELoss()

    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as mf:
        for step in range(1, int(iters) + 1):
            patch_img, patch_mask = sample_patch_3d(
                x,
                y[None, ...],
                patch_size=ps,
                bg_patch_size=None,
                foreground_prob=1.0,
                force_fg=True,
                target_pos_patch_frac=None,
            )
            inp = torch.from_numpy(patch_img[None]).float().to(device)
            tgt = torch.from_numpy(patch_mask[None, None]).float().to(device)

            model.train()
            optim.zero_grad(set_to_none=True)
            out = model(inp)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = criterion(out, tgt)
            loss.backward()
            optim.step()

            with torch.no_grad():
                prob = torch.sigmoid(out)
                pred = (prob >= float(threshold)).float()
                # dice on patch
                inter = (pred * tgt).sum().item()
                denom = pred.sum().item() + tgt.sum().item() + 1e-8
                dice = float(2.0 * inter / denom)

            row_out = {
                **meta_to_dict(meta),
                "task": "overfit_one",
                "case_id": row["case_id"],
                "step": int(step),
                "loss": float(loss.item()),
                "dice_patch": float(dice),
            }
            mf.write(json.dumps(row_out, ensure_ascii=False) + "\n")

            if step % 50 == 0 or step == 1 or step == int(iters):
                print(f"[overfit_one] step {step}/{iters} loss={loss.item():.4f} dice={dice:.4f}", flush=True)

    # Full-volume inference
    model.eval()
    logits = sliding_window_inference_3d(x, model, patch_size=ips, overlap=float(overlap), device=device, aggregate="logits")
    logits = logits[0, 0].astype(np.float32)
    prob = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    pred = (prob >= float(threshold)).astype(np.uint8)

    # Save NIfTIs in the processed (current) space with correct affine.
    save_nifti(prob, img_ref, str(out_dir / "pred_prob.nii.gz"))
    save_nifti(pred.astype(np.float32), img_ref, str(out_dir / "pred_bin.nii.gz"))
    save_nifti(y.astype(np.float32), lbl_ref, str(out_dir / "gt.nii.gz"))

    # Overlays
    base_img = x[0] if x.shape[0] > 0 else None
    overlay_paths = save_overlay_3plane(out_dir, prob=prob, pred=pred, gt=y.astype(np.uint8), base_img=base_img, thr=float(threshold))

    # Summary
    gt_vox = int(y.sum())
    pred_vox = int(pred.sum())
    tp = int(((pred > 0) & (y > 0)).sum())
    fp = int(((pred > 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y > 0)).sum())
    dice_full = float((2 * tp) / (2 * tp + fp + fn + 1e-8))
    prec = float(tp / (tp + fp + 1e-8))
    rec = float(tp / (tp + fn + 1e-8))

    summary = {
        "meta": meta_to_dict(meta),
        "task": "overfit_one",
        "case_id": row["case_id"],
        "config_hash": meta.config_hash,
        "dataset_hash": meta.dataset_hash,
        "settings": {
            "iters": int(iters),
            "lr": float(lr),
            "patch_size": list(ps),
            "infer_patch_size": list(ips),
            "overlap": float(overlap),
            "threshold": float(threshold),
            "base_ch": base_ch,
            "normalize": norm_mode,
        },
        "full_volume": {
            "dice": dice_full,
            "precision": prec,
            "recall": rec,
            "gt_vox": gt_vox,
            "pred_vox": pred_vox,
            "tp_vox": tp,
            "fp_vox": fp,
            "fn_vox": fn,
        },
        "outputs": {
            "metrics_jsonl": str(metrics_path),
            "pred_prob": str(out_dir / "pred_prob.nii.gz"),
            "pred_bin": str(out_dir / "pred_bin.nii.gz"),
            "gt": str(out_dir / "gt.nii.gz"),
            "overlays": [str(p) for p in overlay_paths],
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = "\n".join(
        [
            f"# Overfit One Sample ({meta.run_id})",
            "",
            f"- case_id: `{row['case_id']}` (split={split})",
            f"- iters: {iters}, lr: {lr}",
            f"- full dice/prec/rec: {dice_full:.4f} / {prec:.4f} / {rec:.4f}",
            "",
            "## Outputs",
            f"- `{out_dir / 'pred_prob.nii.gz'}`",
            f"- `{out_dir / 'pred_bin.nii.gz'}`",
            f"- `{out_dir / 'gt.nii.gz'}`",
            f"- overlays: {', '.join(str(p.name) for p in overlay_paths)}",
        ]
    )
    write_text(out_dir / "summary.md", md + "\n")


if __name__ == "__main__":
    app()
