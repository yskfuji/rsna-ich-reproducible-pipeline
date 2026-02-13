"""Probe model probabilities within GT regions for selected cases.

Purpose: diagnose total-miss cases where Dice stays ~0 across thresholds.

This script mirrors the evaluation inference stack (sliding-window + optional TTA)
but reports probability statistics inside GT (and nearby) for a small set of cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import typer
from scipy.ndimage import binary_dilation

from src.datasets.isles_dataset import IslesVolumeDataset
from src.evaluation.evaluate_isles import infer_logits, infer_logits_with_flip_tta, infer_logits_with_tta
from src.models.unet_3d import UNet3D
from src.training.utils_train import prepare_device

app = typer.Typer(add_completion=False)


def _parse_csv_str(s: str) -> list[str]:
    out: list[str] = []
    for p in str(s).split(","):
        p = p.strip()
        if p:
            out.append(p)
    return out


def _parse_patch_size(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("patch_size must be 'X,Y,Z'")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


@dataclass(frozen=True)
class ProbeRow:
    case_id: str
    gt_vox: int
    zooms_mm: str
    slice_spacing_mm: float | None
    tta: str
    patch_size: str
    overlap: float
    p_gt_max: float
    p_gt_p99: float
    p_gt_p95: float
    p_gt_mean: float
    p_near_max: float
    p_near_p99: float
    p_all_max: float
    p_all_p999: float


def _infer_logits(
    *,
    vol: np.ndarray,
    model: torch.nn.Module,
    patch_size: tuple[int, int, int],
    overlap: float,
    device: torch.device,
    tta: str,
) -> np.ndarray:
    tta_m = str(tta).strip().lower()
    if tta_m in {"none", "off", "false", "0"}:
        return infer_logits(vol, model, patch_size=patch_size, overlap=overlap, device=device)
    if tta_m in {"flip"}:
        return infer_logits_with_flip_tta(vol, model, patch_size=patch_size, overlap=overlap, device=device)
    if tta_m in {"full", "on", "true", "1"}:
        return infer_logits_with_tta(vol, model, patch_size=patch_size, overlap=overlap, device=device)
    raise ValueError(f"Unknown tta: {tta!r}")


@app.command()
def main(
    model_path: str = typer.Option(...),
    csv_path: str = typer.Option(...),
    root: str = typer.Option(...),
    split: str = typer.Option("test"),
    out_dir: str = typer.Option(...),
    normalize: str = typer.Option("nonzero_zscore"),
    tta: str = typer.Option("none", help="none|flip|full"),
    patch_size: str = typer.Option("48,48,24"),
    overlap: float = typer.Option(0.5),
    temperature: float = typer.Option(1.0),
    case_ids: str = typer.Option("", help="comma-separated case_id list; empty means auto-pick"),
    exclude_case_ids: str = typer.Option("", help="comma-separated case_ids to exclude (auto-pick only)"),
    auto_pick_k: int = typer.Option(5, help="when case_ids empty: pick k smallest le_3mm cases"),
    near_dilate_iters: int = typer.Option(3, help="dilate GT by this many iters for near-region stats"),
) -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    device = prepare_device()

    ds = IslesVolumeDataset(csv_path, split=split, root=root, transform=None, normalize=normalize)

    # infer in_ch
    first_vol = IslesVolumeDataset(csv_path, split=split, root=root, transform=None, normalize=normalize)[0]["image"]
    in_ch = first_vol.shape[0] if first_vol.ndim == 4 else 1

    state = torch.load(model_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    w = state_dict.get("enc1.0.weight")
    if w is None:
        raise RuntimeError("Cannot infer base_ch from checkpoint; expected enc1.0.weight")
    base_ch = int(w.shape[0])

    model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=base_ch, deep_supervision=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ps = _parse_patch_size(patch_size)

    wanted = _parse_csv_str(case_ids)
    excluded = set(_parse_csv_str(exclude_case_ids))
    if not wanted:
        # auto-pick: le_3mm is exactly 2.0mm in this dataset; choose smallest by gt_vox
        cand: list[tuple[int, str]] = []
        for s in ds:
            meta = s.get("meta") or {}
            z = meta.get("zooms_mm") if isinstance(meta, dict) else None
            max_zoom = float(max(z)) if isinstance(z, (list, tuple)) and len(z) >= 3 else None
            if max_zoom is None or max_zoom > 3.0:
                continue
            gt_vox = int((s["mask"] > 0.5).sum())
            if gt_vox <= 0:
                continue
            cid = str(s["case_id"])
            if cid in excluded:
                continue
            cand.append((gt_vox, cid))
        cand.sort()
        wanted = [cid for _, cid in cand[: max(1, int(auto_pick_k))]]

    by_id = {str(s["case_id"]): s for s in ds if str(s["case_id"]) in set(wanted)}
    missing = [cid for cid in wanted if cid not in by_id]
    if missing:
        raise FileNotFoundError(f"Cases not found in split={split!r}: {missing}")

    rows: list[ProbeRow] = []

    for cid in wanted:
        s = by_id[cid]
        vol = s["image"]
        mask = s["mask"]
        if vol.ndim == 3:
            vol = vol[None, ...]
        vol = vol.astype(np.float32, copy=False)
        gt = (mask > 0.5)
        gt_vox = int(gt.sum())
        if gt_vox <= 0:
            raise ValueError(f"GT is empty for {cid} (this probe expects positives)")

        meta = s.get("meta") or {}
        zooms = None
        slice_spacing = None
        if isinstance(meta, dict):
            z = meta.get("zooms_mm")
            if isinstance(z, (list, tuple)) and len(z) >= 3:
                zooms = [float(z[0]), float(z[1]), float(z[2])]
                slice_spacing = float(max(zooms))

        with torch.inference_mode():
            logits = _infer_logits(vol=vol, model=model, patch_size=ps, overlap=float(overlap), device=device, tta=tta)
        probs = _sigmoid((logits / float(temperature)).astype(np.float32, copy=False))
        p = probs[0, 0]  # (Z,Y,X)

        gt_vals = p[gt]
        if gt_vals.size == 0:
            raise RuntimeError(f"Unexpected empty gt_vals for {cid}")

        near = binary_dilation(gt, iterations=max(0, int(near_dilate_iters)))
        near_vals = p[near]

        rows.append(
            ProbeRow(
                case_id=cid,
                gt_vox=gt_vox,
                zooms_mm="" if zooms is None else json.dumps(zooms),
                slice_spacing_mm=slice_spacing,
                tta=str(tta),
                patch_size=str(patch_size),
                overlap=float(overlap),
                p_gt_max=float(gt_vals.max()),
                p_gt_p99=float(np.percentile(gt_vals, 99.0)),
                p_gt_p95=float(np.percentile(gt_vals, 95.0)),
                p_gt_mean=float(gt_vals.mean()),
                p_near_max=float(near_vals.max()) if near_vals.size else float("nan"),
                p_near_p99=float(np.percentile(near_vals, 99.0)) if near_vals.size else float("nan"),
                p_all_max=float(p.max()),
                p_all_p999=float(np.percentile(p, 99.9)),
            )
        )
        typer.echo(f"done {cid}")

    # write TSV
    cols = [f.name for f in ProbeRow.__dataclass_fields__.values()]
    lines = ["\t".join(cols)]
    for r in rows:
        d = r.__dict__
        lines.append("\t".join(str(d[c]) for c in cols))
    (out_p / "probe.tsv").write_text("\n".join(lines) + "\n")

    # also print table
    typer.echo("\n" + (out_p / "probe.tsv").read_text())


if __name__ == "__main__":
    app()
