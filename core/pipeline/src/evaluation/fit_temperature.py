"""Fit scalar temperature for post-hoc calibration (temperature scaling).

We fit a single scalar T>0 on a calibration split (typically val) by minimizing
voxel-wise BCEWithLogitsLoss over a sampled subset of voxels.

This does NOT change the model weights. At inference time, use logits/T before
sigmoid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple
import json

import numpy as np
import torch
import typer

from ..datasets.isles_dataset import IslesVolumeDataset
from ..models.unet_3d import UNet3D
from ..inference.infer_sliding_window import sliding_window_inference_3d
from ..training.utils_train import prepare_device

app = typer.Typer(add_completion=False)


def infer_logits_with_tta(
    vol: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    """Return logits volume with flip+rot90 TTA (mean over logits)."""
    if vol.ndim == 3:
        vol = vol[None, ...]
    flip_axes: Iterable[Tuple[int, ...]] = [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    acc: np.ndarray | None = None
    num = 0
    for rot_k in (0, 1, 2, 3):
        v_rot = np.rot90(vol, k=rot_k, axes=(2, 3))
        for axes in flip_axes:
            v = np.flip(v_rot, axis=axes) if axes else v_rot
            out = sliding_window_inference_3d(
                v,
                model,
                patch_size=patch_size,
                overlap=overlap,
                device=device,
                aggregate="logits",
            )
            if axes:
                out = np.flip(out, axis=tuple(a + 1 for a in axes))
            if rot_k:
                out = np.rot90(out, k=-rot_k, axes=(3, 4))
            acc = out if acc is None else acc + out
            num += 1
    if acc is None:
        raise RuntimeError("TTA produced no outputs (unexpected)")
    return (acc / float(num)).astype(np.float32)


def _sample_indices(rng: np.random.Generator, idxs: np.ndarray, k: int) -> np.ndarray:
    if idxs.size <= k:
        return idxs
    sel = rng.choice(idxs, size=int(k), replace=False)
    return sel


@app.command()
def main(
    model_path: str = typer.Option(..., help="model checkpoint"),
    csv_path: str = typer.Option(..., help="split csv"),
    root: str = typer.Option(..., help="processed root"),
    split: str = typer.Option("val", help="calibration split (typically val)"),
    out_path: str = typer.Option("", help="output json path (default: next to checkpoint)"),
    patch_size: str = typer.Option("96,96,64", help="patch size"),
    overlap: float = typer.Option(0.5, help="overlap"),
    base_ch: Optional[int] = typer.Option(None, help="base channels; auto-infer from checkpoint when omitted"),
    normalize: str = typer.Option(
        "legacy_zscore",
        help=(
            "input normalization: legacy_zscore | nonzero_zscore | robust_nonzero_zscore | nonzero_minmax01 | none"
        ),
    ),
    allow_missing_label: bool = typer.Option(False, help="treat missing label as all-zero mask"),
    max_cases: int = typer.Option(0, help="max number of cases to use (0=all)"),
    max_pos_vox: int = typer.Option(20000, help="max positive voxels per case"),
    max_neg_vox: int = typer.Option(50000, help="max negative voxels per case"),
    seed: int = typer.Option(42, help="random seed"),
    steps: int = typer.Option(200, help="optimizer steps"),
    lr: float = typer.Option(0.05, help="optimizer learning rate"),
    min_temp: float = typer.Option(0.05, help="minimum temperature"),
    max_temp: float = typer.Option(20.0, help="maximum temperature"),
):
    device = prepare_device()
    ps_list = [int(x) for x in patch_size.split(",") if x.strip()]
    if len(ps_list) != 3:
        raise ValueError(f"patch_size must have 3 ints (D,H,W), got: {patch_size!r}")
    ps: Tuple[int, int, int] = (ps_list[0], ps_list[1], ps_list[2])

    ds = IslesVolumeDataset(
        csv_path,
        split=split,
        root=root,
        transform=None,
        normalize=normalize,
        allow_missing_label=bool(allow_missing_label),
    )
    if len(ds) == 0:
        raise ValueError(f"No cases found for split={split!r} in {csv_path!r}")

    first_vol = ds[0]["image"]
    in_ch = first_vol.shape[0] if first_vol.ndim == 4 else 1

    state = torch.load(model_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    if base_ch is None:
        w = state_dict.get("enc1.0.weight")
        if w is None:
            raise RuntimeError("Cannot infer base_ch; specify --base-ch explicitly")
        base_ch = int(w.shape[0])

    model = UNet3D(in_channels=in_ch, out_channels=1, base_ch=base_ch)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(int(seed))
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    n_use = len(ds) if int(max_cases) <= 0 else min(len(ds), int(max_cases))

    for i in range(n_use):
        sample = ds[i]
        vol = sample["image"]
        mask = sample["mask"]
        logits = infer_logits_with_tta(vol, model, ps, float(overlap), device)
        gt = (mask > 0.5).astype(np.uint8)

        flat_logits = logits[0, 0].reshape(-1)
        flat_gt = gt.reshape(-1)

        pos_idxs = np.flatnonzero(flat_gt > 0)
        neg_idxs = np.flatnonzero(flat_gt == 0)

        pos_sel = _sample_indices(rng, pos_idxs, int(max_pos_vox))
        neg_sel = _sample_indices(rng, neg_idxs, int(max_neg_vox))

        sel = np.concatenate([pos_sel, neg_sel], axis=0)
        if sel.size == 0:
            continue
        rng.shuffle(sel)

        logits_list.append(flat_logits[sel].astype(np.float32))
        labels_list.append(flat_gt[sel].astype(np.float32))

        pos_n = int(pos_sel.size)
        neg_n = int(neg_sel.size)
        print(f"[{i+1}/{n_use}] {sample['case_id']} pos={pos_n} neg={neg_n} sel={int(sel.size)}", flush=True)

    if not logits_list:
        raise RuntimeError("No voxels sampled; check data/labels.")

    logits_all = np.concatenate(logits_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    x = torch.from_numpy(logits_all).float()
    y = torch.from_numpy(labels_all).float()

    # Optimize on CPU for stability and speed (scalar parameter).
    x = x.cpu()
    y = y.cpu()

    def nll(temp: torch.Tensor) -> torch.Tensor:
        t = temp.clamp(min=float(min_temp), max=float(max_temp))
        return torch.nn.functional.binary_cross_entropy_with_logits(x / t, y)

    log_t = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=float(lr))

    with torch.no_grad():
        loss_before = float(nll(torch.tensor(1.0)).item())

    best_loss = None
    best_t = None

    for step in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        temp = torch.exp(log_t)
        loss = nll(temp)
        loss.backward()
        opt.step()

        with torch.no_grad():
            t_val = float(torch.exp(log_t).clamp(min=float(min_temp), max=float(max_temp)).item())
            l_val = float(loss.item())
        if best_loss is None or l_val < best_loss:
            best_loss = l_val
            best_t = t_val
        if (step + 1) % 20 == 0 or step == 0:
            print(f"step {step+1}/{steps} T={t_val:.4f} nll={l_val:.6f}", flush=True)

    if best_t is None or best_loss is None:
        raise RuntimeError("Temperature optimization failed")

    payload = {
        "model_path": str(Path(model_path)),
        "csv_path": str(Path(csv_path)),
        "root": str(Path(root)),
        "split": str(split),
        "patch_size": list(ps),
        "overlap": float(overlap),
        "normalize": str(normalize),
        "allow_missing_label": bool(allow_missing_label),
        "max_cases": int(max_cases),
        "max_pos_vox": int(max_pos_vox),
        "max_neg_vox": int(max_neg_vox),
        "seed": int(seed),
        "n_samples": int(logits_all.size),
        "pos_frac": float(labels_all.mean()),
        "temperature": float(best_t),
        "nll_before": float(loss_before),
        "nll_after": float(best_loss),
    }

    out_p = Path(out_path) if str(out_path).strip() else (Path(model_path).expanduser().resolve().parent / "temperature_best.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {out_p} (T={best_t:.4f})", flush=True)


if __name__ == "__main__":
    app()
