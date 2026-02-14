"""Conditional cascade inference: Stage2 conditioned on Stage1 probmap.

- Input to Stage2: [modalities..., stage1_probs]

Fusion modes:
- max:       final_probs = max(stage1_probs, sigmoid(stage2_logits))
- residual:  final_probs = sigmoid(logit(stage1_probs) + stage2_delta_logits)

This produces .npz probmaps compatible with `src/evaluation/evaluate_isles.py --probs-dir`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.datasets.isles_dataset import IslesVolumeDataset
from src.evaluation.evaluate_isles import (
    _resample_to_max_zoom_mm,
    infer_logits,
    infer_logits_with_flip_tta,
    infer_logits_with_tta,
)
from src.models.unet_3d import UNet3D
from src.training.utils_train import prepare_device


def _parse_ints_csv(s: str, n: int) -> tuple[int, ...]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"expected {n} comma-separated ints, got: {s!r}")
    return tuple(int(x) for x in parts)


def _coerce_probs_zyx(a: Any) -> NDArray[np.float32]:
    arr = np.asarray(a)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"probs must be 3D (Z,Y,X) after squeeze, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _align_probs_to_zyx(probs_zyx: NDArray[np.float32], target_zyx: tuple[int, int, int]) -> NDArray[np.float32]:
    tgt = (int(target_zyx[0]), int(target_zyx[1]), int(target_zyx[2]))
    cur = tuple(int(x) for x in probs_zyx.shape)
    if cur == tgt:
        return probs_zyx

    # Permute axes when it matches exactly.
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        if tuple(int(probs_zyx.shape[i]) for i in perm) == tgt:
            return np.transpose(probs_zyx, axes=perm).astype(np.float32, copy=False)

    # Resize with linear interpolation (best-effort).
    # Use numpy->torch interpolate to avoid scipy dependency in tools.
    zf = float(tgt[0]) / float(max(cur[0], 1))
    yf = float(tgt[1]) / float(max(cur[1], 1))
    xf = float(tgt[2]) / float(max(cur[2], 1))
    t = torch.from_numpy(probs_zyx[None, None].astype(np.float32, copy=False))
    t = torch.nn.functional.interpolate(t, scale_factor=(zf, yf, xf), mode="trilinear", align_corners=False)
    out = t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

    out = out[: tgt[0], : tgt[1], : tgt[2]]
    if out.shape != tgt:
        pad_z = tgt[0] - out.shape[0]
        pad_y = tgt[1] - out.shape[1]
        pad_x = tgt[2] - out.shape[2]
        out = np.pad(out, ((0, max(0, pad_z)), (0, max(0, pad_y)), (0, max(0, pad_x))), mode="constant")
        out = out[: tgt[0], : tgt[1], : tgt[2]]
    return out


def _load_stage1_probs(case_id: str, probs_dir: Path) -> NDArray[np.float32] | None:
    p = probs_dir / f"{case_id}.npz"
    if not p.exists():
        return None
    with np.load(str(p)) as z:
        if "probs" not in z:
            raise KeyError(f"Stage1 probs npz missing key 'probs': {p}")
        probs = _coerce_probs_zyx(z["probs"])
    return np.clip(probs, 0.0, 1.0)


def _logit(p: NDArray[np.float32], eps: float = 1e-4) -> NDArray[np.float32]:
    p = np.clip(p, float(eps), float(1.0 - eps)).astype(np.float32, copy=False)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def _load_model(model_path: Path, in_ch: int, device: torch.device) -> torch.nn.Module:
    state = torch.load(str(model_path), map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    w = sd.get("enc1.0.weight") if hasattr(sd, "get") else None
    if not isinstance(w, torch.Tensor) or w.ndim != 5:
        raise RuntimeError("Cannot infer base_ch from checkpoint (missing enc1.0.weight)")
    base_ch = int(w.shape[0])

    ckpt_has_ds = any((k.startswith("aux2_conv.") or k.startswith("aux3_conv.")) for k in (sd.keys() if hasattr(sd, "keys") else []))
    ds_flag = bool(ckpt_has_ds)

    # NOTE: MPS BatchNorm can be unstable; Stage2 conditional training defaults to InstanceNorm on MPS.
    # For robustness, mirror that choice here.
    model_norm = "instance" if device.type == "mps" else "batch"
    model = UNet3D(
        in_channels=int(in_ch),
        out_channels=1,
        base_ch=base_ch,
        deep_supervision=ds_flag,
        norm=model_norm,
    ).to(device)

    # If in_ch differs, adapt enc1 weight.
    if isinstance(w, torch.Tensor) and int(w.shape[1]) != int(in_ch):
        in_ckpt = int(w.shape[1])
        w_new = torch.zeros((int(w.shape[0]), int(in_ch), int(w.shape[2]), int(w.shape[3]), int(w.shape[4])), dtype=w.dtype)
        cmin = int(min(in_ckpt, int(in_ch)))
        w_new[:, :cmin] = w[:, :cmin].detach().cpu()
        if int(in_ch) > in_ckpt:
            mean = w[:, :cmin].detach().cpu().mean(dim=1, keepdim=True)
            w_new[:, cmin:] = mean.repeat(1, int(int(in_ch) - cmin), 1, 1, 1)
        sd = dict(sd)
        sd["enc1.0.weight"] = w_new

    strict = True
    try:
        model.load_state_dict(sd, strict=strict)
    except RuntimeError:
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-probs-dir", required=True, help="Directory containing <case_id>.npz with key 'probs' (Z,Y,X)")
    ap.add_argument("--stage2-model", required=True, help="Stage2 conditional model checkpoint (.pt)")
    ap.add_argument("--csv-path", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--normalize", default="nonzero_zscore")
    ap.add_argument("--patch-size", default="56,56,24")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tta", default="none", choices=["none", "flip", "full"])
    ap.add_argument("--resample-max-zoom-mm", type=float, default=2.0)
    ap.add_argument("--fusion", default="max", choices=["max", "residual"], help="How to fuse Stage1 and Stage2")
    ap.add_argument("--stage1-logit-eps", type=float, default=1e-4, help="Clamp for Stage1 probs before logit (residual fusion)")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip cases where <out-probs-dir>/<case_id>.npz already exists.",
    )
    ap.add_argument("--out-probs-dir", required=True)
    args = ap.parse_args()

    stage1_dir = Path(args.stage1_probs_dir).expanduser().resolve()
    if not stage1_dir.exists():
        raise FileNotFoundError(f"--stage1-probs-dir not found: {stage1_dir}")

    model_path = Path(args.stage2_model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"--stage2-model not found: {model_path}")

    out_dir = Path(args.out_probs_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = prepare_device()

    ds = IslesVolumeDataset(
        args.csv_path,
        split=args.split,
        root=args.root,
        transform=None,
        normalize=args.normalize,
        allow_missing_label=True,
    )

    first_img = ds[0]["image"]
    base_in_ch = int(first_img.shape[0]) if getattr(first_img, "ndim", 0) == 4 else 1
    in_ch = int(base_in_ch + 1)

    model = _load_model(model_path, in_ch=in_ch, device=device)

    resample_mm = float(args.resample_max_zoom_mm)
    for i in range(len(ds)):
        sample = ds[i]
        case_id = str(sample.get("case_id"))

        out_path = out_dir / f"{case_id}.npz"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            ok_existing = True
            try:
                with np.load(str(out_path)) as z:
                    _ = z["probs"]
            except Exception:
                ok_existing = False
            if ok_existing:
                if (i + 1) % 10 == 0 or (i + 1) == len(ds):
                    print(f"[skip] {i+1}/{len(ds)} exists {case_id}.npz", flush=True)
                continue
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

        vol = sample["image"].astype(np.float32, copy=False)

        meta = sample.get("meta") or {}
        zooms_raw = None
        if isinstance(meta, dict) and "zooms_mm" in meta:
            z = meta.get("zooms_mm")
            if isinstance(z, (list, tuple)) and len(z) >= 3:
                zooms_raw = [float(z[0]), float(z[1]), float(z[2])]

        if resample_mm > 0 and zooms_raw is not None:
            dummy_mask = np.zeros(vol.shape[1:], dtype=np.float32)
            try:
                vol, _, _, _ = _resample_to_max_zoom_mm(
                    vol.astype(np.float32, copy=False),
                    dummy_mask,
                    zooms_mm_xyz=zooms_raw,
                    target_mm=resample_mm,
                )
            except Exception:
                pass

        stage1 = _load_stage1_probs(case_id, stage1_dir)
        if stage1 is None:
            stage1 = np.zeros(vol.shape[1:], dtype=np.float32)
        stage1 = _align_probs_to_zyx(stage1, (int(vol.shape[1]), int(vol.shape[2]), int(vol.shape[3])))

        vol_in = np.concatenate([vol.astype(np.float32, copy=False), stage1[None, ...]], axis=0)

        ps = _parse_ints_csv(args.patch_size, 3)
        overlap = float(np.clip(float(args.overlap), 0.0, 0.95))
        tta = str(args.tta).strip().lower()
        if tta == "none":
            logits = infer_logits(vol_in, model, ps, overlap, device)
        elif tta == "flip":
            logits = infer_logits_with_flip_tta(vol_in, model, ps, overlap, device)
        else:
            logits = infer_logits_with_tta(vol_in, model, ps, overlap, device)

        fusion = str(args.fusion).strip().lower()
        stage2_probs = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)[0, 0]
        if fusion == "residual":
            eps = float(max(1e-8, min(1e-2, float(args.stage1_logit_eps))))
            fused_logits = _logit(stage1, eps=eps) + logits[0, 0].astype(np.float32, copy=False)
            final = (1.0 / (1.0 + np.exp(-fused_logits))).astype(np.float32, copy=False)
        else:
            final = np.maximum(stage1, stage2_probs).astype(np.float32, copy=False)
        tmp_path = out_dir / f"{case_id}.tmp.npz"
        np.savez_compressed(str(tmp_path), probs=final)
        os.replace(str(tmp_path), str(out_path))

        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            print(f"[ok] {i+1}/{len(ds)} wrote {case_id}.npz", flush=True)


if __name__ == "__main__":
    main()
