"""Minimal 3D U-Net training loop with random patch sampling."""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import contextlib
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform, gaussian_filter, distance_transform_cdt
from scipy.ndimage import zoom as nd_zoom
from scipy.ndimage import label as cc_label
from torch.utils.data import DataLoader
import typer
from torch.optim.lr_scheduler import CosineAnnealingLR
from numpy.typing import NDArray
from functools import partial
from ..datasets.isles_dataset import IslesVolumeDataset
from ..models.unet_3d import UNet3D
from .losses import (
    DiceBCELoss,
    DiceFocalLoss,
    DiceOHEMBCELoss,
    TverskyLoss,
    TverskyFocalLoss,
    TverskyOHEMBCELoss,
)
from .utils_train import set_seed, prepare_device, AverageMeter

app = typer.Typer(add_completion=False)


ArrayF = NDArray[np.float32]


def _downsample_like(
    x: torch.Tensor,
    target_size: tuple[int, int, int],
    *,
    kind: str,
) -> torch.Tensor:
    """Downsample 3D tensor (N,C,D,H,W) to match target spatial size.

    We avoid `F.interpolate(..., mode="nearest")` on MPS because it can fall back to CPU and
    sometimes hard-abort the process.

    - `kind="mask"`: adaptive max pooling (preserve any positive voxel)
    - `kind="probs"`: adaptive avg pooling (smooth, stable)
    """
    if tuple(int(s) for s in x.shape[2:]) == tuple(int(s) for s in target_size):
        return x

    in_d, in_h, in_w = (int(s) for s in x.shape[2:])
    tgt_d, tgt_h, tgt_w = (int(s) for s in target_size)

    # Prefer strided slicing when downsample factors are integer (common for UNet deep supervision).
    # This avoids MPS-unsupported 3D pool/upsample ops.
    if (in_d % tgt_d == 0) and (in_h % tgt_h == 0) and (in_w % tgt_w == 0):
        kd, kh, kw = in_d // tgt_d, in_h // tgt_h, in_w // tgt_w
        if kind not in ("mask", "probs"):
            raise ValueError(f"unknown kind={kind!r}")
        y = x[..., ::kd, ::kh, ::kw]
        # Guard against any off-by-one due to integer division.
        return y[..., :tgt_d, :tgt_h, :tgt_w]

    # Rare fallback: do the resize on CPU (inputs are detached/non-differentiable here).
    x_cpu = x.detach().float().cpu()
    if kind == "mask":
        y = F.interpolate(x_cpu, size=(tgt_d, tgt_h, tgt_w), mode="nearest")
    elif kind == "probs":
        y = F.interpolate(x_cpu, size=(tgt_d, tgt_h, tgt_w), mode="trilinear", align_corners=False)
    else:
        raise ValueError(f"unknown kind={kind!r}")
    return y.to(device=x.device, dtype=x.dtype)


def _coerce_probs_zyx(a: Any) -> NDArray[np.float32]:
    arr = np.asarray(a)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"stage1 probs must be 3D (Z,Y,X) after squeeze, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _align_probs_to_zyx(probs_zyx: NDArray[np.float32], target_zyx: tuple[int, int, int]) -> NDArray[np.float32]:
    """Best-effort align Stage1 probs to target Z,Y,X shape.

    - Accepts axis permutations when shapes match.
    - Otherwise applies linear resize to target.
    """
    tgt = (int(target_zyx[0]), int(target_zyx[1]), int(target_zyx[2]))
    cur = tuple(int(x) for x in probs_zyx.shape)
    if cur == tgt:
        return probs_zyx

    # Try permutations when it is just an axis-order issue.
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        if tuple(int(probs_zyx.shape[i]) for i in perm) == tgt:
            return np.transpose(probs_zyx, axes=perm).astype(np.float32, copy=False)

    # Fallback: resize to target (best-effort). Use torch interpolate to avoid SciPy zoom.
    t = torch.from_numpy(probs_zyx[None, None].astype(np.float32, copy=False))
    t = F.interpolate(t, size=tgt, mode="trilinear", align_corners=False)
    out = t[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def _load_stage1_probs(case_id: str, probs_dir: Path) -> NDArray[np.float32] | None:
    p = probs_dir / f"{case_id}.npz"
    if not p.exists():
        return None
    data = np.load(str(p))
    if "probs" not in data:
        raise KeyError(f"Stage1 probs npz missing key 'probs': {p}")
    probs = _coerce_probs_zyx(data["probs"])
    # keep within [0,1]
    probs = np.clip(probs, 0.0, 1.0)
    return probs


def _resample_to_max_zoom_mm(
    vol_czyx: NDArray[np.float32],
    mask_zyx: NDArray[np.float32],
    zooms_mm_xyz: list[float] | tuple[float, float, float],
    target_mm: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[float, float, float], tuple[float, float, float]]:
    """Upsample (never downsample) so max voxel spacing becomes <= target_mm.

    zooms_mm_xyz is expected to be (X,Y,Z) spacing in mm.
    Returns: (vol_rs, mask_rs, new_zooms_mm_zyx, factors_zyx).
    """
    target_mm = float(target_mm)
    if target_mm <= 0:
        zyx = (float(zooms_mm_xyz[2]), float(zooms_mm_xyz[1]), float(zooms_mm_xyz[0]))
        return vol_czyx, mask_zyx, zyx, (1.0, 1.0, 1.0)

    zx, yx, xx = float(zooms_mm_xyz[2]), float(zooms_mm_xyz[1]), float(zooms_mm_xyz[0])
    zooms_mm_zyx = (zx, yx, xx)
    factors_zyx = (
        float(max(1.0, zx / target_mm)),
        float(max(1.0, yx / target_mm)),
        float(max(1.0, xx / target_mm)),
    )
    if factors_zyx == (1.0, 1.0, 1.0):
        return vol_czyx, mask_zyx, zooms_mm_zyx, factors_zyx

    vol_rs = np.stack([nd_zoom(vol_czyx[c], zoom=factors_zyx, order=1) for c in range(vol_czyx.shape[0])], axis=0)
    mask_rs = nd_zoom(mask_zyx, zoom=factors_zyx, order=0)
    new_zooms_mm_zyx = (
        zooms_mm_zyx[0] / factors_zyx[0],
        zooms_mm_zyx[1] / factors_zyx[1],
        zooms_mm_zyx[2] / factors_zyx[2],
    )
    return vol_rs.astype(np.float32, copy=False), mask_rs.astype(np.float32, copy=False), new_zooms_mm_zyx, factors_zyx


def _maybe_resample_case(
    img_czyx: NDArray[np.float32],
    mask_zyx: NDArray[np.float32],
    meta: dict,
    resample_max_zoom_mm: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if resample_max_zoom_mm is None or float(resample_max_zoom_mm) <= 0:
        return img_czyx, mask_zyx
    zooms_mm_xyz = meta.get("zooms_mm", None)
    if zooms_mm_xyz is None or (not hasattr(zooms_mm_xyz, "__len__")) or len(zooms_mm_xyz) != 3:
        return img_czyx, mask_zyx
    img_rs, mask_rs, _, _ = _resample_to_max_zoom_mm(
        img_czyx.astype(np.float32, copy=False),
        mask_zyx.astype(np.float32, copy=False),
        zooms_mm_xyz=zooms_mm_xyz,
        target_mm=float(resample_max_zoom_mm),
    )
    return img_rs, mask_rs


def _load_candidates_jsonl(path: Path) -> dict[str, dict[str, list[list[int]]]]:
    """Load Stage1 candidate bboxes.

    Returns: {case_id: {'pos': [bbox_zyxzyx], 'neg': [bbox_zyxzyx]}}
    """
    out: dict[str, dict[str, list[list[int]]]] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            case_id = str(rec.get("case_id"))
            bbox = rec.get("bbox_zyxzyx")
            if (not isinstance(bbox, list)) or len(bbox) != 6:
                continue
            is_pos = bool(rec.get("is_pos", False))
            bucket = "pos" if is_pos else "neg"
            out.setdefault(case_id, {"pos": [], "neg": []})[bucket].append([int(x) for x in bbox])
    return out


def _crop_patch_center_zyx(
    img_cdhw: ArrayF,
    mask_dhw: ArrayF,
    center_zyx: tuple[int, int, int],
    patch_size_dhw: tuple[int, int, int],
) -> tuple[ArrayF, ArrayF]:
    """Crop (with zero-padding) a patch centered at center_zyx from (C,D,H,W) + (D,H,W)."""
    C, D, H, W = img_cdhw.shape
    pD, pH, pW = (int(patch_size_dhw[0]), int(patch_size_dhw[1]), int(patch_size_dhw[2]))
    cz, cy, cx = (int(center_zyx[0]), int(center_zyx[1]), int(center_zyx[2]))

    z0 = cz - (pD // 2)
    y0 = cy - (pH // 2)
    x0 = cx - (pW // 2)
    z1 = z0 + pD
    y1 = y0 + pH
    x1 = x0 + pW

    sz0, sy0, sx0 = max(0, z0), max(0, y0), max(0, x0)
    sz1, sy1, sx1 = min(D, z1), min(H, y1), min(W, x1)

    dz0, dy0, dx0 = sz0 - z0, sy0 - y0, sx0 - x0
    dz1, dy1, dx1 = dz0 + (sz1 - sz0), dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

    out_img = np.zeros((C, pD, pH, pW), dtype=np.float32)
    out_msk = np.zeros((pD, pH, pW), dtype=np.float32)
    if (sz1 > sz0) and (sy1 > sy0) and (sx1 > sx0):
        out_img[:, dz0:dz1, dy0:dy1, dx0:dx1] = img_cdhw[:, sz0:sz1, sy0:sy1, sx0:sx1]
        out_msk[dz0:dz1, dy0:dy1, dx0:dx1] = mask_dhw[sz0:sz1, sy0:sy1, sx0:sx1]
    return out_img, out_msk


def _floor_to_multiple(x: int, m: int) -> int:
    x = int(x)
    m = int(max(1, m))
    return x - (x % m)


def _effective_patch_and_stride(
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    vol_shape: Tuple[int, int, int],
    multiple: int = 8,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    D, H, W = (int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2]))
    pD, pH, pW = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
    sD, sH, sW = (int(stride[0]), int(stride[1]), int(stride[2]))

    eff_pD = max(multiple, min(pD, _floor_to_multiple(D, multiple)))
    eff_pH = max(multiple, min(pH, _floor_to_multiple(H, multiple)))
    eff_pW = max(multiple, min(pW, _floor_to_multiple(W, multiple)))

    eff_sD = max(1, min(sD, eff_pD))
    eff_sH = max(1, min(sH, eff_pH))
    eff_sW = max(1, min(sW, eff_pW))
    return (eff_pD, eff_pH, eff_pW), (eff_sD, eff_sH, eff_sW)


def sample_patch_3d(
    img: ArrayF,
    mask: ArrayF,
    patch_size: Tuple[int, int, int],
    bg_patch_size: Optional[Tuple[int, int, int]] = None,
    foreground_prob: float = 0.6,
    fg_component_sampling: str = "uniform",
    fg_component_sampling_alpha: float = 1.0,
    target_pos_patch_frac: Optional[float] = None,
    force_fg: bool = False,
    force_bg: bool = False,
    hard_bg_prob: float = 0.0,
    hard_bg_trials: int = 128,
    bg_inside_prob: float = 0.0,
    bg_inside_trials: int = 64,
    ensure_empty_bg: bool = False,
    ensure_empty_bg_trials: int = 256,
    bg_min_dist: Optional[int] = None,
    bg_min_dist_relax: bool = True,
    bg_overlap_trials: int = 0,
    bg_allow_fg_vox: int = 0,
    empty_fg_fallback: str = "bg",
    debug_meta: Optional[Dict[str, Any]] = None,
):
    def _prefix_sum_3d(bin_vol: np.ndarray) -> np.ndarray:
        # Use int64 to avoid overflow in inclusion-exclusion on larger windows.
        v = bin_vol.astype(np.int64, copy=False)
        p = np.pad(v, ((1, 0), (1, 0), (1, 0)), mode="constant", constant_values=0)
        return p.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

    def _window_sum(prefix: np.ndarray, z0: int, y0: int, x0: int, z1: int, y1: int, x1: int) -> int:
        # prefix is padded by 1 on each axis
        z0p, y0p, x0p = z0, y0, x0
        z1p, y1p, x1p = z1, y1, x1
        return int(
            prefix[z1p, y1p, x1p]
            - prefix[z0p, y1p, x1p]
            - prefix[z1p, y0p, x1p]
            - prefix[z1p, y1p, x0p]
            + prefix[z0p, y0p, x1p]
            + prefix[z0p, y1p, x0p]
            + prefix[z1p, y0p, x0p]
            - prefix[z0p, y0p, x0p]
        )

    C, D, H, W = img.shape
    patch_size_fg = tuple(int(x) for x in patch_size)
    patch_size_bg = tuple(int(x) for x in bg_patch_size) if bg_patch_size is not None else None
    hard_bg_prob = float(np.clip(hard_bg_prob, 0.0, 1.0))
    hard_bg_trials = int(max(1, hard_bg_trials))
    bg_inside_prob = float(np.clip(bg_inside_prob, 0.0, 1.0))
    bg_inside_trials = int(max(1, bg_inside_trials))
    bg_overlap_trials = int(max(0, bg_overlap_trials))
    bg_allow_fg_vox = int(max(0, bg_allow_fg_vox))

    empty_fg_fallback = str(empty_fg_fallback).strip().lower()
    fg_component_sampling = str(fg_component_sampling).strip().lower()
    fg_component_sampling_alpha = float(max(0.0, float(fg_component_sampling_alpha)))

    # Robust binarization: masks may contain small non-zero values due to resampling/storage.
    mask_bin = (mask > 0.5)
    has_fg = bool(mask_bin.any())

    force_fg = bool(force_fg)
    force_bg = bool(force_bg)
    if force_fg and force_bg:
        raise ValueError("force_fg and force_bg are mutually exclusive")

    if (not has_fg) and force_fg:
        # In rare cases, upstream mask could be empty. Decide how to handle "force_fg".
        zc, yc, xc = np.random.randint(D), np.random.randint(H), np.random.randint(W)
        use_fg = False
    else:
        if force_bg:
            use_fg = False
        elif target_pos_patch_frac is not None:
            t = float(np.clip(float(target_pos_patch_frac), 0.0, 1.0))
            use_fg = force_fg or (has_fg and (np.random.rand() < t))
        else:
            use_fg = force_fg or (np.random.rand() < foreground_prob and has_fg)

    if use_fg:
        pD, pH, pW = patch_size_fg
        zyx = np.argwhere(mask_bin)
        if fg_component_sampling in {"uniform", "none", "off", "false"}:
            zc, yc, xc = zyx[np.random.randint(len(zyx))]
            if debug_meta is not None:
                debug_meta["fg_sampling"] = "uniform"
        elif fg_component_sampling in {"inverse_size", "inv_size", "small", "small_first"}:
            # Prefer smaller connected components.
            # Sample CC with weight 1/(|CC|^alpha) then sample a voxel within the CC uniformly.
            # alpha=0 reduces to uniform-over-CC.
            lbl = cc_label(mask_bin.astype(np.uint8, copy=False))[0].astype(np.int64, copy=False)
            sizes = np.bincount(lbl.ravel())
            cc_ids = np.unique(lbl[mask_bin])
            cc_ids = cc_ids[cc_ids > 0]
            if cc_ids.size == 0:
                zc, yc, xc = zyx[np.random.randint(len(zyx))]
            else:
                cc_sizes = np.maximum(sizes[cc_ids], 1).astype(np.float64)
                w_cc = 1.0 / np.power(cc_sizes, float(fg_component_sampling_alpha))
                w_sum = float(w_cc.sum())
                if (not np.isfinite(w_sum)) or (w_sum <= 0.0):
                    chosen_cc = int(cc_ids[np.random.randint(int(cc_ids.size))])
                else:
                    w_cc = (w_cc / w_sum).astype(np.float64, copy=False)
                    chosen_cc = int(np.random.choice(cc_ids.astype(np.int64, copy=False), p=w_cc))

                comp_ids_per_vox = lbl[zyx[:, 0], zyx[:, 1], zyx[:, 2]]
                pick_idx = np.flatnonzero(comp_ids_per_vox == chosen_cc)
                if pick_idx.size == 0:
                    zc, yc, xc = zyx[np.random.randint(len(zyx))]
                else:
                    zc, yc, xc = zyx[int(np.random.choice(pick_idx))]

                if debug_meta is not None:
                    debug_meta["fg_sampling"] = "inverse_size"
                    debug_meta["fg_cc_id"] = int(chosen_cc)
                    debug_meta["fg_cc_size"] = int(max(1, sizes[int(chosen_cc)]))
                    debug_meta["fg_cc_alpha"] = float(fg_component_sampling_alpha)
        else:
            raise ValueError(
                f"Unknown fg_component_sampling={fg_component_sampling!r} "
                "(expected: uniform | inverse_size)"
            )
    else:
        # Background-centered sample: optionally use a smaller patch for BG to reduce FG leakage.
        if patch_size_bg is not None:
            pD, pH, pW = patch_size_bg
        else:
            pD, pH, pW = patch_size_fg
        zc, yc, xc = np.random.randint(D), np.random.randint(H), np.random.randint(W)
        picked_bg = False

        # Distance-constrained background sampling (inside-brain & far from FG).
        if has_fg:
            if bg_min_dist is None:
                bg_min_dist_eff = int(max(0, max(pD // 2, pH // 2, pW // 2)))
            else:
                bg_min_dist_eff = int(max(0, int(bg_min_dist)))

            if bg_min_dist_eff > 0:
                brain = (np.abs(img).max(axis=0) > 0.0)
                bg = (~mask_bin) & brain
                dist = distance_transform_cdt((~mask_bin).astype(np.uint8), metric="chessboard")

                if bool(bg_min_dist_relax):
                    thresholds = [
                        bg_min_dist_eff,
                        int(bg_min_dist_eff * 0.75),
                        int(bg_min_dist_eff * 0.5),
                        int(bg_min_dist_eff * 0.25),
                        0,
                    ]
                else:
                    thresholds = [bg_min_dist_eff]
                thresholds = sorted(set(int(max(0, t)) for t in thresholds), reverse=True)

                for t in thresholds:
                    cand = bg & (dist >= int(t))
                    if bool(cand.any()):
                        zyx = np.argwhere(cand)
                        zc, yc, xc = zyx[np.random.randint(len(zyx))]
                        picked_bg = True
                        break

                if bg_overlap_trials > 0:
                    min_dist_for_overlap = 0 if bool(bg_min_dist_relax) else int(bg_min_dist_eff)
                    fg_prefix = _prefix_sum_3d(mask_bin)
                    brain_prefix = _prefix_sum_3d(brain)
                    max_z0 = max(int(D - pD), 0)
                    max_y0 = max(int(H - pH), 0)
                    max_x0 = max(int(W - pW), 0)

                    best_origin = None
                    best_fg = None
                    for _ in range(bg_overlap_trials):
                        z0t = int(np.random.randint(0, max_z0 + 1))
                        y0t = int(np.random.randint(0, max_y0 + 1))
                        x0t = int(np.random.randint(0, max_x0 + 1))
                        z1t, y1t, x1t = z0t + pD, y0t + pH, x0t + pW

                        brain_vox = _window_sum(brain_prefix, z0t, y0t, x0t, z1t, y1t, x1t)
                        if brain_vox <= 0:
                            continue

                        if min_dist_for_overlap > 0:
                            zt = min(z0t + pD // 2, D - 1)
                            yt = min(y0t + pH // 2, H - 1)
                            xt = min(x0t + pW // 2, W - 1)
                            if int(dist[zt, yt, xt]) < int(min_dist_for_overlap):
                                continue

                        fg_vox = _window_sum(fg_prefix, z0t, y0t, x0t, z1t, y1t, x1t)
                        if fg_vox <= bg_allow_fg_vox:
                            best_origin = (z0t, y0t, x0t)
                            best_fg = fg_vox
                            break
                        if best_fg is None or fg_vox < best_fg:
                            best_fg = fg_vox
                            best_origin = (z0t, y0t, x0t)

                    if best_origin is not None:
                        z0t, y0t, x0t = best_origin
                        zc, yc, xc = z0t + pD // 2, y0t + pH // 2, x0t + pW // 2
                        picked_bg = True

        # Hard-negative sampling (approximate)
        use_hard_bg = (not picked_bg) and (np.random.rand() < hard_bg_prob)
        if use_hard_bg:
            best_score = None
            best_zyx = None
            for _ in range(hard_bg_trials):
                z = np.random.randint(D)
                y = np.random.randint(H)
                x = np.random.randint(W)
                if mask_bin[z, y, x]:
                    continue
                v = img[:, z, y, x]
                if float(np.abs(v).max()) <= 0.0:
                    continue
                score = float(v.max())
                if best_score is None or score > best_score:
                    best_score = score
                    best_zyx = (z, y, x)
            if best_zyx is not None:
                zc, yc, xc = best_zyx
            else:
                zc, yc, xc = np.random.randint(D), np.random.randint(H), np.random.randint(W)

        # Optional: bias background centers to be inside-brain
        if (not picked_bg) and (np.random.rand() < bg_inside_prob):
            best = None
            for _ in range(bg_inside_trials):
                z = np.random.randint(D)
                y = np.random.randint(H)
                x = np.random.randint(W)
                v = img[:, z, y, x]
                if float(np.abs(v).max()) <= 0.0:
                    continue
                best = (z, y, x)
                break
            if best is not None:
                zc, yc, xc = best

        # Optionally enforce an empty-background patch
        if bool(ensure_empty_bg) and bool(has_fg):
            best_empty = None
            best_min = None
            best_min_fg = None
            trials = int(max(1, ensure_empty_bg_trials))
            # NOTE: Do NOT require the patch to overlap a "brain" mask here.
            # Some preprocessed/cropped datasets have non-zero voxels only near the lesion,
            # so enforcing inside-brain would make it impossible to find an empty patch.
            fg_prefix = _prefix_sum_3d(mask_bin)
            max_z0 = max(int(D - pD), 0)
            max_y0 = max(int(H - pH), 0)
            max_x0 = max(int(W - pW), 0)

            for _ in range(trials):
                z0t = int(np.random.randint(0, max_z0 + 1))
                y0t = int(np.random.randint(0, max_y0 + 1))
                x0t = int(np.random.randint(0, max_x0 + 1))
                z1t, y1t, x1t = z0t + pD, y0t + pH, x0t + pW

                fg_vox = _window_sum(fg_prefix, z0t, y0t, x0t, z1t, y1t, x1t)
                if fg_vox == 0:
                    best_empty = (z0t, y0t, x0t)
                    break
                if best_min_fg is None or fg_vox < best_min_fg:
                    best_min_fg = fg_vox
                    best_min = (z0t, y0t, x0t)

            if best_empty is not None:
                z0t, y0t, x0t = best_empty
                zc, yc, xc = z0t + pD // 2, y0t + pH // 2, x0t + pW // 2
            elif best_min is not None:
                z0t, y0t, x0t = best_min
                zc, yc, xc = z0t + pD // 2, y0t + pH // 2, x0t + pW // 2

    if debug_meta is not None:
        debug_meta["use_fg"] = bool(use_fg)

    z0 = np.clip(zc - pD // 2, 0, max(D - pD, 0))
    y0 = np.clip(yc - pH // 2, 0, max(H - pH, 0))
    x0 = np.clip(xc - pW // 2, 0, max(W - pW, 0))

    z1, y1, x1 = min(z0 + pD, D), min(y0 + pH, H), min(x0 + pW, W)
    patch_img = img[:, z0:z1, y0:y1, x0:x1]
    patch_mask = mask_bin[z0:z1, y0:y1, x0:x1].astype(np.float32)

    pad_z, pad_y, pad_x = pD - patch_img.shape[1], pH - patch_img.shape[2], pW - patch_img.shape[3]
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        patch_img = np.pad(patch_img, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
        patch_mask = np.pad(patch_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    return patch_img, patch_mask


def _neg_alarm_weight_for_epoch(epoch: int, base_weight: float, warmup_epochs: int) -> float:
    base_weight = float(max(0.0, base_weight))
    warmup_epochs = int(max(0, warmup_epochs))
    if base_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return base_weight
    # linear ramp: 0 -> base_weight over warmup_epochs
    t = float(np.clip(epoch / float(warmup_epochs), 0.0, 1.0))
    return base_weight * t


def augment_patch(img: ArrayF, mask: ArrayF):
    # Simple spatial + intensity jitter for robustness.
    if np.random.rand() < 0.5:
        img = img[:, ::-1]
        mask = mask[::-1]
    if np.random.rand() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]
    if np.random.rand() < 0.5:
        img = img[:, :, :, ::-1]
        mask = mask[:, :, ::-1]

    if np.random.rand() < 0.4:
        # Small affine: isotropic scale and small rotations about each axis.
        scale = np.random.uniform(0.9, 1.1)
        angles = np.radians(np.random.uniform(-5.0, 5.0, size=3))
        sx, sy, sz = np.sin(angles)
        cx, cy, cz = np.cos(angles)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        rot = rot_z @ rot_y @ rot_x
        matrix = rot * scale
        center = 0.5 * (np.array(img.shape[1:]) - 1)
        offset = center - matrix @ center

        warped_img = []
        for c in range(img.shape[0]):
            warped_img.append(
                affine_transform(
                    img[c],
                    matrix=matrix,
                    offset=offset,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
            )
        img = np.stack(warped_img, axis=0)
        mask = affine_transform(
            mask,
            matrix=matrix,
            offset=offset,
            order=0,
            mode="constant",
            cval=0.0,
        )

    if np.random.rand() < 0.4:
        # Contrast/brightness jitter.
        contrast = np.random.uniform(0.85, 1.15)
        brightness = np.random.uniform(-0.05, 0.05)
        img = img * contrast + brightness

    if np.random.rand() < 0.3:
        sigma = np.random.uniform(0.4, 1.0)
        img = gaussian_filter(img, sigma=(0.0, sigma, sigma, sigma))

    if np.random.rand() < 0.4:
        noise_std = np.random.uniform(0.01, 0.05)
        img = img + np.random.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    if np.random.rand() < 0.3:
        gamma = np.random.uniform(0.8, 1.4)
        img = (np.sign(img) * (np.abs(img) ** gamma)).astype(np.float32)

    # Optional: light blur on mask to detect misalignment visually (disabled by default)
    # if np.random.rand() < 0.0:
    #     mask = gaussian_filter(mask, sigma=0.6)

    return img.astype(np.float32), mask.astype(np.float32)


def build_loss(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "dice_bce":
        return DiceBCELoss(
            smooth=params.get("smooth", 1.0),
            pos_weight=params.get("pos_weight", 1.0),
            bce_weight=params.get("bce_weight", 1.0),
        )
    if name == "dice_ohem_bce":
        return DiceOHEMBCELoss(
            smooth=params.get("smooth", 1.0),
            neg_fraction=params.get("neg_fraction", 0.1),
            min_neg=params.get("min_neg", 1024),
            bce_weight=params.get("bce_weight", 1.0),
            pos_weight=params.get("pos_weight", 1.0),
            neg_weight=params.get("neg_weight", 1.0),
        )
    if name == "dice_focal":
        return DiceFocalLoss(alpha=params.get("alpha", 0.25), gamma=params.get("gamma", 2.0), smooth=params.get("smooth", 1.0))
    if name == "tversky":
        return TverskyLoss(alpha=params.get("alpha", 0.3), beta=params.get("beta", 0.7), smooth=params.get("smooth", 1.0))
    if name == "tversky_focal":
        return TverskyFocalLoss(alpha=params.get("alpha", 0.3), beta=params.get("beta", 0.7), gamma=params.get("gamma", 1.33), smooth=params.get("smooth", 1.0))
    if name == "tversky_ohem_bce":
        return TverskyOHEMBCELoss(
            alpha=params.get("alpha", 0.3),
            beta=params.get("beta", 0.7),
            smooth=params.get("smooth", 1.0),
            neg_fraction=params.get("neg_fraction", 0.1),
            min_neg=params.get("min_neg", 1024),
            bce_weight=params.get("bce_weight", 1.0),
            pos_weight=params.get("pos_weight", 1.0),
            neg_weight=params.get("neg_weight", 1.0),
        )
    raise ValueError(f"Unknown loss: {name}")


def _stage2_corrector_ohem_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    stage1_probs: torch.Tensor,
    *,
    fusion: str = "max",
    stage1_logit_eps: float = 1e-4,
    pos_lambda: float,
    neg_lambda: float,
    gamma: float,
    neg_fraction: float,
    min_neg: int,
) -> torch.Tensor:
    """Stage2 'corrector' loss.

    Fusion modes:
    - max: logits are treated as absolute Stage2 logits. (final probs uses max at inference)
    - residual: logits are treated as delta logits to be added to logit(Stage1 probs).

    Weighting intuition:
    - Always emphasize positives where Stage1 prob is low (recover FN).
    - For max-fusion, penalize negatives mainly where Stage1 is low (avoid introducing new FP).
    - For residual-fusion, penalize negatives mainly where Stage1 is high (enable FP suppression).

    Uses weighted BCE + OHEM on negatives.
    """
    fusion = str(fusion).strip().lower()
    p1 = torch.clamp(stage1_probs, 0.0, 1.0)
    g = float(max(0.0, gamma))
    inv = (1.0 - p1).pow(g)
    hi = p1.pow(g)
    pos_l = float(max(0.0, pos_lambda))
    neg_l = float(max(0.0, neg_lambda))
    # weight map
    if fusion == "residual":
        # negatives weighted where Stage1 is confidently positive (to suppress Stage1 FP)
        w_neg = hi
        eps = float(max(1e-8, min(1e-2, float(stage1_logit_eps))))
        p1c = torch.clamp(p1, eps, 1.0 - eps)
        fused_logits = torch.log(p1c / (1.0 - p1c)) + logits
        logits_for_loss = fused_logits
    else:
        # negatives weighted where Stage1 is confidently negative (avoid new FP)
        w_neg = inv
        logits_for_loss = logits

    w = 1.0 + (pos_l * inv) * (targets > 0.5).float() + (neg_l * w_neg) * (targets <= 0.5).float()

    bce = F.binary_cross_entropy_with_logits(logits_for_loss, targets, reduction="none")
    bce = bce * w

    bce_flat = bce.flatten(1)
    t_flat = targets.flatten(1)
    pos_mask = t_flat > 0.5
    neg_mask = ~pos_mask

    pos_loss = bce_flat[pos_mask]
    neg_loss = bce_flat[neg_mask]

    pos_mean = pos_loss.mean() if pos_loss.numel() > 0 else bce_flat.new_tensor(0.0)

    if neg_loss.numel() > 0:
        frac = float(max(0.0, min(1.0, neg_fraction)))
        k = int(max(int(min_neg), round(frac * float(neg_loss.numel()))))
        k = int(min(k, int(neg_loss.numel())))
        hard_neg = neg_loss.topk(k, largest=True).values
        neg_mean = hard_neg.mean()
    else:
        neg_mean = bce_flat.new_tensor(0.0)

    return pos_mean + neg_mean


def sliding_window_predict(
    img: torch.Tensor,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
):
    """Tile the volume into patches, run the model per patch, and aggregate logits."""
    if img.ndim == 4:  # (C, D, H, W)
        img = img.unsqueeze(0)

    _, _, D, H, W = img.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride

    logits_sum = torch.zeros((1, 1, D, H, W), device=img.device)
    counter = torch.zeros_like(logits_sum)

    for z in range(0, D, sD):
        for y in range(0, H, sH):
            for x in range(0, W, sW):
                z1, y1, x1 = min(z + pD, D), min(y + pH, H), min(x + pW, W)
                patch = img[:, :, z:z1, y:y1, x:x1]

                pad_d, pad_h, pad_w = pD - patch.size(2), pH - patch.size(3), pW - patch.size(4)
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    patch = torch.nn.functional.pad(
                        patch,
                        (
                            0, pad_w,
                            0, pad_h,
                            0, pad_d,
                        ),
                        mode="constant",
                    )

                out = model(patch)
                logit = out[0] if isinstance(out, (tuple, list)) else out
                logit = logit[:, :, : z1 - z, : y1 - y, : x1 - x]

                logits_sum[:, :, z:z1, y:y1, x:x1] += logit
                counter[:, :, z:z1, y:y1, x:x1] += 1

    return logits_sum / counter.clamp_min(1.0)


def remove_small_components(pred: NDArray[np.uint8], min_size: int) -> NDArray[np.uint8]:
    if min_size <= 0:
        return pred
    lbl = cc_label(pred)[0]
    lbl = lbl.astype(np.int64, copy=False)
    if lbl.max() == 0:
        return pred
    sizes = np.bincount(lbl.ravel())
    remove = sizes < int(min_size)
    remove[0] = False
    pred = pred.copy()
    pred[remove[lbl]] = 0
    return pred


@app.command()
def main(config: str = typer.Option(..., help="Path to YAML config")):
    cfg = yaml.safe_load(Path(config).read_text())
    # Make config paths independent of the current working directory.
    repo_root = Path(__file__).resolve().parents[2]
    for section, key in [("data", "csv_path"), ("data", "root"), ("log", "out_dir")]:
        raw = cfg.get(section, {}).get(key, None)
        if raw is None:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        cfg[section][key] = str(p)
    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)
    device = prepare_device()

    norm_mode = cfg.get("data", {}).get("normalize", "legacy_zscore")
    allow_missing_label = bool(cfg.get("data", {}).get("allow_missing_label", False))

    # Optional: conditional cascade input channel (Stage1 probability map).
    stage1_probs_dir_train_raw = cfg.get("data", {}).get("stage1_probs_dir_train", None)
    stage1_probs_dir_val_raw = cfg.get("data", {}).get("stage1_probs_dir_val", None)
    stage1_probs_dir_raw = cfg.get("data", {}).get("stage1_probs_dir", None)

    def _resolve_probs_dir(v: Any) -> Path | None:
        if v is None or (isinstance(v, str) and (not v.strip())):
            return None
        p = Path(str(v)).expanduser()
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        return p

    stage1_probs_dir_train = _resolve_probs_dir(stage1_probs_dir_train_raw or stage1_probs_dir_raw)
    stage1_probs_dir_val = _resolve_probs_dir(stage1_probs_dir_val_raw or stage1_probs_dir_raw)
    if stage1_probs_dir_train is not None and (not stage1_probs_dir_train.exists()):
        raise FileNotFoundError(f"data.stage1_probs_dir_train not found: {stage1_probs_dir_train}")
    if stage1_probs_dir_val is not None and (not stage1_probs_dir_val.exists()):
        raise FileNotFoundError(f"data.stage1_probs_dir_val not found: {stage1_probs_dir_val}")

    train_vol = IslesVolumeDataset(
        cfg["data"]["csv_path"],
        split="train",
        root=cfg["data"]["root"],
        transform=None,
        normalize=norm_mode,
        allow_missing_label=allow_missing_label,
    )
    val_vol = IslesVolumeDataset(
        cfg["data"]["csv_path"],
        split="val",
        root=cfg["data"]["root"],
        transform=None,
        normalize=norm_mode,
        allow_missing_label=allow_missing_label,
    )

    print(
        f"[start] exp={cfg.get('experiment_name')} device={device} "
        f"seed={seed} "
        f"modalities={cfg.get('data', {}).get('modalities')} normalize={norm_mode} "
        f"train={len(train_vol)} val={len(val_vol)} root={cfg.get('data', {}).get('root')} csv={cfg.get('data', {}).get('csv_path')} "
        f"stage1_probs_train={None if stage1_probs_dir_train is None else str(stage1_probs_dir_train)} "
        f"stage1_probs_val={None if stage1_probs_dir_val is None else str(stage1_probs_dir_val)}",
        flush=True,
    )

    train_loader = DataLoader(
        train_vol,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: x,  # keep variable-size volumes; patch sampling will handle stacking
    )
    val_loader = DataLoader(
        val_vol,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    first_img = train_vol[0]["image"]
    base_in_ch = int(first_img.shape[0]) if getattr(first_img, "ndim", 0) == 4 else 1
    in_ch = int(base_in_ch + (1 if (stage1_probs_dir_train is not None or stage1_probs_dir_val is not None) else 0))
    base_ch = cfg["train"].get("base_ch", 16)
    deep_supervision = bool(cfg.get("train", {}).get("deep_supervision", False))
    ds_aux2_w = float(cfg.get("train", {}).get("deep_supervision_aux2_weight", 0.5))
    ds_aux3_w = float(cfg.get("train", {}).get("deep_supervision_aux3_weight", 0.25))
    norm_raw = cfg.get("train", {}).get("norm", "auto")
    norm_s = str(norm_raw).strip().lower()
    if norm_s in {"", "auto"}:
        # MPS BatchNorm has known instability; default to InstanceNorm for training on MPS.
        model_norm = "instance" if device.type == "mps" else "batch"
    else:
        model_norm = norm_s

    model = UNet3D(
        in_channels=in_ch,
        out_channels=1,
        base_ch=base_ch,
        deep_supervision=deep_supervision,
        norm=model_norm,
    ).to(device)
    print(f"[model] norm={model_norm}", flush=True)

    init_from = cfg.get("train", {}).get("init_from", None)
    if init_from is not None and str(init_from).strip():
        p = Path(str(init_from)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"train.init_from not found: {p}")
        state = torch.load(str(p), map_location=device)
        sd = state["model"] if isinstance(state, dict) and "model" in state else state

        # If input channels differ (e.g., conditional cascade adds Stage1 probs channel),
        # adapt the first conv weights to avoid size-mismatch errors.
        try:
            w = sd.get("enc1.0.weight") if hasattr(sd, "get") else None
        except Exception:
            w = None
        if isinstance(w, torch.Tensor) and w.ndim == 5:
            in_ch_ckpt = int(w.shape[1])
            in_ch_cur = int(in_ch)
            if in_ch_ckpt != in_ch_cur:
                w_new = torch.zeros((int(w.shape[0]), in_ch_cur, int(w.shape[2]), int(w.shape[3]), int(w.shape[4])), dtype=w.dtype)
                cmin = int(min(in_ch_ckpt, in_ch_cur))
                w_new[:, :cmin] = w[:, :cmin].detach().cpu()
                if in_ch_cur > in_ch_ckpt:
                    mean = w[:, :cmin].detach().cpu().mean(dim=1, keepdim=True)
                    w_new[:, cmin:] = mean.repeat(1, int(in_ch_cur - cmin), 1, 1, 1)
                sd = dict(sd)
                sd["enc1.0.weight"] = w_new
                print(f"[init] adapted enc1.0.weight in_ch {in_ch_ckpt} -> {in_ch_cur}", flush=True)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded weights from {p} (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    criterion = build_loss(cfg["train"]["loss"], cfg["train"].get("loss_params", {}))
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    grad_clip = cfg["train"].get("grad_clip", None)
    device_type = device.type
    use_amp = cfg["train"]["amp"] and device_type == "cuda"
    if use_amp:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        autocast_ctx = partial(torch.amp.autocast, "cuda", enabled=True)
    else:
        scaler = None
        autocast_ctx = contextlib.nullcontext
    scheduler = CosineAnnealingLR(
        optim,
        T_max=cfg["train"]["epochs"],
        eta_min=cfg["train"].get("min_lr", 1e-5),
    )

    out_dir = Path(cfg["log"]["out_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    patch_size = tuple(cfg["data"]["patch_size"])
    val_stride = tuple(cfg["data"].get("val_stride", patch_size))

    bg_patch_size = cfg.get("data", {}).get("bg_patch_size", None)
    if bg_patch_size is not None:
        bg_patch_size = tuple(int(x) for x in bg_patch_size)

    val_min_size = int(cfg.get("data", {}).get("val_min_size", 0))
    val_min_size = int(max(0, val_min_size))

    # Validation thresholding:
    # - numeric `data.val_threshold`: fixed threshold
    # - string `data.val_threshold: auto`: choose threshold that maximizes mean val Dice for the epoch
    val_threshold_raw = cfg.get("data", {}).get("val_threshold", 0.5)
    val_threshold_auto = isinstance(val_threshold_raw, str) and val_threshold_raw.strip().lower() == "auto"
    if val_threshold_auto:
        candidates = cfg.get("data", {}).get("val_threshold_candidates", None)
        if candidates is None:
            candidates = [round(float(x), 2) for x in np.arange(0.05, 0.901, 0.05)]
        val_threshold_candidates = [float(x) for x in candidates]
        val_threshold_candidates = [float(np.clip(x, 0.0, 1.0)) for x in val_threshold_candidates]
        val_threshold_candidates = sorted(set(val_threshold_candidates))
        if not val_threshold_candidates:
            raise ValueError("data.val_threshold_candidates must be non-empty when val_threshold='auto'")
        val_threshold = float(val_threshold_candidates[0])
    else:
        val_threshold_candidates = None
        val_threshold = float(np.clip(float(val_threshold_raw), 0.0, 1.0))

    force_fg_prob = float(cfg.get("data", {}).get("force_fg_prob", 1.0))
    force_fg_prob = float(np.clip(force_fg_prob, 0.0, 1.0))

    patches_per_volume = int(cfg.get("data", {}).get("patches_per_volume", 1) or 1)
    patches_per_volume = int(max(1, patches_per_volume))
    patches_force_one_bg = bool(cfg.get("data", {}).get("patches_force_one_bg", False))

    hard_bg_prob = float(cfg.get("data", {}).get("hard_bg_prob", 0.0))
    hard_bg_prob = float(np.clip(hard_bg_prob, 0.0, 1.0))
    hard_bg_trials = int(cfg.get("data", {}).get("hard_bg_trials", 128))
    hard_bg_trials = int(max(1, hard_bg_trials))

    bg_inside_prob = float(cfg.get("data", {}).get("bg_inside_prob", 0.0))
    bg_inside_prob = float(np.clip(bg_inside_prob, 0.0, 1.0))
    bg_inside_trials = int(cfg.get("data", {}).get("bg_inside_trials", 64))
    bg_inside_trials = int(max(1, bg_inside_trials))

    ensure_empty_bg_prob = float(cfg.get("data", {}).get("ensure_empty_bg_prob", 0.0))
    ensure_empty_bg_prob = float(np.clip(ensure_empty_bg_prob, 0.0, 1.0))
    ensure_empty_bg_trials = int(cfg.get("data", {}).get("ensure_empty_bg_trials", 256))
    ensure_empty_bg_trials = int(max(1, ensure_empty_bg_trials))

    # Sampling controls (for pos_patch_frac):
    # - data.target_pos_patch_frac: if set, controls the probability of sampling FG centers (for positive cases)
    # - data.bg_min_dist: minimum distance from FG for background centers (voxels). If None, defaults to half of min(patch_size).
    target_pos_patch_frac = cfg.get("data", {}).get("target_pos_patch_frac", None)
    if target_pos_patch_frac is not None:
        target_pos_patch_frac = float(np.clip(float(target_pos_patch_frac), 0.0, 1.0))
    bg_min_dist = cfg.get("data", {}).get("bg_min_dist", None)
    bg_min_dist = int(bg_min_dist) if bg_min_dist is not None else None
    bg_min_dist_relax = bool(cfg.get("data", {}).get("bg_min_dist_relax", True))

    bg_overlap_trials = int(cfg.get("data", {}).get("bg_overlap_trials", 0) or 0)
    bg_allow_fg_vox = int(cfg.get("data", {}).get("bg_allow_fg_vox", 0) or 0)

    # Optional: match evaluation-time spacing rule by upsampling (never downsampling)
    # so that the maximum voxel spacing becomes <= data.resample_max_zoom_mm.
    resample_max_zoom_mm = float(cfg.get("data", {}).get("resample_max_zoom_mm", 0.0) or 0.0)
    resample_max_zoom_mm = float(max(0.0, resample_max_zoom_mm))

    # Foreground center selection (for small-lesion emphasis).
    fg_component_sampling = cfg.get("data", {}).get("fg_component_sampling", "uniform")
    fg_component_sampling_alpha = float(cfg.get("data", {}).get("fg_component_sampling_alpha", 1.0))

    # Optional: Stage1 candidate-driven sampling for cascade refinement.
    cand_jsonl = cfg.get("data", {}).get("candidate_jsonl", None)
    cand_use_prob = float(cfg.get("data", {}).get("candidate_use_prob", 0.0))
    cand_use_prob = float(np.clip(cand_use_prob, 0.0, 1.0))
    cand_pos_frac = float(cfg.get("data", {}).get("candidate_pos_fraction", 0.5))
    cand_pos_frac = float(np.clip(cand_pos_frac, 0.0, 1.0))
    candidates_by_case: dict[str, dict[str, list[list[int]]]] | None = None
    if cand_jsonl is not None and str(cand_jsonl).strip():
        p = Path(str(cand_jsonl)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"data.candidate_jsonl not found: {p}")
        candidates_by_case = _load_candidates_jsonl(p)
        print(
            f"[candidates] loaded {sum(len(v['pos'])+len(v['neg']) for v in candidates_by_case.values())} boxes "
            f"for {len(candidates_by_case)} cases from {p} (use_prob={cand_use_prob} pos_frac={cand_pos_frac})",
            flush=True,
        )

    # Smoke helpers: limit steps per epoch / val cases (optional)
    max_steps_per_epoch = cfg.get("train", {}).get("max_steps_per_epoch", None)
    max_steps_per_epoch = int(max_steps_per_epoch) if max_steps_per_epoch is not None else None
    max_val_steps = cfg.get("train", {}).get("max_val_steps", None)
    max_val_steps = int(max_val_steps) if max_val_steps is not None else None

    # Negative-case (empty-mask) alarm penalty:
    # For patches with target all-zero, penalize high logits ("any positive" behavior) using top-k softplus.
    neg_alarm_weight = float(cfg.get("train", {}).get("neg_alarm_weight", 0.0))
    neg_alarm_weight = float(max(0.0, neg_alarm_weight))
    neg_alarm_topk = int(cfg.get("train", {}).get("neg_alarm_topk", 1024))
    neg_alarm_topk = int(max(1, neg_alarm_topk))
    neg_alarm_warmup_epochs = int(cfg.get("train", {}).get("neg_alarm_warmup_epochs", 0))
    neg_alarm_warmup_epochs = int(max(0, neg_alarm_warmup_epochs))

    # Stage2 corrector objective (only meaningful when Stage1 probs are appended as an input channel).
    stage2_corrector_loss = bool(cfg.get("train", {}).get("stage2_corrector_loss", False))
    stage2_corrector_weight = float(cfg.get("train", {}).get("stage2_corrector_weight", 0.0) or 0.0)
    stage2_corrector_weight = float(max(0.0, stage2_corrector_weight))
    stage2_corrector_pos_lambda = float(cfg.get("train", {}).get("stage2_corrector_pos_lambda", 0.0) or 0.0)
    stage2_corrector_neg_lambda = float(cfg.get("train", {}).get("stage2_corrector_neg_lambda", 0.0) or 0.0)
    stage2_corrector_gamma = float(cfg.get("train", {}).get("stage2_corrector_gamma", 1.0) or 1.0)
    stage2_corrector_neg_fraction = float(cfg.get("train", {}).get("stage2_corrector_neg_fraction", 0.02) or 0.02)
    stage2_corrector_min_neg = int(cfg.get("train", {}).get("stage2_corrector_min_neg", 2048) or 2048)
    stage2_corrector_min_neg = int(max(1, stage2_corrector_min_neg))
    stage2_corrector_fusion = str(cfg.get("train", {}).get("stage2_corrector_fusion", "max") or "max").strip().lower()
    if stage2_corrector_fusion not in {"max", "residual"}:
        print(f"[warn] train.stage2_corrector_fusion={stage2_corrector_fusion!r} not in {{'max','residual'}}, falling back to 'max'", flush=True)
        stage2_corrector_fusion = "max"
    stage2_corrector_stage1_logit_eps = float(cfg.get("train", {}).get("stage2_corrector_stage1_logit_eps", 1e-4) or 1e-4)
    stage2_corrector_stage1_logit_eps = float(max(1e-8, min(1e-2, stage2_corrector_stage1_logit_eps)))

    # If true, apply neg_alarm only for truly negative cases (case mask all-zero),
    # not for empty patches sampled from positive cases.
    neg_alarm_only_on_negative_cases = bool(cfg.get("train", {}).get("neg_alarm_only_on_negative_cases", False))

    if neg_alarm_weight > 0.0 and (not neg_alarm_only_on_negative_cases):
        if not allow_missing_label:
            hint = (
                "allow_missing_label=false (no true negative cases), so neg_alarm applies only to empty patches sampled from positive cases. "
                "This setting has tended to increase FP; consider setting train.neg_alarm_only_on_negative_cases: true or disabling neg_alarm."
            )
        else:
            hint = (
                "neg_alarm_only_on_negative_cases=false, so neg_alarm also applies to empty patches sampled from positive cases. "
                "If FP increases, consider setting train.neg_alarm_only_on_negative_cases: true or disabling neg_alarm."
            )
        print(f"[warn] neg_alarm_weight={neg_alarm_weight} with neg_alarm_only_on_negative_cases=false. {hint}", flush=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        meter = AverageMeter()
        # lightweight sanity stats for the first epoch to detect empty masks/patch imbalance
        mask_sums = []
        pos_patches = 0
        total_patches = 0
        fg_forced = 0
        for step_i, batch in enumerate(train_loader, start=1):
            # Ensure patch_size does not exceed any volume in the batch (no padding).
            batch_min_D = min(int(s["image"].shape[1]) for s in batch)
            batch_min_H = min(int(s["image"].shape[2]) for s in batch)
            batch_min_W = min(int(s["image"].shape[3]) for s in batch)
            patch_size_eff, _ = _effective_patch_and_stride(
                patch_size,
                stride=patch_size,  # unused
                vol_shape=(batch_min_D, batch_min_H, batch_min_W),
                multiple=8,
            )

            if bg_patch_size is not None:
                bg_patch_size_eff, _ = _effective_patch_and_stride(
                    bg_patch_size,
                    stride=bg_patch_size,  # unused
                    vol_shape=(batch_min_D, batch_min_H, batch_min_W),
                    multiple=8,
                )
            else:
                bg_patch_size_eff = None

            patches_img, patches_mask = [], []
            patches_case_has_fg = []
            for sample in batch:
                img = sample["image"]
                mask = sample["mask"]
                case_id = str(sample.get("case_id", ""))
                meta = sample.get("meta", {}) or {}

                if resample_max_zoom_mm > 0.0:
                    img, mask = _maybe_resample_case(img, mask, meta, resample_max_zoom_mm)

                if stage1_probs_dir_train is not None:
                    probs = _load_stage1_probs(case_id, stage1_probs_dir_train)
                    if probs is not None:
                        probs = _align_probs_to_zyx(probs, (int(img.shape[1]), int(img.shape[2]), int(img.shape[3])))
                        img = np.concatenate([img.astype(np.float32, copy=False), probs[None, ...]], axis=0)
                    else:
                        # Missing Stage1 probs: fall back to zeros channel.
                        zeros = np.zeros((1, int(img.shape[1]), int(img.shape[2]), int(img.shape[3])), dtype=np.float32)
                        img = np.concatenate([img.astype(np.float32, copy=False), zeros], axis=0)

                mask_sum_val = float((mask > 0.5).sum())
                mask_sums.append(mask_sum_val)
                has_fg = mask_sum_val > 0
                for p_i in range(patches_per_volume):
                    force_fg = bool(has_fg and (np.random.rand() < force_fg_prob))

                    # Optional: ensure we always include one BG-only patch per volume.
                    force_bg = False
                    ensure_empty_bg_this = bool((np.random.rand() < ensure_empty_bg_prob) and (not force_fg))
                    if patches_force_one_bg and patches_per_volume >= 2:
                        if p_i == 0:
                            # First patch: prefer FG when possible (unless force_fg_prob=0).
                            force_bg = False
                        elif p_i == 1:
                            # Second patch: force background (helps control FP in all-positive datasets).
                            force_fg = False
                            force_bg = True
                            ensure_empty_bg_this = True

                    if force_fg:
                        fg_forced += 1

                    used_candidate = False
                    if (
                        candidates_by_case is not None
                        and cand_use_prob > 0.0
                        and (not force_bg)
                        and case_id in candidates_by_case
                        and (np.random.rand() < cand_use_prob)
                    ):
                        cand = candidates_by_case[case_id]
                        want_pos = bool(np.random.rand() < cand_pos_frac)
                        pool = cand.get("pos", []) if (want_pos and cand.get("pos")) else cand.get("neg", [])
                        if not pool:
                            pool = cand.get("pos", []) or cand.get("neg", [])
                        if pool:
                            bbox = pool[int(np.random.randint(len(pool)))]
                            cz = int((int(bbox[0]) + int(bbox[1])) // 2)
                            cy = int((int(bbox[2]) + int(bbox[3])) // 2)
                            cx = int((int(bbox[4]) + int(bbox[5])) // 2)
                            pimg, pmask = _crop_patch_center_zyx(img, mask, (cz, cy, cx), patch_size_eff)
                            used_candidate = True

                    if not used_candidate:
                        pimg, pmask = sample_patch_3d(
                            img,
                            mask,
                            patch_size_eff,
                            bg_patch_size=bg_patch_size_eff,
                            foreground_prob=cfg["data"]["foreground_prob"],
                            fg_component_sampling=fg_component_sampling,
                            fg_component_sampling_alpha=fg_component_sampling_alpha,
                            target_pos_patch_frac=target_pos_patch_frac,
                            force_fg=force_fg,
                            force_bg=force_bg,
                            hard_bg_prob=hard_bg_prob,
                            hard_bg_trials=hard_bg_trials,
                            bg_inside_prob=bg_inside_prob,
                            bg_inside_trials=bg_inside_trials,
                            ensure_empty_bg=ensure_empty_bg_this,
                            ensure_empty_bg_trials=ensure_empty_bg_trials,
                            bg_min_dist=bg_min_dist,
                            bg_min_dist_relax=bg_min_dist_relax,
                            bg_overlap_trials=bg_overlap_trials,
                            bg_allow_fg_vox=bg_allow_fg_vox,
                        )
                    pimg, pmask = augment_patch(pimg, pmask)
                    pos_patches += int(pmask.sum() > 0)
                    total_patches += 1
                    patches_img.append(pimg)
                    patches_mask.append(pmask)
                    patches_case_has_fg.append(bool(has_fg))
            patches_img = torch.from_numpy(np.stack(patches_img, axis=0)).float().to(device)
            # Safety: ensure mask is strictly 0/1 even if upstream preprocessing used 0/255 or float noise.
            patches_mask = torch.from_numpy(np.stack(patches_mask, axis=0)).float().unsqueeze(1).to(device)
            patches_mask = (patches_mask > 0.5).float()

            # If Stage1 probs are appended as the last input channel, reuse them for the corrector objective.
            stage1_patch = None
            if stage1_probs_dir_train is not None and patches_img.size(1) >= 2:
                stage1_patch = patches_img[:, -1:, ...].detach()

            optim.zero_grad()
            with autocast_ctx():
                out = model(patches_img)
                if isinstance(out, (tuple, list)):
                    logits, aux2, aux3 = out

                    logits_main = logits
                    if stage1_patch is not None and stage2_corrector_fusion == "residual":
                        p1c = torch.clamp(stage1_patch, stage2_corrector_stage1_logit_eps, 1.0 - stage2_corrector_stage1_logit_eps)
                        logits_main = torch.log(p1c / (1.0 - p1c)) + logits

                    loss = criterion(logits_main, patches_mask)

                    if (
                        stage2_corrector_loss
                        and stage2_corrector_weight > 0.0
                        and stage1_patch is not None
                        and stage1_probs_dir_train is not None
                    ):
                        loss = loss + (stage2_corrector_weight * _stage2_corrector_ohem_bce(
                            logits,
                            patches_mask,
                            stage1_patch,
                            fusion=stage2_corrector_fusion,
                            stage1_logit_eps=stage2_corrector_stage1_logit_eps,
                            pos_lambda=stage2_corrector_pos_lambda,
                            neg_lambda=stage2_corrector_neg_lambda,
                            gamma=stage2_corrector_gamma,
                            neg_fraction=stage2_corrector_neg_fraction,
                            min_neg=stage2_corrector_min_neg,
                        ))

                    if aux2 is not None and ds_aux2_w > 0.0:
                        m2 = _downsample_like(patches_mask, aux2.shape[2:], kind="mask")
                        aux2_main = aux2
                        if stage1_patch is not None and stage2_corrector_fusion == "residual":
                            p2 = _downsample_like(stage1_patch, aux2.shape[2:], kind="probs")
                            p2c = torch.clamp(p2, stage2_corrector_stage1_logit_eps, 1.0 - stage2_corrector_stage1_logit_eps)
                            aux2_main = torch.log(p2c / (1.0 - p2c)) + aux2
                        loss = loss + (ds_aux2_w * criterion(aux2_main, m2))

                        if (
                            stage2_corrector_loss
                            and stage2_corrector_weight > 0.0
                            and stage1_patch is not None
                            and stage1_probs_dir_train is not None
                        ):
                            p2 = _downsample_like(stage1_patch, aux2.shape[2:], kind="probs")
                            loss = loss + (ds_aux2_w * stage2_corrector_weight * _stage2_corrector_ohem_bce(
                                aux2,
                                m2,
                                p2,
                                fusion=stage2_corrector_fusion,
                                stage1_logit_eps=stage2_corrector_stage1_logit_eps,
                                pos_lambda=stage2_corrector_pos_lambda,
                                neg_lambda=stage2_corrector_neg_lambda,
                                gamma=stage2_corrector_gamma,
                                neg_fraction=stage2_corrector_neg_fraction,
                                min_neg=stage2_corrector_min_neg,
                            ))

                    if aux3 is not None and ds_aux3_w > 0.0:
                        m3 = _downsample_like(patches_mask, aux3.shape[2:], kind="mask")
                        aux3_main = aux3
                        if stage1_patch is not None and stage2_corrector_fusion == "residual":
                            p3 = _downsample_like(stage1_patch, aux3.shape[2:], kind="probs")
                            p3c = torch.clamp(p3, stage2_corrector_stage1_logit_eps, 1.0 - stage2_corrector_stage1_logit_eps)
                            aux3_main = torch.log(p3c / (1.0 - p3c)) + aux3
                        loss = loss + (ds_aux3_w * criterion(aux3_main, m3))

                        if (
                            stage2_corrector_loss
                            and stage2_corrector_weight > 0.0
                            and stage1_patch is not None
                            and stage1_probs_dir_train is not None
                        ):
                            p3 = _downsample_like(stage1_patch, aux3.shape[2:], kind="probs")
                            loss = loss + (ds_aux3_w * stage2_corrector_weight * _stage2_corrector_ohem_bce(
                                aux3,
                                m3,
                                p3,
                                fusion=stage2_corrector_fusion,
                                stage1_logit_eps=stage2_corrector_stage1_logit_eps,
                                pos_lambda=stage2_corrector_pos_lambda,
                                neg_lambda=stage2_corrector_neg_lambda,
                                gamma=stage2_corrector_gamma,
                                neg_fraction=stage2_corrector_neg_fraction,
                                min_neg=stage2_corrector_min_neg,
                            ))
                else:
                    logits = out
                    logits_main = logits
                    if stage1_patch is not None and stage2_corrector_fusion == "residual":
                        p1c = torch.clamp(stage1_patch, stage2_corrector_stage1_logit_eps, 1.0 - stage2_corrector_stage1_logit_eps)
                        logits_main = torch.log(p1c / (1.0 - p1c)) + logits
                    loss = criterion(logits_main, patches_mask)

                    if (
                        stage2_corrector_loss
                        and stage2_corrector_weight > 0.0
                        and stage1_patch is not None
                        and stage1_probs_dir_train is not None
                    ):
                        loss = loss + (stage2_corrector_weight * _stage2_corrector_ohem_bce(
                            logits,
                            patches_mask,
                            stage1_patch,
                            fusion=stage2_corrector_fusion,
                            stage1_logit_eps=stage2_corrector_stage1_logit_eps,
                            pos_lambda=stage2_corrector_pos_lambda,
                            neg_lambda=stage2_corrector_neg_lambda,
                            gamma=stage2_corrector_gamma,
                            neg_fraction=stage2_corrector_neg_fraction,
                            min_neg=stage2_corrector_min_neg,
                        ))

                neg_w = _neg_alarm_weight_for_epoch(epoch, neg_alarm_weight, neg_alarm_warmup_epochs)
                if neg_w > 0.0:
                    # identify empty target patches in the batch
                    per_patch_sum = patches_mask.sum(dim=(1, 2, 3, 4))
                    empty = per_patch_sum <= 0.0

                    if neg_alarm_only_on_negative_cases:
                        # keep only patches coming from negative cases
                        case_has_fg = torch.tensor(patches_case_has_fg, dtype=torch.bool, device=patches_mask.device)
                        empty = empty & (~case_has_fg)

                    if bool(empty.any()):
                        logits_empty = logits[empty]
                        flat = logits_empty.reshape(logits_empty.size(0), -1)
                        k = int(min(neg_alarm_topk, flat.size(1)))
                        topk_logits = torch.topk(flat, k=k, dim=1).values
                        alarm_loss = F.softplus(topk_logits).mean()
                        loss = loss + (neg_w * alarm_loss)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
            meter.update(loss.item(), patches_img.size(0))

            if max_steps_per_epoch is not None and step_i >= max_steps_per_epoch:
                break

        if total_patches > 0:
            pos_patch_frac = float(pos_patches) / float(total_patches)
            if epoch == 1:
                mask_sums_arr = np.array(mask_sums)
                print(
                    "[sanity] epoch1 mask.sum stats min/med/max: "
                    f"{mask_sums_arr.min():.2f} / {np.median(mask_sums_arr):.2f} / {mask_sums_arr.max():.2f}; "
                    f"pos_patch_frac: {pos_patch_frac:.3f} ({pos_patches}/{total_patches}); "
                    f"force_fg_calls: {fg_forced}; force_fg_prob: {force_fg_prob:.3f}; "
                    f"hard_bg_prob: {hard_bg_prob:.3f}; hard_bg_trials: {hard_bg_trials}; "
                    f"bg_inside_prob: {bg_inside_prob:.3f}; bg_inside_trials: {bg_inside_trials}"
                , flush=True)
            else:
                print(
                    f"[train] epoch {epoch} pos_patch_frac: {pos_patch_frac:.3f} ({pos_patches}/{total_patches}); "
                    f"force_fg_calls: {fg_forced}; force_fg_prob: {force_fg_prob:.3f}"
                , flush=True)

        # simple val using center crop of first case to avoid long runtime
        model.eval()
        with torch.no_grad():
            dices = []
            dices_by_thr = {thr: [] for thr in (val_threshold_candidates or [])}
            for v_i, batch in enumerate(val_loader, start=1):
                sample = batch[0]
                case_id = str(sample.get("case_id", ""))
                meta = sample.get("meta", {}) or {}
                img_np = sample["image"].detach().cpu().numpy() if isinstance(sample["image"], torch.Tensor) else sample["image"]
                mask_np = sample["mask"].detach().cpu().numpy() if isinstance(sample["mask"], torch.Tensor) else sample["mask"]

                if resample_max_zoom_mm > 0.0:
                    img_np, mask_np = _maybe_resample_case(img_np, mask_np, meta, resample_max_zoom_mm)

                stage1_np = None
                if stage1_probs_dir_val is not None:
                    p1 = _load_stage1_probs(case_id, stage1_probs_dir_val)
                    if p1 is not None:
                        p1 = _align_probs_to_zyx(p1, (int(img_np.shape[1]), int(img_np.shape[2]), int(img_np.shape[3])))
                        stage1_np = p1
                        img_np = np.concatenate([img_np.astype(np.float32, copy=False), p1[None, ...]], axis=0)
                    else:
                        zeros = np.zeros((1, int(img_np.shape[1]), int(img_np.shape[2]), int(img_np.shape[3])), dtype=np.float32)
                        stage1_np = zeros[0]
                        img_np = np.concatenate([img_np.astype(np.float32, copy=False), zeros], axis=0)

                img = torch.from_numpy(img_np).float().to(device)
                mask = torch.from_numpy(mask_np).float().to(device)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(0)
                mask = (mask > 0.5).float()

                vol_shape = (int(img.shape[1]), int(img.shape[2]), int(img.shape[3]))
                patch_size_eff, val_stride_eff = _effective_patch_and_stride(
                    patch_size,
                    val_stride,
                    vol_shape=vol_shape,
                    multiple=8,
                )
                logits = sliding_window_predict(img, model, patch_size_eff, val_stride_eff)
                if stage1_np is not None and stage2_corrector_fusion == "residual":
                    p1 = torch.from_numpy(stage1_np[None, None].astype(np.float32, copy=False)).to(device)
                    p1c = torch.clamp(p1, stage2_corrector_stage1_logit_eps, 1.0 - stage2_corrector_stage1_logit_eps)
                    logits = torch.log(p1c / (1.0 - p1c)) + logits
                probs = torch.sigmoid(logits)

                probs_np = probs[0, 0].detach().float().cpu().numpy()
                mask_np = mask[0].detach().float().cpu().numpy()

                if val_threshold_auto:
                    for thr in val_threshold_candidates or []:
                        pred_np = (probs_np > float(thr)).astype(np.uint8)
                        if val_min_size > 0:
                            pred_np = remove_small_components(pred_np, val_min_size)
                        inter = float((pred_np * mask_np).sum())
                        den = float(pred_np.sum() + mask_np.sum() + 1e-6)
                        dice = float((2.0 * inter + 1e-6) / den)
                        dices_by_thr[thr].append(dice)
                else:
                    pred_np = (probs_np > float(val_threshold)).astype(np.uint8)
                    if val_min_size > 0:
                        pred_np = remove_small_components(pred_np, val_min_size)
                    inter = float((pred_np * mask_np).sum())
                    den = float(pred_np.sum() + mask_np.sum() + 1e-6)
                    dice = float((2.0 * inter + 1e-6) / den)
                    dices.append(dice)

                if max_val_steps is not None and v_i >= max_val_steps:
                    break

            if val_threshold_auto:
                mean_by_thr = {thr: (float(np.mean(v)) if v else 0.0) for thr, v in dices_by_thr.items()}
                val_threshold = max(mean_by_thr, key=mean_by_thr.get)
                mean_dice = float(mean_by_thr[val_threshold])
            else:
                mean_dice = float(np.mean(dices)) if dices else 0.0

        torch.save(model.state_dict(), out_dir / "last.pt")

        # Persist the validation threshold used for this epoch so evaluation can reuse it.
        (out_dir / "val_threshold_last.json").write_text(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "val_threshold": float(val_threshold),
                    "val_threshold_mode": "auto" if val_threshold_auto else "fixed",
                    "val_min_size": int(val_min_size),
                    "val_dice": float(mean_dice),
                },
                indent=2,
            )
        )

        if mean_dice >= best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), out_dir / "best.pt")
            (out_dir / "val_threshold_best.json").write_text(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "val_threshold": float(val_threshold),
                        "val_threshold_mode": "auto" if val_threshold_auto else "fixed",
                        "val_min_size": int(val_min_size),
                        "val_dice": float(mean_dice),
                    },
                    indent=2,
                )
            )
        if val_threshold_auto:
            print(f"epoch {epoch} loss {meter.avg:.4f} val_dice {mean_dice:.4f} val_thr(auto) {val_threshold:.2f}", flush=True)
        else:
            print(f"epoch {epoch} loss {meter.avg:.4f} val_dice {mean_dice:.4f} val_thr {val_threshold:.2f}", flush=True)
        scheduler.step()


if __name__ == "__main__":
    app()
