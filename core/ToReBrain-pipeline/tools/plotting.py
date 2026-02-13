from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

# Headless-safe backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_curve_png(
    out_path: Path,
    xs: Iterable[float],
    ys: Iterable[float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array(list(xs), dtype=np.float32)
    y = np.array(list(ys), dtype=np.float32)
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def save_hist_png(
    out_path: Path,
    values: np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 80,
    logy: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v = values.astype(np.float64).ravel()
    v = v[np.isfinite(v)]
    plt.figure(figsize=(7, 4))
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def save_overlay_3plane(
    out_dir: Path,
    prob: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    base_img: np.ndarray | None = None,
    thr: float = 0.5,
) -> list[Path]:
    """Save simple 3-plane overlays.

    prob/pred/gt are (D,H,W). base_img optional (D,H,W).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    prob3 = prob
    pred3 = pred
    gt3 = gt
    if base_img is None:
        base_img = prob3

    # pick a slice: center of GT if available else center of volume
    if gt3.any():
        zyx = np.argwhere(gt3 > 0)
        cz, cy, cx = np.round(zyx.mean(axis=0)).astype(int).tolist()
    else:
        cz, cy, cx = [s // 2 for s in gt3.shape]

    planes = {
        "axial": (cz, base_img[cz], prob3[cz], pred3[cz], gt3[cz]),
        "coronal": (cy, base_img[:, cy, :], prob3[:, cy, :], pred3[:, cy, :], gt3[:, cy, :]),
        "sagittal": (cx, base_img[:, :, cx], prob3[:, :, cx], pred3[:, :, cx], gt3[:, :, cx]),
    }

    out_paths: list[Path] = []
    for name, (idx, img2d, prob2d, pred2d, gt2d) in planes.items():
        plt.figure(figsize=(7, 4))
        v = img2d.astype(np.float32)
        # robust normalize to [0,1]
        lo, hi = np.percentile(v[np.isfinite(v)], [1, 99]) if np.isfinite(v).any() else (0.0, 1.0)
        v = np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        plt.imshow(v, cmap="gray", interpolation="nearest")
        # prob heatmap
        p = np.clip(prob2d.astype(np.float32), 0.0, 1.0)
        plt.imshow(p, cmap="magma", alpha=0.35, interpolation="nearest")
        # contours
        if gt2d.any():
            plt.contour(gt2d.astype(np.float32), levels=[0.5], colors="lime", linewidths=1.0)
        if pred2d.any():
            plt.contour(pred2d.astype(np.float32), levels=[0.5], colors="cyan", linewidths=1.0)
        plt.title(f"{name} idx={idx} thr={thr}")
        plt.axis("off")
        plt.tight_layout()
        out_path = out_dir / f"overlay_{name}.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        out_paths.append(out_path)

    return out_paths
