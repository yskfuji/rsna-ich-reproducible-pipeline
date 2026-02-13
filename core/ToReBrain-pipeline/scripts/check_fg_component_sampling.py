from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import label as cc_label

from src.datasets.isles_dataset import IslesVolumeDataset


def main() -> None:
    cfg_path = Path(
        "configs/train_3d_unet_smoke2_dwi_adc_flair_fp_ohem_balanced_fg033_ps562424_autoshrink_dist_target033_bgps404816_fgccinv_a1.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text())

    train_vol = IslesVolumeDataset(
        cfg["data"]["csv_path"],
        split="train",
        root=cfg["data"]["root"],
        transform=None,
        normalize=cfg["data"].get("normalize", "nonzero_zscore"),
        allow_missing_label=bool(cfg["data"].get("allow_missing_label", False)),
    )

    sample = None
    for i in range(len(train_vol)):
        item = train_vol[i]
        mask = item["mask"]
        m = (mask[0] > 0.5).astype(np.uint8)
        if m.sum() == 0:
            continue
        lbl, n = cc_label(m)
        if n >= 2:
            sample = (i, m, lbl, n)
            break

    if sample is None:
        raise SystemExit("No multi-component sample found")

    i, m, lbl, n = sample
    sizes = np.bincount(lbl.ravel())

    comp_ids = np.arange(1, n + 1)
    comp_sizes = sizes[comp_ids]

    vox_cc = lbl[m.astype(bool)]

    rng = np.random.default_rng(0)
    N = 20000

    # Uniform over voxels => component probability proportional to size
    choices_uniform = rng.integers(0, vox_cc.shape[0], size=N)
    cc_uniform = vox_cc[choices_uniform]

    # inverse_size(alpha=1): per-voxel weight 1/|CC| => total weight per component ~ 1
    alpha = 1.0
    w_comp = 1.0 / (comp_sizes.astype(np.float64) ** alpha)
    w_vox = w_comp[vox_cc - 1]
    p_vox = w_vox / w_vox.sum()
    cc_inv = rng.choice(vox_cc, size=N, replace=True, p=p_vox)

    cu = Counter(cc_uniform.tolist())
    ci = Counter(cc_inv.tolist())

    def summarize(counter: Counter[int]) -> tuple[float, float, float]:
        probs = np.array([counter.get(int(k), 0) / N for k in comp_ids], dtype=np.float64)
        return float(probs.min()), float(probs.mean()), float(probs.max())

    u_min, u_mean, u_max = summarize(cu)
    i_min, i_mean, i_max = summarize(ci)

    mean_size_u = float(np.mean([sizes[int(k)] for k in cc_uniform]))
    mean_size_i = float(np.mean([sizes[int(k)] for k in cc_inv]))

    print(
        "case_index=%d n_components=%d comp_size_min/med/max=%d/%.1f/%d"
        % (i, n, int(comp_sizes.min()), float(np.median(comp_sizes)), int(comp_sizes.max()))
    )
    print(
        "uniform_voxel: comp_prob min/mean/max = %.4f/%.4f/%.4f  mean_selected_cc_size=%.1f"
        % (u_min, u_mean, u_max, mean_size_u)
    )
    print(
        "inverse_size(a=1): comp_prob min/mean/max = %.4f/%.4f/%.4f  mean_selected_cc_size=%.1f"
        % (i_min, i_mean, i_max, mean_size_i)
    )


if __name__ == "__main__":
    main()
