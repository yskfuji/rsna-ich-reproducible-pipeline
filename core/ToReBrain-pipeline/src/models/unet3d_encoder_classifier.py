from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .blocks_unet import conv_block_3d


class UNet3DEncoderClassifier(nn.Module):
    """UNet3D encoder-only classifier.

    - Reuses the exact encoder/bottleneck module naming from `UNet3D` so that
      ISLES checkpoints (`best.pt` which stores `model.state_dict()`) can be loaded
      directly (or with minimal adaptation).
    - Uses multi-scale pooled features (e3 + bottleneck) to improve stability.

    Input:  (N, C, D, H, W)
    Output: (N, num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 6,
        base_ch: int = 16,
        norm: str = "batch",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = str(norm).strip().lower()

        self.enc1 = conv_block_3d(in_channels, base_ch, norm=self.norm)
        # RSNA slice-level input uses D=1; do not downsample depth.
        self.down1 = nn.Conv3d(base_ch, base_ch, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2 = conv_block_3d(base_ch, base_ch * 2, norm=self.norm)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc3 = conv_block_3d(base_ch * 2, base_ch * 4, norm=self.norm)
        self.down3 = nn.Conv3d(base_ch * 4, base_ch * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.bottleneck = conv_block_3d(base_ch * 4, base_ch * 8, norm=self.norm)

        feat_ch = base_ch * 4 + base_ch * 8
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.head = nn.Linear(feat_ch, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        # Global average pooling over (D,H,W)
        e3p = e3.mean(dim=(2, 3, 4))
        bp = b.mean(dim=(2, 3, 4))
        feat = torch.cat([e3p, bp], dim=1)
        feat = self.dropout(feat)
        return self.head(feat)


def load_isles_unet_weights_into_classifier(
    model: UNet3DEncoderClassifier,
    ckpt_path: str,
    device: torch.device | str = "cpu",
    allow_input_channel_adapt: bool = True,
) -> tuple[list[str], list[str]]:
    """Load an ISLES `UNet3D` checkpoint (`best.pt` = state_dict) into this classifier.

    Returns (missing_keys, unexpected_keys).
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    state: Any = torch.load(str(ckpt_path), map_location=dev)
    sd_any: Any = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(sd_any, Mapping):
        raise TypeError(f"Unsupported checkpoint format: {type(sd_any)}")
    sd: Mapping[str, Any] = sd_any

    if allow_input_channel_adapt:
        w = sd.get("enc1.0.weight")
        if isinstance(w, torch.Tensor) and w.ndim == 5:
            in_ch_ckpt = int(w.shape[1])
            in_ch_cur = int(model.enc1[0].weight.shape[1])
            if in_ch_ckpt != in_ch_cur:
                w_new = torch.zeros(
                    (
                        int(w.shape[0]),
                        in_ch_cur,
                        int(w.shape[2]),
                        int(w.shape[3]),
                        int(w.shape[4]),
                    ),
                    dtype=w.dtype,
                )
                cmin = int(min(in_ch_ckpt, in_ch_cur))
                w_new[:, :cmin] = w[:, :cmin].detach().cpu()
                if in_ch_cur > in_ch_ckpt:
                    mean = w[:, :cmin].detach().cpu().mean(dim=1, keepdim=True)
                    w_new[:, cmin:] = mean.repeat(1, int(in_ch_cur - cmin), 1, 1, 1)
                sd_mut = dict(sd)
                sd_mut["enc1.0.weight"] = w_new
                sd = sd_mut

    # Filter out keys with shape mismatch (PyTorch raises even with strict=False).
    cur_sd = model.state_dict()
    filtered: dict[str, Any] = {}
    for k, v in sd.items():
        if k not in cur_sd:
            continue
        cur_v = cur_sd[k]
        if isinstance(v, torch.Tensor) and isinstance(cur_v, torch.Tensor) and v.shape == cur_v.shape:
            filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return list(missing), list(unexpected)
