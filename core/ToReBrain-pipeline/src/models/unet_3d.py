import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks_unet import conv_block_3d


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_ch: int = 16,
        mps_upsample_mode: str = "cpu_trilinear",
        norm: str = "batch",
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.mps_upsample_mode = str(mps_upsample_mode).strip().lower()
        self.norm = str(norm).strip().lower()
        self.deep_supervision = bool(deep_supervision)
        self.enc1 = conv_block_3d(in_channels, base_ch, norm=self.norm)
        self.down1 = nn.Conv3d(base_ch, base_ch, kernel_size=2, stride=2)
        self.enc2 = conv_block_3d(base_ch, base_ch * 2, norm=self.norm)
        self.down2 = nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.enc3 = conv_block_3d(base_ch * 2, base_ch * 4, norm=self.norm)
        self.down3 = nn.Conv3d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)

        self.bottleneck = conv_block_3d(base_ch * 4, base_ch * 8, norm=self.norm)

        # MPS does not support ConvTranspose3d or trilinear upsample; use 1x1 conv + conditional upsample.
        self.up3_conv = nn.Conv3d(base_ch * 8, base_ch * 4, kernel_size=1)
        self.dec3 = conv_block_3d(base_ch * 8, base_ch * 4, norm=self.norm)
        self.up2_conv = nn.Conv3d(base_ch * 4, base_ch * 2, kernel_size=1)
        self.dec2 = conv_block_3d(base_ch * 4, base_ch * 2, norm=self.norm)
        self.up1_conv = nn.Conv3d(base_ch * 2, base_ch, kernel_size=1)
        self.dec1 = conv_block_3d(base_ch * 2, base_ch, norm=self.norm)

        if self.deep_supervision:
            self.aux2_conv = nn.Conv3d(base_ch * 2, out_channels, 1)
            self.aux3_conv = nn.Conv3d(base_ch * 4, out_channels, 1)

        self.out_conv = nn.Conv3d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self._upsample(self.up3_conv(b), scale_factor=2)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = self._align_to(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        aux3 = self.aux3_conv(d3) if self.deep_supervision else None

        d2 = self._upsample(self.up2_conv(d3), scale_factor=2)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = self._align_to(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        aux2 = self.aux2_conv(d2) if self.deep_supervision else None

        d1 = self._upsample(self.up1_conv(d2), scale_factor=2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = self._align_to(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        if self.deep_supervision:
            return out, aux2, aux3
        return out

    @staticmethod
    def _align_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # Center crop or pad to match `ref` spatial dimensions without interpolation.
        rd, rh, rw = ref.shape[2:]
        diff_d, diff_h, diff_w = rd - x.size(2), rh - x.size(3), rw - x.size(4)

        pad = [
            max(diff_w, 0) // 2 + max(diff_w, 0) % 2,
            max(diff_w, 0) // 2,
            max(diff_h, 0) // 2 + max(diff_h, 0) % 2,
            max(diff_h, 0) // 2,
            max(diff_d, 0) // 2 + max(diff_d, 0) % 2,
            max(diff_d, 0) // 2,
        ]
        if any(pad):
            x = F.pad(x, pad)

        d, h, w = x.shape[2:]
        start_d = max((d - rd) // 2, 0)
        start_h = max((h - rh) // 2, 0)
        start_w = max((w - rw) // 2, 0)
        return x[:, :, start_d : start_d + rd, start_h : start_h + rh, start_w : start_w + rw]

    def _upsample(self, x: torch.Tensor, scale_factor: int) -> torch.Tensor:
        # NOTE: MPS may not support 3D interpolate ops. Historically this model used CPU trilinear
        # (x.cpu() -> interpolate -> to(mps)), and existing checkpoints were trained under that behavior.
        if x.device.type == "mps":
            mode = self.mps_upsample_mode
            sf = int(scale_factor)
            if sf <= 0:
                raise ValueError(f"scale_factor must be positive, got: {scale_factor}")

            if mode in {"cpu_trilinear", "trilinear_cpu", "legacy"}:
                return F.interpolate(x.cpu(), scale_factor=sf, mode="trilinear", align_corners=False).to(x.device)

            if mode in {"repeat_nearest", "nearest_repeat", "repeat"}:
                x = x.repeat_interleave(sf, dim=2)
                x = x.repeat_interleave(sf, dim=3)
                x = x.repeat_interleave(sf, dim=4)
                return x

            raise ValueError(
                f"Unknown mps_upsample_mode={self.mps_upsample_mode!r}. Expected: cpu_trilinear | repeat_nearest"
            )

        return F.interpolate(x, scale_factor=scale_factor, mode="trilinear", align_corners=False)
