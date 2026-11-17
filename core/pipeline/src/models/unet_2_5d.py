import torch
import torch.nn as nn
from .blocks_unet import conv_block


class UNet2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = conv_block(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
