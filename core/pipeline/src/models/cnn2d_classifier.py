from __future__ import annotations

import torch
from torch import nn


def _norm2d(kind: str, num_features: int) -> nn.Module:
    k = str(kind).strip().lower()
    if k == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if k == "batch":
        return nn.BatchNorm2d(num_features)
    raise ValueError(f"Unknown norm: {kind}")


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            _norm2d(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            _norm2d(norm, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN2DClassifier(nn.Module):
    """Small 2D CNN baseline for RSNA multi-label classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_ch: int = 32,
        norm: str = "batch",
        dropout: float = 0.1,
    ):
        super().__init__()

        c1 = int(base_ch)
        c2 = int(base_ch) * 2
        c3 = int(base_ch) * 4
        c4 = int(base_ch) * 8

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, kernel_size=7, stride=2, padding=3, bias=False),
            _norm2d(norm, c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = _ConvBlock(c1, c1, norm=norm, stride=1, dropout=dropout)
        self.stage2 = _ConvBlock(c1, c2, norm=norm, stride=2, dropout=dropout)
        self.stage3 = _ConvBlock(c2, c3, norm=norm, stride=2, dropout=dropout)
        self.stage4 = _ConvBlock(c3, c4, norm=norm, stride=2, dropout=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, c4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(c4, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.head(x)
