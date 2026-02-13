import torch
import torch.nn as nn


def _pick_groupnorm_groups(num_channels: int) -> int:
    for g in (16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


def _norm2d(kind: str, ch: int) -> nn.Module:
    k = str(kind).strip().lower()
    if k in {"batch", "bn", "batchnorm"}:
        return nn.BatchNorm2d(ch)
    if k in {"instance", "in", "instancenorm"}:
        return nn.InstanceNorm2d(ch, affine=True, track_running_stats=True)
    if k in {"group", "gn", "groupnorm"}:
        return nn.GroupNorm(_pick_groupnorm_groups(ch), ch)
    raise ValueError(f"Unknown norm kind: {kind!r}")


def _norm3d(kind: str, ch: int) -> nn.Module:
    k = str(kind).strip().lower()
    if k in {"batch", "bn", "batchnorm"}:
        return nn.BatchNorm3d(ch)
    if k in {"instance", "in", "instancenorm"}:
        return nn.InstanceNorm3d(ch, affine=True, track_running_stats=True)
    if k in {"group", "gn", "groupnorm"}:
        return nn.GroupNorm(_pick_groupnorm_groups(ch), ch)
    raise ValueError(f"Unknown norm kind: {kind!r}")


def conv_block(in_ch: int, out_ch: int, norm: str = "batch") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        _norm2d(norm, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        _norm2d(norm, out_ch),
        nn.ReLU(inplace=True),
    )


def conv_block_3d(in_ch: int, out_ch: int, norm: str = "batch") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        _norm3d(norm, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        _norm3d(norm, out_ch),
        nn.ReLU(inplace=True),
    )
