import numpy as np
import torch

from src.inference.infer_sliding_window import sliding_window_inference_3d


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,D,H,W) -> logits (B,1,D,H,W)
        b, _, d, h, w = x.shape
        return torch.zeros((b, 1, d, h, w), dtype=x.dtype, device=x.device)


def test_sliding_window_output_shape_matches_input():
    vol = np.random.randn(3, 37, 41, 29).astype(np.float32)  # (C,D,H,W)
    model = _DummyModel().eval()
    out = sliding_window_inference_3d(
        volume=vol,
        model=model,
        patch_size=(64, 64, 48),
        overlap=0.5,
        device=torch.device("cpu"),
        aggregate="logits",
    )
    assert out.shape == (1, 1, 37, 41, 29)


def test_sliding_window_handles_non_divisible_edges():
    vol = np.random.randn(1, 17, 19, 23).astype(np.float32)
    model = _DummyModel().eval()
    out = sliding_window_inference_3d(
        volume=vol,
        model=model,
        patch_size=(16, 16, 16),
        overlap=0.25,
        device=torch.device("cpu"),
        aggregate="logits",
    )
    assert out.shape == (1, 1, 17, 19, 23)
