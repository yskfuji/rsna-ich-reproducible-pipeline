"""Simple sliding window inference for 3D volumes."""
from typing import List, Tuple, Literal
import numpy as np
import torch

def sliding_window_inference_3d(
    volume: np.ndarray,
    model,
    patch_size: Tuple[int, int, int],
    overlap: float,
    device: torch.device,
    aggregate: Literal["probs", "logits"] = "probs",
):
    model.eval()
    C, D, H, W = volume.shape
    ps_d, ps_h, ps_w = patch_size
    stride_d = int(ps_d * (1 - overlap))
    stride_h = int(ps_h * (1 - overlap))
    stride_w = int(ps_w * (1 - overlap))
    out = np.zeros((1, 1, D, H, W), dtype=np.float32)
    count = np.zeros((1, 1, D, H, W), dtype=np.float32)

    def positions(size: int, patch: int, stride: int) -> List[int]:
        # Ensure full coverage, including tail regions smaller than the patch.
        if size <= patch:
            return [0]
        stride = max(stride, 1)
        pos = list(range(0, size - patch + 1, stride))
        if pos[-1] != size - patch:
            pos.append(size - patch)
        return pos

    for z in positions(D, ps_d, stride_d):
        z_end = min(z + ps_d, D)
        for y in positions(H, ps_h, stride_h):
            y_end = min(y + ps_h, H)
            for x in positions(W, ps_w, stride_w):
                x_end = min(x + ps_w, W)
                patch = volume[:, z:z_end, y:y_end, x:x_end]
                pad_z, pad_y, pad_x = ps_d - patch.shape[1], ps_h - patch.shape[2], ps_w - patch.shape[3]
                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    patch = np.pad(patch, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
                # torch.from_numpy does not support arrays with negative strides (e.g. from views/rotations).
                patch = np.ascontiguousarray(patch)
                patch_t = torch.from_numpy(patch[None]).float().to(device)
                with torch.no_grad():
                    out_t = model(patch_t)
                    if isinstance(out_t, (tuple, list)):
                        out_t = out_t[0]
                    logits = out_t.float().cpu().numpy()
                if aggregate == "logits":
                    tile = logits
                else:
                    tile = 1.0 / (1.0 + np.exp(-logits))
                out[:, :, z:z_end, y:y_end, x:x_end] += tile[:, :, : z_end - z, : y_end - y, : x_end - x]
                count[:, :, z:z_end, y:y_end, x:x_end] += 1

    count[count == 0] = 1
    out = out / count
    return out
