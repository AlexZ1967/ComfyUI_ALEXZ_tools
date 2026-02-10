"""Utility/support module: `utils/utils.py`."""
import numpy as np

import torch
import torch.nn.functional as F
from . import color_match_utils
try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None


def to_numpy_uint8(image_tensor):
    """Convert image tensor to uint8 NumPy array in HWC layout."""
    image = image_tensor.detach().cpu().clamp(0, 1).numpy()
    return (image * 255.0).round().astype(np.uint8)


def to_torch_image(image_np):
    """Convert NumPy image array to float torch tensor in HWC layout."""
    return torch.from_numpy(image_np.astype(np.float32) / 255.0)


def select_batch_item(batch, index):
    """Return selected item from batched tensor input with bounds checks."""
    return batch[min(index, batch.shape[0] - 1)]


def mask_to_uint8(mask_tensor, target_hw):
    """Convert float mask tensor to uint8 mask image."""
    if mask_tensor is None:
        return None
    mask_np = mask_tensor.detach().cpu().clamp(0, 1).numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    mask_np = (mask_np * 255.0).round().astype(np.uint8)
    if mask_np.shape != target_hw:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for mask resizing.")
        mask_np = cv2.resize(mask_np, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask_np


def round_to_multiple(value, multiple):
    """Round value up or down to the nearest multiple."""
    return max(multiple, int(round(value / multiple)) * multiple)


def ensure_hwc(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor layout to HWC, dropping batch if present."""
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.shape[0] == 3 and t.shape[-1] != 3:
        t = t.permute(1, 2, 0)
    return t


def resize_to_hw(image: torch.Tensor, target_hw):
    """Resize HWC tensor to (H, W) with bilinear interpolation."""
    h, w = target_hw
    if image.shape[0] == h and image.shape[1] == w:
        return image
    return (
        F.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )


def image_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute absolute difference between two images (HWC or BCHW/BHWC)."""
    a = torch.clamp(ensure_hwc(a).float(), 0.0, 1.0)
    b = torch.clamp(ensure_hwc(b).float(), 0.0, 1.0)

    if a.shape[:2] != b.shape[:2]:
        area_a = a.shape[0] * a.shape[1]
        area_b = b.shape[0] * b.shape[1]
        if area_a >= area_b:
            b = resize_to_hw(b, a.shape[:2])
        else:
            a = resize_to_hw(a, b.shape[:2])

    return torch.abs(a - b)


def normalize_to_reference(image: torch.Tensor, reference: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize image to reference using color match modes."""
    if mode == "none":
        return image
    supported = {"mean_std", "linear", "hist", "pca_cov", "lab_l", "lab_l_cdf", "lab_full", "lab_cdf"}
    if mode not in supported:
        raise ValueError(f"Unsupported normalize mode: {mode}")

    img = torch.clamp(ensure_hwc(image).float(), 0.0, 1.0)
    ref = torch.clamp(ensure_hwc(reference).float(), 0.0, 1.0)
    mask = torch.zeros((ref.shape[0], ref.shape[1]), dtype=ref.dtype, device=ref.device)
    out = color_match_utils.apply_color_match(img.unsqueeze(0), ref.unsqueeze(0), mask, mode)
    return out[0].to(image.device)
