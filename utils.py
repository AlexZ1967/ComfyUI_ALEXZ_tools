import numpy as np
import torch
try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None


def to_numpy_uint8(image_tensor):
    image = image_tensor.detach().cpu().clamp(0, 1).numpy()
    return (image * 255.0).round().astype(np.uint8)


def to_torch_image(image_np):
    return torch.from_numpy(image_np.astype(np.float32) / 255.0)


def select_batch_item(batch, index):
    return batch[min(index, batch.shape[0] - 1)]


def mask_to_uint8(mask_tensor, target_hw):
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
    return max(multiple, int(round(value / multiple)) * multiple)
