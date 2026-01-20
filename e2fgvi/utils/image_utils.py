import numpy as np
import torch


def convert_image_to_frames(images: torch.Tensor) -> list[np.ndarray]:
    frames = []
    for image in images:
        np_frame = image.detach().cpu().numpy()
        if np_frame.max() <= 1.0:
            np_frame = (np_frame * 255).clip(0, 255).astype(np.uint8)
        else:
            np_frame = np_frame.clip(0, 255).astype(np.uint8)
        frames.append(np_frame)
    return frames


def convert_mask_to_frames(masks: torch.Tensor) -> list[np.ndarray]:
    frames = []
    for mask in masks:
        mask = mask.detach().cpu()
        if mask.dtype != torch.uint8:
            mask = (mask * 255).clamp(0, 255).byte()
        np_mask = mask.numpy()
        if np_mask.ndim == 3:
            np_mask = np_mask.squeeze()
        frames.append(np_mask)
    return frames


def resize_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    width, height = size
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in frames]


def resize_masks(masks: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    width, height = size
    return [cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) for mask in masks]


def dilate_masks(masks: list[np.ndarray], iterations: int) -> list[np.ndarray]:
    if iterations <= 0:
        return masks
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return [cv2.dilate(mask, kernel, iterations=iterations) for mask in masks]


def prepare_tensors(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray], int, int]:
    if len(masks) == 1:
        masks = masks * len(frames)

    binary_masks = [np.expand_dims((mask > 0).astype(np.uint8), 2) for mask in masks]
    masks_stack = np.stack(binary_masks, axis=0).astype(np.float32)
    masks_tensor = torch.from_numpy(masks_stack).unsqueeze(1)

    frames_stack = np.stack(frames, axis=0).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_stack).permute(0, 3, 1, 2)
    frames_tensor = frames_tensor.unsqueeze(0) * 2 - 1

    return (
        frames_tensor.to(device),
        masks_tensor.unsqueeze(0).to(device),
        binary_masks,
        frames[0].shape[0],
        frames[0].shape[1],
    )
