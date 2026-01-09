import numpy as np
import torch
try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None

from .matcher import detect_and_match, estimate_affine
from .utils import mask_to_uint8, select_batch_item, to_numpy_uint8, to_torch_image


def align_overlay_to_background(
    background,
    overlay,
    background_mask,
    overlay_mask,
    feature_count,
    good_match_percent,
    ransac_thresh,
    opacity,
    matcher_type,
    logger,
):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for this node. Please install opencv-python.")

    batch_size = max(background.shape[0], overlay.shape[0])
    aligned_results = []
    composite_results = []

    for index in range(batch_size):
        bg_tensor = select_batch_item(background, index)
        ov_tensor = select_batch_item(overlay, index)
        bg_mask_tensor = select_batch_item(background_mask, index) if background_mask is not None else None
        ov_mask_tensor = select_batch_item(overlay_mask, index) if overlay_mask is not None else None

        bg_np = to_numpy_uint8(bg_tensor)
        ov_np = to_numpy_uint8(ov_tensor)
        bg_mask_np = mask_to_uint8(bg_mask_tensor, bg_np.shape[:2])
        ov_mask_np = mask_to_uint8(ov_mask_tensor, ov_np.shape[:2])

        aligned_np, status = _align_overlay_to_background(
            bg_np,
            ov_np,
            feature_count,
            good_match_percent,
            ransac_thresh,
            bg_mask_np,
            ov_mask_np,
        )
        if aligned_np is None:
            logger.warning("Alignment failed: %s", status)
            aligned_np = cv2.resize(ov_np, (bg_np.shape[1], bg_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        if aligned_np.shape[2] == 4:
            alpha = aligned_np[:, :, 3:4].astype(np.float32) / 255.0
            aligned_rgb_uint8 = aligned_np[:, :, :3]
            aligned_rgb = aligned_rgb_uint8.astype(np.float32)
            bg_rgb = bg_np[:, :, :3].astype(np.float32)
            composite_np = aligned_rgb * alpha + bg_rgb * (1.0 - alpha)
        else:
            aligned_rgb_uint8 = aligned_np[:, :, :3]
            aligned_rgb = aligned_rgb_uint8.astype(np.float32)
            bg_rgb = bg_np[:, :, :3].astype(np.float32)
            composite_np = aligned_rgb * opacity + bg_rgb * (1.0 - opacity)

        aligned_results.append(to_torch_image(aligned_rgb_uint8))
        composite_results.append(to_torch_image(np.clip(composite_np, 0, 255).astype(np.uint8)))

    return (torch.stack(aligned_results, dim=0), torch.stack(composite_results, dim=0))


def _align_overlay_to_background(
    background_np,
    overlay_np,
    feature_count,
    good_match_percent,
    ransac_thresh,
    bg_mask_np,
    ov_mask_np,
):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for feature alignment. Please install opencv-python.")

    ov_points, bg_points, status = detect_and_match(
        background_np,
        overlay_np,
        bg_mask_np,
        ov_mask_np,
        feature_count,
        good_match_percent,
        matcher_type,
    )
    if ov_points is None:
        return None, status

    matrix, status = estimate_affine(ov_points, bg_points, ransac_thresh)
    if matrix is None:
        return None, status

    height, width = background_np.shape[:2]
    aligned = cv2.warpAffine(
        overlay_np,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned, "ok"
