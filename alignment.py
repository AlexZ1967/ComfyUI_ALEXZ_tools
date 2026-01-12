import json
import math
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
    scale_mode,
    allow_rotation,
    logger,
):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for this node. Please install opencv-python.")

    batch_size = max(background.shape[0], overlay.shape[0])
    aligned_results = []
    composite_results = []
    difference_results = []
    transform_results = []

    for index in range(batch_size):
        bg_tensor = select_batch_item(background, index)
        ov_tensor = select_batch_item(overlay, index)
        bg_mask_tensor = select_batch_item(background_mask, index) if background_mask is not None else None
        ov_mask_tensor = select_batch_item(overlay_mask, index) if overlay_mask is not None else None

        bg_np = to_numpy_uint8(bg_tensor)
        ov_np = to_numpy_uint8(ov_tensor)
        bg_mask_np = mask_to_uint8(bg_mask_tensor, bg_np.shape[:2])
        ov_mask_np = mask_to_uint8(ov_mask_tensor, ov_np.shape[:2])

        aligned_np, matrix, status = _align_overlay_to_background(
            bg_np,
            ov_np,
            feature_count,
            good_match_percent,
            ransac_thresh,
            bg_mask_np,
            ov_mask_np,
            matcher_type,
            scale_mode,
            allow_rotation,
        )
        if aligned_np is None:
            logger.warning("Alignment failed: %s", status)
            aligned_np = cv2.resize(ov_np, (bg_np.shape[1], bg_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            matrix = None

        if aligned_np.shape[2] == 4:
            alpha = aligned_np[:, :, 3:4].astype(np.float32) / 255.0
            if opacity < 1.0:
                alpha = np.clip(alpha * opacity, 0.0, 1.0)
            aligned_rgb_uint8 = aligned_np[:, :, :3]
            aligned_rgb = aligned_rgb_uint8.astype(np.float32)
            bg_rgb = bg_np[:, :, :3].astype(np.float32)
            composite_np = aligned_rgb * alpha + bg_rgb * (1.0 - alpha)
        else:
            aligned_rgb_uint8 = aligned_np[:, :, :3]
            aligned_rgb = aligned_rgb_uint8.astype(np.float32)
            bg_rgb = bg_np[:, :, :3].astype(np.float32)
            if matrix is not None:
                ov_alpha = np.ones(ov_np.shape[:2], dtype=np.uint8) * 255
                aligned_mask = cv2.warpAffine(
                    ov_alpha,
                    matrix,
                    (bg_np.shape[1], bg_np.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                alpha = (aligned_mask.astype(np.float32) / 255.0) * opacity
                alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
                composite_np = aligned_rgb * alpha + bg_rgb * (1.0 - alpha)
            else:
                composite_np = aligned_rgb * opacity + bg_rgb * (1.0 - opacity)

        aligned_results.append(to_torch_image(aligned_rgb_uint8))
        composite_results.append(to_torch_image(np.clip(composite_np, 0, 255).astype(np.uint8)))
        diff_np = np.abs(aligned_rgb - bg_rgb)
        difference_results.append(to_torch_image(np.clip(diff_np, 0, 255).astype(np.uint8)))
        transform_json = _format_transform_json(
            matrix,
            ov_np.shape[1],
            ov_np.shape[0],
            bg_np.shape[1],
            bg_np.shape[0],
        )
        transform_results.append(transform_json)
        logger.info("Transform JSON: %s", transform_json)

    return (
        torch.stack(aligned_results, dim=0),
        torch.stack(composite_results, dim=0),
        torch.stack(difference_results, dim=0),
        transform_results,
    )


def _align_overlay_to_background(
    background_np,
    overlay_np,
    feature_count,
    good_match_percent,
    ransac_thresh,
    bg_mask_np,
    ov_mask_np,
    matcher_type,
    scale_mode,
    allow_rotation,
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
        return None, None, status

    matrix, status = estimate_affine(ov_points, bg_points, ransac_thresh, scale_mode)
    if matrix is None:
        return None, None, status
    if not allow_rotation:
        matrix = _remove_rotation(matrix, overlay_np.shape[1], overlay_np.shape[0], scale_mode)

    height, width = background_np.shape[:2]
    aligned = cv2.warpAffine(
        overlay_np,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned, matrix, "ok"


def _remove_rotation(matrix, overlay_width, overlay_height, scale_mode):
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    if scale_mode == "preserve_aspect":
        scale_x = scale_y = (scale_x + scale_y) * 0.5

    center_x = overlay_width * 0.5
    center_y = overlay_height * 0.5
    pos_x = a * center_x + b * center_y + tx
    pos_y = c * center_x + d * center_y + ty

    tx = pos_x - scale_x * center_x
    ty = pos_y - scale_y * center_y
    return np.array([[scale_x, 0.0, tx], [0.0, scale_y, ty]], dtype=np.float32)


def _format_transform_json(matrix, overlay_width, overlay_height, background_width, background_height):
    if matrix is None:
        payload = {
            "overlay_scale": {"x": None, "y": None},
            "overlay_position": {"x": None, "y": None},
            "overlay_rotation_angle": None,
            "overlay_position_pixels": {"x": None, "y": None},
        }
        return json.dumps(payload, ensure_ascii=True)

    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    rotation_rad = math.atan2(c, a)
    rotation_deg = math.degrees(rotation_rad)

    center_x = overlay_width * 0.5
    center_y = overlay_height * 0.5
    pos_x = a * center_x + b * center_y + tx
    pos_y = c * center_x + d * center_y + ty
    bg_w = max(1.0, background_width - 1.0)
    bg_h = max(1.0, background_height - 1.0)
    norm_x = float(pos_x / bg_w)
    norm_y = float(1.0 - (pos_y / bg_h))
    resolve_rotation = -rotation_deg

    payload = {
        "overlay_scale": {"x": round(float(scale_x), 3), "y": round(float(scale_y), 3)},
        "overlay_rotation_angle": round(float(resolve_rotation), 3),
        "overlay_position_pixels": {"x": round(float(pos_x), 3), "y": round(float(pos_y), 3)},
        "fusion_position": {"x": round(norm_x, 6), "y": round(norm_y, 6)},
        "resolve_position_edit": _format_resolve_edit_position(
            norm_x,
            norm_y,
            background_width,
            background_height,
            overlay_width,
            overlay_height,
        ),
    }
    return json.dumps(payload, ensure_ascii=True)


def _format_resolve_edit_position(
    norm_x,
    norm_y,
    background_width,
    background_height,
    overlay_width,
    overlay_height,
):
    scale_x = (background_width * background_width) / max(1.0, float(overlay_width))
    scale_y = (background_height * background_height) / max(1.0, float(overlay_height))
    pos_x = (norm_x - 0.5) * scale_x
    pos_y = (norm_y - 0.5) * scale_y
    return {"x": round(float(pos_x), 3), "y": round(float(pos_y), 3)}
