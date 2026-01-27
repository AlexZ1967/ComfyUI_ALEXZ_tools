import numpy as np
import torch
import torch.nn.functional as torch_nn_func

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    max_val = float(mask.max()) if mask.numel() else 0.0
    if max_val > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def ensure_mask_batch(mask: torch.Tensor, frame_count: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.size(0) == 1 and frame_count > 1:
        mask = mask.repeat(frame_count, 1, 1)
    return mask


def resize_mask_to_output(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[-2] == height and mask.shape[-1] == width:
        return mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(1)
    resized = torch_nn_func.interpolate(mask, size=(height, width), mode="nearest")
    return resized.squeeze(1)


def resize_images_to_size(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if images.shape[1] == height and images.shape[2] == width:
        return images
    images_bchw = images.permute(0, 3, 1, 2)
    resized = torch_nn_func.interpolate(
        images_bchw,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1)


def _match_mean_std_channel(src: np.ndarray, ref: np.ndarray, keep: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    src_keep = src[keep]
    ref_keep = ref[keep]
    if src_keep.size < 10 or ref_keep.size < 10:
        return src
    src_mean = float(src_keep.mean())
    ref_mean = float(ref_keep.mean())
    src_std = float(src_keep.std())
    ref_std = float(ref_keep.std())
    scale = ref_std / max(src_std, eps)
    return (src - src_mean) * scale + ref_mean


def _match_linear_channel(src: np.ndarray, ref: np.ndarray, keep: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    src_keep = src[keep]
    ref_keep = ref[keep]
    if src_keep.size < 10 or ref_keep.size < 10:
        return src
    mean_x = float(src_keep.mean())
    mean_y = float(ref_keep.mean())
    var_x = float(((src_keep - mean_x) ** 2).mean())
    if var_x < eps:
        a = 1.0
        b = mean_y - mean_x
    else:
        cov_xy = float(((src_keep - mean_x) * (ref_keep - mean_y)).mean())
        a = cov_xy / var_x
        b = mean_y - a * mean_x
    return src * a + b


def _match_histogram_channel(
    src: np.ndarray,
    ref: np.ndarray,
    keep: np.ndarray,
    bins: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    src_keep = src[keep]
    ref_keep = ref[keep]
    if src_keep.size < 10 or ref_keep.size < 10:
        return src
    src_hist, bin_edges = np.histogram(src_keep, bins=bins, range=value_range, density=True)
    ref_hist, _ = np.histogram(ref_keep, bins=bins, range=value_range, density=True)
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)
    if src_cdf[-1] > 0:
        src_cdf = src_cdf / src_cdf[-1]
    if ref_cdf[-1] > 0:
        ref_cdf = ref_cdf / ref_cdf[-1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    interp_values = np.interp(src_cdf, ref_cdf, bin_centers)
    indices = np.searchsorted(bin_edges, src.ravel(), side="right") - 1
    indices = np.clip(indices, 0, bins - 1)
    matched = interp_values[indices].reshape(src.shape)
    return matched


def _match_mean_std_rgb(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    for c in range(3):
        out_np[..., c] = _match_mean_std_channel(out_np[..., c], ref_np[..., c], keep)
    return out_np


def _match_linear_rgb(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    for c in range(3):
        out_np[..., c] = _match_linear_channel(out_np[..., c], ref_np[..., c], keep)
    return out_np


def _match_hist_rgb(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray) -> np.ndarray:
    for c in range(3):
        out_np[..., c] = _match_histogram_channel(
            out_np[..., c], ref_np[..., c], keep, bins=256, value_range=(0.0, 1.0)
        )
    return out_np


def _match_lab_l(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, use_cdf: bool) -> np.ndarray:
    if cv2 is None:
        return out_np

    out_lab = []
    ref_lab = []
    for idx in range(out_np.shape[0]):
        out_lab.append(cv2.cvtColor(out_np[idx], cv2.COLOR_RGB2LAB))
        ref_lab.append(cv2.cvtColor(ref_np[idx], cv2.COLOR_RGB2LAB))
    out_lab = np.stack(out_lab, axis=0)
    ref_lab = np.stack(ref_lab, axis=0)

    out_l = out_lab[..., 0]
    ref_l = ref_lab[..., 0]
    if use_cdf:
        out_l = _match_histogram_channel(out_l, ref_l, keep, bins=256, value_range=(0.0, 100.0))
    else:
        out_l = _match_mean_std_channel(out_l, ref_l, keep)
    out_lab[..., 0] = np.clip(out_l, 0.0, 100.0)

    out_rgb = []
    for idx in range(out_lab.shape[0]):
        out_rgb.append(cv2.cvtColor(out_lab[idx], cv2.COLOR_LAB2RGB))
    out_rgb = np.stack(out_rgb, axis=0)
    return np.clip(out_rgb, 0.0, 1.0)


def _match_lab_full(out_np: np.ndarray, ref_np: np.ndarray, keep: np.ndarray, use_cdf: bool) -> np.ndarray:
    if cv2 is None:
        return out_np

    out_lab = []
    ref_lab = []
    for idx in range(out_np.shape[0]):
        out_lab.append(cv2.cvtColor(out_np[idx], cv2.COLOR_RGB2LAB))
        ref_lab.append(cv2.cvtColor(ref_np[idx], cv2.COLOR_RGB2LAB))
    out_lab = np.stack(out_lab, axis=0)
    ref_lab = np.stack(ref_lab, axis=0)

    ranges = [(0.0, 100.0), (-127.0, 127.0), (-127.0, 127.0)]
    for ch in range(3):
        if use_cdf:
            out_lab[..., ch] = _match_histogram_channel(
                out_lab[..., ch], ref_lab[..., ch], keep, bins=256, value_range=ranges[ch]
            )
        else:
            out_lab[..., ch] = _match_mean_std_channel(
                out_lab[..., ch], ref_lab[..., ch], keep
            )

    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)
    out_lab[..., 1] = np.clip(out_lab[..., 1], -127.0, 127.0)
    out_lab[..., 2] = np.clip(out_lab[..., 2], -127.0, 127.0)

    out_rgb = []
    for idx in range(out_lab.shape[0]):
        out_rgb.append(cv2.cvtColor(out_lab[idx], cv2.COLOR_LAB2RGB))
    out_rgb = np.stack(out_rgb, axis=0)
    return np.clip(out_rgb, 0.0, 1.0)


def apply_color_match(
    output_images: torch.Tensor,
    reference_images: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "none":
        return output_images

    mask = normalize_mask(mask)
    mask = ensure_mask_batch(mask, output_images.size(dim=0))
    mask = resize_mask_to_output(mask, output_images.shape[1], output_images.shape[2])
    keep = (mask < 0.5).detach().cpu().numpy()
    if keep.sum() < 10:
        return output_images

    if reference_images.shape[1:3] != output_images.shape[1:3]:
        reference_images = resize_images_to_size(
            reference_images, output_images.shape[1], output_images.shape[2]
        )

    out_np = output_images.detach().cpu().numpy().astype(np.float32)
    ref_np = reference_images.detach().cpu().numpy().astype(np.float32)

    if mode == "mean_std":
        out_np = _match_mean_std_rgb(out_np, ref_np, keep)
    elif mode == "linear":
        out_np = _match_linear_rgb(out_np, ref_np, keep)
    elif mode == "hist":
        out_np = _match_hist_rgb(out_np, ref_np, keep)
    elif mode == "lab_l":
        out_np = _match_lab_l(out_np, ref_np, keep, use_cdf=False)
    elif mode == "lab_l_cdf":
        out_np = _match_lab_l(out_np, ref_np, keep, use_cdf=True)
    elif mode == "lab_full":
        out_np = _match_lab_full(out_np, ref_np, keep, use_cdf=False)
    elif mode == "lab_cdf":
        out_np = _match_lab_full(out_np, ref_np, keep, use_cdf=True)
    else:
        return output_images

    out_np = np.clip(out_np, 0.0, 1.0)
    return torch.from_numpy(out_np)
