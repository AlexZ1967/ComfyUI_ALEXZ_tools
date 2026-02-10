"""Node implementation module: `nodes/image_scopes.py`."""
import json


import torch
import torch.nn.functional as F

from ..utils.utils import ensure_hwc


def _resize_width(img: torch.Tensor, width: int) -> torch.Tensor:
    """Internal helper: `_resize_width`."""
    h, w, _ = img.shape
    if w == width:
        return img
    out = F.interpolate(
        img.permute(2, 0, 1).unsqueeze(0),
        size=(h, width),
        mode="nearest",
    )
    return out.squeeze(0).permute(1, 2, 0)


def _build_waveform(img: torch.Tensor, mode: str, width: int, height: int, gain: float, log_scale: bool) -> torch.Tensor:
    """Internal helper: `_build_waveform`."""
    img = torch.clamp(ensure_hwc(img).float(), 0.0, 1.0)
    img = _resize_width(img, int(width))

    if mode == "parade":
        chans = [img[..., 0], img[..., 1], img[..., 2]]
        colors = img.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    else:
        luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        chans = [luma]
        colors = img.new_tensor([[1.0, 1.0, 1.0]])

    h_scope = int(height)
    w_scope = img.shape[1]
    canvas = torch.zeros((h_scope, w_scope, 3), dtype=img.dtype, device=img.device)
    levels = torch.clamp(torch.stack(chans, dim=-1) * (h_scope - 1), 0, h_scope - 1).long()

    for ci in range(len(chans)):
        bins = levels[..., ci]
        counts = torch.zeros((h_scope, w_scope), dtype=img.dtype, device=img.device)
        for x in range(w_scope):
            hist = torch.bincount(bins[:, x], minlength=h_scope).float()
            counts[:, x] = hist
        if log_scale:
            counts = torch.log1p(counts)
        counts = counts * float(gain)
        max_per_col = torch.clamp(counts.max(dim=0, keepdim=True).values, min=1e-6)
        counts = counts / max_per_col
        canvas[..., 0] += counts * colors[ci, 0]
        canvas[..., 1] += counts * colors[ci, 1]
        canvas[..., 2] += counts * colors[ci, 2]

    return torch.flip(torch.clamp(canvas, 0.0, 1.0), dims=[0])


def _hist_counts(flat_channel: torch.Tensor, bins: int, log_scale: bool) -> torch.Tensor:
    """Internal helper: `_hist_counts`."""
    hist = torch.histc(flat_channel, bins=bins, min=0.0, max=1.0)
    if log_scale:
        hist = torch.log1p(hist)
    max_val = torch.clamp(hist.max(), min=1e-6)
    return hist / max_val


def _draw_hist_curve(
    canvas: torch.Tensor,
    hist: torch.Tensor,
    color: torch.Tensor,
    x0: int,
    x1: int,
    fill: bool = True,
    blend: str = "add",
    thickness: int = 1,
) -> None:
    """Internal helper: `_draw_hist_curve`."""
    h, _, _ = canvas.shape
    width = max(1, x1 - x0)
    xs = torch.linspace(0, hist.numel() - 1, steps=width, device=hist.device)
    ys = hist[xs.long()]
    levels = (ys * (h - 1)).long()
    for i in range(width):
        x = x0 + i
        y = h - 1 - int(levels[i].item())
        if fill:
            y0, y1 = y, h
        else:
            t = max(1, int(thickness))
            y0, y1 = max(0, y - t + 1), min(h, y + t)

        if blend == "max":
            patch = canvas[y0:y1, x, :]
            canvas[y0:y1, x, :] = torch.maximum(
                patch, color.view(1, 3).expand_as(patch)
            )
        else:
            canvas[y0:y1, x, :] += color


def _build_histogram(img: torch.Tensor, mode: str, bins: int, width: int, height: int, log_scale: bool):
    """Internal helper: `_build_histogram`."""
    img = torch.clamp(ensure_hwc(img).float(), 0.0, 1.0)
    h, w = int(height), int(width)
    canvas = torch.zeros((h, w, 3), dtype=img.dtype, device=img.device)
    flat = img.reshape(-1, 3)

    if mode == "luma":
        luma = 0.2126 * flat[:, 0] + 0.7152 * flat[:, 1] + 0.0722 * flat[:, 2]
        hist = _hist_counts(luma, bins, log_scale)
        _draw_hist_curve(canvas, hist, img.new_tensor([1.0, 1.0, 1.0]), 0, w)
        info = {"mode": mode, "bins": int(bins), "peak": round(float(hist.max().item()), 4)}
    elif mode == "rgb_overlay":
        canvas[:] = 0.02
        h_r = _hist_counts(flat[:, 0], bins, log_scale)
        h_g = _hist_counts(flat[:, 1], bins, log_scale)
        h_b = _hist_counts(flat[:, 2], bins, log_scale)
        _draw_hist_curve(
            canvas,
            h_r,
            img.new_tensor([1.0, 0.0, 0.0]),
            0,
            w,
            fill=False,
            blend="max",
            thickness=2,
        )
        _draw_hist_curve(
            canvas,
            h_g,
            img.new_tensor([0.0, 1.0, 0.0]),
            0,
            w,
            fill=False,
            blend="max",
            thickness=2,
        )
        _draw_hist_curve(
            canvas,
            h_b,
            img.new_tensor([0.0, 0.0, 1.0]),
            0,
            w,
            fill=False,
            blend="max",
            thickness=2,
        )
        info = {
            "mode": mode,
            "bins": int(bins),
            "peak_r": round(float(h_r.max().item()), 4),
            "peak_g": round(float(h_g.max().item()), 4),
            "peak_b": round(float(h_b.max().item()), 4),
            "peak_bin_r": int(torch.argmax(h_r).item()),
            "peak_bin_g": int(torch.argmax(h_g).item()),
            "peak_bin_b": int(torch.argmax(h_b).item()),
            "channel_order": ["R", "G", "B"],
        }
    else:  # rgb_split
        part = max(1, w // 3)
        h_r = _hist_counts(flat[:, 0], bins, log_scale)
        h_g = _hist_counts(flat[:, 1], bins, log_scale)
        h_b = _hist_counts(flat[:, 2], bins, log_scale)
        _draw_hist_curve(canvas, h_r, img.new_tensor([1.0, 0.0, 0.0]), 0, part)
        _draw_hist_curve(canvas, h_g, img.new_tensor([0.0, 1.0, 0.0]), part, min(2 * part, w))
        _draw_hist_curve(canvas, h_b, img.new_tensor([0.0, 0.0, 1.0]), min(2 * part, w), w)
        info = {
            "mode": mode,
            "bins": int(bins),
            "peak_r": round(float(h_r.max().item()), 4),
            "peak_g": round(float(h_g.max().item()), 4),
            "peak_b": round(float(h_b.max().item()), 4),
        }

    return torch.clamp(canvas, 0.0, 1.0), info


class ImageWaveformScope:
    """ComfyUI node class: `ImageWaveformScope`."""
    @classmethod
    def INPUT_TYPES(cls):
        """Execute `INPUT_TYPES` routine."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Входная картинка для waveform scope."}),
                "mode": (["luma", "parade"], {"default": "parade", "tooltip": "Luma waveform или RGB parade."}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "tooltip": "Ширина scope. Больше ширина = медленнее расчёт."}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1024, "tooltip": "Высота scope. Больше высота = больше VRAM/CPU."}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Усиление яркости точек scope."}),
                "log_scale": ("BOOLEAN", {"default": True, "tooltip": "Логарифмическая шкала плотности. Чуть медленнее, но информативнее в тенях/хайлайтах."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("waveform",)
    FUNCTION = "build"
    CATEGORY = "image/analysis"

    def build(self, image, mode, width, height, gain, log_scale):
        """Execute `build` routine."""
        out = []
        for i in range(image.shape[0]):
            wf = _build_waveform(image[i], mode, int(width), int(height), float(gain), bool(log_scale))
            out.append(wf.cpu())
        return (torch.stack(out, dim=0),)


class ImageHistogramScope:
    """ComfyUI node class: `ImageHistogramScope`."""
    @classmethod
    def INPUT_TYPES(cls):
        """Execute `INPUT_TYPES` routine."""
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Входная картинка для histogram scope."}),
                "mode": (["rgb_overlay", "rgb_split", "luma"], {"default": "rgb_overlay", "tooltip": "Наложенные RGB, разделенные RGB или luma histogram."}),
                "bins": ("INT", {"default": 256, "min": 16, "max": 2048, "tooltip": "Количество бинов гистограммы. 64/128 быстрее, 256+ точнее."}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "tooltip": "Ширина scope. Больше ширина = медленнее."}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1024, "tooltip": "Высота scope. Больше высота = больше VRAM/CPU."}),
                "log_scale": ("BOOLEAN", {"default": False, "tooltip": "Логарифмическая шкала плотности. Немного медленнее."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("histogram", "hist_json")
    FUNCTION = "build"
    CATEGORY = "image/analysis"

    def build(self, image, mode, bins, width, height, log_scale):
        """Execute `build` routine."""
        images = []
        infos = []
        for i in range(image.shape[0]):
            hist_img, info = _build_histogram(
                image[i],
                mode,
                int(bins),
                int(width),
                int(height),
                bool(log_scale),
            )
            images.append(hist_img.cpu())
            infos.append(json.dumps(info, ensure_ascii=True))
        return (torch.stack(images, dim=0), infos)
