"""Node implementation module: `nodes/qr_code_generate.py`."""

import numpy as np
import torch
from PIL import Image


def _error_correction(level: str) -> int:
    """Map UI error-correction level to qrcode constant."""
    import qrcode

    mapping = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    return mapping.get(level, qrcode.constants.ERROR_CORRECT_M)


class GenerateQRCode:
    """ComfyUI node class: `GenerateQRCode`."""

    @classmethod
    def INPUT_TYPES(cls):
        """Execute `INPUT_TYPES` routine."""
        return {
            "required": {
                "url": ("STRING", {"default": "https://example.com", "multiline": False, "tooltip": "Ссылка или текст для QR-кода."}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": "Итоговое разрешение квадратного QR-изображения."}),
                "error_correction": (["L", "M", "Q", "H"], {"default": "M", "tooltip": "Уровень коррекции ошибок: L(7%), M(15%), Q(25%), H(30%)."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/utils"

    def generate(self, url: str, resolution: int, error_correction: str):
        """Execute `generate` routine."""
        payload = (url or "").strip()
        if not payload:
            raise ValueError("`url` must not be empty.")

        try:
            import qrcode
        except ImportError as exc:
            raise RuntimeError("Package `qrcode` is required. Install with: pip install qrcode") from exc

        qr = qrcode.QRCode(
            version=None,
            error_correction=_error_correction(error_correction),
            box_size=10,
            border=4,
        )
        qr.add_data(payload)
        qr.make(fit=True)

        image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        nearest = getattr(Image, "Resampling", Image).NEAREST
        image = image.resize((int(resolution), int(resolution)), resample=nearest)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image_np).unsqueeze(0)
        return (image_t,)
