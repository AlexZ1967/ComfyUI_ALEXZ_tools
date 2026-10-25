"""Display one middle frame while preserving the full IMAGE pass-through contract."""

from .video_silent_film_finish import _validate_video_batch


class VideoPreviewFrame:
    """Prevent a bypassed preview connection from rendering a whole video gallery."""

    @classmethod
    def INPUT_TYPES(cls):
        """Match the installed core PreviewImage input and metadata contract."""
        return {"required": {"images": ("IMAGE", {"tooltip": "Показывает один средний кадр входного батча. Если предыдущий этап в bypass, это кадр предыдущей работающей обработки. Выход images передаёт весь входной батч дальше без изменения."})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "video/preview"

    def save_images(self, images, prompt=None, extra_pnginfo=None):
        """Use core preview encoding on one frame and forward the original video unchanged."""
        from nodes import PreviewImage
        _validate_video_batch(images, normalize=False)
        middle = len(images) // 2
        result = PreviewImage().save_images(images[middle:middle + 1], filename_prefix="ALEXZ_Middle", prompt=prompt, extra_pnginfo=extra_pnginfo)
        result["result"] = (images,)
        return result
