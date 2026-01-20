from .image_align import ImageAlignOverlayToBackground
from .image_prepare import ImagePrepareForQwenEditOutpaint
from .json_output import JsonDisplayAndSave
from .video_inpaint import VideoInpaintWatermark


NODE_CLASS_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": ImagePrepareForQwenEditOutpaint,
    "ImageAlignOverlayToBackground": ImageAlignOverlayToBackground,
    "JsonDisplayAndSave": JsonDisplayAndSave,
    "VideoInpaintWatermark": VideoInpaintWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": "Image Prepare for QwenEdit Outpaint",
    "ImageAlignOverlayToBackground": "Align Overlay To Background",
    "JsonDisplayAndSave": "Show/Save JSON",
    "VideoInpaintWatermark": "Video Inpaint (ProPainter/E2FGVI)",
}
