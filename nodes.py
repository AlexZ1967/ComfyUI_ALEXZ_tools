from .image_align import ImageAlignOverlayToBackground
from .image_prepare import ImagePrepareForQwenEditOutpaint


NODE_CLASS_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": ImagePrepareForQwenEditOutpaint,
    "ImageAlignOverlayToBackground": ImageAlignOverlayToBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": "Image Prepare for QwenEdit Outpaint",
    "ImageAlignOverlayToBackground": "Align Overlay To Background",
}
