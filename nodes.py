from .image_align import ImageAlignOverlayToBackground
from .image_prepare import ImagePrepareForQwenEditOutpaint
from .json_output import JsonDisplayAndSave, JsonPreview


NODE_CLASS_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": ImagePrepareForQwenEditOutpaint,
    "ImageAlignOverlayToBackground": ImageAlignOverlayToBackground,
    "JsonDisplayAndSave": JsonDisplayAndSave,
    "JsonPreview": JsonPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePrepare_for_QwenEdit_outpaint": "Image Prepare for QwenEdit Outpaint",
    "ImageAlignOverlayToBackground": "Align Overlay To Background",
    "JsonDisplayAndSave": "Show/Save JSON",
    "JsonPreview": "Show JSON",
}
