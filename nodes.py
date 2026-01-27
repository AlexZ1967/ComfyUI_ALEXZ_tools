import importlib
import logging
import traceback

_LOGGER = logging.getLogger("ALEXZ_tools")

_NODE_SPECS = [
    ("ImagePrepare_for_QwenEdit_outpaint", "Image Prepare for QwenEdit Outpaint", ".image_prepare", "ImagePrepareForQwenEditOutpaint"),
    ("ImageAlignOverlayToBackground", "Align Overlay To Background", ".image_align", "ImageAlignOverlayToBackground"),
    ("JsonDisplayAndSave", "Show/Save JSON", ".json_output", "JsonDisplayAndSave"),
    ("VideoInpaintWatermark", "Remove Static Watermark from Video", ".video_inpaint", "VideoInpaintWatermark"),
    ("ImageColorMatchToReference", "Color Match To Reference", ".image_color_match", "ImageColorMatchToReference"),
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
LOAD_RESULTS = {"ok": [], "fail": []}
_LOG_LINES = []


def _load_node(name: str, display: str, module: str, attr: str):
    try:
        mod = importlib.import_module(module, __name__.rsplit(".", 1)[0])
        cls = getattr(mod, attr)
        NODE_CLASS_MAPPINGS[name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[name] = display
        LOAD_RESULTS["ok"].append(name)
        _LOG_LINES.append(f"✅ {display} loaded")
    except Exception as exc:  # pragma: no cover - diagnostic
        LOAD_RESULTS["fail"].append({"name": name, "reason": str(exc)})
        _LOG_LINES.append(f"❌ {display} failed: {exc}")
        _LOGGER.error("Failed to load node %s: %s\n%s", name, exc, traceback.format_exc())


for _name, _disp, _mod, _attr in _NODE_SPECS:
    _load_node(_name, _disp, _mod, _attr)

# Compact startup log
header = "ALEXZ_tools loading..."
_LOGGER.info(header)
for line in _LOG_LINES:
    _LOGGER.info(line)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "LOAD_RESULTS"]
