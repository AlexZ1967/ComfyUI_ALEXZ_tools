"""Node registry for ComfyUI_ALEXZ_tools.

Imports node modules, builds `NODE_CLASS_MAPPINGS` and
`NODE_DISPLAY_NAME_MAPPINGS`, and prints compact load status lines.
"""

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
    ("VideoFrameMatch", "Find Closest Video Frame", ".video_frame_match", "VideoFrameMatch"),
    ("VideoCutMatch", "Match Video Cut Point", ".video_cut_match", "VideoCutMatch"),
    ("ImageDifference", "Image Difference", ".image_difference", "ImageDifference"),
    ("ImageWaveformScope", "Image Waveform Scope", ".image_scopes", "ImageWaveformScope"),
    ("ImageHistogramScope", "Image Histogram Scope", ".image_scopes", "ImageHistogramScope"),
    ("GenerateQRCode", "Generate QR Code", ".qr_code_generate", "GenerateQRCode"),
    ("ALEXZTestNode", "ALEXZ Test Node", ".test_node", "ALEXZTestNode"),
]

_NODE_UI_METADATA = {
    "ImagePrepare_for_QwenEdit_outpaint": {
        "description": "Подготавливает изображение и latent под QwenEdit Outpaint.",
        "output_tooltips": ["Подготовленное изображение.", "Пустой latent для KSampler."],
        "search_aliases": ["qwen", "outpaint", "prepare"],
    },
    "ImageAlignOverlayToBackground": {
        "description": "Выравнивает оверлей относительно фона по ключевым точкам.",
        "output_tooltips": ["Выровненный оверлей.", "Композит оверлея на фоне.", "Карта разницы.", "Параметры трансформации JSON."],
        "search_aliases": ["align", "overlay", "homography"],
    },
    "JsonDisplayAndSave": {
        "description": "Показывает и сохраняет JSON в читаемом виде.",
        "output_tooltips": ["Текст JSON для UI."],
        "search_aliases": ["json", "save", "display"],
    },
    "VideoInpaintWatermark": {
        "description": "Удаляет статический вотермарк/объект из видео (ProPainter).",
        "output_tooltips": ["Превью кадра с результатом.", "JSON параметров кропа/позиционирования."],
        "search_aliases": ["video", "inpaint", "watermark", "propainter"],
    },
    "ImageColorMatchToReference": {
        "description": "Подгоняет цвет и тон изображения под референс.",
        "output_tooltips": ["Цветокорректированное изображение.", "JSON параметров и метрик качества."],
        "search_aliases": ["color", "match", "reference", "grade"],
    },
    "VideoFrameMatch": {
        "description": "Ищет наиболее похожий кадр в видео для входной картинки.",
        "output_tooltips": ["Лучший найденный кадр.", "Номер лучшего кадра.", "JSON с оценками и top-k."],
        "search_aliases": ["video", "frame", "match", "closest"],
    },
    "VideoCutMatch": {
        "description": "Подбирает оптимальную пару кадров для склейки двух видео.",
        "output_tooltips": ["Лучший кадр из A.", "Лучший кадр из B.", "Номер кадра A.", "Номер кадра B.", "JSON с cut-point и top-k."],
        "search_aliases": ["video", "cut", "stitch", "match"],
    },
    "ImageDifference": {
        "description": "Строит абсолютную разницу двух изображений.",
        "output_tooltips": ["Изображение абсолютной разницы."],
        "search_aliases": ["difference", "diff", "compare"],
    },
    "ImageWaveformScope": {
        "description": "Строит waveform/parade scope для анализа яркости и каналов.",
        "output_tooltips": ["Waveform scope изображение."],
        "search_aliases": ["waveform", "scope", "analysis"],
    },
    "ImageHistogramScope": {
        "description": "Строит RGB/Luma гистограмму изображения.",
        "output_tooltips": ["Гистограмма как изображение.", "JSON статистики гистограммы."],
        "search_aliases": ["histogram", "scope", "analysis"],
    },
    "GenerateQRCode": {
        "description": "Генерирует QR-код из ссылки или текста.",
        "output_tooltips": ["Сгенерированный QR-код."],
        "search_aliases": ["qr", "qrcode", "link"],
    },
    "ALEXZTestNode": {
        "description": "Тестовая нода для проверки загрузки пакета и Module Nodes.",
        "output_tooltips": ["Текстовый тестовый выход.", "Числовой тестовый выход."],
        "search_aliases": ["test", "debug", "alexz"],
    },
}

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
LOAD_RESULTS = {"ok": [], "fail": []}
_LOG_LINES = []


def _load_node(name: str, display: str, module: str, attr: str):
    """Load one node class and store it in ComfyUI mappings."""
    try:
        mod = importlib.import_module(module, __name__)
        cls = getattr(mod, attr)
        NODE_CLASS_MAPPINGS[name] = cls
        NODE_DISPLAY_NAME_MAPPINGS[name] = display
        LOAD_RESULTS["ok"].append(name)
        _LOG_LINES.append(f"✅ {display} loaded")
    except Exception as exc:  # pragma: no cover - diagnostic
        LOAD_RESULTS["fail"].append({"name": name, "reason": str(exc)})
        _LOG_LINES.append(f"❌ {display} failed: {exc}")
        _LOGGER.error("Failed to load node %s: %s\n%s", name, exc, traceback.format_exc())


def _apply_node_ui_metadata() -> None:
    """Attach optional UI metadata used by newer ComfyUI node cards."""
    for node_name, node_cls in NODE_CLASS_MAPPINGS.items():
        meta = _NODE_UI_METADATA.get(node_name)
        if not meta:
            continue
        if not getattr(node_cls, "DESCRIPTION", ""):
            node_cls.DESCRIPTION = meta["description"]
        if not hasattr(node_cls, "OUTPUT_TOOLTIPS"):
            node_cls.OUTPUT_TOOLTIPS = list(meta["output_tooltips"])
        if not hasattr(node_cls, "SEARCH_ALIASES"):
            node_cls.SEARCH_ALIASES = list(meta["search_aliases"])


for _name, _disp, _mod, _attr in _NODE_SPECS:
    _load_node(_name, _disp, _mod, _attr)

_apply_node_ui_metadata()

for line in _LOG_LINES:
    _LOGGER.info(line)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "LOAD_RESULTS"]
