"""
Module: nodes/node_registry.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Canonical registry manifest for ComfyUI_ALEXZ_tools node specs and UI metadata.

Purpose:
    Provides a single source of truth for node definitions, so adding/removing
    nodes does not require touching loader logic in multiple files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    """Declarative node registration descriptor used by the loader."""

    type_name: str
    display_name: str
    module_import: str
    class_name: str


NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec("ImagePrepare_for_QwenEdit_outpaint", "Image Prepare for QwenEdit Outpaint", ".image_prepare", "ImagePrepareForQwenEditOutpaint"),
    NodeSpec("ImageAlignOverlayToBackground", "Align Overlay To Background", ".image_align", "ImageAlignOverlayToBackground"),
    NodeSpec("JsonDisplayAndSave", "Show/Save JSON", ".json_output", "JsonDisplayAndSave"),
    NodeSpec("VideoInpaintWatermark", "Remove Static Watermark from Video", ".video_inpaint", "VideoInpaintWatermark"),
    NodeSpec("ImageColorMatchToReference", "Color Match To Reference", ".image_color_match", "ImageColorMatchToReference"),
    NodeSpec("VideoFrameMatch", "Find Closest Video Frame", ".video_frame_match", "VideoFrameMatch"),
    NodeSpec("VideoCutMatch", "Match Video Cut Point", ".video_cut_match", "VideoCutMatch"),
    NodeSpec("ImageDifference", "Image Difference", ".image_difference", "ImageDifference"),
    NodeSpec("ImageWaveformScope", "Image Waveform Scope", ".image_scopes", "ImageWaveformScope"),
    NodeSpec("ImageHistogramScope", "Image Histogram Scope", ".image_scopes", "ImageHistogramScope"),
    NodeSpec("GenerateQRCode", "Generate QR Code", ".qr_code_generate", "GenerateQRCode"),
    NodeSpec("ALEXZTestNode", "ALEXZ Test Node", ".test_node", "ALEXZTestNode"),
)


NODE_UI_METADATA = {
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


def iter_node_specs():
    """Yield node specs as plain tuples for backward-compatible loaders."""
    for spec in NODE_SPECS:
        yield (spec.type_name, spec.display_name, spec.module_import, spec.class_name)

