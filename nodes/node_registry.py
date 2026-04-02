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
    NodeSpec("ImageLookMatchResolve", "Look Match Resolve", ".image_look_match", "ImageLookMatchResolve"),
    NodeSpec("ImageLookMatchNukeBuild", "Look Match Nuke Build", ".image_look_match", "ImageLookMatchNukeBuild"),
    NodeSpec("ImageLookMatchNukeApply", "Look Match Nuke Apply", ".image_look_match", "ImageLookMatchNukeApply"),
    NodeSpec("ImageSeamMatchToReference", "Seam Match To Reference (Legacy)", ".image_seam_match", "ImageSeamMatchToReference"),
    NodeSpec("ImageSeamMatchV1AffineToReference", "Seam Match v1 Affine", ".image_seam_match", "ImageSeamMatchV1AffineToReference"),
    NodeSpec("ImageSeamMatchV2TonalToReference", "Seam Match v2 Tonal", ".image_seam_match", "ImageSeamMatchV2TonalToReference"),
    NodeSpec("ImageSeamMatchV3HybridToReference", "Seam Match v3 Hybrid", ".image_seam_match", "ImageSeamMatchV3HybridToReference"),
    NodeSpec("ImageSeamMatchV4LUTToReference", "Seam Match v4 LUT", ".image_seam_match", "ImageSeamMatchV4LUTToReference"),
    NodeSpec("VideoFrameMatch", "Find Closest Video Frame", ".video_frame_match", "VideoFrameMatch"),
    NodeSpec("VideoCutMatch", "Match Video Cut Point", ".video_cut_match", "VideoCutMatch"),
    NodeSpec("ImageDifference", "Image Difference", ".image_difference", "ImageDifference"),
    NodeSpec("ImageWaveformScope", "Image Waveform Scope", ".image_scopes", "ImageWaveformScope"),
    NodeSpec("ImageHistogramScope", "Image Histogram Scope", ".image_scopes", "ImageHistogramScope"),
    NodeSpec("ImageEstimateRasterPeriod", "Estimate Raster Period", ".image_descreen_adaptive", "ImageEstimateRasterPeriod"),
    NodeSpec("ImageDescreenScalePreview", "Descreen Scale Preview", ".image_descreen_adaptive", "ImageDescreenScalePreview"),
    NodeSpec("ImageDescreenAdaptiveScale", "Descreen By Adaptive Scale (Deprecated)", ".image_descreen_adaptive", "ImageDescreenAdaptiveScale"),
    NodeSpec("ImageDescreenApplyPercent", "Apply Descreen Percent", ".image_descreen_adaptive", "ImageDescreenApplyPercent"),
    NodeSpec("GenerateQRCode", "Generate QR Code", ".qr_code_generate", "GenerateQRCode"),
    NodeSpec("ImageDownloadDZITiles", "Download DZI Tiles Image", ".image_download_dzi_tiles", "ImageDownloadDZITiles"),
    NodeSpec("ImageDownloadDZITilesBatchSave", "Download DZI Tiles Batch Save", ".image_download_dzi_tiles", "ImageDownloadDZITilesBatchSave"),
    NodeSpec("ImageDownloadIIIFImage", "Download IIIF Image", ".image_download_iiif", "ImageDownloadIIIFImage"),
    NodeSpec("SearchTroveImageIDs", "Search Trove Image IDs", ".trove_search_ids", "SearchTroveImageIDs"),
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
    "ImageLookMatchResolve": {
        "description": "Resolve-style монолитный перенос look (Phase B MVP).",
        "output_tooltips": ["Скорректированное изображение.", "JSON контракта/диагностики.", "Опциональный .cube текст."],
        "search_aliases": ["look", "resolve", "match", "grade"],
    },
    "ImageLookMatchNukeBuild": {
        "description": "Nuke-style build: собирает reusable look-модель из пары source/reference.",
        "output_tooltips": ["JSON look-модели.", "Опциональный .cube текст."],
        "search_aliases": ["look", "nuke", "build", "model"],
    },
    "ImageLookMatchNukeApply": {
        "description": "Nuke-style apply: применяет look-модель к кадру/батчу.",
        "output_tooltips": ["Скорректированное изображение.", "JSON применения/диагностики."],
        "search_aliases": ["look", "nuke", "apply", "model"],
    },
    "ImageSeamMatchToReference": {
        "description": "Legacy: универсальная seam-match нода с выбором модели через seam_model.",
        "output_tooltips": ["Скорректированное изображение под seam-match.", "JSON с параметрами оптимизации и метриками."],
        "search_aliases": ["seam", "match", "reference", "color", "legacy"],
    },
    "ImageSeamMatchV1AffineToReference": {
        "description": "Seam-match v1: быстрый глобальный affine матчинг.",
        "output_tooltips": ["Скорректированное изображение (v1 affine).", "JSON с параметрами оптимизации и метриками."],
        "search_aliases": ["seam", "v1", "affine", "match"],
    },
    "ImageSeamMatchV2TonalToReference": {
        "description": "Seam-match v2: отдельные трансформы для теней/середины/светов.",
        "output_tooltips": ["Скорректированное изображение (v2 tonal).", "JSON с параметрами оптимизации и метриками."],
        "search_aliases": ["seam", "v2", "tonal", "match"],
    },
    "ImageSeamMatchV3HybridToReference": {
        "description": "Seam-match v3: глобальный affine + тональные residual-поправки.",
        "output_tooltips": ["Скорректированное изображение (v3 hybrid).", "JSON с параметрами оптимизации и метриками."],
        "search_aliases": ["seam", "v3", "hybrid", "match"],
    },
    "ImageSeamMatchV4LUTToReference": {
        "description": "Seam-match v4: высокоточный 3D LUT матчинг (медленнее).",
        "output_tooltips": ["Скорректированное изображение (v4 LUT).", "JSON с параметрами оптимизации и метриками."],
        "search_aliases": ["seam", "v4", "lut", "match"],
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
    "ImageEstimateRasterPeriod": {
        "description": "Измеряет шаг печатного растра по ROI и выдает прогноз базового процента уменьшения.",
        "output_tooltips": ["ROI превью для контроля анализа.", "Оцененный шаг растра в пикселях.", "Прогноз базового процента уменьшения.", "JSON с ROI, period estimate и predicted scale."],
        "search_aliases": ["descreen", "estimate", "period", "halftone", "raster"],
    },
    "ImageDescreenScalePreview": {
        "description": "Строит scale-sheet от заданного базового процента вверх для визуального выбора лучшего уменьшения.",
        "output_tooltips": ["Подборочный scale-sheet с подписями процентов.", "JSON с ROI, методом ресемплинга и списком scale-вариантов."],
        "search_aliases": ["descreen", "preview", "scale", "sheet", "halftone", "raster"],
    },
    "ImageDescreenAdaptiveScale": {
        "description": "Deprecated legacy all-in-one: оценивает шаг растра, строит preview и сразу делает descreen через downscale/upscale.",
        "output_tooltips": ["Обработанное изображение.", "Подборочный scale-sheet по ROI с подписями процентов.", "Рекомендуемый масштаб в процентах.", "Оцененный шаг растра в пикселях.", "JSON с ROI, period estimate, scale-sheet и таблицей кандидатов."],
        "search_aliases": ["descreen", "halftone", "raster", "moire", "scan", "legacy", "deprecated"],
    },
    "ImageDescreenApplyPercent": {
        "description": "Применяет уже найденный descreen percent как final downscale без обратного апскейла.",
        "output_tooltips": ["Уменьшенное обработанное изображение.", "Фактически примененный масштаб в процентах.", "JSON с параметрами fixed downscale descreen."],
        "search_aliases": ["descreen", "apply", "percent", "halftone", "raster"],
    },
    "GenerateQRCode": {
        "description": "Генерирует QR-код из ссылки или текста.",
        "output_tooltips": ["Сгенерированный QR-код."],
        "search_aliases": ["qr", "qrcode", "link"],
    },
    "ImageDownloadDZITiles": {
        "description": "Скачивает DZI-тайлы и собирает итоговое изображение.",
        "output_tooltips": ["Собранное изображение из тайлов DZI."],
        "search_aliases": ["dzi", "tiles", "download", "zoom"],
    },
    "ImageDownloadDZITilesBatchSave": {
        "description": "Батч-скачивание DZI изображений со сохранением на диск и manifest JSON.",
        "output_tooltips": ["JSON manifest по всему батчу.", "JSON списка сохраненных путей.", "Количество успешно сохраненных изображений.", "Количество ошибок."],
        "search_aliases": ["dzi", "tiles", "download", "batch", "save"],
    },
    "ImageDownloadIIIFImage": {
        "description": "Скачивает изображение из IIIF Image API сервиса, включая London Museum object pages.",
        "output_tooltips": ["Скачанное IIIF изображение.", "JSON с IIIF service/info/image URL и метаданными."],
        "search_aliases": ["iiif", "download", "image", "london museum", "viewer"],
    },
    "SearchTroveImageIDs": {
        "description": "Ищет `nla.obj-...` id в Trove Images через best-effort headless Chrome search.",
        "output_tooltips": ["IDs по одному на строку.", "JSON с URL поиска и диагностикой.", "Количество найденных ids."],
        "search_aliases": ["trove", "search", "nla", "ids", "images"],
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
