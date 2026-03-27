# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.29.3

## Overview
Набор кастомных нод для ComfyUI: подготовка под Qwen Outpaint, выравнивание оверлея, цветокоррекция по референсу, видео-инструменты, waveform/histogram анализ, генерация QR-кода и отображение/сохранение JSON.

Changelog: [CHANGELOG.md](CHANGELOG.md)
Refactoring plan (RU): [PLAN_REFACTORING_ADOPTED_RU.md](refactoring_plan/PLAN_REFACTORING_ADOPTED_RU.md)
Look Match roadmap (RU): [ROADMAP_LOOK_MATCH_0_22_RU.md](refactoring_plan/ROADMAP_LOOK_MATCH_0_22_RU.md)

## Install
1. Клонируйте в `ComfyUI/custom_nodes/`:  
   `git clone https://github.com/AlexZ1967/ComfyUI_ALEXZ_tools.git`
2. Установите зависимости ноды:  
   `pip install -r requirements.txt`
3. Перезапустите ComfyUI.

## Docs Check
- Проверка синхронизации параметров/выходов нод и документации:  
  `python utils/docs_check.py`

## Development Environment
- Для локальных проверок используйте Conda-окружение `p313`.
- Рекомендуемый префикс команд: `conda run -n p313 ...`
- Быстрые команды: `make docs-check`, `make seam-smoke`, `make smoke`, `make js-check`

## Runtime notes
- `Color Match To Reference` preset `perceptual_vgg_fast` использует `torchvision` из базовой среды ComfyUI.
- `Find Closest Video Frame` при `max_frames > 0` использует `ffmpeg` (должен быть в `PATH`).
  Linux: `sudo apt install ffmpeg`  
  Windows: `choco install ffmpeg`  
  macOS: `brew install ffmpeg`
- Ноды дополнены UI-метаданными (`DESCRIPTION`, `OUTPUT_TOOLTIPS`, `SEARCH_ALIASES`) для совместимости с новым дизайном карточек (Nodes 2.0), при этом полностью совместимы со старым UI.
- Диагностика динамического скрытия параметров в UI:
  [GUIDE_WIDGET_VISIBILITY_PROFILES.md](guides/GUIDE_WIDGET_VISIBILITY_PROFILES.md)

## UI Tool: Module Node Picker
- Инструмент показывает ноды по модулям (`Core/Extras/API/Custom`), дает быстрый поиск, статус обновлений и вставку выбранной ноды в workflow.
- Подробное описание UI и всех кнопок: [GUIDE_MODULE_NODE_PICKER.md](guides/GUIDE_MODULE_NODE_PICKER.md)
- Известные проблемы и обходные пути: [![Known Issue](https://img.shields.io/badge/Known%20Issue-red)](guides/GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md) [GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md](guides/GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md)

## Nodes (jump to details)
- [Image Prepare for QwenEdit Outpaint](#image-prepare-for-qwenedit-outpaint)
- [Align Overlay To Background](#align-overlay-to-background)
- [Color Match To Reference](#color-match-to-reference)
- [Look Match Resolve](#look-match-resolve)
- [Look Match Nuke Build](#look-match-nuke-build)
- [Look Match Nuke Apply](#look-match-nuke-apply)
- [Seam Match To Reference (Legacy)](#seam-match-to-reference-legacy)
- [Find Closest Video Frame](#find-closest-video-frame)
- [Match Video Cut Point](#match-video-cut-point)
- [Image Difference](#image-difference)
- [Generate QR Code](#generate-qr-code)
- [Download DZI Tiles Image](#download-dzi-tiles-image)
- [Download DZI Tiles Batch Save](#download-dzi-tiles-batch-save)
- [Download IIIF Image](#download-iiif-image)
- [Search Trove Image IDs](#search-trove-image-ids)
- [Image Waveform Scope](#image-waveform-scope)
- [Image Histogram Scope](#image-histogram-scope)
- [Remove Static Watermark from Video](#remove-static-watermark-from-video)
- [Show/Save JSON](#showsave-json)
- [ALEXZ Test Node](#alexz-test-node)

---

## Image Prepare for QwenEdit Outpaint
Масштабирует и центрирует изображение под нужное соотношение сторон, возвращает подготовленное изображение и пустой латент (KSampler). Целевая площадь ~1328×1328.

- Display name: Image Prepare for QwenEdit Outpaint  
- Type name: ImagePrepare_for_QwenEdit_outpaint  
- Category: image/qwen

Inputs: `image`, `aspect_ratio` (as_is, 1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)  
Outputs: `image`, `latent`
Guide: [GUIDE_IMAGE_PREP.md](guides/GUIDE_IMAGE_PREP.md)

---

## Align Overlay To Background
Ищет фичи (ORB/AKAZE/SIFT), выравнивает оверлей к фону (масштаб/поворот/сдвиг), возвращает aligned/composite/difference и transform_json (Fusion/Resolve координаты).

- Display name: Align Overlay To Background  
- Type name: ImageAlignOverlayToBackground  
- Category: image/alignment

Основные входы: background/overlay (+маски), feature_count, good_match_percent, ransac_thresh, opacity, matcher_type, min_matches, min_inliers, scale_mode, allow_rotation, color_mode.  
Выходы: `aligned_overlay`, `composite`, `difference`, `transform_json`  
Guide: [GUIDE_ALIGN.md](guides/GUIDE_ALIGN.md)

---

## Color Match To Reference
Цветокоррекция по образцу с пресетами `mean_std`, `linear`, `tone_curve`, `adain`, `optimal_transport`, `lab_cdf`, `oklab_cdf`, `auto_optimal`, `perceptual_vgg_fast`. Подробно: [GUIDE_COLOR_MATCH_DETAILED.md](guides/GUIDE_COLOR_MATCH_DETAILED.md).

- Display name: Color Match To Reference  
- Type name: ImageColorMatchToReference  
- Category: image/color

Пресеты: `mean_std`, `linear`, `tone_curve`, `adain`, `optimal_transport`, `lab_cdf`, `oklab_cdf`, `auto_optimal`, `perceptual_vgg_fast`.  
Для `auto_optimal`: `auto_optimal_metric` = `mse` / `mse_ssim` / `mse_ssim_lpips`.  
Для видео: `auto_temporal_stability` + `auto_temporal_alpha` + `auto_switch_threshold` стабилизируют выбор метода между кадрами.  
Для сложных кадров в `auto_optimal`: `auto_quality_fallback` + `auto_fallback_method` + `auto_fallback_threshold` включают fallback к более тяжелому методу.  
`quality_metrics_mode`: `off` / `fast` / `full`; legacy `compute_quality_metrics=false` принудительно ставит `off`.  
`spatial_grid` (NxN) включает локальный матчинг по плиткам для `linear` / `mean_std` / `adain` / `auto_optimal`.  
`skin_tone_protection` + `skin_protection_strength` уменьшают цветовой сдвиг в skin-областях.  
`match_json.quality`: метрики до/после (`mse`, `ssim`, `delta_e76`, `lpips_alex`) и `improvement_pct`.  
`delta_e76` теперь считается torch-формулой (быстрее, без обязательной зависимости от `cv2`).  
`compute_quality_metrics=false` отключает расчёт метрик для ускорения обработки батчей.  
`export_lut=true` сохраняет `.cube` (параметры: `lut_size`, `lut_output_dir`, `lut_name`) и возвращает путь в `match_json.lut`.  
Выходы: `matched_image`, `match_json`.  
Guide: [GUIDE_COLOR_MATCH_DETAILED.md](guides/GUIDE_COLOR_MATCH_DETAILED.md), кратко — [GUIDE_COLOR_MATCH.md](guides/GUIDE_COLOR_MATCH.md)

---

## Look Match Resolve
Resolve-style монолитная нода для look transfer.  
Текущий статус: Phase B MVP (реальный staged match: exposure/WB -> tone -> palette -> optional skin protection).

- Display name: Look Match Resolve  
- Type name: ImageLookMatchResolve  
- Category: image/color

Ключевые входы: `reference`, `image`, `strength`, `compute_device`, `working_space`, `downscale_long_side`, `tone_model`, `palette_model`, `lut_size`, `w_exposure`, `w_tone`, `w_chroma`, `skin_protection`, `skin_protection_strength`, `subject_mask`, `sky_mask`, `ground_mask`, `export_lut_cube`.  
Выходы: `matched_image`, `look_json`, `cube_text`.
Guide: [GUIDE_LOOK_MATCH.md](guides/GUIDE_LOOK_MATCH.md)

---

## Look Match Nuke Build
Nuke-style нода построения reusable look-модели (Build часть).  
Текущий статус: Phase A contract baseline.

- Display name: Look Match Nuke Build  
- Type name: ImageLookMatchNukeBuild  
- Category: image/color

Ключевые входы: `reference`, `source`, `compute_device`, `working_space`, `downscale_long_side`, `fit_global`, `fit_tone`, `fit_hue_sectors`, `fit_local_regions`, `skin_mask`, `sky_mask`, `ground_mask`, `subject_mask`, `export_lut_cube`, `lut_size`.  
Выходы: `look_model_json`, `cube_text`.
Guide: [GUIDE_LOOK_MATCH.md](guides/GUIDE_LOOK_MATCH.md)

---

## Look Match Nuke Apply
Nuke-style нода применения look-модели (Apply часть).  
Текущий статус: Phase A contract baseline.

- Display name: Look Match Nuke Apply  
- Type name: ImageLookMatchNukeApply  
- Category: image/color

Ключевые входы: `image`, `look_model_json`, `strength`, `compute_device`, `temporal_stabilization`, `temporal_alpha`, `shot_change_threshold`.  
Выходы: `matched_image`, `apply_json`.
Guide: [GUIDE_LOOK_MATCH.md](guides/GUIDE_LOOK_MATCH.md)

---

## Seam Match To Reference (Legacy)
Оптимизированная подгонка кадра к референсу с приоритетом минимальной видимости стыка (минимальный diff) на всей картинке.

- Display name: Seam Match To Reference (Legacy)  
- Type name: ImageSeamMatchToReference  
- Category: image/color

Seam family variants (fixed mode, compact UI):
- `ImageSeamMatchV1AffineToReference` (`Seam Match v1 Affine`)
- `ImageSeamMatchV2TonalToReference` (`Seam Match v2 Tonal`)
- `ImageSeamMatchV3HybridToReference` (`Seam Match v3 Hybrid`)
- `ImageSeamMatchV4LUTToReference` (`Seam Match v4 LUT`)

Ключевые входы (универсальная legacy-нода): `reference`, `image`, `strength`, `compute_device` (`auto`/`cpu`/`cuda`), `color_space` (`rgb`/`oklab`), `downscale_long_side` (`as_is`/`1080p`/`720p`/`480p`), `seam_model` (`v2_tonal`/`v3_hybrid`/`v4_lut`/`v1_affine`), `steps`, `lr`, `w_mse`, `w_ssim`, `w_grad`, `reg_weight`, `robust_delta`, `hybrid_residual_strength`, `hybrid_residual_reg`, `hybrid_coherence_reg`, `lut_size`, `lut_identity_reg`, `lut_smooth_reg`, `lut_lr_scale`.  
Альфа-канал сохраняется автоматически, если присутствует на входе (параметр `preserve_alpha` удален).  
Выходы: `matched_image`, `seam_json`.  
Guide: [GUIDE_SEAM_MATCH.md](guides/GUIDE_SEAM_MATCH.md)

---

## Remove Static Watermark from Video
Инпейнтинг водяных знаков/объектов в видео на базе ProPainter (веса автозагружаются). Стриминг с кэшем, вывод preview и transform_json, опционально полноразмерные кадры.

- Display name: Remove Static Watermark from Video  
- Type name: VideoInpaintWatermark  
- Category: video/inpaint

Ключевые входы: mask, mask_dilates/flow_mask_dilates, ref_stride, neighbor_length, subvideo_length, raft_iter, fp16, throughput_mode, crop_padding, color_match_mode, cache_dir, output_dir, output_name, video, preview_frame, write_fullframes.  
Выходы: `preview_image`, `transform_json`
Guide: [GUIDE_VIDEO_INPAINT.md](guides/GUIDE_VIDEO_INPAINT.md)

---

## Find Closest Video Frame
По заданной картинке ищет наиболее похожий кадр в видео (метрики: MSE/SSIM/LPIPS). Если задан `max_frames`, анализируются только последние N кадров.

- Display name: Find Closest Video Frame  
- Type name: VideoFrameMatch  
- Category: video/utils  
Входы: `image`, `video`, `max_frames` (0=все, иначе последние N), `metric`, `normalize`.  
Выходы: `best_frame`, `best_frame_number`, `scores_json` (объект с metric/normalize, `best{index,score,confidence}`, `top_k`; для `lpips_*` включает двухпроходный поиск с refine-метаданными).
Примечание: при `max_frames > 0` требуется `ffmpeg` в `PATH`.
Guide: [GUIDE_VIDEO_FRAME_MATCH.md](guides/GUIDE_VIDEO_FRAME_MATCH.md)

---

## Match Video Cut Point
Ищет лучшую пару кадров для монтажа между двумя видео: хвост `video_a` и начало `video_b`.

- Display name: Match Video Cut Point  
- Type name: VideoCutMatch  
- Category: video/utils  
Входы: `video_a`, `video_b`, `search_tail_a`, `search_head_b`, `metric`, `normalize`, `top_k`.  
Выходы: `best_frame_a`, `best_frame_b`, `best_frame_a_number`, `best_frame_b_number`, `match_json`.  
Для удобства в ноде есть две отдельные кнопки загрузки: `choose video_a to upload` и `choose video_b to upload`.
`match_json` содержит `best`, `top_k`, `confidence` и `cut_hint` для склейки.
Guide: [GUIDE_VIDEO_CUT_MATCH.md](guides/GUIDE_VIDEO_CUT_MATCH.md)

---

## Image Difference
Абсолютная разница между двумя картинками. Если размеры различаются, меньшая автоматически приводится к большей (по площади).

- Display name: Image Difference  
- Type name: ImageDifference  
- Category: image/utils  
Входы: `image_a`, `image_b` (авторесайз меньшей к большей).  
Выходы: `difference` (|A−B|).  
Guide: [GUIDE_IMAGE_DIFFERENCE.md](guides/GUIDE_IMAGE_DIFFERENCE.md)

---

## Generate QR Code
Генерирует QR-код из ссылки или текста.

- Display name: Generate QR Code  
- Type name: GenerateQRCode  
- Category: image/utils  
Входы: `url`, `resolution`, `error_correction` (L/M/Q/H).  
Выходы: `image`.  
Guide: [GUIDE_QR_CODE.md](guides/GUIDE_QR_CODE.md)

---

## Download DZI Tiles Image
Скачивает тайлы DeepZoom (DZI) и склеивает их в одну итоговую картинку.

- Display name: Download DZI Tiles Image  
- Type name: ImageDownloadDZITiles  
- Category: image/io  
Входы: `site`, `mw`, `level`, `transport`, `proxy_url`, `tile_extension`, `output_dir`, `output_extension`, `filename_mode`.  
`site`: выпадающий список сайтов из `config/dzi_sites.json`.  
`mw`: можно вводить только цифры, тогда нода сама добавит префикс сайта; полный идентификатор тоже принимается.  
Если `mw` пустой, используется `default_mw` выбранного сайта.  
Если `level = -1`, используется `default_level` выбранного сайта.  
`tile_extension`: `jpg` / `jpeg` / `png` / `webp` (используется только выбранный формат, без перебора остальных).  
Если `proxy_url` пустой, нода автоматически пытается подобрать рабочий маршрут через env/system proxy настройки.  
Если `output_dir` задан, нода сохраняет итоговую собранную картинку на диск.  
`filename_mode`: `mw` или `title_or_mw`. Для `title_or_mw` нода пытается взять title страницы объекта и использует `mw` как fallback.  
Если `output_dir` пустой, файл не записывается и картинка только отдается через выход `image`.  
Новые сайты можно добавлять через `config/dzi_sites.json`, если указать `object_url_template`, `dzi_url_template` и `tile_url_template`.  
Выходы: `image`.  
Guide: [GUIDE_DZI_TILES_DOWNLOAD.md](guides/GUIDE_DZI_TILES_DOWNLOAD.md)

---

## Download DZI Tiles Batch Save
Батч-скачивание DZI изображений со сохранением на диск и manifest JSON.

- Display name: Download DZI Tiles Batch Save  
- Type name: ImageDownloadDZITilesBatchSave  
- Category: image/io  
Входы: `site`, `ids_text`, `output_dir`, `level`, `transport`, `proxy_url`, `tile_extension`, `output_extension`, `filename_template`, `overwrite_mode`, `continue_on_error`, `save_mode`.  
`ids_text`: multiline список ID, поддерживаются также `,` и `;`, строки с `#` игнорируются.  
`output_extension`: `png` / `jpg` / `jpeg` / `webp` для итоговых файлов на диске.  
`filename_template` поддерживает: `{index}`, `{raw_id}`, `{mw}`, `{id}`, `{title}`, `{site}`, `{site_key}`, `{level}`. `{title}` пытается взять title страницы объекта и использует `mw` как fallback.  
Выходы: `manifest_json`, `saved_paths_json`, `count_ok`, `count_failed`.  
Guide: [GUIDE_DZI_TILES_DOWNLOAD.md](guides/GUIDE_DZI_TILES_DOWNLOAD.md)

---

## Download IIIF Image
Скачивает изображение из IIIF Image API сервиса, включая London Museum object pages.

- Display name: Download IIIF Image  
- Type name: ImageDownloadIIIFImage  
- Category: image/io  
Входы: `site`, `source_url`, `size_mode`, `requested_width`, `output_dir`, `filename_mode`, `delivery_mode`, `output_format`.  
`London Museum Object Page`: принимает URL страницы объекта и сама извлекает IIIF service URL.  
`Generic IIIF Service URL`: принимает direct service URL, `info.json` URL или HTML-страницу с встроенным IIIF viewer.  
Если `output_dir` задан, нода сохраняет итоговую картинку на диск, а имя файла берет из последнего meaningful сегмента входного `source_url`.  
`filename_mode`: `source_url_slug` или `title_or_slug`. Во втором режиме нода пытается взять title страницы `source_url`, а при неудаче использует slug из URL.  
Выходы: `image`, `info_json`.  
Guide: [GUIDE_IIIF_IMAGE_DOWNLOAD.md](guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md)

---

## Search Trove Image IDs
Best-effort поиск `nla.obj-...` в Trove Images через headless Chrome render публичной search-страницы.

- Display name: Search Trove Image IDs  
- Type name: SearchTroveImageIDs  
- Category: image/io  
Входы: `query`, `category`, `max_results`, `virtual_time_budget_ms`.  
Выходы: `ids_text`, `result_json`, `count`.  
Ограничение: это не официальный API-режим; Trove anti-bot/UI flow может мешать стабильному поиску без API key.  
Guide: [GUIDE_TROVE_SEARCH_IDS.md](guides/GUIDE_TROVE_SEARCH_IDS.md)

---

## Image Waveform Scope
Строит waveform scope по изображению: Luma или RGB parade.

- Display name: Image Waveform Scope  
- Type name: ImageWaveformScope  
- Category: image/analysis  
Входы: `image`, `mode` (luma/parade), `width`, `height`, `gain`, `log_scale`.  
Выходы: `waveform`.  
Guide: [GUIDE_IMAGE_WAVEFORM.md](guides/GUIDE_IMAGE_WAVEFORM.md)

---

## Image Histogram Scope
Строит гистограмму изображения в режимах RGB overlay, RGB split или Luma.

- Display name: Image Histogram Scope  
- Type name: ImageHistogramScope  
- Category: image/analysis  
Входы: `image`, `mode` (rgb_overlay/rgb_split/luma), `bins`, `width`, `height`, `log_scale`.  
`rgb_overlay`: тонкие RGB-кривые, с max-blend (меньше визуального смешивания).  
Выходы: `histogram`, `hist_json`.  
Guide: [GUIDE_IMAGE_HISTOGRAM.md](guides/GUIDE_IMAGE_HISTOGRAM.md)

---

## Show/Save JSON
Узловой вывод красиво отформатированного JSON и (опционально) сохранение в файл/директорию.

- Display name: Show/Save JSON  
- Type name: JsonDisplayAndSave  
- Category: utils/json  
Inputs: `json_text`, optional `output_path`  
Outputs: UI-only (без выходного порта)
Guide: [GUIDE_JSON.md](guides/GUIDE_JSON.md)

---

## ALEXZ Test Node
Простая тестовая нода для проверки загрузки пакета и отслеживания изменений в `Module Nodes`.

- Display name: ALEXZ Test Node  
- Type name: ALEXZTestNode  
- Category: utils/debug  
Inputs: `text`, `value`  
Outputs: `text_out`, `value_out`
