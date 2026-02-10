# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.15.6

## Overview
Набор кастомных нод для ComfyUI: подготовка под Qwen Outpaint, выравнивание оверлея, цветокоррекция по референсу, видео-инструменты, waveform/histogram анализ, генерация QR-кода и отображение/сохранение JSON.

Changelog: [CHANGELOG.md](CHANGELOG.md)

## Install
1. Клонируйте в `ComfyUI/custom_nodes/`:  
   `git clone https://github.com/AlexZ1967/ComfyUI_ALEXZ_tools.git`
2. Установите зависимости ноды:  
   `pip install -r requirements.txt`
3. Перезапустите ComfyUI.

## Docs Check
- Проверка синхронизации параметров/выходов нод и документации:  
  `python utils/docs_check.py`

## Runtime notes
- `Color Match To Reference` preset `perceptual` использует `torchvision` из базовой среды ComfyUI.
- `Find Closest Video Frame` при `max_frames > 0` использует `ffmpeg` (должен быть в `PATH`).
  Linux: `sudo apt install ffmpeg`  
  Windows: `choco install ffmpeg`  
  macOS: `brew install ffmpeg`
- Ноды дополнены UI-метаданными (`DESCRIPTION`, `OUTPUT_TOOLTIPS`, `SEARCH_ALIASES`) для совместимости с новым дизайном карточек (Nodes 2.0), при этом полностью совместимы со старым UI.

## UI Tool: Module Node Picker
- В современных версиях ComfyUI инструмент появляется как вкладка `Module Nodes` в боковой панели (Sidebar).
- В старых версиях, где Sidebar API недоступен, появляется fallback-кнопка `Module Nodes` в меню.
- Интерфейс разбит на 2 dropdown:
  1) выбор группы `Core_Nodes`, `Core_Extras_Nodes`, `API_Nodes` или `Custom_Nodes`,
  2) выбор python-модуля из выбранной группы.
- Добавлен текстовый фильтр модулей (по подстроке), чтобы быстро находить нужный пакет в больших списках.
- Для `Custom_Nodes` во втором списке показываются имена пакетов как они лежат в `ComfyUI/custom_nodes` (без путей).
- Даже если custom-модуль установлен, но его ноды не загрузились в рантайм, модуль все равно показывается в списке (с `0` нод) для быстрой диагностики.
- Для `Core_Extras_Nodes` и `API_Nodes` также показываются короткие имена модулей (без путей).
- Внутри выбранного custom-пакета показываются все его ноды, включая ноды из подпапок пакета.
- В dropdown модулей `Custom_Nodes` добавлены метки статуса: `✅` — модуль обновился между запусками ComfyUI, `🟥` — для модуля подтверждено обновление по git (`behind > 0`).
- В верхней части `Module Nodes` показывается красное предупреждение, если у установленного ComfyUI есть новая версия на GitHub (по git `behind > 0`).
- В этом предупреждении появляется кнопка `Update ComfyUI`, которая выполняет `git pull --ff-only` для ComfyUI.
- По кнопке `Обновить информацию о модулях` выполняется `git fetch --quiet` для custom-модулей с настроенным upstream, чтобы статус `🟥` (`behind > 0`) отражал актуальное состояние GitHub.
- Добавлена отдельная кнопка `Обновить информацию о ComfyUI` (без полного module-refresh), чтобы быстро обновить только статус самого ComfyUI.
- Прогресс refresh показывается в виджете одной строкой (`текущий/всего`, `осталось`, текущий модуль), без спама в консоли.
- При выборе модуля показывается карточка модуля: краткое описание, owner (клик по owner открывает GitHub модуля), а также статус обновления (локально установленная версия/дата, последнее удалённое обновление, `Update available` / `Up to date` / `Unknown`).
- Список нод выбранного модуля открывается по клику на карточку модуля (сначала показывается только карточка).
- В карточке модуля добавлена кнопка `Обновить информацию о модуле` для точечного refresh статуса выбранного модуля.
- В карточке custom-модуля при наличии обновления появляется кнопка `Update module` (выполняет `git pull --ff-only` для выбранного модуля).
- Если доступна новая версия, в карточке custom-модуля выводится красный статус `модуль требует обновления`; для ComfyUI аналогично показывается красный алерт `ComfyUI требует обновления` и кнопка `Update ComfyUI`.
- Вверху раздела `Custom_Nodes` при наличии обновлений появляется кнопка `Update all custom_nodes (N)` (выполняет `git pull --ff-only` по всем custom-модулям, где есть update).
- Если после update изменился `requirements.txt`, виджет предлагает установить зависимости. При подтверждении запускается `pip install -r requirements.txt` в том же Python-окружении, где запущен ComfyUI (`sys.executable`).
- Если после `Update ComfyUI` изменился `ComfyUI/requirements.txt`, виджет также предложит обновить зависимости в текущем окружении ComfyUI.
- При старте ComfyUI выполняется проверка ранее отслеженных модулей: если module commit изменился вручную между запусками, эта информация показывается в карточке модуля в `Module Nodes` (`Updated between runs: <old> -> <new>`).
- Новый модуль, появившийся между запусками ComfyUI, помечается `✅` в списке модулей и в карточке как `Detected between runs: new module`.
- Для нод внутри выбранного модуля (во всех 4 категориях) подсвечивается рамка: красная — новая нода между запусками, зеленая — обновленная нода между запусками.
- Это одинаково работает для `Custom_Nodes`, `Core_Nodes`, `Core_Extras_Nodes` и `API_Nodes`, в том числе после обновления самого ComfyUI.
- Кнопка `Обновить информацию о модулях` в `Module Nodes` обновляет внутренний кэш статусов/снимков модулей без перезапуска ComfyUI, затем перезагружает список модулей и нод.
- Дополнительно, при первом открытии `Module Nodes` после старта ComfyUI статусы/снимки модулей обновляются автоматически (без ручного нажатия `Обновить`).
- После выбора модуля показывается список нод; клик по ноде сразу вставляет её в workflow (в центр видимой области).

## Nodes (jump to details)
- [Image Prepare for QwenEdit Outpaint](#image-prepare-for-qwenedit-outpaint)
- [Align Overlay To Background](#align-overlay-to-background)
- [Color Match To Reference](#color-match-to-reference)
- [Find Closest Video Frame](#find-closest-video-frame)
- [Match Video Cut Point](#match-video-cut-point)
- [Image Difference](#image-difference)
- [Generate QR Code](#generate-qr-code)
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
Цветокоррекция по образцу с пресетами качества: fast (mean/std), balanced (linear), quality (LAB CDF), perceptual (VGG). Подробно: [COLOR_MATCH_GUIDE.md](guides/COLOR_MATCH_GUIDE.md).

- Display name: Color Match To Reference  
- Type name: ImageColorMatchToReference  
- Category: image/color

Пресеты: `fast`=mean/std, `balanced`=linear, `quality`=LAB CDF, `perceptual`=VGG.  
`match_json.quality`: метрики до/после (`mse`, `ssim`, `delta_e76`, `lpips_alex`) и `improvement_pct`.  
Выходы: `matched_image`, `match_json`.  
Guide: [COLOR_MATCH_GUIDE.md](guides/COLOR_MATCH_GUIDE.md), кратко — [GUIDE_COLOR_MATCH.md](guides/GUIDE_COLOR_MATCH.md)

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
