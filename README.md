# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.6.3

## Русский
Набор кастомных нод для ComfyUI. Включает подготовку изображения для Qwen
Outpaint и ноду выравнивания оверлея по бэкграунду с выводом параметров
трансформации.

- 2026-01-27 | v0.6.4 | Color Match: добавлен perceptual_vgg (VGG19, без скачивания весов вручную).
- 2026-01-27 | v0.6.3 | Color Match: torch-only (GPU) обработка, подсказки по скорости.
- 2026-01-27 | v0.6.2 | Color Match: waveform/parade, ΔE метрики, heatmap, расширенный JSON.
- 2026-01-27 | v0.6.1 | Color Match To Reference: добавлены режимы PCA/strength, LUT экспорт; документация обновлена.
- 2026-01-27 | v0.6.0 | New node: Color Match To Reference (цветокоррекция по образцу, difference и JSON для GIMP/Resolve/Fusion).
- 2026-01-21 | v0.5.4 | VideoInpaintWatermark: выбор видео через Upload/список input, пути к кэшу/выходу вводятся вручную (упрощено).
- 2026-01-21 | v0.5.3 | VideoInpaintWatermark: выходы упрощены до preview_image + transform_json (без маски).
- 2026-01-21 | v0.5.2 | VideoInpaintWatermark: запись полноразмерных кадров (fullframe_*) при стриминге.
- 2026-01-21 | v0.5.1 | VideoInpaintWatermark: двухфазный стриминг с кэшем на диск, отдельные RGB/маска файлы, preview_frame для контроля.
- 2026-01-20 | v0.5.0 | VideoInpaintWatermark: предобрезка по маске и режимы коррекции цвета (color_match_mode).
- 2026-01-19 | v0.4.9 | VideoInpaintWatermark: встроен E2FGVI (e2fgvi/e2fgvi_hq).
- 2026-01-19 | v0.4.8 | ProPainter weights перенесены в propainter/weights, добавлена авто-загрузка.
- 2026-01-19 | v0.4.7 | VideoInpaintWatermark: встроенная реализация ProPainter, без внешних нод.
- 2026-01-19 | v0.4.6 | VideoInpaintWatermark: добавлена нода для видео-инпейнтинга (ProPainter/E2FGVI).
- 2026-01-19 | v0.4.5 | ImageAlignOverlayToBackground: добавлены min_matches/min_inliers и lab_channels.
- 2026-01-19 | v0.4.4 | Image Prepare for QwenEdit Outpaint: убрана опция size_rounding.
- 2026-01-19 | v0.4.3 | ImageAlignOverlayToBackground: добавлен color_mode (gray/lab_l/lab), use_color помечен устаревшим.
- 2026-01-19 | v0.4.2 | Image Prepare for QwenEdit Outpaint: добавлена опция size_rounding (none/32).
- 2026-01-19 | v0.4.1 | ImageAlignOverlayToBackground: transform_json теперь всегда содержит status и overlay_position.
- 2026-01-19 | v0.4.1 | Image Prepare for QwenEdit Outpaint: as_is округляет размеры до кратных 32.
- 2026-01-19 | v0.4.0 | JsonDisplayAndSave: сохранение в файл стало опциональным; поддержка пути к директории.
- 2026-01-19 | v0.4.0 | JsonDisplayAndSave: исправлено отображение JSON без экранирования.
- 2026-01-19 | v0.4.0 | pyproject.toml: обновлены name/description проекта.
- 2026-01-13 | v0.3.1 | JsonDisplayAndSave: объединено отображение и сохранение, output_path стал опциональным (JsonPreview удалена).
- 2026-01-13 | v0.3.0 | JsonDisplayAndSave: вывод JSON на экран и сохранение в файл.
- 2026-01-13 | v0.3.0 | ImageAlignOverlayToBackground: опция use_color для поиска фич по цвету.
- 2026-01-13 | v0.3.0 | Example workflow: восстановление фото + Align Overlay.
- 2026-01-13 | v0.2.0 | ImageAlignOverlayToBackground: матчинг ORB/AKAZE/SIFT, маски, композит, difference и JSON трансформации (Fusion/Resolve).
- 2026-01-13 | v0.2.0 | ImageAlignOverlayToBackground: опции запрета поворота и независимого масштаба по X/Y.

### Установка
1. Склонируйте репозиторий в `ComfyUI/custom_nodes/`:
   `git clone https://github.com/AlexZ1967/ComfyUI_ALEXZ_tools.git`
2. Перезапустите ComfyUI.

### Ноды
#### Image Prepare for QwenEdit Outpaint
Нода подготовки для Qwen Image Edit Outpaint. Масштабирует и центрирует
изображение под выбранное соотношение сторон, затем возвращает пустой латент
нужного размера для KSampler. Целевая площадь ~1328x1328.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Входы:
- **image** (IMAGE)
- **aspect_ratio** (as_is, 1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

Разрешения (image):
- as_is: сохраняет пропорции, масштабирование по площади ~1328x1328
- 1x1: 1328x1328
- 16x9: 1664x928
- 9x16: 928x1664
- 4x3: 1472x1104
- 3x4: 1104x1472
- 3x2: 1584x1056
- 2x3: 1056x1584

Выходы:
- **image** (IMAGE, подготовленное изображение)
- **latent** (LATENT, пустой, соответствующий размеру)

#### Align Overlay To Background
Находит соответствия между двумя изображениями и выравнивает оверлей по
бэкграунду (масштаб/поворот/сдвиг). Возвращает выровненный оверлей, композит,
разницу и JSON с параметрами трансформации.

Принцип работы:
1. Детектор (ORB/AKAZE/SIFT) находит ключевые точки и дескрипторы в фоне и оверлее.
2. Мэтчер сопоставляет дескрипторы и оставляет лучшие совпадения.
3. RANSAC оценивает матрицу трансформации по подмножеству совпадений и отбрасывает выбросы.
4. Оверлей приводится к масштабу/повороту/сдвигу и композитится с фоном.

- Display name: Align Overlay To Background
- Type name: ImageAlignOverlayToBackground
- Category: image/alignment

Входы:
- **background** (IMAGE)
- **overlay** (IMAGE)
- **background_mask** (MASK, optional)
- **overlay_mask** (MASK, optional)
- **feature_count** (INT)
- **good_match_percent** (FLOAT)
- **ransac_thresh** (FLOAT)
- **opacity** (FLOAT)
- **matcher_type** (orb/akaze/sift)
- **min_matches** (INT)
- **min_inliers** (INT)
- **scale_mode** (preserve_aspect/independent_xy)
- **allow_rotation** (BOOLEAN)
- **color_mode** (gray/lab_l/lab)
- **lab_channels** (l/lab)
- **use_color** (BOOLEAN, optional, deprecated)

Описание входов:
- **background**: фоновое изображение, в координатах которого выполняется выравнивание.
- **overlay**: изображение, которое будет масштабировано/повернуто/сдвинуто.
- **background_mask**: маска области совпадений на фоне (белое=использовать).
- **overlay_mask**: маска области совпадений на оверлее (белое=использовать).
- **feature_count**: количество ключевых точек для детектора.
- **good_match_percent**: доля лучших совпадений, используемых для оценки трансформации.
- **ransac_thresh**: порог RANSAC (в пикселях) для отбрасывания выбросов.
- **opacity**: прозрачность оверлея в композите (0..1).
- **matcher_type**: выбор детектора/дескриптора (orb/akaze/sift).
- **min_matches**: минимум совпадений ключевых точек для старта оценки.
- **min_inliers**: минимум inliers после RANSAC (совпадений, согласованных с трансформацией).
- **scale_mode**: масштабирование с сохранением пропорций или по X/Y отдельно.
- **allow_rotation**: разрешить поворот оверлея.
- **color_mode**: режим цвета для детектора (серый/lab_l/lab).
- **lab_channels**: какие каналы LAB использовать при color_mode=lab (l или lab).
- **use_color**: устаревший флаг, эквивалент color_mode=lab.

Рекомендации по параметрам (единицы и примеры):
- **matcher_type**: ORB быстрый и устойчивый; AKAZE подходит для шума; SIFT точнее, но медленнее.
- **feature_count**: количество ключевых точек; примеры: 800–1500 (быстро), 2000–4000 (детальнее).
- **good_match_percent**: доля лучших совпадений (0..1); примеры: 0.1–0.3 типично, 0.4–0.6 при шуме.
- **min_matches**: минимальное число совпадений ключевых точек (целое); примеры: 8–20 простые сцены, 20–50 сложные.
- **min_inliers**: минимальное число inliers после RANSAC (целое); обычно близко к min_matches.
- **ransac_thresh**: порог RANSAC (в пикселях); примеры: 2–5 точнее, 6–10 устойчивее к шуму.
- **scale_mode**: preserve_aspect — обычно верно для фотографии; independent_xy полезен при деформациях.
- **allow_rotation**: отключайте, если оверлей не должен вращаться.
- **opacity**: прозрачность композита (0..1); примеры: 0.5 = 50% оверлея.
- **color_mode**: gray — универсальный; lab_l — устойчивее к цветовым артефактам; lab — лучше на цветных текстурах.
- **lab_channels**: l — только яркость; lab — яркость+цвет (актуально при color_mode=lab).

Выходы:
- **aligned_overlay** (IMAGE)
- **composite** (IMAGE)
- **difference** (IMAGE)
- **transform_json** (STRING)

Поля transform_json:
- **status**: статус выравнивания (ok или сообщение об ошибке).
- **overlay_scale**: масштаб по X/Y.
- **overlay_rotation_angle**: угол поворота в градусах (плюс = против часовой).
- **overlay_position_pixels**: позиция центра оверлея в пикселях бэкграунда.
- **overlay_position**: позиция центра в нормированных координатах 0..1 (0/0 = левый нижний).
- **fusion_position**: позиция центра в координатах Fusion (0..1, 0/0 = левый нижний).
- **resolve_position_edit**: значения Position X/Y для Inspector → Edit в DaVinci Resolve (центр = 0/0; расчет зависит от размеров бэкграунда/овэрлея).
- **pixel_center**: центр в пикселях:
  **top_left** — origin в левом верхнем; **center** — origin в центре кадра.
- **normalized_center**: центр в 0..1:
  **top_left** — origin в левом верхнем; **bottom_left** — origin в левом нижнем.

#### Color Match To Reference
Подгоняет цвет/яркость изображения под образец. Выводит скорректированное изображение,
карту разницы и JSON с параметрами для GIMP, DaVinci Resolve и Fusion.

- Display name: Color Match To Reference
- Type name: ImageColorMatchToReference
- Category: image/color

Входы:
- **reference** (IMAGE) — образец.
- **image** (IMAGE) — картинка для коррекции.
- **mode** (levels/mean_std/linear/hist/pca_cov/lab_l/lab_full/lab_l_cdf/lab_cdf/hsv_shift)
- **percentile** (FLOAT) — обрезка хвостов для levels (0..5%).
- **strength** (FLOAT) — сила применения коррекции (0..1, смешивание с исходником).
- **clip** (BOOLEAN) — обрезать результат 0..1.
- **match_mask** (MASK, optional) — где считать статистику (белое=учитывать).
- **apply_mask** (MASK, optional) — где применять коррекцию (белое=применить).
- **preserve_alpha** (BOOLEAN) — сохранить альфу, если есть.
- **export_lut / lut_size** — сгенерировать 1D LUT (.cube).
- **waveform_enabled / waveform_mode / waveform_width / waveform_gain / waveform_log** — вывод waveform/parade для контроля.
- **deltae_heatmap** — вывод теплоты ΔE как отдельного изображения.

Выходы:
- **matched_image** (IMAGE) — скорректированное изображение.
- **difference** (IMAGE) — |matched - reference| в RGB.
- **deltae_heatmap** (IMAGE) — тепловая карта ΔE (если включено, иначе 1x1).
- **waveform_ref** (IMAGE) — waveform/parade референса (если включено).
- **waveform_matched** (IMAGE) — waveform/parade результата (если включено).
- **match_json** (STRING) — параметры коррекции и пресеты.

Поля match_json:
- **status/mode** — итог и выбранный метод.
- **gimp_levels** — per-channel input black/white/gamma (+black_out/white_out).
- **gimp_hsv** — hue_shift_deg, saturation_mul, value_mul (для hsv_shift).
- **resolve/fusion/linear** — коэффициенты для Resolve/Fusion/линейной аппроксимации.
- **lut_1d_cube/lut_size** — текст 1D LUT (.cube), если export_lut=true.
- **presets** — блоки gimp/resolve/fusion с подсказками.
- **stats** — средние, σ и **delta_e** (mean/median/p95/under2/under5/max), mask_used.
  
Как оценивать результат:
- Смотрите **delta_e** в stats: mean < 2 и p95 < 5 обычно значит «очень близко». under2/under5 — доля пикселей в пределах порогов.
- Визуально: difference и deltae_heatmap подсветят проблемные зоны; waveform/parade покажут совпадение уровней (полосы должны накладываться).

Применение:
- GIMP: Colors → Levels (R/G/B: black, white, gamma), при необходимости Colors → Hue-Saturation (Hue shift, Saturation %, Value %).
- Resolve/Fusion: в Primaries/ColorCorrector выставить Gain (scale), Lift (offset), Gamma (power) по каналам; значения в match_json.

Производительность (torch-only):
- Нода работает на том же устройстве, что входные тензоры (GPU ускоряет все режимы, включая waveform и ΔE).
- Самые дешёвые режимы: `mean_std`, `linear`. Дороже: `hist`, `pca_cov`, `lab_*` (особенно cdf). Waveform/heatmap добавляют вычисления, выключайте при неиспользовании.
- При сильных батчах/высоких разрешениях уменьшите `waveform_width` и отключите `deltae_heatmap` для экономии времени.

#### Remove Static Watermark from Video
Нода для удаления объектов/водяных знаков на видео через инпейтинг. Варианты
ProPainter и E2FGVI встроены. Веса хранятся в `propainter/weights/` и
`e2fgvi/weights/` (при отсутствии скачиваются автоматически).

- Display name: Remove Static Watermark from Video
- Type name: VideoInpaintWatermark
- Category: video/inpaint

Входы:
- **mask** (MASK)
- **method** (propainter/e2fgvi/e2fgvi_hq)
- **mask_dilates** (INT)
- **flow_mask_dilates** (INT)
- **ref_stride** (INT)
- **neighbor_length** (INT)
- **subvideo_length** (INT)
- **raft_iter** (INT)
- **fp16** (enable/disable)
- **throughput_mode** (enable/disable)
- **cudnn_benchmark** (default/enable/disable)
- **tf32** (default/enable/disable)
- **crop_padding** (INT)
- **color_match_mode** (none/mean_std/linear/hist/lab_l/lab_l_cdf/lab_full/lab_cdf)
- **cache_dir** (STRING)
- **output_dir** (STRING)
- **output_name** (STRING)
- **video** (STRING)
- **preview_frame** (INT)
- **write_fullframes** (BOOLEAN)
- **fullframe_prefix** (STRING)

Описание входов:
- **mask**: маска области удаления (1 кадр или batch).
- **method**: выбор алгоритма (propainter/e2fgvi/e2fgvi_hq).
- **mask_dilates**: расширение маски (в пикселях, итерации дилатации).
- **flow_mask_dilates**: расширение маски для оптического потока.
- **ref_stride**: шаг выбора опорных кадров (E2FGVI).
- **neighbor_length**: окно соседних кадров для обработки.
- **subvideo_length**: длина батча/подвидео для ProPainter.
- **raft_iter**: число итераций RAFT (ProPainter).
- **fp16**: ускорение и экономия VRAM ценой точности.
- **throughput_mode**: пропускать очистку кэша GPU (быстрее, но больше памяти).
- **cudnn_benchmark**: оптимизация cuDNN под размер входа.
- **tf32**: разрешить TF32 матмулы (быстрее, менее точно).
- **crop_padding**: паддинг вокруг маски в пикселях.
- **color_match_mode**: подгонка цвета по чистой области (вне маски).
  Варианты: `none` — без коррекции; `mean_std` — выравнивание по среднему/σ (RGB);
  `linear` — линейная подгонка `a*x+b` (RGB); `hist` — совпадение гистограмм (RGB);
  `lab_l` — корректирует только яркость L (LAB); `lab_l_cdf` — CDF‑matching по L;
  `lab_full` — корректирует L+a+b (LAB) по среднему/σ; `lab_cdf` — CDF‑matching по L+a+b.
- **cache_dir**: папка для кэша обрезанного входа (RGB `input_0000.png` + маска `mask_0000.png`).
- **output_dir**: папка для сохранения результата (PNG с альфой, имена `output_name0000.png`), также сохраняет `output_name` + `transform.json`.
- **output_name**: префикс имени файлов (например `patch_`).
- **video**: файл видео из `input/` (можно загрузить через Upload).
- **preview_frame**: индекс кадра для превью (0 = первый обработанный, -1 = не выводить).
- **write_fullframes**: записать полные кадры с наложенным патчем в **output_dir**.
- **fullframe_prefix**: префикс файлов полноразмерных кадров (например `fullframe_0000.png`).

Примечание:
- Нода всегда работает в режиме стриминга с кешированием и всегда делает pre‑crop по маске.
- Режим save‑only включен постоянно (результат пишется на диск).
- Параметры стрима фиксированы: chunk=30, start=0, end=0, stride=1.

Рекомендации по параметрам:
- **method**: `propainter` обычно лучше на сложных сценах; `e2fgvi_hq` — для произвольных разрешений.
- **mask_dilates/flow_mask_dilates**: 4–12 для небольших логотипов, 10–20 для крупных.
- **ref_stride**: 5–15 для типовых видео; меньше = точнее, но медленнее.
- **neighbor_length**: 5–15 типично; больше помогает при сложной динамике.
- **subvideo_length**: 40–120 для длинных роликов; меньше при нехватке VRAM.
- **raft_iter**: 10–30 типично; больше = точнее, но медленнее.
- **fp16**: включайте на GPU с ограниченной памятью.
- **throughput_mode**: включайте при стабильной памяти (без OOM).
- **crop_padding**: 8–32 пикселя для контекста вокруг водяного знака.
- **color_match_mode**: `lab_l`/`lab_l_cdf` для яркости/гаммы; `mean_std`/`linear` для быстрых правок цвета;
  `lab_full` или `lab_cdf` — самый точный, но медленный вариант для сложного цветового дрейфа.
- **cache_dir**: папка для записи обрезанного входа на диск; экономит RAM, но медленнее.
- **output_dir**: папка для PNG‑патчей на диске; рядом будет `transform.json`.
- **output_name**: например `patch_` даст `patch_0000.png`.
- **preview_frame**: используйте, чтобы посмотреть конкретный кадр без сборки всего видео в интерфейсе.

Выходы:
- **preview_image** (IMAGE): один кадр превью (композит патча поверх кэша).
- **transform_json** (STRING): JSON с позицией и масштабом для повторного совмещения (формат как у Align).

Примечание: нода всегда работает в стриминге с кешем и делает pre‑crop по маске,
результат всегда пишется на диск. Если **preview_frame >= 0**, она возвращает один
кадр для контроля. Если **preview_frame = -1**, выход IMAGE = 1x1. При
**write_fullframes=true** сохраняются полноразмерные кадры с наложенным патчем.
Параметры стрима фиксированы: chunk=30, start=0, end=0, stride=1.

Поля transform_json:
- **status**: ok или empty_mask (если маска пуста).
- **overlay_scale**: всегда 1.0/1.0 (масштаб уже учтен в кадре).
- **overlay_rotation_angle**: всегда 0.0.
- **overlay_position_pixels**: центр обрезки в пикселях полного кадра.
- **overlay_position**: центр обрезки в нормированных координатах 0..1.
- **fusion_position**: те же координаты для Fusion (0..1, 0/0 = левый нижний).
- **resolve_position_edit**: значения Position X/Y для Inspector → Edit в DaVinci Resolve
  (расчет зависит от размеров полного кадра и обрезки).
- **pixel_center**: центр обрезки в пикселях:
  **top_left** — origin в левом верхнем; **center** — origin в центре кадра.
- **normalized_center**: центр обрезки в нормированных координатах 0..1:
  **top_left** — origin в левом верхнем; **bottom_left** — origin в левом нижнем.
- **color_space**: цветовое пространство PNG‑патча (сейчас `srgb`).
- **alpha_mode**: тип альфы (`straight`).
- **levels**: уровни (`full`).

Примечание для DaVinci Resolve:
- Для PNG‑патча установите Alpha = **Straight** и Data Levels = **Full**.
- В проекте с цветовым менеджментом сопоставьте `color_space` (обычно sRGB → project space).

#### Show/Save JSON
Нода для аккуратного отображения JSON и записи в файл по заданному пути
(если путь указан).

- Display name: Show/Save JSON
- Type name: JsonDisplayAndSave
- Category: utils/json

Входы:
- **json_text** (ANY)
- **output_path** (STRING, optional, пусто = без сохранения)

Выходы:
- **json_pretty** (STRING)

## English
A set of custom nodes for ComfyUI. Includes image preparation for Qwen
Outpaint and an overlay alignment node with transformation export.

- 2026-01-27 | v0.6.4 | Color Match: new mode perceptual_vgg (optimizes 3x3+bias by VGG19 perceptual loss).
- 2026-01-27 | v0.6.3 | Color Match: torch-only pipeline (GPU/CPU), perf tips.
- 2026-01-27 | v0.6.2 | Color Match: added waveform/parade outputs, ΔE stats/heatmap, richer JSON.
- 2026-01-27 | v0.6.1 | Color Match To Reference: new PCA mode, strength blending, optional 1D LUT export; docs updated.
- 2026-01-27 | v0.6.0 | New node: Color Match To Reference (sample-based color match, difference, JSON for GIMP/Resolve/Fusion).
- 2026-01-21 | v0.5.4 | VideoInpaintWatermark: video selection via Upload/input list; cache/output paths remain manual (simplified).
- 2026-01-21 | v0.5.3 | VideoInpaintWatermark: outputs simplified to preview_image + transform_json (no mask).
- 2026-01-21 | v0.5.2 | VideoInpaintWatermark: full-frame output (fullframe_*) in streaming mode.
- 2026-01-21 | v0.5.1 | VideoInpaintWatermark: two-pass streaming with disk cache, separate RGB/mask cache files, preview_frame output.
- 2026-01-20 | v0.5.0 | VideoInpaintWatermark: pre-crop by mask and color matching modes (color_match_mode).
- 2026-01-19 | v0.4.9 | VideoInpaintWatermark: embedded E2FGVI (e2fgvi/e2fgvi_hq).
- 2026-01-19 | v0.4.8 | ProPainter weights moved to propainter/weights with auto-download.
- 2026-01-19 | v0.4.7 | VideoInpaintWatermark: built-in ProPainter implementation (no external nodes).
- 2026-01-19 | v0.4.6 | VideoInpaintWatermark: added video inpainting node (ProPainter/E2FGVI).
- 2026-01-19 | v0.4.5 | ImageAlignOverlayToBackground: added min_matches/min_inliers and lab_channels.
- 2026-01-19 | v0.4.4 | Image Prepare for QwenEdit Outpaint: removed size_rounding option.
- 2026-01-19 | v0.4.3 | ImageAlignOverlayToBackground: added color_mode (gray/lab_l/lab), use_color marked deprecated.
- 2026-01-19 | v0.4.2 | Image Prepare for QwenEdit Outpaint: added size_rounding option (none/32).
- 2026-01-19 | v0.4.1 | ImageAlignOverlayToBackground: transform_json always includes status and overlay_position.
- 2026-01-19 | v0.4.1 | Image Prepare for QwenEdit Outpaint: as_is sizes are rounded to multiples of 32.
- 2026-01-19 | v0.4.0 | JsonDisplayAndSave: optional file save; directory paths are supported.
- 2026-01-19 | v0.4.0 | JsonDisplayAndSave: JSON display fixed (no escaped slashes).
- 2026-01-19 | v0.4.0 | pyproject.toml: updated project name/description.
- 2026-01-13 | v0.3.1 | JsonDisplayAndSave: merged preview/save, output_path is optional (JsonPreview removed).
- 2026-01-13 | v0.3.0 | JsonDisplayAndSave: display JSON and save to file.
- 2026-01-13 | v0.3.0 | ImageAlignOverlayToBackground: use_color option for color-based feature detection.
- 2026-01-13 | v0.3.0 | Example workflow: photo restoration + Align Overlay.
- 2026-01-13 | v0.2.0 | ImageAlignOverlayToBackground: ORB/AKAZE/SIFT matching, masks, composite, difference, transform JSON (Fusion/Resolve).
- 2026-01-13 | v0.2.0 | ImageAlignOverlayToBackground: rotation lock and independent X/Y scaling options.

### Install
1. Clone the repo into `ComfyUI/custom_nodes/`:
   `git clone https://github.com/AlexZ1967/ComfyUI_ALEXZ_tools.git`
2. Restart ComfyUI.

### Nodes
#### Image Prepare for QwenEdit Outpaint
Prepares inputs for Qwen Image Edit Outpaint. Resizes and centers the image to
the selected aspect ratio, then outputs an empty latent for KSampler. Target
area is ~1328x1328.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Inputs:
- **image** (IMAGE)
- **aspect_ratio** (as_is, 1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

Resolutions (image):
- as_is: keeps aspect ratio, scales to ~1328x1328 area
- 1x1: 1328x1328
- 16x9: 1664x928
- 9x16: 928x1664
- 4x3: 1472x1104
- 3x4: 1104x1472
- 3x2: 1584x1056
- 2x3: 1056x1584

Outputs:
- **image** (IMAGE, prepared)
- **latent** (LATENT, empty, matching size)

#### Align Overlay To Background
Finds feature matches between two images and aligns the overlay to the
background (scale/rotation/translation). Outputs aligned overlay, composite,
difference, and transform JSON.

How it works:
1. A detector (ORB/AKAZE/SIFT) finds keypoints and descriptors in background and overlay.
2. The matcher pairs descriptors and keeps the best matches.
3. RANSAC estimates the transform from a subset of matches and rejects outliers.
4. The overlay is warped (scale/rotation/translation) and composited with the background.

- Display name: Align Overlay To Background
- Type name: ImageAlignOverlayToBackground
- Category: image/alignment

Inputs:
- **background** (IMAGE)
- **overlay** (IMAGE)
- **background_mask** (MASK, optional)
- **overlay_mask** (MASK, optional)
- **feature_count** (INT)
- **good_match_percent** (FLOAT)
- **ransac_thresh** (FLOAT)
- **opacity** (FLOAT)
- **matcher_type** (orb/akaze/sift)
- **min_matches** (INT)
- **min_inliers** (INT)
- **scale_mode** (preserve_aspect/independent_xy)
- **allow_rotation** (BOOLEAN)
- **color_mode** (gray/lab_l/lab)
- **lab_channels** (l/lab)
- **use_color** (BOOLEAN, optional, deprecated)

Input descriptions:
- **background**: background image used as the alignment reference.
- **overlay**: image that will be scaled/rotated/translated.
- **background_mask**: mask of matching region on background (white=use).
- **overlay_mask**: mask of matching region on overlay (white=use).
- **feature_count**: number of keypoints to detect.
- **good_match_percent**: fraction of best matches used to estimate transform.
- **ransac_thresh**: RANSAC pixel threshold for rejecting outliers.
- **opacity**: overlay opacity in composite (0..1).
- **matcher_type**: detector/descriptor type (orb/akaze/sift).
- **min_matches**: minimum keypoint matches to start estimation.
- **min_inliers**: minimum RANSAC inliers (matches consistent with transform).
- **scale_mode**: preserve aspect or scale X/Y independently.
- **allow_rotation**: allow overlay rotation.
- **color_mode**: detector color mode (gray/lab_l/lab).
- **lab_channels**: LAB channels used when color_mode=lab (l or lab).
- **use_color**: deprecated flag, same as color_mode=lab.

Parameter guidance (units and examples):
- **matcher_type**: ORB is fast and robust; AKAZE handles noise well; SIFT is most accurate but slower.
- **feature_count**: number of keypoints; examples: 800–1500 (fast), 2000–4000 (more detail).
- **good_match_percent**: fraction of best matches (0..1); examples: 0.1–0.3 typical, 0.4–0.6 for noisy scenes.
- **min_matches**: minimum keypoint matches (integer); examples: 8–20 simple scenes, 20–50 complex.
- **min_inliers**: minimum RANSAC inliers (integer); usually close to min_matches.
- **ransac_thresh**: RANSAC pixel threshold; examples: 2–5 for precision, 6–10 for tolerance.
- **scale_mode**: preserve_aspect for normal photos; independent_xy for non-uniform scaling.
- **allow_rotation**: disable if the overlay must not rotate.
- **opacity**: composite opacity (0..1); example: 0.5 = 50% overlay.
- **color_mode**: gray is general; lab_l is more stable with color artifacts; lab can help on colorful textures.
- **lab_channels**: l = luminance only; lab = luminance+color (used when color_mode=lab).

Outputs:
- **aligned_overlay** (IMAGE)
- **composite** (IMAGE)
- **difference** (IMAGE)
- **transform_json** (STRING)

transform_json fields:
- **status**: alignment status (ok or error message).
- **overlay_scale**: scale X/Y.
- **overlay_rotation_angle**: rotation angle in degrees (positive = counter-clockwise).
- **overlay_position_pixels**: overlay center in background pixels.
- **overlay_position**: center in normalized coordinates 0..1 (0/0 = bottom-left).
- **fusion_position**: center in Fusion coordinates (0..1, 0/0 = bottom-left).
- **resolve_position_edit**: Position X/Y for DaVinci Resolve Inspector → Edit (center = 0/0; computed from background/overlay sizes).
- **pixel_center**: center in pixels:
  **top_left** origin; **center** origin.
- **normalized_center**: center in 0..1:
  **top_left** origin; **bottom_left** origin.

#### Color Match To Reference
Matches color/brightness of an image to a reference. Outputs corrected image,
difference map, and JSON with parameters for GIMP, DaVinci Resolve, and Fusion.
[Полное руководство](COLOR_MATCH_GUIDE.md)

- Display name: Color Match To Reference
- Type name: ImageColorMatchToReference
- Category: image/color

- **mode** (levels/mean_std/linear/hist/pca_cov/lab_l/lab_full/lab_l_cdf/lab_cdf/hsv_shift/perceptual_vgg/perceptual_adain/perceptual_ltct/perceptual_lut3d/perceptual_unet)
- **perceptual_steps / perceptual_lr** — VGG19 perceptual mode (optimizes 3x3+bias); AdaIN/LTCT/LUT3D/UNet используют свои веса (см. примечания ниже).

Outputs:
- **matched_image** (IMAGE) — corrected image.
- **difference** (IMAGE) — |matched - reference| in RGB.
- **deltae_heatmap** (IMAGE) — ΔE heatmap (if enabled, else 1x1).
- **waveform_ref** (IMAGE) — reference waveform/parade (if enabled).
- **waveform_matched** (IMAGE) — matched waveform/parade (if enabled).
- **match_json** (STRING) — correction params and presets.

match_json fields:
- **status/mode** — result and chosen method.
- **gimp_levels/gimp_hsv** — inputs for GIMP Levels / Hue-Sat.
- **resolve/fusion/linear** — gain/offset/gamma and linear scale/offset.
- **lut_1d_cube/lut_size** — text LUT if requested.
- **deep** — параметры perceptual_vgg/adain; другие deep-моды требуют весов.
- **presets** — blocks gimp/resolve/fusion with short hints.
- **stats** — means, std, and **delta_e** (mean/median/p95/under2/under5/max), mask_used.

How to judge quality:
- Look at **delta_e** stats: mean < 2 and p95 < 5 usually means very close. under2/under5 show pixel ratios within thresholds.
- Visually: use **difference** and **deltae_heatmap** to spot problem areas; waveform/parade should largely overlap between ref and matched if exposure/contrast align.

How to apply:
- GIMP: Colors → Levels (R/G/B: black, white, gamma); optionally Colors → Hue-Saturation (Hue shift, Saturation %, Value %).
- Resolve/Fusion: in Primaries/ColorCorrector set Gain (scale), Lift (offset), Gamma (power) per channel; values are in match_json.

#### Remove Static Watermark from Video
Node for removing objects/watermarks on video via inpainting. ProPainter and
E2FGVI are embedded. Weights live in `propainter/weights/` and `e2fgvi/weights/`
(auto-downloaded if missing).

- Display name: Remove Static Watermark from Video
- Type name: VideoInpaintWatermark
- Category: video/inpaint

Inputs:
- **mask** (MASK)
- **method** (propainter/e2fgvi/e2fgvi_hq)
- **mask_dilates** (INT)
- **flow_mask_dilates** (INT)
- **ref_stride** (INT)
- **neighbor_length** (INT)
- **subvideo_length** (INT)
- **raft_iter** (INT)
- **fp16** (enable/disable)
- **throughput_mode** (enable/disable)
- **cudnn_benchmark** (default/enable/disable)
- **tf32** (default/enable/disable)
- **crop_padding** (INT)
- **color_match_mode** (none/mean_std/linear/hist/lab_l/lab_l_cdf/lab_full/lab_cdf)
- **cache_dir** (STRING)
- **output_dir** (STRING)
- **output_name** (STRING)
- **video** (STRING)
- **preview_frame** (INT)
- **write_fullframes** (BOOLEAN)
- **fullframe_prefix** (STRING)

Input descriptions:
- **mask**: removal mask (single frame or batch).
- **method**: algorithm choice (propainter/e2fgvi/e2fgvi_hq).
- **mask_dilates**: mask dilation iterations (pixels/iterations).
- **flow_mask_dilates**: flow-mask dilation iterations.
- **ref_stride**: reference frame stride (E2FGVI).
- **neighbor_length**: window of neighboring frames.
- **subvideo_length**: subvideo batch length (ProPainter).
- **raft_iter**: RAFT iterations (ProPainter).
- **fp16**: faster, lower VRAM, slightly less accurate.
- **throughput_mode**: skip GPU cache cleanup (faster, more VRAM).
- **cudnn_benchmark**: cuDNN tuning for fixed sizes.
- **tf32**: enable TF32 matmuls (faster, slightly less precise).
- **crop_padding**: padding around the mask in pixels.
- **color_match_mode**: color matching on clean area (outside mask).
  Modes: `none` (no correction); `mean_std` (mean/std matching, RGB);
  `linear` (linear fit `a*x+b`, RGB); `hist` (histogram matching, RGB);
  `lab_l` (L‑only in LAB); `lab_l_cdf` (CDF matching on L);
  `lab_full` (mean/std on L+a+b); `lab_cdf` (CDF matching on L+a+b).
- **cache_dir**: directory for cached cropped input (RGB `input_0000.png` + mask `mask_0000.png`).
- **output_dir**: directory to save output PNG patches (names `output_name0000.png`), also writes `output_name` + `transform.json`.
- **output_name**: filename prefix (e.g. `patch_`).
- **video**: video file from `input/` (use Upload to add).
- **preview_frame**: preview frame index (0 = first processed, -1 = disable preview output).
- **write_fullframes**: write full frames with the patch composited into **output_dir**.
- **fullframe_prefix**: prefix for full-frame files (e.g. `fullframe_0000.png`).

Parameter guidance:
- **method**: `propainter` usually best on complex scenes; `e2fgvi_hq` for arbitrary resolutions.
- **mask_dilates/flow_mask_dilates**: 4–12 for small logos, 10–20 for large.
- **ref_stride**: 5–15 typical; lower = more accurate, slower.
- **neighbor_length**: 5–15 typical; higher helps with complex motion.
- **subvideo_length**: 40–120 for longer videos; lower if VRAM is tight.
- **raft_iter**: 10–30 typical; higher = more accurate, slower.
- **fp16**: enable for limited VRAM GPUs.
- **throughput_mode**: enable if you have VRAM headroom.
- **crop_padding**: 8–32 pixels for extra context around the watermark.
- **color_match_mode**: try `lab_l`/`lab_l_cdf` for brightness/gamma; `mean_std`/`linear` for fast RGB matching;
  `lab_full` or `lab_cdf` for the most accurate but slowest correction.
- **cache_dir**: directory for cached cropped input on disk (slower, lower RAM).
- **output_dir**: directory for PNG patches; `transform.json` is written alongside.
- **output_name**: e.g. `patch_` -> `patch_0000.png`.
- **preview_frame**: use to inspect a specific frame without loading the whole sequence in UI.

Outputs:
- **preview_image** (IMAGE): single preview frame (patch composited over cached input).
- **transform_json** (STRING): JSON with placement data (same format as Align).

Note: the node always writes results to disk and always uses streaming cache with pre-crop.
If **preview_frame >= 0**, it returns one frame for preview. If **preview_frame = -1**,
IMAGE output is 1x1. With **write_fullframes=true**, full-size frames are saved
with the patch composited. Stream parameters are fixed: chunk=30, start=0, end=0, stride=1.

transform_json fields:
- **status**: ok or empty_mask (mask has no pixels).
- **overlay_scale**: always 1.0/1.0 (scale is already baked into the crop).
- **overlay_rotation_angle**: always 0.0.
- **overlay_position_pixels**: crop center in full-frame pixels.
- **overlay_position**: crop center in normalized 0..1 coordinates.
- **fusion_position**: same for Fusion (0..1, 0/0 = bottom-left).
- **resolve_position_edit**: Position X/Y for Inspector → Edit in DaVinci Resolve
  (depends on full-frame and crop sizes).
- **pixel_center**: crop center in pixels:
  **top_left** origin; **center** origin.
- **normalized_center**: crop center in 0..1 coordinates:
  **top_left** origin; **bottom_left** origin.
- **color_space**: PNG patch color space (currently `srgb`).
- **alpha_mode**: alpha type (`straight`).
- **levels**: data levels (`full`).

Resolve note:
- Set PNG patch Alpha = **Straight** and Data Levels = **Full**.
- If color management is enabled, map `color_space` (usually sRGB → project space).

#### Show/Save JSON
Node to display JSON neatly and save it to a file path (if provided).

- Display name: Show/Save JSON
- Type name: JsonDisplayAndSave
- Category: utils/json

Inputs:
- **json_text** (ANY)
- **output_path** (STRING, optional, empty = no save)

Outputs:
- **json_pretty** (STRING)
Примечания:
- Для ProPainter нужны веса в `propainter/weights/` (см. `propainter/weights/README.txt`).
- Если весов нет, они будут скачаны при первом запуске (нужен доступ в интернет).
- Для E2FGVI нужны веса в `e2fgvi/weights/` (см. `e2fgvi/weights/README.txt`).
- Если весов нет, они будут скачаны с Google Drive (нужен доступ в интернет).
- E2FGVI-HQ поддерживает произвольные разрешения, но требует больше памяти.
Notes:
- ProPainter requires weights in `propainter/weights/` (see `propainter/weights/README.txt`).
- Missing weights are downloaded on first run (internet required).
- E2FGVI requires weights in `e2fgvi/weights/` (see `e2fgvi/weights/README.txt`).
- Missing E2FGVI weights are downloaded from Google Drive (internet required).
- E2FGVI-HQ supports arbitrary resolutions but uses more VRAM.
