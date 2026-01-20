# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.4.5

## Русский
Набор кастомных нод для ComfyUI. Включает подготовку изображения для Qwen
Outpaint и ноду выравнивания оверлея по бэкграунду с выводом параметров
трансформации.

### Изменения
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

Рекомендации по параметрам:
- **matcher_type**: ORB быстрый и устойчивый; AKAZE подходит для шума; SIFT точнее, но медленнее.
- **feature_count**: увеличивайте для детализированных сцен; снижайте для скорости.
- **good_match_percent**: 0.1–0.3 для типовых случаев, выше — больше устойчивости при шуме.
- **min_matches**: минимальное число совпадений ключевых точек (features) между оверлеем и фоном.
- **min_inliers**: минимальное число inliers после RANSAC (совпадений, которые хорошо описываются найденной трансформацией).
- **ransac_thresh**: порог RANSAC (в пикселях) — ниже = точнее, выше = устойчивее к шуму/сдвигам.
- **RANSAC**: алгоритм, который оценивает трансформацию по подмножеству совпадений и отбрасывает выбросы.
- **scale_mode**: preserve_aspect — обычно верно для фотографии; independent_xy полезен при деформациях.
- **allow_rotation**: отключайте, если оверлей не должен вращаться.
- **opacity**: влияет только на композит; не влияет на расчет матрицы.
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

### Changelog
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

Parameter guidance:
- **matcher_type**: ORB is fast and robust; AKAZE handles noise well; SIFT is most accurate but slower.
- **feature_count**: increase for detailed scenes; lower for speed.
- **good_match_percent**: 0.1–0.3 for typical cases; higher adds robustness to noise.
- **min_matches**: minimum number of keypoint matches (features) between overlay and background.
- **min_inliers**: minimum number of RANSAC inliers (matches consistent with the estimated transform).
- **ransac_thresh**: RANSAC pixel threshold — lower = more precise, higher = more tolerant to noise/misalignment.
- **RANSAC**: an algorithm that fits the transform on subsets of matches and rejects outliers.
- **scale_mode**: preserve_aspect for normal photos; independent_xy for non-uniform scaling.
- **allow_rotation**: disable if the overlay must not rotate.
- **opacity**: affects composite only; does not affect alignment.
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
