# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.4.3

## Русский
Набор кастомных нод для ComfyUI. Включает подготовку изображения для Qwen
Outpaint и ноду выравнивания оверлея по бэкграунду с выводом параметров
трансформации.

### Изменения
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
нужного размера для KSampler. Целевая площадь ~1328x1328, округление до кратных 32.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Входы:
- image (IMAGE)
- aspect_ratio (as_is, 1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)
- size_rounding (none/32)

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
- image (IMAGE, подготовленное изображение)
- latent (LATENT, пустой, соответствующий размеру)

#### Align Overlay To Background
Находит соответствия между двумя изображениями и выравнивает оверлей по
бэкграунду (масштаб/поворот/сдвиг). Возвращает выровненный оверлей, композит,
разницу и JSON с параметрами трансформации.

- Display name: Align Overlay To Background
- Type name: ImageAlignOverlayToBackground
- Category: image/alignment

Входы:
- background (IMAGE)
- overlay (IMAGE)
- background_mask (MASK, optional)
- overlay_mask (MASK, optional)
- feature_count (INT)
- good_match_percent (FLOAT)
- ransac_thresh (FLOAT)
- opacity (FLOAT)
- matcher_type (orb/akaze/sift)
- scale_mode (preserve_aspect/independent_xy)
- allow_rotation (BOOLEAN)
- color_mode (gray/lab_l/lab)
- use_color (BOOLEAN, optional, deprecated)

Выходы:
- aligned_overlay (IMAGE)
- composite (IMAGE)
- difference (IMAGE)
- transform_json (STRING)

Поля transform_json:
- status: статус выравнивания (ok или сообщение об ошибке).
- overlay_scale: масштаб по X/Y.
- overlay_rotation_angle: угол поворота в градусах (плюс = против часовой).
- overlay_position_pixels: позиция центра оверлея в пикселях бэкграунда.
- overlay_position: позиция центра в нормированных координатах 0..1 (0/0 = левый нижний).
- fusion_position: позиция центра в координатах Fusion (0..1, 0/0 = левый нижний).
- resolve_position_edit: значения Position X/Y для Inspector → Edit в DaVinci Resolve (центр = 0/0; расчет зависит от размеров бэкграунда/овэрлея).

#### Show/Save JSON
Нода для аккуратного отображения JSON и записи в файл по заданному пути
(если путь указан).

- Display name: Show/Save JSON
- Type name: JsonDisplayAndSave
- Category: utils/json

Входы:
- json_text (ANY)
- output_path (STRING, optional, пусто = без сохранения)

Выходы:
- json_pretty (STRING)

## English
A set of custom nodes for ComfyUI. Includes image preparation for Qwen
Outpaint and an overlay alignment node with transformation export.

### Changelog
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
area is ~1328x1328, rounded to multiples of 32.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Inputs:
- image (IMAGE)
- aspect_ratio (as_is, 1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)
- size_rounding (none/32)

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
- image (IMAGE, prepared)
- latent (LATENT, empty, matching size)

#### Align Overlay To Background
Finds feature matches between two images and aligns the overlay to the
background (scale/rotation/translation). Outputs aligned overlay, composite,
difference, and transform JSON.

- Display name: Align Overlay To Background
- Type name: ImageAlignOverlayToBackground
- Category: image/alignment

Inputs:
- background (IMAGE)
- overlay (IMAGE)
- background_mask (MASK, optional)
- overlay_mask (MASK, optional)
- feature_count (INT)
- good_match_percent (FLOAT)
- ransac_thresh (FLOAT)
- opacity (FLOAT)
- matcher_type (orb/akaze/sift)
- scale_mode (preserve_aspect/independent_xy)
- allow_rotation (BOOLEAN)
- color_mode (gray/lab_l/lab)
- use_color (BOOLEAN, optional, deprecated)

Outputs:
- aligned_overlay (IMAGE)
- composite (IMAGE)
- difference (IMAGE)
- transform_json (STRING)

transform_json fields:
- status: alignment status (ok or error message).
- overlay_scale: scale X/Y.
- overlay_rotation_angle: rotation angle in degrees (positive = counter-clockwise).
- overlay_position_pixels: overlay center in background pixels.
- overlay_position: center in normalized coordinates 0..1 (0/0 = bottom-left).
- fusion_position: center in Fusion coordinates (0..1, 0/0 = bottom-left).
- resolve_position_edit: Position X/Y for DaVinci Resolve Inspector → Edit (center = 0/0; computed from background/overlay sizes).

#### Show/Save JSON
Node to display JSON neatly and save it to a file path (if provided).

- Display name: Show/Save JSON
- Type name: JsonDisplayAndSave
- Category: utils/json

Inputs:
- json_text (ANY)
- output_path (STRING, optional, empty = no save)

Outputs:
- json_pretty (STRING)
