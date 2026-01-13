# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.2.0

## Русский
Набор кастомных нод для ComfyUI. Включает подготовку изображения для Qwen
Outpaint и ноду выравнивания оверлея по бэкграунду с выводом параметров
трансформации.

### Изменения
- Добавлена нода выравнивания с матчингом (ORB/AKAZE/SIFT), масками и композитом.
- Добавлен JSON с трансформациями для Fusion (0..1) и Resolve Edit Inspector.
- Добавлены опции запрета поворота и независимого масштаба по X/Y.

### Установка
1. Поместите папку в `ComfyUI/custom_nodes/`.
2. Перезапустите ComfyUI.

### Ноды
#### Image Prepare for QwenEdit Outpaint
Нода подготовки для Qwen Image Edit Outpaint. Масштабирует и центрирует
изображение под выбранное соотношение сторон, затем возвращает пустой латент
нужного размера для KSampler. Целевая площадь ~1024x1024, округление до кратных 32.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Входы:
- image (IMAGE)
- aspect_ratio (1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

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

Выходы:
- aligned_overlay (IMAGE)
- composite (IMAGE)
- difference (IMAGE)
- transform_json (STRING)

## English
A set of custom nodes for ComfyUI. Includes image preparation for Qwen
Outpaint and an overlay alignment node with transformation export.

### Changelog
- Added alignment node with feature matching (ORB/AKAZE/SIFT), masks, and compositing.
- Added transform JSON output for Fusion (normalized 0..1) and Resolve Edit Inspector.
- Added options for rotation lock and independent X/Y scaling.

### Install
1. Place this folder under `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

### Nodes
#### Image Prepare for QwenEdit Outpaint
Prepares inputs for Qwen Image Edit Outpaint. Resizes and centers the image to
the selected aspect ratio, then outputs an empty latent for KSampler. Target
area is ~1024x1024, rounded to multiples of 32.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Inputs:
- image (IMAGE)
- aspect_ratio (1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

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

Outputs:
- aligned_overlay (IMAGE)
- composite (IMAGE)
- difference (IMAGE)
- transform_json (STRING)
