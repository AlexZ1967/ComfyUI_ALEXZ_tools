# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.9.0

## Overview
Набор кастомных нод для ComfyUI: подготовка под Qwen Outpaint, выравнивание оверлея, цветокоррекция по референсу, инпейнтинг водяных знаков в видео и отображение/сохранение JSON.

Changelog: [CHANGELOG.md](CHANGELOG.md)

## Install
1. Клонируйте в `ComfyUI/custom_nodes/`:  
   `git clone https://github.com/AlexZ1967/ComfyUI_ALEXZ_tools.git`
2. Перезапустите ComfyUI.

## Nodes (jump to details)
- [Image Prepare for QwenEdit Outpaint](#image-prepare-for-qwenedit-outpaint)
- [Align Overlay To Background](#align-overlay-to-background)
- [Color Match To Reference](#color-match-to-reference)
- [Find Closest Video Frame](#find-closest-video-frame)
- [Image Difference](#image-difference)
- [Remove Static Watermark from Video](#remove-static-watermark-from-video)
- [Show/Save JSON](#showsave-json)

---

## Image Prepare for QwenEdit Outpaint
Масштабирует и центрирует изображение под нужное соотношение сторон, возвращает подготовленное изображение и пустой латент (KSampler). Целевая площадь ~1328×1328.

- Display name: Image Prepare for QwenEdit Outpaint  
- Type name: ImagePrepare_for_QwenEdit_outpaint  
- Category: image/qwen

Inputs: `image`, `aspect_ratio` (as_is, 1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)  
Outputs: `image`, `latent`
Guide: [GUIDE_IMAGE_PREP.md](GUIDE_IMAGE_PREP.md)

---

## Align Overlay To Background
Ищет фичи (ORB/AKAZE/SIFT), выравнивает оверлей к фону (масштаб/поворот/сдвиг), возвращает aligned/composite/difference и transform_json (Fusion/Resolve координаты).

- Display name: Align Overlay To Background  
- Type name: ImageAlignOverlayToBackground  
- Category: image/alignment

Основные входы: background/overlay (+маски), feature_count, good_match_percent, ransac_thresh, opacity, matcher_type, min_matches, min_inliers, scale_mode, allow_rotation, color_mode.  
Выходы: `aligned_overlay`, `composite`, `difference`, `transform_json`  
Guide: [GUIDE_ALIGN.md](GUIDE_ALIGN.md)

---

## Color Match To Reference
Цветокоррекция по образцу с пресетами качества: fast (mean/std), balanced (linear), quality (LAB CDF), perceptual (VGG). Подробно: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md).

- Display name: Color Match To Reference  
- Type name: ImageColorMatchToReference  
- Category: image/color

Пресеты: `fast`=mean/std, `balanced`=linear, `quality`=LAB CDF, `perceptual`=VGG.  
Выходы: `matched_image`, `match_json`.  
Guide: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md), кратко — [GUIDE_COLOR_MATCH.md](GUIDE_COLOR_MATCH.md)

---

## Remove Static Watermark from Video
Инпейнтинг водяных знаков/объектов в видео (встроены ProPainter и E2FGVI; веса автозагружаются). Стриминг с кэшем, вывод preview и transform_json, опционально полноразмерные кадры.

- Display name: Remove Static Watermark from Video  
- Type name: VideoInpaintWatermark  
- Category: video/inpaint

Ключевые входы: mask, method (propainter/e2fgvi/e2fgvi_hq), mask_dilates/flow_mask_dilates, ref_stride, neighbor_length, subvideo_length, raft_iter, fp16, throughput_mode, crop_padding, color_match_mode, cache_dir, output_dir, output_name, video, preview_frame, write_fullframes.  
Выходы: `preview_image`, `transform_json`
Guide: [GUIDE_VIDEO_INPAINT.md](GUIDE_VIDEO_INPAINT.md)

---

## Find Closest Video Frame
По заданной картинке ищет наиболее похожий кадр в видео (метрики: MSE/SSIM/LPIPS). Если задан `max_frames`, анализируются только последние N кадров.

- Display name: Find Closest Video Frame  
- Type name: VideoFrameMatch  
- Category: video/utils  
Входы: `image`, `video`, `max_frames` (0=все, иначе последние N), `metric`, `normalize`.  
Выходы: `best_frame`, `best_frame_number`, `scores_json` (объект с metric/normalize и первыми 500 оценок).
Guide: [GUIDE_VIDEO_FRAME_MATCH.md](GUIDE_VIDEO_FRAME_MATCH.md)

---

## Image Difference
Абсолютная разница между двумя картинками, при разных размерах можно выбрать сторону для ресайза.

- Display name: Image Difference  
- Type name: ImageDifference  
- Category: image/utils  
Входы: `image_a`, `image_b` (авторесайз меньшей к большей).  
Выходы: `difference` (|A−B|).  
Guide: [GUIDE_IMAGE_DIFFERENCE.md](GUIDE_IMAGE_DIFFERENCE.md)

---

## Show/Save JSON
Узловой вывод красиво отформатированного JSON и (опционально) сохранение в файл/директорию.

- Display name: Show/Save JSON  
- Type name: JsonDisplayAndSave  
- Category: utils/json  
Inputs: `json_text`, optional `output_path`  
Outputs: `json_pretty`
Guide: [GUIDE_JSON.md](GUIDE_JSON.md)
