# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.6.4

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
Цветокоррекция по образцу с выводом matched, difference, ΔE heatmap, waveform, JSON для GIMP/Resolve/Fusion и LUT. Подробно: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md).

- Display name: Color Match To Reference  
- Type name: ImageColorMatchToReference  
- Category: image/color

Режимы: levels / mean_std / linear / hist / pca_cov / lab_l / lab_full / lab_l_cdf / lab_cdf / hsv_shift / perceptual_vgg / perceptual_vgg_fast / perceptual_adain / perceptual_ltct / perceptual_lut3d / perceptual_unet (последние три требуют веса).  
Выходы: `matched_image`, `difference`, `deltae_heatmap`, `waveform_ref`, `waveform_matched`, `match_json`
Guide: [COLOR_MATCH_GUIDE.md](COLOR_MATCH_GUIDE.md)

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

## Show/Save JSON
Узловой вывод красиво отформатированного JSON и (опционально) сохранение в файл/директорию.

- Display name: Show/Save JSON  
- Type name: JsonDisplayAndSave  
- Category: utils/json  
Inputs: `json_text`, optional `output_path`  
Outputs: `json_pretty`
Guide: [GUIDE_JSON.md](GUIDE_JSON.md)
