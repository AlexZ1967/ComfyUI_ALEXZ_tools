# ALEXZ_tools (Custom Nodes for ComfyUI)

Version: 0.2.0

A small набор кастомных нод для ComfyUI.

## Changelog
- Added alignment node with feature matching (ORB/AKAZE/SIFT), masks, and compositing outputs.
- Added transform JSON output for Fusion (normalized 0..1) and Resolve Edit Inspector.
- Added options for rotation lock and independent X/Y scaling.

## Install
1. Place this folder under `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

## Nodes
### Image Prepare for QwenEdit Outpaint
Custom node for preparing Qwen Image Edit outpaint inputs. It resizes and
centers the input image onto a canvas sized to the selected aspect ratio,
then outputs an empty latent with matching resolution for KSampler. Target
pixel area is ~1024x1024, rounded to multiples of 32.

- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

Inputs:
- image (IMAGE)
- aspect_ratio (1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

Outputs:
- image (IMAGE, padded to target size, centered, gray background)
- latent (LATENT, empty, matching target size)

### Align Overlay To Background
Finds feature matches between two images and aligns the overlay onto the
background with scale/rotation/translation. Outputs aligned overlay, composite,
difference, and transform JSON (Fusion normalized position + Resolve Edit Position).

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
