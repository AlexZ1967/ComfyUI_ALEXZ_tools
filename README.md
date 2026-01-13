# ComfyUI-QWen_Edit_outpaint

Version: 0.2.0

## Changelog
- Added alignment node with feature matching (ORB/AKAZE/SIFT), masks, and compositing outputs.
- Added transform JSON output for Fusion (normalized 0..1) and Resolve Edit Inspector.
- Added options for rotation lock and independent X/Y scaling.

Custom node for preparing Qwen Image Edit outpaint inputs. It resizes and
centers the input image onto a canvas sized to the selected aspect ratio,
then outputs an empty latent with matching resolution for KSampler. Target
pixel area is ~1024x1024, rounded to multiples of 32.

## Node
- Display name: Image Prepare for QwenEdit Outpaint
- Type name: ImagePrepare_for_QwenEdit_outpaint
- Category: image/qwen

## Install
1. Place this folder under `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

## Inputs
- image (IMAGE)
- aspect_ratio (1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

## Output
- image (IMAGE, padded to target size, centered, gray background)
- latent (LATENT, empty, matching target size)
