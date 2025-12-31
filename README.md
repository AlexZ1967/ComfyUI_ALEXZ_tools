# ComfyUI-QWen_Edit_outpaint

Custom node for preparing Qwen Image Edit outpaint inputs. It resizes and
centers the input image onto a canvas sized to the selected aspect ratio,
then outputs an empty latent with matching resolution for KSampler. Target
pixel area is ~1024x1024, rounded to multiples of 32.

## Node
- Display name: ComfyUI-QWen_Edit_outpaint
- Type name: QWen_Edit_outpaint
- Category: QWen

## Install
1. Place this folder under `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

## Inputs
- image (IMAGE)
- aspect_ratio (1x1, 16x9, 9x16, 2x3, 3x2, 4x3, 3x4)

## Output
- image (IMAGE, padded to target size, centered, gray background)
- latent (LATENT, empty, matching target size)
