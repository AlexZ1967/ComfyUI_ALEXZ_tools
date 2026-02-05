# Image Histogram Scope - Guide

## Purpose
Histogram visualization for quick exposure and channel distribution checks.

## Inputs
- `image`: source image.
- `mode`: `rgb_overlay`, `rgb_split`, or `luma`.
- `bins`: histogram bins count.
- `width`, `height`: output scope size.
- `log_scale`: log density scale.

## Outputs
- `histogram`: histogram image.
- `hist_json`: short JSON with mode/bins/peak values.

## Usage
1) Use `mode=rgb_overlay` for quick color balance checks.
2) Use `mode=rgb_split` when channels overlap too much.
3) Use `mode=luma` to inspect exposure distribution only.

## Notes
- Start with `bins=256`.
- For noisy images, reduce bins to `64` or `128`.
- Enable `log_scale` when shadows/highlights are too sparse.
