# Image Waveform Scope - Guide

## Purpose
Visual control of luminance and channel distribution in an image.

## Inputs
- `image`: source image.
- `mode`: `luma` or `parade`.
- `width`, `height`: scope resolution.
- `gain`: point intensity multiplier.
- `log_scale`: log density scale.

## Outputs
- `waveform`: scope image in `IMAGE` format.

## Usage
1) Start with `mode=parade`, `width=512`, `height=256`, `gain=1.0`, `log_scale=true`.
2) Compare reference and processed images by running both through the same settings.
3) If waveform is too dim, increase `gain`.

## Notes
- `parade` helps detect channel imbalance.
- `luma` is better for contrast/exposure checks.
- Large scope sizes increase compute time.
