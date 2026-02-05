# Color Match To Reference - Practical Guide

## Purpose
Match color of `image` to `reference` using simple presets.

## Quick Start
1) Connect `reference` and `image`.
2) Select `preset`:
- `fast`: mean/std match (fastest).
- `balanced`: linear channel fit (default).
- `quality`: LAB CDF match (better tone transfer, slower).
- `perceptual`: VGG perceptual fast (slowest).
3) Tune `strength` if result is too strong (`0.6..0.8` is usually safe).
4) Check `match_json` and, if needed, compare with `Image Difference` node.

## Masks
- `match_mask`: where statistics are computed.
- `apply_mask`: where correction is applied.

## Outputs
- `matched_image`: corrected image.
- `match_json`: correction parameters and stats.

## Quality Check
- Compare `ref_mean/ref_std` and `img_mean/img_std` in `match_json.stats`.
- If channel means are close and visual difference is small, match is usually good.

## Common Fixes
- Over-correction: lower `strength`.
- Weak correction: switch `balanced` -> `quality` or `perceptual`.
- Problem only in region: use `match_mask` and `apply_mask`.
