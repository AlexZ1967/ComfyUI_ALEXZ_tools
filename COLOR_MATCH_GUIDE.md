# Color Match To Reference - Practical Guide

## Purpose
Match color of `image` to `reference` using simple presets.

## Dependencies
- `preset=perceptual` uses VGG19 from `torchvision`.
- In normal ComfyUI installs `torchvision` is already available in the base environment.
- If your custom Python env does not include it, install `torchvision` manually in that env.

## Quick Start
1. Connect `reference` and `image`.
2. Select `preset`:
   - `fast`: mean/std match (fastest).
   - `balanced`: linear channel fit (default).
   - `quality`: LAB CDF match (better tone transfer, slower).
   - `perceptual`: VGG perceptual fast (slowest).
3. Tune `strength` if result is too strong (`0.6..0.8` is usually safe).
4. Check `match_json` and, if needed, compare with `Image Difference` node.

## Masks
- `match_mask`: where statistics are computed.
- `apply_mask`: where correction is applied.

## Outputs
- `matched_image`: corrected image.
- `match_json`: correction parameters and stats.
- `match_json.quality.before`: `mse`, `ssim`, `delta_e76`, `lpips_alex` до коррекции.
- `match_json.quality.after`: те же метрики после коррекции.
- `match_json.quality.improvement_pct`: улучшение в процентах (для `mse/delta_e76/lpips` — уменьшение ошибки, для `ssim` — рост).

## Quality Check
- Compare `ref_mean/ref_std` and `img_mean/img_std` in `match_json.stats`.
- Check `match_json.quality`: good match usually means `mse`, `delta_e76`, `lpips_alex` go down and `ssim` goes up.

## Common Fixes
- Over-correction: lower `strength`.
- Weak correction: switch `balanced` -> `quality` or `perceptual`.
- Problem only in region: use `match_mask` and `apply_mask`.
