# Changelog — ALEXZ_tools

## 0.10.6 — 2026-02-05
- VideoFrameMatch: added `best.confidence` and `top_k` to `scores_json`.
- VideoFrameMatch: added explicit `top_k_source` and kept LPIPS two-pass metadata for analysis.
- Color Match To Reference: added quality metrics in `match_json` (`before/after/improvement_pct` for `mse`, `ssim`, `delta_e76`, `lpips_alex`).
- UI tooltips: added compact performance hints for VideoFrameMatch, Color Match, Waveform Scope, and Histogram Scope.
- Tests: added smoke tests (`tests/test_smoke_nodes.py`) for core utilities and JSON contract checks.

## 0.10.5 — 2026-02-05
- VideoFrameMatch: added automatic two-pass search for `lpips_alex/lpips_vgg` (coarse MSE prefilter + LPIPS refine on top-k candidates).
- VideoFrameMatch: reduced memory usage of long runs by storing only first 500 coarse scores in `scores_json`.
- VideoFrameMatch: extended LPIPS `scores_json` with refine metadata (`search`, `coarse_metric`, `coarse_max_side`, `refine_candidates`, `refined_scores`).
- Docs: updated VideoFrameMatch README/guide for two-pass LPIPS behavior.

## 0.10.4 — 2026-02-05
- Image Histogram Scope: improved `rgb_overlay` readability (line curves instead of filled overlap, max blending).
- Image Histogram Scope: extended `hist_json` for overlay with `peak_bin_r/g/b` and `channel_order`.

## 0.10.3 — 2026-02-05
- VideoFrameMatch: removed stale `seek_ok` logging and dead code paths after ffmpeg-tail switch.
- VideoFrameMatch: reduced repeated ffprobe stream probing by reusing stream info in one pass.
- VideoFrameMatch: updated LPIPS dependency error text for `lpips_alex` / `lpips_vgg` metrics.
- Docs: added explicit `ffmpeg` requirement and install commands for `max_frames > 0`.
- Docs: clarified `torchvision` runtime dependency for Color Match preset `perceptual`.

## 0.10.2 — 2026-02-05
- Requirements: removed `tqdm` and `torchvision` from plugin requirements (already provided by base ComfyUI).

## 0.10.1 — 2026-02-05
- Color Match: removed unused legacy functions and kept only active preset pipeline.
- Requirements: added explicit `torchvision` dependency for perceptual preset.

## 0.10.0 — 2026-02-05
- New node: Image Waveform Scope (`ImageWaveformScope`).
- New node: Image Histogram Scope (`ImageHistogramScope`).
- Docs: README and guides updated for new analysis nodes.

## 0.9.1 — 2026-02-05
- Color Match: fix missing `_lab_match_torch` helper for quality preset.

## 0.9.0 — 2026-02-05
- Image Difference: removed `match_target`; auto-resize smaller image to larger by area.

## 0.8.9 — 2026-02-05
- VideoFrameMatch: reference is always resized to video resolution; removed parameter from UI.

## 0.8.8 — 2026-02-05
- VideoFrameMatch: option to resize reference to video resolution before comparison.

## 0.8.6 — 2026-02-05
- VideoFrameMatch: when `max_frames > 0` always use ffmpeg tail decode; require `ffmpeg` in PATH.
- VideoFrameMatch: improved ffmpeg-not-found error message with install hints.
- VideoFrameMatch: metric now uses `lpips_alex`/`lpips_vgg` instead of separate `lpips_net` param.

## 0.8.5 — 2026-02-05
- VideoFrameMatch: optional ffmpeg tail decoding when seek fails.

## 0.8.4 — 2026-02-05
- VideoFrameMatch: when seek fails but total_frames is known, fast-skip to last N frames using `cap.grab()`.

## 0.8.3 — 2026-02-05
- VideoFrameMatch: enforce max_frames limit when tail search is requested.
- VideoFrameMatch: verify actual seek position, fallback if seek is ineffective.

## 0.8.2 — 2026-02-05
- VideoFrameMatch: ffprobe fallback for frame count to speed up tail-only search.
- VideoFrameMatch: log total_frames/start_idx/seek_ok for debugging.

## 0.8.1 — 2026-02-05
- Fix: VideoFrameMatch SSIM path missing `torch.nn.functional` import (`NameError: F`).

## 0.8.0 — 2026-02-05
- Color Match To Reference: simplified inputs to presets (`fast`/`balanced`/`quality`/`perceptual`).
- Color Match To Reference: reduced outputs to `matched_image` / `match_json`.

## 0.7.0 — 2026-02-04
- VideoFrameMatch: removed CLIP options/outputs and `metric_size`; simplified outputs (`best_frame_number` only).

## 0.6.9 — 2026-02-04
- VideoFrameMatch: logging for CLIP loading and frame processing progress.

## 0.6.8 — 2026-02-04
- Fix: VideoFrameMatch fallback when seeking to last frames fails (avoids "No frames processed" with `max_frames > 0`).

## 0.6.7 — 2026-02-04
- VideoFrameMatch: added SSIM/LPIPS/CLIP metrics and color normalization before comparison.
- VideoFrameMatch: `scores_json` now includes metadata (`metric`/`normalize`).

## 0.6.6 — 2026-02-04
- VideoFrameMatch: removed `stride`; `max_frames` now means "last N frames" (`0` = all).

## 0.6.5 — 2026-02-04
- Fix: VideoFrameMatch normalizes image/frame tensors to HWC before MSE to avoid dimension mismatch errors.
- New: Image Difference node (+shared diff helper) to compute `|A−B|` with auto-resize.

## 0.6.4 — 2026-01-27
- Color Match: added mode `perceptual_vgg` (VGG19, no manual weight placement).
- Added mode `perceptual_adain` (auto-download weights).
- GPU torch-only implementation and performance tips.
- Color Match Guide moved to a separate file.
- Added stub modes `perceptual_ltct`/`lut3d`/`unet` (require external weights).
- README simplified; guides moved to dedicated files with links from node descriptions.
- Added compact node load logging at startup (loaded/failed visibility).
- Fixed perceptual_vgg: disabled inference_mode for correct backward pass.
- Added tqdm progress for perceptual_vgg and model download logs.
- Additional perceptual_vgg fixes for inference tensors and optimization context.
- Added node Find Closest Video Frame (MSE, returns frame/index/scores).

## 0.6.3 — 2026-01-27
- Torch-only pipeline (GPU/CPU), performance tips for Color Match.

## 0.6.2 — 2026-01-27
- Waveform/parade outputs, delta metrics, heatmap, expanded JSON.

## 0.6.1 — 2026-01-27
- PCA/strength modes, 1D LUT export; documentation updated.

## 0.6.0 — 2026-01-27
- New node: Color Match To Reference.

## 0.5.4–0.2.0 (2026-01-21…2026-01-19)
- VideoInpaintWatermark improvements (ProPainter/E2FGVI, cache, full frames).
- ImageAlignOverlayToBackground improvements (LAB modes, min_matches/min_inliers).
- JsonDisplayAndSave merged and simplified.

## 0.3.0–0.2.0 (2026-01-13)
- Example workflow: photo restore + Align Overlay.
- Added `use_color`, rotation lock, independent scaling.
