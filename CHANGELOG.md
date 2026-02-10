# Changelog — ALEXZ_tools

## 0.15.1 — 2026-02-10
- Added cross-UI node metadata compatibility for old and new node card design (Nodes 2.0).
- Node registry now injects `DESCRIPTION`, `OUTPUT_TOOLTIPS`, and `SEARCH_ALIASES` for all package nodes (`nodes/__init__.py`).
- Added smoke-test for UI metadata presence on loaded nodes (`tests/test_smoke_nodes.py`).
- Updated `README.md` runtime notes to document Nodes 2.0 compatibility.

## 0.15.0 — 2026-02-10
- Added new node `Generate QR Code` (`GenerateQRCode`) for QR generation from link/text.
- New node inputs: `url`, `resolution`, `error_correction` (`L/M/Q/H`), output: `image`.
- Added guide `guides/GUIDE_QR_CODE.md`.
- Updated docs synchronization checks to include QR node (`utils/docs_check.py`).
- Added dependency `qrcode` to `requirements.txt`.

## 0.14.0 — 2026-02-10
- Minor-version bump due large internal refactor of project layout (`nodes/`, `guides/`, `utils/`).
- `Remove Static Watermark from Video` migrated to ProPainter-only pipeline.
- Removed `e2fgvi` and `e2fgvi_hq` execution paths and deleted `method` input from `VideoInpaintWatermark`.
- Removed internal `e2fgvi` package from repository (`e2fgvi/*`).
- Updated `README.md` and `guides/GUIDE_VIDEO_INPAINT.md` to match new ProPainter-only behavior.

## 0.13.18 — 2026-02-10
- `Module Nodes`: added `Update ComfyUI` button in ComfyUI update alert (shown when `behind > 0`).
- `Update ComfyUI` now runs `git pull --ff-only` for ComfyUI and then refreshes module/node snapshots.
- Added requirements flow for ComfyUI update: if `ComfyUI/requirements.txt` changed, widget asks for confirmation and installs via current interpreter (`sys.executable -m pip install -r ...`).
- Update job API (`/alexz_tools/module_update`) now supports `scope=comfyui`; status payload includes `requirements_changed` for both module and ComfyUI update flows.
- Node change highlighting remains unified for all groups (`Core_Nodes`, `Core_Extras_Nodes`, `API_Nodes`, `Custom_Nodes`): red frame for new nodes, green frame for updated nodes between runs.

## 0.13.17 — 2026-02-10
- `Module Nodes`: added module update actions in UI:
  - `Update module` in custom-module card (shown only when update is available),
  - `Update all custom_nodes (N)` in header for `Custom_Nodes` group.
- Added backend update APIs:
  - `POST /alexz_tools/module_update`,
  - `GET /alexz_tools/module_update_status`,
  - `POST /alexz_tools/module_install_requirements`.
- Update flow now performs `git pull --ff-only`; if `requirements.txt` changed between old/new commit, widget prompts to install dependencies.
- Dependency install runs in current ComfyUI Python environment via `sys.executable -m pip install -r <module>/requirements.txt`.
- `node_catalog` now returns `custom_modules_need_update` for dynamic visibility of `Update all`.

## 0.13.16 — 2026-02-09
- Startup log order adjusted: first line is now always `ALEXZ_tools loading...`, followed by widget/backend and node load lines.
- Removed duplicate package header logging from `nodes.py` to keep startup output clean.

## 0.13.15 — 2026-02-09
- Startup load order adjusted: Module Nodes backend is initialized before node table import.
- Added startup log line `✅ Module Nodes widget backend loaded` so widget registration is visible and appears first in package startup logs.

## 0.13.14 — 2026-02-09
- `Module Nodes` refresh progress moved from console to widget status line (single-line progress with `current/total`, `remaining`, current module).
- Refresh API switched to background job mode with status polling (`POST /alexz_tools/module_refresh`, `GET /alexz_tools/module_refresh_status`) to keep UI responsive.
- Removed verbose console progress output for module refresh.
- Added regression test for refresh progress callback emission.

## 0.13.13 — 2026-02-09
- Fixed duplicated refresh logs in ComfyUI console: progress lines are now emitted once per event.

## 0.13.12 — 2026-02-09
- Added compact console progress logs for Module Nodes refresh:
  - startup initialization message,
  - per-module upstream sync progress (when enabled),
  - snapshot recomputation and completion messages.

## 0.13.11 — 2026-02-09
- Hardened git subprocess calls with non-interactive environment (`GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=echo`) to prevent UI/API hangs on credential/passphrase prompts during status refresh.

## 0.13.10 — 2026-02-09
- Fixed Node Picker freeze on initial open: startup catalog load no longer performs `git fetch` across all custom modules.
- Upstream sync is now explicit in `POST /alexz_tools/module_refresh` (`sync_upstreams=1` by default), keeping first-load UI responsive.
- Added regression test to ensure default refresh path does not sync upstreams unless requested.

## 0.13.9 — 2026-02-09
- `Module Nodes`: before status recomputation, backend now syncs custom-module upstream refs via `git fetch --quiet` (for modules with configured upstream).
- Fixes false `up_to_date` states caused by stale local upstream refs when modules were updated on GitHub but not fetched locally.
- Added regression test ensuring refresh triggers upstream sync for discovered custom modules.

## 0.13.8 — 2026-02-08
- `Module Nodes` update-tracking fixed for modules that were not previously in cache: startup scan now tracks all installed custom modules, not only modules already seen in `module_state_cache.json`.
- Added custom-module name canonicalization in git/status paths to avoid case/alias duplicates (e.g. `comfyui-AGSoft` vs `ComfyUI-AGSoft`).
- Added regression test for unseen-module commit change detection between runs.

## 0.13.7 — 2026-02-08
- `Module Nodes`: added ComfyUI update check (git upstream) and top red warning banner when a newer ComfyUI version is available on GitHub (`behind > 0`).
- Backend `node_catalog` now returns `comfyui` status block; refresh endpoint updates this status too.
- Added regression test for ComfyUI update-status computation (`can_update` when behind).

## 0.13.6 — 2026-02-08
- `Module Nodes`: when a module is marked as `new module between runs`, all node cards in that module are now highlighted with green frame (`updated`) for consistent visual semantics.

## 0.13.5 — 2026-02-08
- Fixed `Module Nodes` new-module marker application: `startup_new_modules` is now applied even when there are no node-level `startup_changes` for the module.
- Added regression tests for startup module-change tracking:
  - new module marker without node diffs,
  - `startup_new_modules` detection from module-set diff.

## 0.13.4 — 2026-02-08
- `Module Nodes`: added module-name filter field (substring search) above module dropdown for faster navigation in large module lists.
- Improves discoverability of newly installed modules that may be hard to find by scrolling.

## 0.13.3 — 2026-02-08
- `Module Nodes` classification hardened for edge-case custom modules: if `RELATIVE_PYTHON_MODULE` is missing/ambiguous, group/module is now resolved by source file path.
- Added canonicalization of custom module names against actual `custom_nodes` directory names (handles dashed/normalized naming differences).
- This fixes cases where loaded third-party modules were visible in ComfyUI but not shown under `Custom_Nodes` in Node Picker.

## 0.13.2 — 2026-02-08
- `Module Nodes`: custom module list now includes installed directories from `ComfyUI/custom_nodes` even when their nodes are not loaded in runtime (shown as `(<0>)`).
- Added startup detection for newly appeared modules between runs (`startup_new_modules`), reused by module marker `✅`.
- Module card UX: new module now shows `Detected between runs: new module`; commit transition row is shown only for real commit-to-commit updates.

## 0.13.1 — 2026-02-08
- Fixed `Module Nodes` startup markers disappearing after ComfyUI restart.
- Removed duplicate early snapshot scan on import; now startup change detection runs once via lazy refresh on first Module Nodes API access.

## 0.13.0 — 2026-02-08
- Added new test node `ALEXZ Test Node` (`ALEXZTestNode`) in category `utils/debug`.
- Node is intended for quick validation of package loading and Module Nodes change tracking.

## 0.12.32 — 2026-02-08
- Extended node change tracking to all Module Nodes categories (`Core_Nodes`, `Core_Extras_Nodes`, `API_Nodes`, `Custom_Nodes`), not only custom modules.
- Node card frame highlighting (new/updated) now works for modules from all four categories.
- Startup/runtime node snapshots are now stored by `group + module`, preventing cross-category collisions.

## 0.12.31 — 2026-02-08
- Fixed startup detection timing for node/module changes in `Module Nodes`.
- On first `Module Nodes` API access after ComfyUI start, backend now performs automatic runtime refresh of snapshots/statuses (same effect as pressing `Обновить` once).

## 0.12.30 — 2026-02-08
- `Show/Save JSON` (`JsonDisplayAndSave`): removed output port `json_pretty`.
- Node now works as UI/output-only formatter+saver (shows prettified JSON in UI and writes file when `output_path` is set).

## 0.12.29 — 2026-02-08
- `Image Histogram Scope`: changed default value of `log_scale` to `False`.

## 0.12.28 — 2026-02-08
- Added runtime refresh API `POST /alexz_tools/module_refresh` to recompute module/node status snapshots without restarting ComfyUI.
- `Module Nodes` refresh button now triggers backend status refresh first, then reloads module and node lists.

## 0.12.27 — 2026-02-08
- Added node-level change tracking between ComfyUI runs for custom modules.
- In `Module Nodes`, node cards are now highlighted by frame color:
  - red frame = new node,
  - green frame = updated node.
- Node change detection is based on persisted module snapshots in `module_state_cache.json` (`node_name` + source signature).

## 0.12.26 — 2026-02-08
- `Module Nodes`: red marker `🟥` is now shown only for git-confirmed updates (`git behind > 0`).
- Removed Manager timestamp-based update inference from update status logic to avoid false-positive remote update markers.
- Added git status fields in module info payload: `git_has_upstream`, `git_ahead`, `git_behind`.

## 0.12.25 — 2026-02-08
- Updated remote-update marker in `Module Nodes` module list to plain red square `🟥` (without additional check mark).

## 0.12.24 — 2026-02-08
- Updated red update marker in `Module Nodes` module list to `🟥✔` (check mark on red square style).

## 0.12.23 — 2026-02-08
- Module list markers in `Module Nodes` were adjusted to a more compact style:
- `✅` module updated between ComfyUI runs,
- `🟥✔` update available on remote.

## 0.12.22 — 2026-02-08
- Module list in `Module Nodes` now shows status markers for `Custom_Nodes` modules:
- `🟢✓` module updated between ComfyUI runs,
- `🔴✓` update available on remote.
- Marker state is loaded asynchronously per module and refreshed for the selected module after opening its info card.

## 0.12.21 — 2026-02-08
- Startup module update detection remains enabled, but notifications were moved from console to `Module Nodes` widget.
- Module card now shows `Updated between runs: <old> -> <new>` when tracked module commit changed between ComfyUI launches.

## 0.12.20 — 2026-02-08
- Module Node Picker backend: added startup check for tracked custom modules (`module_state_cache.json`) to detect manual updates between ComfyUI runs.
- On startup, when tracked module commit changed, console now logs `ALEXZ_tools module updated: <module>: <old> -> <new>`.
- Module git state optimized: `remote_head` now uses local upstream ref (`git rev-parse @{u}`), removing network `ls-remote` call.

## 0.12.19 — 2026-02-07
- Module Node Picker: `Remote updated` now prefers git upstream timestamp (`git log -1 @{u}`) for consistency with git-based update status.
- ComfyUI-Manager `github-stats.json` timestamp is now used only as fallback when upstream git timestamp is unavailable.

## 0.12.18 — 2026-02-07
- Module Node Picker: added module update status block in module card (`Installed`, `Remote updated`, `Status`).
- Module metadata backend now combines local git state + ComfyUI-Manager `github-stats.json` to infer `can_update` / `up_to_date` / `unknown`.
- Added persistent module state cache (`module_state_cache.json`) for tracking last check/local change timestamps.

## 0.12.17 — 2026-02-06
- Module Node Picker: removed duplicated `Repository` line in module card.
- Module card now shows owner (clickable GitHub link) + description only.

## 0.12.16 — 2026-02-06
- Module Node Picker: added module metadata card before node list (title, description, owner, repository link).
- Added new API `GET /alexz_tools/module_info` used by sidebar picker.
- For `Custom_Nodes`, metadata is resolved using ComfyUI-Manager `custom-node-list.json` + local git remote fallback for repository mapping.

## 0.12.15 — 2026-02-06
- `Color Match To Reference`: added batch progress indication in console via `tqdm` (`ColorMatch[<preset>]`) during node execution.

## 0.12.14 — 2026-02-06
- `VideoCutMatch`: added two explicit upload buttons in node UI (`choose video_a to upload`, `choose video_b to upload`) so both source videos can be uploaded directly from the node.
- `VideoCutMatch`: removed ambiguous built-in `video_upload` flags from `video_a`/`video_b` combo fields and switched to dedicated frontend uploader for each input.
- Docs: updated README and `GUIDE_VIDEO_CUT_MATCH.md` for dual-upload workflow.

## 0.12.13 — 2026-02-06
- Module Node Picker: simplified group/module classification to ComfyUI-native `RELATIVE_PYTHON_MODULE` first, with lightweight core fallback.
- Fixes missing module buckets after recent classifier changes and restores complete `Custom_Nodes` population.
- `ComfyUI-RMBG` / `AILab_*` nodes are now reliably grouped under `Custom_Nodes -> ComfyUI-RMBG` even when class file path is unavailable.

## 0.12.12 — 2026-02-06
- Group classification switched to ComfyUI-native `node_cls.RELATIVE_PYTHON_MODULE` as primary source for external nodes.
- This restores stable population of `Custom_Nodes`, `Core_Extras_Nodes`, and `API_Nodes` while keeping `Core_Nodes` for built-ins.
- Fix targets missing/empty group lists and incorrect bucket assignment (including `ComfyUI-RMBG`).

## 0.12.11 — 2026-02-06
- Fixed node grouping regression caused by source indexing by `node_name`.
- Classification now uses reverse index by node class identity (`id(class_obj)`) from loaded `NODE_CLASS_MAPPINGS`, preventing cross-pack collisions and restoring full group/module lists.

## 0.12.10 — 2026-02-06
- Node source classification improved using loaded modules' `NODE_CLASS_MAPPINGS` indexes before path-based fallback.
- Fixes mis-grouping cases where custom node classes are wrapped/exported and file-path detection alone is insufficient (e.g. `ComfyUI-RMBG` / `AILab_*` under `Core_Nodes`).

## 0.12.9 — 2026-02-06
- Node picker groups expanded from 2 to 4: `Core_Nodes`, `Core_Extras_Nodes`, `API_Nodes`, `Custom_Nodes`.
- Group detection now maps nodes by source roots: `nodes` / `comfy_extras` / `comfy_api_nodes` / `custom_nodes`.
- For `Custom_Nodes`, `Core_Extras_Nodes`, and `API_Nodes`, module list now shows short module names without path fragments.

## 0.12.8 — 2026-02-06
- Custom/Core detection hardened for third-party nodes: custom-pack name is now resolved from both class file path and python module file path.
- Fixes cases where nodes from custom packs (e.g. wrapped/exported classes) were incorrectly shown under `Core_Nodes`.

## 0.12.7 — 2026-02-06
- Custom module grouping refined: for `Custom_Nodes` the second dropdown now uses package directory names from `ComfyUI/custom_nodes` (no path fragments).
- Node grouping for custom packs now aggregates all nodes of the pack across subdirectories into one module bucket.

## 0.12.6 — 2026-02-06
- Module Node Picker UI adjusted: second dropdown now shows module names (not node names) for selected group.
- Restored click-to-add flow: selecting a module shows node list, clicking a node inserts it into workflow immediately.
- Removed explicit "Добавить ноду в workflow" button from picker.

## 0.12.5 — 2026-02-06
- Module Node Picker UI redesigned to two-step selection: `Core_Nodes | Custom_Nodes` then node list for selected group.
- Added `GET /alexz_tools/node_catalog` endpoint with grouped node catalog for Sidebar picker.
- Picker now shows selected node annotation, category, module and explicit "add node" action.

## 0.12.4 — 2026-02-06
- Module browser API: removed all slicing from fallback output parsing (`RETURN_NAMES`) to support custom container types from third-party nodes.
- Fixed repeated errors in `/alexz_tools/module_list` and `/alexz_tools/module_nodes`: `TypeError: '>' not supported between instances of 'slice' and 'int'`.

## 0.12.3 — 2026-02-06
- Module Node Picker: added dropdown of currently loaded ComfyUI modules (with node counts).
- Added API endpoint `GET /alexz_tools/module_list` for loaded module names and counts.
- Module picker input now supports datalist suggestions from loaded modules.

## 0.12.2 — 2026-02-06
- Module browser API: fixed fallback annotation generation for nodes with non-list `RETURN_NAMES` (no slicing on custom index objects).
- Fixed runtime error: `'>' not supported between instances of 'slice' and 'int'` in `/alexz_tools/module_nodes`.

## 0.12.1 — 2026-02-06
- Module Node Picker: moved from top-menu-only UI to Sidebar tab (`Module Nodes`) via `registerSidebarTab`.
- Added robust fallback: if Sidebar API is unavailable, `Module Nodes` button is added to menu container or as floating button.
- Docs: updated README section for Sidebar-based usage.

## 0.12.0 — 2026-02-05
- New UI tool: `Module Node Picker` (button `Module Nodes` in ComfyUI top menu).
- Added API endpoint `GET /alexz_tools/module_nodes` for module-based node listing with short annotations.
- Added frontend panel to search by python module name and insert node into workflow by click.
- README: documented Module Node Picker usage.

## 0.11.2 — 2026-02-05
- Startup logs: restored single package header `ALEXZ_tools loading...` before compact `✅ <Node> loaded` lines.
- Startup logs: kept duplicate module-level load messages disabled (only centralized logger in `nodes.py`).

## 0.11.1 — 2026-02-05
- Startup logs: removed duplicate per-module `Loaded ... NODE_CLASS_MAPPINGS` messages.
- Startup logs: removed duplicate `✅ ... loaded` messages emitted by `ImageDifference` and `ImageScopes` modules.
- Startup logs: removed extra `ALEXZ_tools loading...` header; now only one compact list with `✅ <Node> loaded` remains.

## 0.11.0 — 2026-02-05
- New node: `Match Video Cut Point` (`VideoCutMatch`) for A-tail vs B-head cut-point matching between two videos.
- VideoCutMatch: outputs best pair frames, frame numbers, top-k candidates, confidence, and cut hints in JSON.
- VideoCutMatch: supports `mse` / `ssim` / `lpips_alex` / `lpips_vgg`, optional normalization, and LPIPS two-pass search.
- Docs: added `GUIDE_VIDEO_CUT_MATCH.md` and README node section.

## 0.10.7 — 2026-02-05
- Docs: all node guides unified to a single structure (purpose -> when to use -> 3-step scenario -> params table -> decision helper -> outputs -> errors -> performance).
- Docs: added decision helpers and numeric quality thresholds/examples for Color Match, Video Frame Match, Difference, scopes, and other nodes.
- Docs: translated `GUIDE_IMAGE_WAVEFORM.md` and `GUIDE_IMAGE_HISTOGRAM.md` fully to Russian.
- Tooling: added `scripts/docs_check.py` to validate guide template and sync of key params/outputs with node API + README sections.
- README: added `Docs Check` section with command to run docs validation.

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
