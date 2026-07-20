# Changelog — ALEXZ_tools

## 0.37.2 — 2026-07-20
- Phase 3 `Module Node Picker` stabilization continued without API changes:
  - extracted catalog orchestration seams from
    `utils/module_node_browser_api.py` to
    `utils/module_browser_api/catalog_ops.py`;
  - kept the legacy facade methods (`_collect_nodes`, `_build_catalog`,
    `_build_group_catalog`, `_build_group_modules`, `_filter_modules`,
    `_build_group_payload`, `_build_module_list_payload`,
    `_build_module_nodes_payload`) as thin compatibility delegates;
  - added direct unit coverage for the new helper seam in
    `tests/test_module_browser_api_catalog_ops.py`.
- Updated H2 stabilization roadmap progress in
  `refactoring_plan/ROADMAP_2026_H2_RU.md` for the completed facade-thinning
  substeps.
- Validation:
  - `conda run -n p313 pytest -q tests/test_module_browser_api_catalog_ops.py tests/test_module_browser_catalog.py tests/test_module_browser_catalog_payload_ops.py tests/test_phase0_baseline.py tests/test_module_browser_api_routes.py`;
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.37.2` in `pyproject.toml` and `README.md`.

## 0.37.1 — 2026-07-20
- Fixed `Module Node Picker` ComfyUI release-check behavior for detached-head
  release installs:
  - removed the false `releases -> commits` fallback that could report
    `behind=N` even when the selected `ComfyUI check: releases` mode should
    only compare against the latest release tag;
  - added local-tag fallback when GitHub latest-release metadata is
    temporarily unavailable, so release-check can still resolve an installed
    release from local git tags;
  - updated the status-card messaging so degraded release checks render as
    neutral availability diagnostics instead of misleading update warnings.
- Validation:
  - `conda run -n p313 pytest -q tests/test_module_browser_comfyui_git_status_ops.py`;
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.37.1` in `pyproject.toml` and `README.md`.

## 0.37.0 — 2026-07-19
- Improved Registry / packaging metadata for current ComfyUI publishing flow:
  - expanded `pyproject.toml` with `requires-python`, project URLs,
    classifiers, optional dependency grouping, and `requires-comfyui`;
  - added registry-facing assets `assets/registry/icon.svg` and
    `assets/registry/banner.svg`;
  - added `.comfyignore` and clarified install/dependency flow in `README.md`.
- Switched `Module Node Picker` requirements follow-up to advisory/manual mode:
  - backend routes for module and ComfyUI requirements now return structured
    manual-install payloads instead of attempting runtime `pip install`;
  - update-status payloads now keep precise `requirements.txt` paths so the UI
    can show the exact command to run manually in the ComfyUI Python
    environment;
  - frontend status cards and follow-up prompts now describe dependency
    changes as manual actions instead of offering auto-install behavior.
- Added operational follow-up artifacts for the new stabilization track:
  - added roadmap document `refactoring_plan/ROADMAP_2026_H2_RU.md`;
  - added helper script `scripts/module_picker_requirements_demo.py` for
    seeding/clearing demo `requirements pending` markers during manual UI
    validation.
- Validation:
  - `conda run -n p313 pytest -q tests/test_module_browser_api_routes.py tests/test_module_browser_widget_mode_ops.py`;
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 pytest -q tests/test_module_browser_tracker.py -k "module_update_job_supports_comfyui_scope or install_requirements_for_modules_aggregates_results or install_comfyui_requirements_endpoint_helper"`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k seam_match`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.37.0` in `pyproject.toml` and `README.md`.

## 0.36.0 — 2026-05-08
- Added new `Silent Film Finish` video stylization node:
  - applies projection/print-era finishing such as monochrome toning, soft
    highlight rolloff, flicker, gate weave, softness, and grain;
  - supports optional `cadence_json` input from `Silent Film Cadence` so
    flicker and weave can lock to the same hand-cranked capture rhythm instead
    of being layered as unrelated noise;
  - exposes `finish_json.sync_mode` to show whether cadence synchronization is
    active or whether the node fell back to standalone finish behavior.
- Upgraded `Silent Film Cadence` blur quality and workflow defaults:
  - added `blur_mode=flow_integrated` for motion-compensated optical-flow
    shutter integration, reducing obvious double/triple-frame ghosting versus
    simple temporal averaging;
  - extended `cadence_json` with full group ids, fps values, and phase-bias
    data so downstream nodes can synchronize to the virtual hand-cranked
    capture intervals;
  - changed node defaults to the recommended 24 fps source -> 25 fps silent
    projection setup (`undercrank_projected_25fps`, `flow_integrated`, higher
    motion blur strength, updated finish defaults).
- Wired docs, registry, and smoke coverage for the second-stage silent-film workflow:
  - registered `Silent Film Finish` in the canonical node registry/UI metadata;
  - added README section and dedicated guide
    `guides/GUIDE_VIDEO_SILENT_FILM_FINISH.md`;
  - extended smoke tests to cover cadence undercrank, finish alpha
    preservation, cadence-locked finish sync, and `flow_integrated` blur mode.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.36.0` in `pyproject.toml` and `README.md`.

## 0.35.0 — 2026-05-08
- Added new `Silent Film Cadence` video stylization node:
  - accepts VHS/Comfy `IMAGE` frame batches and emulates silent-era cadence
    with target capture ranges such as `16-20 fps`, fps drift, and temporal
    exposure blur;
  - includes two playback modes: `preserve_duration_25fps` for cadence-only
    stylization and `undercrank_projected_25fps` for historically accurate
    speed-up when the shortened batch is assembled back at `25 fps`;
  - returns diagnostic `cadence_json` with effective fps, interval/group
    sizes, durations, and preview values for drift/bias behavior.
- Hardened the new cadence node after real workflow feedback:
  - fixed ComfyUI output contract for the `STRING` socket so the node no
    longer appears to run successfully with an empty result;
  - replaced exact intra-group duplicate frames with phase-shifted temporal
    samples when blur is enabled, reducing the "double frame" look;
  - fixed an `IndexError` in `undercrank_projected_25fps` by deriving blur
    span from interval duration instead of preserve-mode group maps.
- Wired release docs and validation around the new node:
  - registered the node in canonical registry/UI metadata;
  - added README section and dedicated guide
    `guides/GUIDE_VIDEO_SILENT_FILM_CADENCE.md`;
  - extended smoke coverage for both preserve-duration and undercrank modes.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.35.0` in `pyproject.toml` and `README.md`.

## 0.34.5 — 2026-05-07
- Reduced project-side Pillow deprecation risk ahead of Pillow 13:
  - removed deprecated `mode=` usage from `Image.fromarray(...)` in the
    affected production paths for adaptive descreening, video inpaint frame
    export, and DZI tensor-to-PIL conversion;
  - synchronized the touched NYPL/DZI smoke tests and the helper script under
    `guides/assets/color_match_examples/` to the same Pillow-safe pattern;
  - full test warning count is now down to the environment-level
    `PytestCacheWarning` caused by a read-only `.pytest_cache`.
- Validation:
  - `conda run -n p313 pytest -q`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.34.5` in `pyproject.toml` and `README.md`.

## 0.34.4 — 2026-05-03
- Added conservative NYPL tile-size clamping in `Download IIIF Image`:
  - when NYPL `info.json` advertises large tiles (for example `1024x1024`),
    the node now clamps tile assembly to a safe maximum of `512x512`;
  - this avoids `403` failures on NYPL region requests that are formally
    advertised but not actually accepted by the backend;
  - dedicated NYPL coverage was added for the tile-profile clamp.
- Validation:
  - `conda run -n p313 pytest -q tests/test_iiif_nypl.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.34.4` in `pyproject.toml` and `README.md`.

## 0.34.3 — 2026-05-03
- Hardened NYPL IIIF tile cache handling in `Download IIIF Image`:
  - invalid cached tile bytes are now discarded automatically and refetched
    instead of being reused and failing later in Pillow decode;
  - raw tile responses are checked with a lightweight image-signature guard
    before being stored in the persistent cache;
  - smoke coverage now includes invalid-cache replacement for IIIF tile fetches.
- Validation:
  - `conda run -n p313 pytest -q tests/test_iiif_nypl.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.34.3` in `pyproject.toml` and `README.md`.

## 0.34.2 — 2026-05-03
- Hardened NYPL IIIF tile delivery in `Download IIIF Image`:
  - NYPL sessions now send browser-like image request headers
    (`Referer`, `Accept`, `Accept-Language`) for `iiif.nypl.org` tile fetches;
  - tile responses are now checked for non-image `content-type` before decode,
    so HTML/service pages fail earlier with a clearer error;
  - synchronized NYPL troubleshooting notes in `README.md` and
    `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`.
- Validation:
  - `conda run -n p313 pytest -q tests/test_iiif_nypl.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.34.2` in `pyproject.toml` and `README.md`.

## 0.34.1 — 2026-05-03
- Simplified NYPL support in `Download IIIF Image`:
  - for `site = The New York Public Library (NYPL) Digital Collections`,
    `source_url` can now be a plain NYPL image id such as `57538105`,
    `NIJINSKY_2032V`, or `NIJINSKY_2033V`;
  - the node builds `https://iiif.nypl.org/iiif/3/<image_id>` directly from
    that value;
  - the temporary separate `nypl_image_id` UI input was removed again to keep
    NYPL on the same single-field flow as other IIIF sources.
- Fixed NYPL tile assembly for services that expect explicit tile size in the
  region request path:
  - tile URLs now use `region/<w>,<h>/0/default.<fmt>` instead of
    `region/max/0/default.<fmt>`;
  - this aligns the node with working NYPL region-request URLs and avoids
    non-image responses on some NYPL items.
- NYPL docs were synchronized across:
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
- Validation:
  - `conda run -n p313 pytest -q tests/test_iiif_nypl.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.34.1` in `pyproject.toml` and `README.md`.

## 0.34.0 — 2026-04-23
- Minor-version release for the safe stabilization pass and follow-up baseline
  tooling.
- Restored legacy `utils.module_browser` import compatibility through thin shim
  modules so older import paths keep working after the internal refactor.
- Removed import-time Comfy/CUDA side effects from the targeted nodes:
  - `nodes/image_prepare.py`
  - `nodes/video_inpaint.py`
  - GPU availability remains a runtime requirement instead of breaking package
    import in headless or CI contexts.
- Improved node-loader diagnostics in `nodes/__init__.py`:
  - failure records now distinguish `runtime_unavailable` from `import_error`;
  - public node mappings remain unchanged.
- Added follow-up safety coverage and release tooling:
  - dedicated smoke test for `image_prepare`;
  - `scripts/save_baseline.sh` helper with configurable output path;
  - `make save-baseline` target;
  - refreshed `baseline.json`.
- Plan execution record:
  - [SAFE_STABILIZATION_EXECUTION_TRACKER_RU.md](refactoring_plan/SAFE_STABILIZATION_EXECUTION_TRACKER_RU.md)
- Validation:
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 node --check web/widget_visibility_profiles.js`;
  - `conda run -n p313 pytest -q`.
- Version updated to `0.34.0` in `pyproject.toml` and `README.md`.

## 0.33.6 — 2026-04-22
- Renamed the NYPL IIIF site option in `Download IIIF Image` to a natural
  user-facing label:
  - `The New York Public Library (NYPL) Digital Collections`
- Synchronized the new label across:
  - `nodes/image_download_iiif.py`
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
  - `tests/test_smoke_nodes.py`
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.6` in `pyproject.toml` and `README.md`.

## 0.33.5 — 2026-04-22
- Added explicit NYPL support to `Download IIIF Image`:
  - new `site = NYPL IIIF Info URL`;
  - intended for direct `iiif.nypl.org/.../info.json` or service URLs;
  - keeps the existing generic resolver path, but exposes a clearer UX for
    NYPL sources discovered from the viewer.
- Updated IIIF docs to distinguish NYPL item pages from direct IIIF sources:
  - `digitalcollections.nypl.org/items/...` pages are not recommended as direct
    node input because the site is fronted by Imperva/Incapsula;
  - `iiif.nypl.org/.../info.json` is the stable input path for high-resolution
    downloads.
- Extended smoke coverage:
  - IIIF site dropdown now includes `NYPL IIIF Info URL`;
  - direct resolver contract is covered for
    `https://iiif.nypl.org/iiif/3/57538105/info.json`.
- Updated:
  - `nodes/image_download_iiif.py`
  - `nodes/node_registry.py`
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
  - `tests/test_smoke_nodes.py`
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.5` in `pyproject.toml` and `README.md`.

## 0.33.4 — 2026-04-07
- Improved `Download IIIF Image` support for `Gallica BnF Object Page`:
  - fixed Gallica ARK parsing so `.r=...`, `.zoom`, query strings, and similar
    suffixes no longer pollute the IIIF service URL;
  - cleaned `source_url_slug` naming for Gallica so fallback filenames use the
    stable ARK id instead of technical URL tails;
  - improved `title_or_slug` naming for Gallica:
    - first prefer IIIF `manifest.json` metadata;
    - then prefer Gallica page title blocks such as `.title`;
    - ignore generic page titles like `Gallica`;
    - only then fall back to the human-readable `.r=...` query fragment.
- Filename sanitization now strips both square and round brackets from extracted
  page titles before saving the final image.
- Updated:
  - `nodes/image_download_iiif.py`
  - `tests/test_smoke_nodes.py`
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
  - `nodes/node_registry.py`
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.4` in `pyproject.toml` and `README.md`.

## 0.33.3 — 2026-04-03
- Updated the safe default for `Remove Static Watermark from Video`:
  - `crop_padding` default changed from `16` to `64`;
  - tooltip and docs now describe `64` as the practical safe-start value for
    watermark removal, based on the observed black masked-area failure with
    smaller crop context.
- Updated:
  - `nodes/video_inpaint.py`
  - `README.md`
  - `guides/GUIDE_VIDEO_INPAINT.md`
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "video_inpaint or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.3` in `pyproject.toml` and `README.md`.

## 0.33.2 — 2026-04-03
- Hardened `Remove Static Watermark from Video` against black masked-area failures:
  - changed the safe default to `fp16=disable`;
  - added automatic FP32 retry when an FP16 chunk collapses to an almost-black
    result inside the masked region;
  - fixed the masked-area validation path to handle resized ProPainter
    `process_size` outputs without shape mismatches.
- Added a second safety fallback for ProPainter output staging:
  - when `feature_propagation` returns an invalid dark patch but
    `image_propagation` already contains a sane fill, the node now uses the
    `image_propagation` result for that chunk.
- Improved patch compositing to reduce the visible circular seam:
  - added `feather_radius`;
  - patch alpha is now feathered before saving RGBA patch frames, which also
    improves `preview_image` and `write_fullframes` compositing.
- Updated docs for the new recommended workflow:
  - `patch_*.png` are explicitly documented as RGBA patches, not final frames;
  - `write_fullframes=true` is now the recommended way to inspect final output;
  - documented the practical guidance that black patch failures often require
    larger `crop_padding` (commonly `64+`) and that seam visibility is addressed
    with `feather_radius`.
- Added smoke coverage for:
  - safe default `fp16=disable`;
  - resized masked-area validation;
  - fallback from invalid `feature_propagation` output to
    `image_propagation`;
  - non-binary feathered alpha edges.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "video_inpaint or node_ui_metadata_compat"`;
  - `conda run -n p313 python -m py_compile nodes/video_inpaint.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.2` in `pyproject.toml` and `README.md`.

## 0.33.1 — 2026-04-02
- Marked the old all-in-one descreen node as deprecated everywhere it appears:
  - picker display name changed to `Descreen By Adaptive Scale (Deprecated)`;
  - node description in registry now explicitly states deprecated legacy status;
  - README and the legacy guide now warn that new workflows should use the
    split chain instead.
- Updated documentation links/checks to match the new deprecated heading:
  - `README.md`
  - `guides/GUIDE_IMAGE_DESCREEN_ADAPTIVE.md`
  - `nodes/node_registry.py`
  - `utils/docs_check.py`
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "descreen or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.1` in `pyproject.toml` and `README.md`.

## 0.33.0 — 2026-04-02
- Split the practical descreen workflow into separate nodes with clearer responsibilities:
  - added `Estimate Raster Period` for ROI-based period measurement and
    predicted base scale calculation;
  - added `Descreen Scale Preview` for visual scale-sheet generation from a
    chosen base percent upward;
  - kept the old all-in-one workflow as `Descreen By Adaptive Scale (Legacy)`.
- Updated descreen node registration and UI metadata:
  - the legacy node is now explicitly marked as legacy in the picker;
  - new estimate/preview nodes have dedicated descriptions and output tooltips.
- Added documentation for the new split workflow:
  - `guides/GUIDE_IMAGE_RASTER_PERIOD_ESTIMATE.md`
  - `guides/GUIDE_IMAGE_DESCREEN_SCALE_PREVIEW.md`
- Updated existing descreen docs:
  - `README.md`
  - `guides/GUIDE_IMAGE_DESCREEN_ADAPTIVE.md`
- Tests:
  - added smoke coverage for `Estimate Raster Period`;
  - added smoke coverage for `Descreen Scale Preview`;
  - updated node registration compatibility coverage.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_descreen_adaptive.py nodes/node_registry.py tests/test_smoke_nodes.py utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "descreen or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.33.0` in `pyproject.toml` and `README.md`.

## 0.32.0 — 2026-04-02
- Reworked the descreen workflow around the practical scale-based path:
  - `Descreen By Adaptive Scale` now supports explicit resampler choice
    (`lanczos` / `bicubic`);
  - the node now returns a labeled `scale_sheet` built from the calculated
    base scale upward, instead of the older compare preview;
  - `scale_sheet` no longer includes the original card and now supports an
    independent preview zone via `sheet_zone_mode` and `sheet_zone_*`;
  - percentage labels in the scale sheet were made more readable.
- Simplified `Apply Descreen Percent` into a final downscale-only node:
  - removed the old pre-blur option;
  - added `resample_mode`;
  - the node now outputs the reduced final image and no longer upscales back
    to the original size.
- Removed experimental descreen nodes that did not justify their maintenance:
  - `Descreen FFT Notch`;
  - `Descreen Tonal Hybrid`.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_IMAGE_DESCREEN_ADAPTIVE.md`
  - `guides/GUIDE_IMAGE_DESCREEN_APPLY.md`
- Docs removed:
  - `guides/GUIDE_IMAGE_DESCREEN_FFT.md`
  - `guides/GUIDE_IMAGE_DESCREEN_TONAL_HYBRID.md`
- Tests:
  - updated smoke coverage for the new adaptive `scale_sheet` contract and
    the new downscale-only `Apply Descreen Percent`;
  - removed smoke coverage for deleted experimental descreen nodes.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_descreen_adaptive.py nodes/node_registry.py tests/test_smoke_nodes.py utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "descreen or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.32.0` in `pyproject.toml` and `README.md`.

## 0.31.3 — 2026-03-27
- Improved `Download DZI Tiles Image` title-based filename stability:
  - when `filename_mode=title_or_mw`, the saved filename now appends the stable
    object id (`mw...` / `nla.obj-...`) to the human-readable title;
  - typical result is now `title + stable_id`, for example
    `Anna_Pavlova_as_the_Dying_swan_Melbourne_1926_nla.obj-138204672.png`;
  - this aligns DZI single-save naming with the stable-id strategy already used
    for IIIF downloads.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
- Tests:
  - updated DZI smoke coverage for the new single-save title naming contract.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "dzi"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.31.3` in `pyproject.toml` and `README.md`.

## 0.31.2 — 2026-03-27
- Improved `Download IIIF Image` filename stability:
  - saved filenames now append a stable identifier extracted from
    `source_url` and/or `service_url` when available;
  - typical result is now `slug_or_title + stable_id`, for example
    `anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house_object-443337.jpg`;
  - this reduces cross-item collisions even before numeric suffix fallback.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
- Tests:
  - updated IIIF smoke coverage for new filename contract and collision suffixes.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.31.2` in `pyproject.toml` and `README.md`.

## 0.31.1 — 2026-03-27
- Improved non-destructive save behavior for image download nodes:
  - `Download DZI Tiles Image` no longer overwrites an existing file when
    `output_dir` is set;
  - `Download IIIF Image` no longer overwrites an existing file when
    `output_dir` is set;
  - instead, both single-image nodes now append numeric suffixes `_2`, `_3`,
    ... to the filename automatically.
- Adjusted batch DZI unique-save naming:
  - `overwrite_mode=unique` now uses human-readable suffixes `_2`, `_3`, ...
    instead of zero-padded `_001`, `_002`, ...
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
- Tests:
  - added smoke coverage for filename collision handling in both single-save
    DZI and IIIF flows.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "dzi or iiif"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.31.1` in `pyproject.toml` and `README.md`.

## 0.31.0 — 2026-03-27
- Added new companion node `Apply Descreen Percent`:
  - applies a known descreen percent to a full image or batch without
    repeating FFT analysis;
  - designed for production flow where `recommended_percent` is measured once
    and then reused for a whole series of frames;
  - returns processed image, applied percent, and fixed-mode analysis JSON.
- Updated descreen docs and registry:
  - `README.md`
  - `guides/GUIDE_IMAGE_DESCREEN_APPLY.md`
  - `nodes/node_registry.py`
  - `utils/docs_check.py`
- Tests:
  - added smoke coverage for `ImageDescreenApplyPercent`;
  - updated node registration compatibility coverage.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_descreen_adaptive.py tests/test_smoke_nodes.py utils/docs_check.py nodes/node_registry.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "descreen or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.31.0` in `pyproject.toml` and `README.md`.

## 0.30.0 — 2026-03-27
- Added new node `Descreen By Adaptive Scale`:
  - estimates halftone screen period from a selected ROI via FFT;
  - searches adaptive downscale candidates in a configurable percent range;
  - chooses the best compromise between screen suppression and structure retention;
  - returns processed image, side-by-side ROI preview, recommended percent,
    estimated period, and diagnostic JSON.
- Added node docs and guide:
  - `README.md`
  - `guides/GUIDE_IMAGE_DESCREEN_ADAPTIVE.md`
- Updated registry and docs contracts:
  - `nodes/node_registry.py`
  - `utils/docs_check.py`
- Tests:
  - added smoke coverage for `ImageDescreenAdaptiveScale` output contract and
    node registration visibility.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_descreen_adaptive.py tests/test_smoke_nodes.py utils/docs_check.py nodes/node_registry.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "descreen or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.30.0` in `pyproject.toml` and `README.md`.

## 0.29.4 — 2026-03-27
- `Download IIIF Image`:
  - added tile recovery cache for `delivery_mode=tile_assemble_full`;
  - downloaded IIIF tile bytes are cached on disk so interrupted/failed runs can
    resume without re-downloading completed tiles;
  - cache is scoped per assembled image rather than shared blindly;
  - after a fully successful tile assembly, the cache for that specific image is
    removed automatically;
  - `delivery` metadata now includes cache diagnostics:
    `cache_dir`, `cache_hits`, `cache_misses`, `cache_stores`, `cache_cleared`.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
- Tests:
  - added smoke coverage for IIIF tile cache reuse and automatic cache cleanup
    after successful assembly.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.29.4` in `pyproject.toml` and `README.md`.

## 0.29.3 — 2026-03-27
- `Download IIIF Image`:
  - improved network resilience for `delivery_mode=tile_assemble_full`;
  - IIIF requests now reuse one shared `requests.Session` for the whole run;
  - added retry/backoff for transient network failures:
    `ReadTimeout`, `ConnectTimeout`, `ConnectionError`, `SSLError`;
  - this prevents a single transient timeout during tile assembly from
    aborting the whole London Museum / IIIF full-resolution download too
    aggressively.
- Tests:
  - added smoke coverage for IIIF HTTP retry after transient timeout.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.29.3` in `pyproject.toml` and `README.md`.

## 0.29.2 — 2026-03-27
- `Download DZI Tiles Image`:
  - added optional `output_dir` and `output_extension` save-to-disk flow;
  - when `output_dir` is empty, the node only returns `IMAGE`;
  - when `output_dir` is set, the final assembled image is saved to disk;
  - added `filename_mode = mw | title_or_mw`;
  - `title_or_mw` tries to resolve a human-readable page title and falls back
    to `mw` when the title is unavailable.
- `Download DZI Tiles Batch Save`:
  - `filename_template` now supports `{title}`;
  - `{title}` tries to resolve a human-readable page title and falls back to
    `mw`.
- `Download IIIF Image`:
  - added `filename_mode = source_url_slug | title_or_slug`;
  - `title_or_slug` tries to resolve the page title from `source_url` and
    falls back to the URL slug.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
- Tests:
  - added smoke coverage for DZI single save-to-disk, DZI title-derived
    filenames, batch `{title}` template handling, and IIIF title-derived
    filenames.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "dzi or iiif"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.29.2` in `pyproject.toml` and `README.md`.

## 0.29.1 — 2026-03-27
- `Download IIIF Image` now supports optional save-to-disk output via
  `output_dir`.
- `nodes/image_download_iiif.py`:
  - added `output_dir` parameter;
  - when `output_dir` is empty, the node does not save any file and only
    returns the `IMAGE` output;
  - when `output_dir` is set, the final image is written to disk and the path
    is returned in `info_json.saved_path`;
  - output filename is derived from the last meaningful segment of the input
    `source_url`, for example:
    `.../anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house/`
    -> `anna-pavlova-posed-in-day-dress-by-urn-in-the-garden-of-ivy-house.jpg`.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
- Tests:
  - added smoke coverage for `output_dir` save behavior and URL-derived
    filename generation.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.29.1` in `pyproject.toml` and `README.md`.

## 0.29.0 — 2026-03-27
- Added new node `Download IIIF Image` for IIIF Image API sources, with
  first-class support for London Museum object pages.
- `nodes/image_download_iiif.py`:
  - resolves IIIF service URLs from either London Museum object pages or
    direct/generic IIIF service URLs;
  - downloads images via IIIF Image API single-request mode;
  - exposes `info_json` with `service_url`, `info_url`, `image_url`, and IIIF
    metadata;
  - now reports `source.width/height` separately from delivered image size;
  - now reports service-side limits in `limits`, including `maxArea`-derived
    diagnostics when single-request delivery is downscaled by server policy;
  - added `delivery_mode=tile_assemble_full` to assemble full-resolution output
    from IIIF tiles when the service exposes `tiles` and `scaleFactor=1`.
- London Museum verification:
  - object pages embed IIIF service URLs in `data-clover-image-viewer`
    `data-src`;
  - `.../info.json` works as standard IIIF Image API 3 metadata;
  - single-request `full/max` is limited by service policy (`maxArea`), while
    tile assembly can be used for full-resolution output.
- `nodes/node_registry.py`:
  - registered `ImageDownloadIIIFImage`;
  - added UI metadata, aliases, and output tooltips.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_IIIF_IMAGE_DOWNLOAD.md`
  - README node list now includes `Download IIIF Image`.
- Tests/docs contracts updated:
  - added IIIF node contract smoke test;
  - added IIIF tile-assembly contract smoke test;
  - extended `utils/docs_check.py` for the new node.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "iiif or node_ui_metadata_compat"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_iiif.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.29.0` in `pyproject.toml` and `README.md`.

## 0.28.1 — 2026-03-26
- `Download DZI Tiles Image` and `Download DZI Tiles Batch Save` now respect
  `Cancel current run` in ComfyUI.
- `nodes/image_download_dzi_tiles.py`:
  - added interrupt checks at start, during proxy/transport preflight,
    DZI probing, tile download loops, and batch save flow;
  - batch mode no longer swallows `InterruptProcessingException` as a normal
    per-item failure;
  - interrupt events now log explicit cancellation messages for single and
    batch runs.
- `utils/interrupt.py`:
  - added `is_interrupt_exception(...)` helper for safe interrupt propagation.
- Tests:
  - added smoke coverage for single-node interrupt propagation;
  - added smoke coverage for batch interrupt propagation.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "dzi"`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py utils/interrupt.py tests/test_smoke_nodes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.28.1` in `pyproject.toml` and `README.md`.

## 0.28.0 — 2026-03-26
- Added new node `Search Trove Image IDs` for best-effort discovery of
  `nla.obj-...` identifiers from Trove public image search.
- Added new node `Download DZI Tiles Batch Save` for multi-ID DZI downloads
  with direct save-to-disk workflow and manifest output.
- `nodes/trove_search_ids.py`:
  - added headless Chrome-based Trove search flow;
  - returns newline-separated ids, diagnostics JSON, and count;
  - designed as a non-API best-effort mode because Trove search without an API
    key is SPA/anti-bot constrained.
- `nodes/image_download_dzi_tiles.py`:
  - added batch helpers for parsing `ids_text`, filename templating, overwrite
    policies, output path resolution, and image saving;
  - added `ImageDownloadDZITilesBatchSave` with:
    `output_dir`, `output_extension`, `filename_template`, `overwrite_mode`,
    `continue_on_error`, and `save_mode`;
  - batch mode now writes manifest JSON and saved-paths JSON outputs for
    downstream automation.
- `nodes/node_registry.py`:
  - registered `SearchTroveImageIDs`;
  - registered `ImageDownloadDZITilesBatchSave`;
  - added node metadata, aliases, and output tooltips for both nodes.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
  - `guides/GUIDE_TROVE_SEARCH_IDS.md`
- Tests/docs contracts updated:
  - added Trove search node contract smoke test;
  - added batch DZI save smoke test;
  - extended `utils/docs_check.py` for both new nodes.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k "dzi or trove_search_ids_node_contract or node_ui_metadata_compat"`;
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 python -m py_compile nodes/trove_search_ids.py nodes/node_registry.py utils/docs_check.py`.
- Version updated to `0.28.0` in `pyproject.toml` and `README.md`.

## 0.27.17 — 2026-03-26
- `Download DZI Tiles Image`: moved provider URL logic to config-driven
  templates so new archives can be added through JSON without Python changes.
- `config/dzi_sites.json`:
  - added `object_url_template`;
  - added `dzi_url_template`;
  - added `tile_url_template`;
  - kept `mw_prefix`, `default_mw`, `default_level`, and `url_scheme`.
- `nodes/image_download_dzi_tiles.py`:
  - added generic DZI template renderer with placeholders:
    `{base_url}`, `{mw}`, `{level}`, `{x}`, `{y}`, `{ext}`;
  - `_build_dzi_source_urls(...)` now prefers config templates over hardcoded
    provider branches;
  - `_tile_url(...)` now supports `template` mode;
  - probe/download paths now pass template context (`base_url`, `mw`) through
    the full tile-fetch pipeline;
  - config loader no longer restricts providers to only `npg` / `nla`.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
- Tests:
  - added smoke coverage for a fully template-driven custom provider in
    `tests/test_smoke_nodes.py`.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.17` in `pyproject.toml` and `README.md`.

## 0.27.16 — 2026-03-26
- `Download DZI Tiles Image`: switched site selection to JSON-backed catalog.
- Added `config/dzi_sites.json` with site metadata:
  - display name;
  - base URL;
  - provider;
  - `default_mw`;
  - `default_level`;
  - `mw_prefix`;
  - reference `url_scheme`.
- `nodes/image_download_dzi_tiles.py`:
  - replaced manual `base_url` UI input with `site` dropdown populated from
    `config/dzi_sites.json`;
  - `mw=""` now falls back to site `default_mw`;
  - `level=-1` now falls back to site `default_level`;
  - digits-only `mw` input is now normalized per site prefix:
    - NPG: `207134` -> `mw207134`
    - NLA: `138204672` -> `nla.obj-138204672`
  - full object ids are still accepted unchanged;
  - preserved compatibility for legacy direct-URL callers in tests/code paths.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
- Tests:
  - extended DZI smoke coverage for JSON dropdown config and numeric `mw`
    normalization in `tests/test_smoke_nodes.py`.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.16` in `pyproject.toml` and `README.md`.

## 0.27.15 — 2026-03-26
- `Download DZI Tiles Image`: added provider-aware DZI URL support for
  `nla.gov.au` (National Library of Australia) in addition to existing
  `collectionimages.npg.org.uk` support.
- `nodes/image_download_dzi_tiles.py`:
  - added `provider` input: `auto` / `npg` / `nla`;
  - introduced provider-aware URL builder for DZI XML and tile endpoints;
  - `auto` now detects NLA by `base_url`/`mw` (`nla.obj-...`) and uses
    query-based tile URLs:
    `https://nla.gov.au/<mw>/dzi?tile=<level>/<x>_<y>.<ext>`;
  - existing NPG file-based URL scheme remains unchanged.
- Docs updated:
  - `README.md`
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`
- Tests:
  - added NLA URL builder and mocked NLA provider smoke coverage in
    `tests/test_smoke_nodes.py`.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi`;
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`.
- Version updated to `0.27.15` in `pyproject.toml` and `README.md`.

## 0.27.14 — 2026-03-10
- `Node Picker`: fixed widget freeze after `Refresh Custom Nodes Info`.
- Root cause:
  - backend refresh job could finish with `phase="done"` while leaving
    `refresh.running=true`, so frontend polling never observed terminal state
    and kept the widget locked until full page reload.
- Backend fix:
  - `utils/module_browser_api/handlers_refresh.py`
  - successful refresh completion now explicitly stores:
    - `running=false`
    - `phase="done"`
    - `message="done"`
    - final refresh counters and `refreshed_at`
- Frontend hardening kept in place:
  - busy-state reset path for manual custom refresh now force-clears stale UI
    locks even if picker lifecycle changes during async completion;
  - successful global refresh now also clears frontend
    `updatedModulesSession` markers so stale green check marks do not persist.
- Tests:
  - added `tests/test_module_browser_refresh_handler.py` to verify refresh job
    transitions into terminal `running=false` state;
  - updated `tests/js/test_module_node_picker_frontend_behavior.mjs` for
    forced busy reset and custom refresh finalization behavior.
- Validation:
  - `conda run -n p313 pytest -q tests/test_module_browser_refresh_handler.py`;
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.14` in `pyproject.toml` and `README.md`.

## 0.27.13 — 2026-03-09
- `Node Picker`: corrected visible-canvas center insertion logic.
- `web/ui/module_node_picker_node_factory.js`:
  - viewport center is now derived from current `ds.offset`, `ds.scale`, and
    canvas size, matching the actual visible area on screen;
  - kept fallback to `visible_area` only when direct viewport parameters are
    unavailable.
- `web/ui/module_node_picker_renderers.js`:
  - restored `centerOnNode(...)` after fixing position math, so inserted node is
    immediately visible without jumping to an incorrect location.
- Frontend regression coverage updated:
  - `tests/js/test_module_node_picker_frontend_behavior.mjs`
  - center-placement test now covers `offset + scale + canvas size` path.
- Validation:
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 node --check web/ui/module_node_picker_node_factory.js`;
  - `conda run -n p313 node --check web/ui/module_node_picker_renderers.js`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.13` in `pyproject.toml` and `README.md`.

## 0.27.12 — 2026-03-09
- `Node Picker`: fixed node insertion position so selected nodes land in the
  center of the visible canvas instead of an offset graph position.
- `web/ui/module_node_picker_node_factory.js`:
  - normalized visible-area center calculation in graph space;
  - removed incorrect `devicePixelRatio` adjustment from `ds.visible_area`
    center computation.
- `web/ui/module_node_picker_renderers.js`:
  - after successful insertion, selected node is now explicitly centered and
    focused in canvas.
- Frontend regression coverage updated:
  - `tests/js/test_module_node_picker_frontend_behavior.mjs`
  - added center-placement assertion for visible canvas insertion.
- Validation:
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 node --check web/ui/module_node_picker_node_factory.js`;
  - `conda run -n p313 node --check web/ui/module_node_picker_renderers.js`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.12` in `pyproject.toml` and `README.md`.

## 0.27.11 — 2026-03-06
- Refactoring continuity: completed the last `Phase 4 Remaining` deliverable
  for frontend behavioral checks.
- Added lightweight frontend behavior smoke test:
  - `tests/js/test_module_node_picker_frontend_behavior.mjs`
  - covers tab-relay runtime state transitions and refresh/update progress
    polling behavior in pure JS modules.
- CI update:
  - `.github/workflows/ci-lite.yml` now runs frontend behavioral check via
    `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`.
- Updated refactoring status docs:
  - `refactoring_plan/PLAN_REFACTORING_ADOPTED.md`
  - `refactoring_plan/PLAN_REFACTORING_ADOPTED_RU.md`
  - marked "frontend behavioral checks" as completed and
    `Phase 4 Remaining` status as completed.
- Validation:
  - `conda run -n p313 node tests/js/test_module_node_picker_frontend_behavior.mjs`;
  - `conda run -n p313 pytest -q tests/test_module_browser_api_routes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.11` in `pyproject.toml` and `README.md`.

## 0.27.10 — 2026-03-06
- Refactoring continuity: progressed `Phase 4 Remaining` quality coverage.
- Added critical-route integration tests for extracted routes module:
  - `tests/test_module_browser_api_routes.py`
  - validates idempotent route registration and critical endpoint behavior:
    `module_refresh`, `module_update` (info-only rejection), `module_info`,
    `comfyui_info`.
- Updated refactoring status docs:
  - `refactoring_plan/PLAN_REFACTORING_ADOPTED.md`
  - `refactoring_plan/PLAN_REFACTORING_ADOPTED_RU.md`
  - marked "integration tests for critical routes" as completed.
- Validation:
  - `conda run -n p313 pytest -q tests/test_module_browser_api_routes.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.10` in `pyproject.toml` and `README.md`.

## 0.27.9 — 2026-03-06
- Refactoring continuity: Phase 4 Remaining quality task progressed.
- Added lightweight CI workflow:
  - `.github/workflows/ci-lite.yml`
  - Conda `p313` setup + syntax checks + docs check + module-browser contract tests.
- Updated refactoring plan status:
  - `refactoring_plan/PLAN_REFACTORING_ADOPTED.md`
  - `refactoring_plan/PLAN_REFACTORING_ADOPTED_RU.md`
  - marked CI deliverable in "Phase 4 Remaining — Quality and Coverage" as completed.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py utils/module_node_browser_api.py`;
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_jobs.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_tracker.py -k "module_update_job_supports_comfyui_scope or resolve_update_targets_all_filters_modules"`;
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py -k "refresh_status_contract or update_status_contract or runtime_warmup_status_contract or node_catalog_group_payload_contract or module_list_payload_filter_contract or module_nodes_payload_contract"`.
- Version updated to `0.27.9` in `pyproject.toml` and `README.md`.

## 0.27.8 — 2026-03-06
- `Download DZI Tiles Image`: UI simplified for everyday use.
- `nodes/image_download_dzi_tiles.py`:
  - removed optional UI inputs `pac_url`, `cookie`, `disable_env_proxy`, `referer`;
  - added explicit `tile_extension` selector (`jpg|jpeg|png|webp`);
  - first-tile probing now uses selected extension only (no full extension sweep);
  - kept automatic route/proxy discovery when `proxy_url` is empty
    (env + system proxy settings + PAC/env PAC + common local proxy endpoints).
- Documentation updated:
  - `README.md` section for `Download DZI Tiles Image`;
  - `guides/GUIDE_DZI_TILES_DOWNLOAD.md`.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi_tiles`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.8` in `pyproject.toml` and `README.md`.

## 0.27.7 — 2026-03-06
- `Download DZI Tiles Image`: added explicit transport error diagnostics for
  `status=0` cases (network/proxy failures without HTTP response).
- `nodes/image_download_dzi_tiles.py`:
  - deduplicated console logging for fetch exceptions by transport
    (`requests/urllib/curl/cloudscraper`);
  - first-tile error now includes proxy hint when `proxy_url` is set and all
    attempts return status `0`.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi_tiles`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.7` in `pyproject.toml` and `README.md`.

## 0.27.6 — 2026-03-06
- `Download DZI Tiles Image`: fixed proxy propagation for `cloudscraper` transport.
- `nodes/image_download_dzi_tiles.py`:
  - `proxy_url` from session is now explicitly applied to cloudscraper client
    (`scraper.proxies`), so `transport=cloudscraper` respects node proxy input.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi_tiles`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.6` in `pyproject.toml` and `README.md`.

## 0.27.5 — 2026-03-06
- `Download DZI Tiles Image`: added explicit network-control inputs for VPN/proxy
  environments where browser works but backend HTTP gets `403`.
- New optional node inputs in `nodes/image_download_dzi_tiles.py`:
  - `transport`: `auto|requests|cloudscraper|urllib|curl`
  - `proxy_url`: explicit proxy URL
  - `cookie`: raw Cookie header (e.g. `cf_clearance=...`)
  - `disable_env_proxy`: ignore env HTTP(S)_PROXY for session
  - `referer`: explicit Referer override
- Behavior updates:
  - retry matrix now supports referer variants + selected transport strategy;
  - preserves prior transport/proxy fallbacks and adds explicit user control;
  - compatibility wrappers keep smoke tests stable when helpers are monkeypatched.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi_tiles`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.5` in `pyproject.toml` and `README.md`.

## 0.27.4 — 2026-03-06
- `Download DZI Tiles Image`: additional anti-403 hardening for environments
  where tile requests behave differently than browser.
- `nodes/image_download_dzi_tiles.py`:
  - added `cloudscraper` transport fallback in first-tile probing chain;
  - first-tile probing now tries multiple referers
    (`.../zoom/<mw>/`, origin root, NPG fallback);
  - improved compatibility wrappers for monkeypatched helper signatures used by
    smoke tests (`_new_session`, `_download_tile`, `_probe_axis_count`);
  - keeps previously added `requests/urllib/curl` fallback and proxy-bypass
    retry path.
- Validation:
  - `conda run -n p313 pytest -q tests/test_smoke_nodes.py -k dzi_tiles`;
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.4` in `pyproject.toml` and `README.md`.

## 0.27.3 — 2026-03-06
- `Download DZI Tiles Image`: changed probe order to avoid hard dependency on
  `.dzi` metadata availability.
- Fix in `nodes/image_download_dzi_tiles.py`:
  - first tile probe is now performed before DZI metadata request;
  - tile extension candidates are probed directly (`jpg/jpeg/png/webp`) even if
    `zoomXML.dzi` is blocked;
  - DZI metadata is parsed only after working tile transport/extension is found,
    with existing transport fallbacks preserved.
- Why:
  - some hosts may return `403` for `zoomXML.dzi` while still serving tile
    images; previous order could fail too early in that scenario.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`.
- Version updated to `0.27.3` in `pyproject.toml` and `README.md`.

## 0.27.2 — 2026-03-06
- `Download DZI Tiles Image`: added multi-transport HTTP fallback for strict
  network environments where `requests` gets `403`.
- `nodes/image_download_dzi_tiles.py` changes:
  - added transport backends: `requests`, `urllib`, `curl`;
  - first-tile probe now tries extension+transport matrix
    (`jpg/jpeg/png/webp` x transport);
  - selected working transport is reused for tile downloads and DZI metadata;
  - DZI parsing also has transport fallback path;
  - improved diagnostics in first-tile error message
    (`<ext>@<transport>:<status>`).
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`;
  - manual transport probe for
    `https://collectionimages.npg.org.uk/zoom/mw251004/zoomXML_files/11/0_0.jpg`
    confirms `200` for `requests/urllib/curl`.
- Version updated to `0.27.2` in `pyproject.toml` and `README.md`.

## 0.27.1 — 2026-03-06
- `Download DZI Tiles Image` robustness fix for false `403`/first-tile failures.
- Node improvements in `nodes/image_download_dzi_tiles.py`:
  - tile extension is now selected dynamically (DZI `Format` first, then
    fallback candidates `jpg/jpeg/png/webp`);
  - first tile probe now logs per-extension status and includes detailed status
    hints in the error message;
  - browser-like request headers were expanded (`Accept`, `Accept-Language`,
    `Origin`, `Sec-Fetch-*`) and `Referer` is now host-aware;
  - added retry path with proxy bypass (`session.trust_env = False`) when first
    tile returns `403`, to avoid environment-proxy false denials.
- Validation:
  - `conda run -n p313 python -m py_compile nodes/image_download_dzi_tiles.py`;
  - `conda run -n p313 python utils/docs_check.py`;
  - manual probe against
    `https://collectionimages.npg.org.uk/zoom/mw251004/zoomXML_files/11/0_0.jpg`
    confirms `200` on primary extension candidates.
- Version updated to `0.27.1` in `pyproject.toml` and `README.md`.

## 0.27.0 — 2026-03-05
- Refactoring roadmap execution: completed Phase 4 for Module Node Picker backend.
- Extracted PromptServer route wiring from `utils/module_node_browser_api.py`
  into dedicated module:
  - `utils/module_browser_api/routes.py`
- Converted backend facade to thin route-registration shim:
  - `utils/module_node_browser_api.py` now delegates route registration via
    `register_routes(...)` and preserves route constants/helper surface for
    compatibility.
- Added idempotent route-registration guard in route module:
  - duplicate imports/reloads no longer re-register identical endpoints.
- Planning/docs:
  - marked Phase 4 as completed in:
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API.md`,
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API_RU.md`.
- Validation:
  - `conda run -n p313 python -m py_compile utils/module_browser_api/routes.py utils/module_node_browser_api.py`;
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_jobs.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_tracker.py -k "module_update_job_supports_comfyui_scope or resolve_update_targets_all_filters_modules"`;
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py -k "refresh_status_contract or update_status_contract or runtime_warmup_status_contract or node_catalog_group_payload_contract or module_list_payload_filter_contract or module_nodes_payload_contract"`.
- Note:
  - full `tests/test_phase0_baseline.py` currently contains an unrelated frontend
    marker assertion mismatch in `test_frontend_tab_relay_contract_markers_exist`
    (`createNodeFromCatalogInfo` signature string check).
- Version updated to `0.27.0` in `pyproject.toml` and `README.md`.

## 0.26.0 — 2026-03-05
- Refactoring roadmap execution: completed Phase 3 for Module Node Picker backend.
- Extracted catalog/introspection API glue from `utils/module_node_browser_api.py`
  into dedicated modules:
  - `utils/module_browser_api/node_introspection.py`
  - `utils/module_browser_api/handlers_catalog.py`
- Rewired facade wrappers/routes to delegate while preserving existing contracts:
  - `_node_mappings(...)`
  - `_build_node_snapshots(...)`
  - `/alexz_tools/node_catalog`
  - `/alexz_tools/module_list`
  - `/alexz_tools/module_nodes`
- Compatibility:
  - restored legacy module-level aliases in `utils/module_browser/__init__.py`
    for direct imports used by tests/tooling:
    - `catalog_payload_ops`
    - `node_snapshot_ops`
- Planning/docs:
  - marked Phase 3 as completed in:
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API.md`,
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API_RU.md`.
- Validation:
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_catalog.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_catalog_payload_ops.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_node_snapshot_ops.py`;
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py -k "node_catalog_group_payload_contract or module_list_payload_filter_contract or module_nodes_payload_contract"`;
  - `conda run -n p313 pytest -q tests/test_module_browser_tracker.py -k "module_update_job_supports_comfyui_scope or resolve_update_targets_all_filters_modules"`.
- Version updated to `0.26.0` in `pyproject.toml` and `README.md`.

## 0.25.0 — 2026-03-05
- Refactoring roadmap execution: completed Phase 2 for Module Node Picker backend.
- Extracted refresh/update handler orchestration from
  `utils/module_node_browser_api.py` into dedicated modules:
  - `utils/module_browser_api/handlers_refresh.py`
  - `utils/module_browser_api/handlers_update.py`
- Rewired API facade wrappers to delegate refresh/update job startup logic to
  extracted handlers while keeping private wrapper API names stable:
  - `_start_refresh_job(...)`
  - `_start_module_update_job(...)`
- Preserved existing behavior:
  - route payload contracts unchanged;
  - thread names/locking semantics preserved;
  - compatibility wrappers for tests and internal callers remain in place.
- Planning/docs:
  - Phase status updated in:
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API.md`,
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API_RU.md`.
- Validation:
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_jobs.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_tracker.py -k "module_update_job_supports_comfyui_scope or resolve_update_targets_all_filters_modules"`;
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py -k "refresh_status_contract or update_status_contract or runtime_warmup_status_contract or start_runtime_warmup_noop_when_already_ready"`.
- Version updated to `0.25.0` in `pyproject.toml` and `README.md`.

## 0.24.0 — 2026-03-05
- Refactoring roadmap execution: completed Phase 1 for Module Node Picker backend.
- Extracted backend runtime state into dedicated module:
  - added `utils/module_browser_api/state.py`;
  - moved refresh/update locks, status payload templates, and runtime flags into
    `ModuleBrowserApiState` singleton.
- Extracted refresh/update console logging helpers:
  - added `utils/module_browser_api/logging_ops.py`;
  - moved update/refresh console-log mode handling and de-duplicated refresh log
    output into shared helpers.
- Rewired `utils/module_node_browser_api.py` to use extracted state/logging
  modules while preserving existing route contracts and behavior.
- Compatibility:
  - kept legacy warmup mirrors (`_LAZY_REFRESH_DONE`,
    `_RUNTIME_WARMUP_THREAD`) synchronized for baseline test compatibility.
- Planning/docs:
  - updated refactoring plan status in:
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API.md`,
    `refactoring_plan/PROPOSAL_REFACTOR_MODULE_NODE_BROWSER_API_RU.md`.
- Validation:
  - `conda run -n p313 python utils/docs_check.py`;
  - `conda run -n p313 pytest -q tests/test_module_browser_jobs.py tests/test_module_browser_tracker.py`;
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py -k "runtime_warmup_status_contract or start_runtime_warmup_noop_when_already_ready"`.
- Version updated to `0.24.0` in `pyproject.toml` and `README.md`.

## 0.23.5 — 2026-03-05
- Color Match To Reference: added experimental palette-transfer presets inspired
  by KJNodes/ColorMatchV2 method family, integrated into existing safe pipeline:
  - `reinhard_lab_fast` (fast Lab mean/std transfer),
  - `hm` (RGB histogram match),
  - `mkl` (MKL-like covariance transfer),
  - `mvgd` (MVGD-like affine fit),
  - `hm-mkl-hm` and `hm-mvgd-hm` (compound palette-transfer chains).
- Implementation details:
  - kept existing defaults and prior presets unchanged;
  - preserved existing behavior for masks/strength/apply-mask/JSON payload flow;
  - added stable torch implementations with numeric guards and fallback-safe paths.
- Tests:
  - added smoke coverage for new experimental presets in
    `tests/test_smoke_nodes.py`.
- Version updated to `0.23.5` in `pyproject.toml` and `README.md`.

## 0.23.4 — 2026-03-05
- Module Node Picker: unknown-update module names are now visible in UI status.
  - backend now returns explicit unknown-name list:
    `unknown_update_modules` in refresh status and
    `custom_modules_unknown_update_modules` in node catalog;
  - refresh progress and top Custom Nodes alert card now show module name preview
    for `could not be checked` state (first 3 names + `+N more`);
  - flow/composer wiring extended with dedicated state setter/getter for unknown
    module-name list.
- Result:
  - users can immediately see which specific modules failed update-status checks
    without opening logs.
- Version updated to `0.23.4` in `pyproject.toml` and `README.md`.

## 0.23.3 — 2026-03-04
- Module refresh update-detection fix for custom nodes:
  - removed forced `up_to_date` suppression for modules with ComfyUI-Manager
    `.git/.cnr-id` marker in refresh/update evaluation;
  - `modules_need_update` now uses git/remote comparison for CNR-marked modules
    as well (no blanket exclusion).
- Affected backend paths:
  - `utils/module_browser/module/module_update_state_ops.py`
  - `utils/module_browser/tracking/tracker_ops.py`
  - `utils/module_browser/module/module_info.py`
- Result:
  - reduces undercount cases where `Refresh Custom Nodes Info` could report too
    few updatable custom modules versus actual manager-driven updates.
- Version updated to `0.23.3` in `pyproject.toml` and `README.md`.

## 0.23.2 — 2026-02-26
- DZI node runtime logging/diagnostics update (`ImageDownloadDZITiles`):
  - added structured console logs with `DZI` prefix for start/URLs/geometry/result;
  - added tile download progress bar (`DZI Tiles`) in ComfyUI console;
  - added explicit error logging for:
    tile HTTP failures, tile decode exceptions, DZI metadata fetch/parse errors;
  - added full traceback logging on node failure for faster debugging.
- Docs:
  - updated DZI guide with note about console progress and error logs.
- Version updated to `0.23.2` in `pyproject.toml` and `README.md`.

## 0.23.1 — 2026-02-26
- DZI node URL contract update (`ImageDownloadDZITiles`):
  - `base_url` now expects site root without `/zoom`
    (e.g. `https://collectionimages.npg.org.uk`);
  - node now appends `/zoom` internally for all requests.
- DZI assembly correctness fix:
  - fixed duplicated/wide output on NPG-like endpoints by prioritizing
    level geometry derived from `.dzi` metadata (when available);
  - probe-based axis scan is now used as fallback path only when `.dzi`
    metadata is unavailable.
- UX/docs/tests:
  - updated Russian tooltips for `base_url`, `mw`, `level`;
  - updated DZI guide examples for new `base_url` format;
  - added smoke coverage for URL normalization and DZI-priority geometry.
- Version updated to `0.23.1` in `pyproject.toml` and `README.md`.

## 0.23.0 — 2026-02-26
- New node: `Download DZI Tiles Image` (`ImageDownloadDZITiles`):
  - added DZI tile download/assembly node with inputs `base_url`, `mw`, `level`;
  - outputs assembled ComfyUI `IMAGE` tensor (`[1,H,W,3]`, float32, `0..1`);
  - added registration and UI metadata in central node registry.
- DZI probe/network robustness:
  - replaced probe status checks that used `stream=True` (caused transient
    `ConnectionError` on some endpoints) with standard request status checks;
  - added small retry logic and robust axis probing to avoid false `1x1`
    assembly on unstable responses.
- Docs/tests:
  - added guide: `guides/GUIDE_DZI_TILES_DOWNLOAD.md`;
  - updated README node index/details for DZI node;
  - added smoke test for deterministic mocked `2x2` DZI assembly;
  - docs consistency check mapping extended (`utils/docs_check.py`).
- Version updated to `0.23.0` in `pyproject.toml` and `README.md`.

## 0.22.1 — 2026-02-26
- Module Node Picker (Nodes 2.0 placement fix):
  - fixed node insertion anchor mismatch where top Node 2.0 menu was centered,
    but node body was shifted up-left;
  - switched picker insertion flow to native-style positioning at creation time
    (`createNode(..., { pos })`) and removed post-insert recenter shift;
  - stabilized graph insertion across Node 1.x/2.0 APIs.
- Custom Nodes refresh/update status alignment improvements:
  - added ComfyUI-Manager based update overrides during refresh
    (`/customnode/installed` + `/customnode/getlist`) to reduce undercount of
    modules requiring update versus Manager;
  - extended refresh diagnostics with manager-override counters in backend logs;
  - improved UI status text/cards to show both counters together when relevant:
    `need_update` and `unknown_update`.
- Version updated to `0.22.1` in `pyproject.toml` and `README.md`.

## 0.22.0 — 2026-02-18
- Opened new minor line `0.22.x` for advanced Look Match development.
- Added formal implementation roadmap:
  `refactoring_plan/ROADMAP_LOOK_MATCH_0_22_RU.md`.
- Scope of roadmap:
  - `ImageLookMatchResolve` (Resolve-style monolithic look transfer),
  - `ImageLookMatchNukeBuild` + `ImageLookMatchNukeApply` (Nuke-style model pipeline),
  - contracts/tests/docs/performance checkpoints for strong-reference scenarios.
- Phase A contract baseline implemented:
  - added new nodes:
    `ImageLookMatchResolve`,
    `ImageLookMatchNukeBuild`,
    `ImageLookMatchNukeApply`;
  - added stable JSON schema contracts:
    `alexz.look_match.resolve`,
    `alexz.look_model.nuke_build`,
    `alexz.look_apply.nuke_apply` (version `1`);
  - added initial guide:
    `guides/GUIDE_LOOK_MATCH.md`;
  - registered nodes in central registry and added smoke tests for contracts.
- Phase B Resolve MVP implemented:
  - `ImageLookMatchResolve` now performs staged fitting and application:
    exposure/WB -> tone (`monotonic_spline` / `gamma_gain_lift`) ->
    palette affine (`lut3d` / `rbf`) -> optional skin protection;
  - `export_lut_cube` now bakes fitted resolve pipeline into `cube_text`
    (non-identity LUT for current fit);
  - extended resolve diagnostics JSON with fit/transform stage data.
- Version updated to `0.22.0` in `pyproject.toml` and `README.md`.

## 0.21.2 — 2026-02-18
- Seam Match node split and alpha behavior cleanup:
  - added dedicated compact nodes with fixed models:
    `ImageSeamMatchV1AffineToReference`,
    `ImageSeamMatchV2TonalToReference`,
    `ImageSeamMatchV3HybridToReference`,
    `ImageSeamMatchV4LUTToReference`;
  - kept universal legacy node `ImageSeamMatchToReference` with `seam_model` selector;
  - removed `preserve_alpha` parameter from seam nodes;
  - alpha is now preserved automatically when input is RGBA.
- Registry/tests/docs:
  - registered new seam nodes in `nodes/node_registry.py` (+ UI metadata);
  - extended seam smoke tests for variant nodes and auto-alpha preservation;
  - updated seam docs (`README.md`, `guides/GUIDE_SEAM_MATCH.md`).
  - added UI troubleshooting guide for dynamic widget visibility:
    `guides/GUIDE_WIDGET_VISIBILITY_PROFILES.md`.
  - marked universal seam node display as legacy:
    `Seam Match To Reference (Legacy)`.

## 0.21.1 — 2026-02-18
- Seam Match quality update:
  - stabilized `v3_hybrid` optimization with phased training
    (global warmup + joint residual phase), gradient clipping and
    safe fallback to global affine when hybrid branch is worse;
  - added new high-precision model `seam_model=v4_lut`:
    differentiable 3D LUT optimization (`rgb`/`oklab`) with identity/smoothness regularization;
  - added LUT controls:
    `lut_size`, `lut_identity_reg`, `lut_smooth_reg`, `lut_lr_scale`;
  - extended seam diagnostics with LUT/hybrid fallback flags and eval losses
    (`optimization.lut_fallback_to_affine`, `transform.lut.*`).
- Tests/docs:
  - extended seam smoke tests for `v4_lut`;
  - updated seam docs and README parameter list.

## 0.21.0 — 2026-02-18
- Seam Match v3 hybrid update (`ImageSeamMatchToReference`):
  - added new `seam_model=v3_hybrid`:
    global affine + tonal residuals (shadow/midtone/highlight) optimized jointly;
  - added v3-specific tuning controls:
    `hybrid_residual_strength`, `hybrid_residual_reg`,
    `hybrid_coherence_reg`;
  - extended seam diagnostics JSON with `transform.hybrid`
    (`global`, `residual_bands`, `residual_strength`);
  - kept compatibility with existing models:
    `v2_tonal` and `v1_affine`.
- Tests/docs:
  - extended seam smoke tests for `v3_hybrid` mode and JSON payload fields;
  - updated seam docs and README input list for hybrid parameters.

## 0.20.0 — 2026-02-18
- Seam Match v2 (`ImageSeamMatchToReference`) for harder seam cases:
  - added `seam_model` selector:
    `v2_tonal` (new default) / `v1_affine` (legacy behavior);
  - implemented `v2_tonal` optimization with 3 smooth tonal bands
    (shadow/midtone/highlight), each with its own affine transform;
  - blended band transforms by luminance weights to better fit nonlinear
    tonal shifts (e.g. color-balance style offsets by range);
  - extended `seam_json` diagnostics:
    `optimization.seam_model`, `transform.tonal_bands`,
    `transform.band_weights_mean`.
- Tests/docs/version:
  - added smoke test coverage for `seam_model` options and v2 JSON fields;
  - updated `README.md` and `guides/GUIDE_SEAM_MATCH.md` with new parameter and usage hints;
  - bumped package version to `0.20.0`.

## 0.19.0 — 2026-02-18
- Unified interrupt handling across long-running nodes:
  - added shared helper `utils/interrupt.py` (`check_interrupt`);
  - switched long loops in `image_seam_match`, `image_color_match`,
    `video_frame_match`, `video_cut_match`, and `video_inpaint` to shared
    interrupt checks for consistent `Interrupt` button behavior.
- New node: `Seam Match To Reference` (`ImageSeamMatchToReference`):
  - added seam-focused color optimization mode for minimizing visible cuts/diff;
  - supports optimization spaces `rgb`/`oklab`;
  - added `compute_device` selection:
    `auto` / `cpu` / `cuda` (with safe CPU fallback when CUDA unavailable);
  - added fixed optimization size options:
    `downscale_long_side` = `as_is` / `1080p` / `720p` / `480p`;
  - outputs optimization diagnostics in `seam_json`.
  - fixed ComfyUI inference-mode compatibility:
    seam optimization now converts inference tensors to regular tensors before
    autograd, preventing runtime error
    "Inference tensors cannot be saved for backward".
  - added ComfyUI interrupt checks inside seam-match optimization loop,
    so `Interrupt` (red stop button) reacts during long runs.
- Color Match To Reference UI/UX and docs update (Stage 4):
  - added dedicated preset comparison matrix in
    `guides/GUIDE_COLOR_MATCH_DETAILED.md` with columns:
    method, speed, color accuracy, detail preservation;
  - added visual before/after section for all presets on a single shared case
    (`guides/assets/color_match_examples/case01/`);
  - added reproducible example generator:
    `guides/assets/color_match_examples/generate_case01.py`;
  - expanded `INPUT_TYPES` tooltips with runtime/dependency hints
    (CPU/GPU expectations, `torchvision`/`lpips` notes).
  - added real-pair visual example set `case02`
    (`warm western -> cold cinematic`) in
    `guides/assets/color_match_examples/case02/`.
- Color Match To Reference advanced matching update:
  - added optional local matching by tiles with `spatial_grid` (NxN) for
    `linear`, `mean_std`, `adain`, and `auto_optimal`;
  - extended `auto_optimal` with quality-gated fallback:
    `auto_quality_fallback`, `auto_fallback_method`,
    `auto_fallback_threshold`, `auto_fallback_margin`;
  - extended diagnostics (`match_json.deep.auto_optimal` and `stats`) with
    fallback and spatial-grid fields.
- Color Match To Reference metric backend improvement:
  - switched `delta_e76` full-metric calculation to torch-first Lab distance
    (no mandatory `cv2` dependency for this metric path).
- Color Match To Reference quality/stability update:
  - added `quality_metrics_mode`: `off` / `fast` / `full`;
  - kept backward compatibility: `compute_quality_metrics=false` forces metrics mode to `off`;
  - `fast` mode computes only `mse` + `ssim` (skips `delta_e76` + `lpips_alex`) for faster evaluation.
- Color Match To Reference temporal and portrait improvements:
  - extended `auto_optimal` with temporal stabilization:
    `auto_temporal_stability`, `auto_temporal_alpha`, `auto_switch_threshold`;
  - added EMA + hysteresis method switching logic for video frame consistency;
  - added optional skin-tone protection:
    `skin_tone_protection`, `skin_protection_strength`;
  - extended JSON diagnostics with `quality_mode`, skin-mask stats,
    and temporal auto-optimal score fields.
- Color Match To Reference performance and correctness update:
  - vectorized batch processing for `mean_std` and `linear` presets in `nodes/image_color_match.py`;
  - added VGG19 feature extractor cache for `perceptual_vgg_fast` (prevents repeated model reloads);
  - moved `tone_curve`, `adain`, `optimal_transport`, `lab_cdf`, `oklab_cdf` node path to torch-side processing, reducing CPU roundtrips;
  - added optional input `compute_quality_metrics` (default `true`) to skip MSE/SSIM/DeltaE/LPIPS computation for faster batch runs;
  - fixed `match_mask` semantics for node presets routed via color-match utils (`white = include`);
  - fixed `lab_cdf` execution path in `utils/color_match_utils.py` and stabilized output dtype/device.
- Tests and docs:
  - updated and extended smoke tests for color-match presets, mask semantics, batch padding, and metric-disable mode;
  - synced README and color-match guides with current preset names and `compute_quality_metrics` parameter;
  - docs consistency check passes.
- Color Match To Reference feature expansion:
  - added new preset `auto_optimal` (automatic method choice between `linear` and `oklab_cdf`);
  - added configurable auto-selection strategy `auto_optimal_metric`:
    `mse`, `mse_ssim`, `mse_ssim_lpips`;
  - added `match_json.deep.auto_optimal` diagnostics with per-candidate metrics/scores and selected mode;
  - added LUT export to `.cube` with new optional inputs:
    `export_lut`, `lut_size`, `lut_output_dir`, `lut_name`;
  - LUT generation supports exact linear transform for linear-like modes and polynomial bake for nonlinear modes.
- Color Match To Reference refactoring (Stage 2, architecture cleanup):
  - unified shared torch helpers in `utils/color_match_utils.py` for:
    batch padding, mask preparation, mask weights, `linear` fit/match, `mean_std` fit/match;
  - switched node-side duplicated math/shape plumbing in `nodes/image_color_match.py`
    to shared `utils/color_match_utils.py` helpers;
  - standardized match/apply mask preprocessing via one pipeline (`prepare_match_and_apply_masks`);
  - added explicit empty `match_mask` handling:
    frame returns original image, logs warning, and marks mode with `empty_match_mask`;
  - updated smoke tests and guides for empty-mask behavior.

## 0.18.8 — 2026-02-18
- Pre-Phase 4 structural improvement:
  - reorganized `utils/module_browser/` monolithic module into functional submodules:
    - `catalog/` — catalog building, component registry, payload builders;
    - `git/` — git operations, pull helpers, subprocess/git commands;
    - `module/` — module metadata, identity, information, classification;
    - `comfyui/` — ComfyUI-specific tracking, state, manager data;
    - `state/` — state storage, pending-state mutations, runtime refresh;
    - `jobs/` — job execution, status handling, requirements operations;
    - `tracking/` — module novelty tracking and change detection;
    - `bootstrap/` — repository bootstrap helpers;
    - `core/` — core utilities (paths, values, manifest checks, widget mode);
  - all 36 core module files now organized by responsibility (9 thematic folders);
  - created dedicated `__init__.py` in each subfolder with re-exports;
  - maintained backward-compatible public API: all external imports continue to work;
  - updated internal cross-folder imports and import paths;
  - tests: `4 jobs tests + all Phase 3 baseline` pass without regressions.
- Rationale:
  - improved first-time contributor navigation (clear folder semantics);
  - easier test organization per folder;
  - reduced cognitive load during maintenance;
  - no breaking changes to API or public contract.

## 0.18.7 — 2026-02-13
- Phase 3 backend split continued (no API changes):
  - extracted GitHub latest-release helper into
    `utils/module_browser/release_ops.py` (`github_latest_release`);
  - extracted module/comfy update-state decision helpers into
    `utils/module_browser/module_update_state_ops.py`:
    `module_needs_update_now`, `count_custom_modules_need_update`,
    `count_custom_modules_unknown_update`, `comfyui_needs_update_now`;
  - extracted repository bootstrap helpers into
    `utils/module_browser/repo_bootstrap_ops.py`:
    `comfyui_requirements_path`, `bootstrap_module_remote_from_manager`;
  - extracted node classification/annotation helpers into
    `utils/module_browser/node_classification_ops.py`:
    `module_root`, `classify_by_source_path`, `classify_by_relative_module`,
    `fallback_annotation`;
  - switched API facade wrappers in `utils/module_node_browser_api.py`
    to use extracted helpers.
- Tests:
  - added `tests/test_module_browser_release_ops.py`;
  - added `tests/test_module_browser_module_update_state_ops.py`;
  - added `tests/test_module_browser_repo_bootstrap_ops.py`;
  - added `tests/test_module_browser_node_classification_ops.py`;
  - regression check: `78 passed`
    (`node_classification_ops + repo_bootstrap_ops + release_ops + module_update_state_ops + path_ops + requirements_pending_ops + module_browser_tracker + comfyui_git_status_ops + phase0_baseline`).
- Plan/docs sync:
  - marked Phase 3 Steps 27, 28, 29 and 30 completed in both adopted plans.

## 0.18.6 — 2026-02-13
- Phase 3 backend split continued (no API changes):
  - extracted requirements pending-state helpers into
    `utils/module_browser/requirements_pending_ops.py`:
    `set_comfyui_requirements_pending`, `set_module_requirements_pending`;
  - extracted path resolution helpers into
    `utils/module_browser/path_ops.py`:
    `custom_nodes_roots`, `manager_custom_db_path`,
    `manager_github_stats_path`, `module_dir`, `comfyui_root`;
  - switched API facade wrappers in `utils/module_node_browser_api.py`
    to use extracted requirements-pending/path helper modules.
- Tests:
  - added `tests/test_module_browser_requirements_pending_ops.py`;
  - added `tests/test_module_browser_path_ops.py`;
  - regression check: `76 passed`
    (`path_ops + requirements_pending_ops + value_ops + catalog_payload_ops + widget_mode_ops + module_browser_tracker + phase0_baseline`).
- Plan/docs sync:
  - marked Phase 3 Steps 25 and 26 completed in both adopted plans.

## 0.18.5 — 2026-02-13
- Phase 3 backend split continued (no API changes):
  - extracted pure value/date/repository helpers into
    `utils/module_browser/value_ops.py`:
    `short_commit`, `normalize_repo_url`, `github_id`, `repo_name`,
    `pick_repo_url`, `parse_datetime`, `to_iso`, `now_iso`,
    `normalize_comfyui_mode`;
  - switched API facade wrappers in `utils/module_node_browser_api.py`
    to use new helper module.
- Tests:
  - added `tests/test_module_browser_value_ops.py`;
  - regression check: `69 passed`
    (`value_ops + catalog_payload_ops + widget_mode_ops + module_browser_tracker + phase0_baseline`).
- Plan/docs sync:
  - marked Phase 3 Step 24 completed in both adopted plans.

## 0.18.4 — 2026-02-13
- Phase 3 backend split continued (no API changes):
  - extracted catalog-route payload builders to `utils/module_browser/catalog_payload_ops.py`:
    `build_group_payload`, `build_module_list_payload`, `build_module_nodes_payload`;
  - extracted widget-mode/log-mode helpers to `utils/module_browser/widget_mode_ops.py`:
    `custom_update_checked_flag`, `info_only_rejection_payload`,
    `set_custom_update_checked`, `normalize_log_mode`;
  - kept facade-compatible wrappers in `utils/module_node_browser_api.py`.
- Package exports:
  - added new helper exports in `utils/module_browser/__init__.py`.
- Plan/docs sync:
  - marked Phase 3 Steps 22 and 23 completed in both adopted plans.
- Tests:
  - added `tests/test_module_browser_catalog_payload_ops.py`;
  - added `tests/test_module_browser_widget_mode_ops.py`;
  - regression check: `63 passed`
    (`catalog_payload_ops + widget_mode_ops + module_browser_tracker + phase0_baseline`).

## 0.18.3 — 2026-02-13
- Phase 3 backend split continued (no API changes):
  - extracted subprocess/git command helpers from `utils/module_node_browser_api.py`
    into `utils/module_browser/command_ops.py`:
    `run_command`, `run_git`, `extract_git_repo_from_args`,
    `is_git_dubious_ownership_error`, `try_mark_git_safe_directory`, `tail_lines`;
  - kept facade-compatible wrappers in `utils/module_node_browser_api.py` so existing
    routes and monkeypatch points remain stable.
- Package exports:
  - added command helper exports in `utils/module_browser/__init__.py`.
- Plan/docs sync:
  - marked Phase 3 Step 21 completed in both adopted plans
    (`PLAN_REFACTORING_ADOPTED.md`, `PLAN_REFACTORING_ADOPTED_RU.md`).
- Tests:
  - added `tests/test_module_browser_command_ops.py`;
  - regression check target: `test_module_browser_command_ops + test_module_browser_tracker`.

## 0.18.2 — 2026-02-12
- Slice 0 tooling extension:
  - added standalone manifest validator `utils/module_browser/manifest_check.py`;
  - added `run_manifest_check()` export in `utils/module_browser/__init__.py`;
  - validator report includes registry summary, `manifest_signature`, and strict/soft status modes.
- API/manifest contract strengthening:
  - `api_manifest.py` now exposes `ALL_API_ROUTES` and `iter_all_api_routes()`;
  - health checks validate component API routes against full route manifest and detect duplicates.
- Tests:
  - added `tests/test_module_browser_manifest_check.py`;
  - extended slice0 tests for all-routes + health-checked fields;
  - regression pass: `65 passed` (`manifest_check + slice0_registry + module_browser_jobs + phase0_baseline + module_browser_tracker`).

## 0.18.1 — 2026-02-12
- Slice 0 hardening for component lifecycle tracking:
  - added manifest health validation module `utils/module_browser/health.py`;
  - `/alexz_tools/component_registry` now returns `health` report (issues, counts, checked manifests);
  - added deterministic component-manifest signature fields:
    `manifest_signature` and `manifest_changed` in registry payload;
  - persisted component-registry tracker now stores `manifest_signature`.
- API manifest contract hardening:
  - added `ALL_API_ROUTES` + `iter_all_api_routes()` in `utils/module_browser/api_manifest.py`;
  - health checks now validate consistency of `COMPONENT_API_ROUTES` against all declared routes.
- Phase 3 progress:
  - extracted refresh/update job helpers into `utils/module_browser/jobs.py` and kept
    `utils/module_node_browser_api.py` as API-compatible facade.
- Tests:
  - added `tests/test_module_browser_jobs.py`,
  - expanded `tests/test_slice0_registry.py` (health/signature/API-manifest coverage),
  - regression pass: `63 passed` (`slice0_registry + module_browser_jobs + phase0_baseline + module_browser_tracker`).

## 0.18.0 — 2026-02-12
- Slice 0 (Extensibility Foundation) completed:
  - centralized node lifecycle in `nodes/node_registry.py` (single node manifest source);
  - added backend API manifest `utils/module_browser/api_manifest.py` and switched route decorators to manifest constants;
  - added widget manifest `utils/module_browser/widget_manifest.py` and integrated it into component registry;
  - added versioned component-registry contract fields (`schema_name`, `schema_version`);
  - added component snapshot diff-tracking (`added`/`removed` for `node/widget/api`) in `/alexz_tools/component_registry`.
- Stability fixes:
  - fixed unknown-status badge gating so `🟨` is hidden until explicit custom refresh and does not conflict with local-update marker.
- Plan/docs sync:
  - marked Slice 0 as completed in `PLAN_REFACTORING_ADOPTED.md` and `PLAN_REFACTORING_ADOPTED_RU.md`.
- Tests:
  - extended `tests/test_slice0_registry.py` (manifests, schema, diff-tracking),
  - full regression pass: `58 passed` (`slice0_registry + phase0_baseline + module_browser_tracker`).

## 0.17.2 — 2026-02-12
- Slice 0 extensibility foundation (phase-3 prep, no breaking API changes):
  - added central node registry manifest `nodes/node_registry.py` (`NodeSpec`, `NODE_SPECS`, `NODE_UI_METADATA`), and switched node loader to use this manifest.
  - added internal module-browser helper package `utils/module_browser/`:
    - `contracts.py` with versioned state-cache schema normalization (`schema_name`, `schema_version`),
    - `component_registry.py` with reusable node/widget/api registry primitives.
  - backend now normalizes persisted `module_state_cache.json` schema metadata on load/save.
  - added diagnostic endpoint `GET /alexz_tools/component_registry` returning current registry snapshot (`nodes`, `widgets`, `apis`).
- Tests:
  - added `tests/test_slice0_registry.py` to validate registry add/remove lifecycle and schema normalization.
  - regression suite passes: `53 passed` for `test_slice0_registry + phase0_baseline + module_browser_tracker`.
- Metadata sync:
  - version updated to `0.17.2` in `pyproject.toml` and `README.md`.

## 0.17.1 — 2026-02-12
- Module Node Picker UI consistency:
  - aligned selection-legend hint placement/spacing with module-card hint behavior for uniform UX near controls.
- Refactoring plan progress:
  - Phase 2 marked as completed in adopted plans (`RU`/`EN`) with explicit scope note:
    external `NodesMap` direct-switch issue remains documented as third-party known issue and workaround.
- Metadata sync:
  - package/version fields updated to `0.17.1` in `pyproject.toml` and `README.md`.

## 0.17.0 — 2026-02-12
- Minor-version release for large Phase 2 refactor package in Module Node Picker:
  - orchestration/UI/relay logic split into smaller modules and folders with stable wrappers;
  - startup/warmup/default-selection flow stabilized (faster first open, persistent state behavior);
  - refresh/update-status UX normalized (single-card progress/result behavior, less noisy UI state churn);
  - monitoring statuses expanded and clarified (`✅` local update, `🟥` update available, `🟨` unknown status);
  - CSS and UI composition structure cleaned up for easier further maintenance.
- Documentation/metadata sync:
  - package version aligned in `pyproject.toml` and `README.md` to `0.17.0`;
  - changelog date/version baseline moved to `2026-02-12`.

## 0.16.39 — 2026-02-11
- Module Node Picker unknown-update UX improvements:
  - added `🟨` module badge for `update_status=unknown` (no upstream/remote).
  - selected module card now uses yellow border for unknown update status.
  - module card status text for unknown state is now explicit:
    `update status unknown (no upstream/remote)`.
  - module legend now includes explanation for the yellow marker.
  - backend `node_catalog` module entries now include `update_status`, so unknown-state badges render reliably in module dropdown.
  - refresh summary log now prints unknown modules list in summary mode:
    `unknown update status modules: ...`.

## 0.16.38 — 2026-02-11
- Module Node Picker switched to monitoring-only mode for updates:
  - removed in-widget update/install actions for modules and ComfyUI (`Update module`, `Update Custom Nodes`, `Update ComfyUI`, requirements install buttons).
  - status cards now only inform about update availability and unknown-check states.
  - module card now keeps only informational and refresh actions (no pull/pip actions).
- Updated Module Node Picker guide to match monitoring-only workflow.

## 0.16.37 — 2026-02-11
- Module update UX fix for `requirements.txt` follow-up action:
  - `Install updated requirements` prompt is now shown after update-flow busy-lock is released, so the first click is not ignored while controls are still disabled.
  - applies to post-update requirements prompt shown after `Update module` / `Update all`.
- Backend install logging improvement:
  - module and ComfyUI requirements installation now prints pip output/warnings tail to ComfyUI console for easier diagnostics of what was installed.

## 0.16.36 — 2026-02-11
- Custom update-card startup behavior fix:
  - on first picker open after ComfyUI restart, custom update summary card is now hidden until explicit `Refresh Custom Nodes Info`.
  - stale cached custom update counters from previous runs are no longer shown on initial load.
  - backend now gates custom update counters with explicit-check flag (`custom_update_checked`) and resets that flag on startup/warmup.
  - local startup scan (`local_only`) no longer injects manager-remote update inference before explicit refresh.

## 0.16.35 — 2026-02-11
- Module update robustness improvements for repositories with local changes and missing remotes:
  - custom module update now auto-detects git pull conflicts caused by local uncommitted changes and retries after auto-stash (`git stash push -u`), similar to ComfyUI-Manager behavior.
  - merge-conflict detector now supports localized git output (including Russian), so auto-stash retry works in non-English git environments.
  - pull result now carries explicit stash marker metadata (`stashed_local_changes`, `stash_ref`) and clearer status messages.
  - improved module update console logs: failed module updates now print concrete per-module error reasons (not only aggregate counters).
- Update-check compatibility improvements:
  - kept fallback update detection via ComfyUI-Manager metadata/statistics for modules with missing upstream tracking (e.g. `crt-nodes`-like cases).
- Tests:
  - added regression test for auto-stash + retry flow in `tests/test_module_browser_tracker.py`.

## 0.16.34 — 2026-02-11
- Module refresh correctness fix for non-trackable custom repos:
  - backend now distinguishes between:
    - `modules_need_update` (confirmed updates),
    - `modules_unknown_update` (cannot be checked: missing git remote/upstream or no git state).
  - `Refresh Custom Nodes Info` no longer reports global "no updates required" when some modules are uncheckable.
  - `node_catalog` API now returns `custom_modules_unknown_update`.
  - custom status card now shows warning:
    - `N custom modules could not be checked for updates (missing git remote/upstream).`
  - refresh progress/status line now shows unknown-check results where applicable.
- Backend tracking updates:
  - per-module state now persists `update_status` (`can_update` / `up_to_date` / `unknown`).
  - modules without valid remote/upstream are explicitly marked `update_status=unknown` and `update_available=null`.
- Tests:
  - added backend tests for unknown-update counting and missing-git-state behavior in `tests/test_module_browser_tracker.py`.
  - updated baseline refresh-status contract for `modules_unknown_update` in `tests/test_phase0_baseline.py`.

## 0.16.33 — 2026-02-11
- Phase 2 relay runtime modular cleanup (no UX/behavior changes):
  - extracted relay bind-state lifecycle helpers to:
    - `web/orchestration/relay/module_node_picker_tab_relay_lifecycle.js`;
    and switched facade unbind/bind-state creation to shared lifecycle helpers.
  - extracted relay diagnostics payload + deduplicated emitter to:
    - `web/orchestration/relay/module_node_picker_tab_relay_diagnostics.js`;
    and switched relay runtime diagnostics path to shared diagnostics helpers.
  - extracted relay DOM ownership (attach/detach + mount-host recovery) to:
    - `web/orchestration/relay/module_node_picker_tab_relay_dom_ownership.js`;
    and switched relay runtime visibility path to shared ownership controller.
  - extracted composer relay bind wiring to:
    - `web/orchestration/core/composition/module_node_picker_relay_bridge.js`;
    and switched composer to bridge-based relay binding call.
  - extended frontend baseline markers in `tests/test_phase0_baseline.py` for new relay modules.
- Validation:
  - `conda run -n p313 node --check` on changed relay modules,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (41 passed).

## 0.16.32 — 2026-02-11
- Phase 2 relay consistency and lightweight diagnostics:
  - added shared relay reason/timing constants module:
    - `web/orchestration/relay/module_node_picker_tab_relay_constants.js`;
  - switched relay `runtime/intent/tick/facade` modules to shared constants (no behavior changes);
  - added shared `isImmediateRelayReason()` helper in relay constants and switched runtime debounce bypass checks to that helper;
  - updated relay baseline markers in `tests/test_phase0_baseline.py` for constants-based reason flow;
  - added browser-context guards in relay facade bind/unbind to avoid non-browser initialization crashes;
  - added explicit relay `mountHost` wiring (`composer -> relay facade -> relay runtime`) so root re-attach can recover even when initial parent container changes;
  - normalized relay bind input in facade (`root` must be `Element`) and improved runtime host re-attach logic to move connected root back under preferred mount host when needed;
  - extracted relay DOM candidate/context checks into:
    - `web/orchestration/relay/module_node_picker_tab_relay_dom.js`;
    and switched `module_node_picker_tab_relay_helpers.js` to shared DOM helper usage.
  - gated relay diagnostics rendering in composer to debug-enabled mode only.
- Validation:
  - `conda run -n p313 node --check` on changed relay/composer modules,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (41 passed).

## 0.16.31 — 2026-02-11
- Phase 2 relay-state centralization:
  - added shared relay state helper module:
    - `web/orchestration/relay/module_node_picker_tab_relay_state.js`;
  - switched relay facade to use shared state helper for read/write/clear of global relay runtime state key.
- Validation:
  - `conda run -n p313 node --check web/module_node_picker_tab_relay.js web/orchestration/relay/module_node_picker_tab_relay_facade.js web/orchestration/relay/module_node_picker_tab_relay_state.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (41 passed).

## 0.16.30 — 2026-02-11
- Phase 2 relay-entrypoint cleanup:
  - extracted relay bind/unbind implementation into:
    - `web/orchestration/relay/module_node_picker_tab_relay_facade.js`;
  - converted `web/module_node_picker_tab_relay.js` into a stable public re-export shim that keeps existing import path unchanged.
- Frontend baseline contract updates:
  - updated `tests/test_phase0_baseline.py` relay markers to validate facade wiring and root-level relay re-export contract.
- Validation:
  - `conda run -n p313 node --check web/module_node_picker_tab_relay.js web/orchestration/relay/module_node_picker_tab_relay_facade.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (41 passed).

## 0.16.29 — 2026-02-11
- Phase 2 reliability hardening for Module Node Picker initialization:
  - removed direct deep import of `scripts/app.js` from `web/orchestration/core/composition/module_node_picker_composer.js`,
  - `app` instance is now injected from entrypoint (`web/module_node_picker.js`) via:
    - `renderModuleNodePicker(container, { appInstance: app })`,
  - added explicit guard in composer for missing injected app instance to avoid silent startup failures.
- Test/contract updates:
  - updated frontend baseline marker in `tests/test_phase0_baseline.py` to support both legacy and injected app center-node wiring markers.
- Validation:
  - `conda run -n p313 node --check web/module_node_picker.js web/orchestration/core/composition/module_node_picker_composer.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (41 passed).

## 0.16.28 — 2026-02-11
- Phase 2 orchestration-core structure cleanup continued:
  - split `web/orchestration/core/` into semantic subfolders:
    - `web/orchestration/core/composition/`,
    - `web/orchestration/core/infra/`.
  - moved modules:
    - to `core/composition/`:
      - `module_node_picker_composer.js`,
      - `module_node_picker_context_builders.js`,
      - `module_node_picker_stage_bridge.js`;
    - to `core/infra/`:
      - `module_node_picker_bindings.js`,
      - `module_node_picker_error_utils.js`,
      - `module_node_picker_registration.js`.
  - updated dependent imports in runtime/bootstrap and flow modules.
  - updated module header paths and baseline contract path markers in `tests/test_phase0_baseline.py`.
  - updated plan document path references for moved core modules.
- Validation:
  - `conda run -n p313 node --check web/module_node_picker.js web/orchestration/core/composition/module_node_picker_composer.js web/orchestration/core/composition/module_node_picker_context_builders.js web/orchestration/core/composition/module_node_picker_stage_bridge.js web/orchestration/core/infra/module_node_picker_bindings.js web/orchestration/core/infra/module_node_picker_registration.js web/orchestration/core/infra/module_node_picker_error_utils.js web/orchestration/runtime/bootstrap/module_node_picker_runtime_bootstrap.js web/orchestration/runtime/bootstrap/module_node_picker_runtime_bootstrap_bindings.js web/orchestration/flow/actions/module_node_picker_actions.js web/orchestration/flow/progress/module_node_picker_update_flow.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (40 passed).

## 0.16.27 — 2026-02-11
- Phase 2 flow-structure cleanup continued:
  - introduced semantic flow subfolders:
    - `web/orchestration/flow/stage/`,
    - `web/orchestration/flow/panel/`.
  - moved modules:
    - to `flow/stage/`:
      - `module_node_picker_flow_stage.js`,
      - `module_node_picker_flow_wiring.js`;
    - to `flow/panel/`:
      - `module_node_picker_module_panel_controller.js`.
  - updated dependent imports in composer and flow wiring/modules.
  - updated module header paths and baseline contract path markers in `tests/test_phase0_baseline.py`.
- Validation:
  - `find web/orchestration/flow -type f -name '*.js' | xargs conda run -n p313 node --check`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (40 passed).

## 0.16.26 — 2026-02-11
- Phase 2 flow-structure cleanup continued:
  - introduced semantic flow subfolders:
    - `web/orchestration/flow/actions/`,
    - `web/orchestration/flow/catalog/`.
  - moved modules:
    - to `flow/actions/`:
      - `module_node_picker_actions.js`,
      - `module_node_picker_action_flows.js`;
    - to `flow/catalog/`:
      - `module_node_picker_catalog_controller.js`,
      - `module_node_picker_data_flow.js`.
  - updated dependent imports in flow wiring and moved modules.
  - updated module header paths and baseline contract path markers in `tests/test_phase0_baseline.py`.
- Validation:
  - `find web/orchestration/flow -type f -name '*.js' | xargs conda run -n p313 node --check`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (40 passed).

## 0.16.25 — 2026-02-11
- Phase 2 runtime-structure cleanup continued:
  - introduced semantic runtime subfolders:
    - `web/orchestration/runtime/bootstrap/`,
    - `web/orchestration/runtime/lifecycle/`.
  - moved modules:
    - to `runtime/bootstrap/`:
      - `module_node_picker_runtime_setup.js`,
      - `module_node_picker_runtime_bootstrap.js`,
      - `module_node_picker_runtime_bootstrap_bindings.js`,
      - `module_node_picker_runtime_projection.js`,
      - `module_node_picker_startup_flow.js`,
      - `module_node_picker_warmup_controller.js`;
    - to `runtime/lifecycle/`:
      - `module_node_picker_lifecycle.js`,
      - `module_node_picker_lifecycle_guard.js`.
  - updated dependent imports in composer, flow modules, and runtime modules.
  - updated module header paths and baseline contract path markers in `tests/test_phase0_baseline.py`.
- Validation:
  - `conda run -n p313 node --check` on changed orchestration files,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (40 passed).

## 0.16.24 — 2026-02-11
- Phase 2 flow-structure cleanup continued:
  - introduced semantic flow subfolders:
    - `web/orchestration/flow/progress/`,
    - `web/orchestration/flow/resume/`.
  - moved modules:
    - to `flow/progress/`:
      - `module_node_picker_update_flow.js`,
      - `module_node_picker_polling_controller.js`;
    - to `flow/resume/`:
      - `module_node_picker_resume_flow.js`,
      - `module_node_picker_resume_custom_refresh.js`,
      - `module_node_picker_resume_module_update.js`,
      - `module_node_picker_resume_comfy_refresh.js`.
  - updated dependent imports in flow wiring/actions and updated module header paths for moved files.
  - updated baseline contract file-path markers in `tests/test_phase0_baseline.py`.
- Validation:
  - `conda run -n p313 node --check web/orchestration/flow/module_node_picker_action_flows.js web/orchestration/flow/module_node_picker_flow_wiring.js`,
  - `find web/orchestration/flow -type f -name '*.js' | xargs conda run -n p313 node --check`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (40 passed).

## 0.16.23 — 2026-02-11
- Phase 2 orchestration structure cleanup continued:
  - introduced semantic subfolders for remaining flat orchestration modules:
    - `web/orchestration/core/`,
    - `web/orchestration/ui/`,
    - `web/orchestration/api/`.
  - moved modules:
    - to `core/`:
      - `module_node_picker_bindings.js`,
      - `module_node_picker_composer.js`,
      - `module_node_picker_context_builders.js`,
      - `module_node_picker_error_utils.js`,
      - `module_node_picker_registration.js`,
      - `module_node_picker_stage_bridge.js`;
    - to `ui/`:
      - `module_node_picker_busy_ui.js`,
      - `module_node_picker_debug_ui.js`,
      - `module_node_picker_selection_controller.js`,
      - `module_node_picker_status_cards.js`,
      - `module_node_picker_ui_controllers.js`,
      - `module_node_picker_ui_stage.js`,
      - `module_node_picker_view_helpers.js`;
    - to `api/`:
      - `module_node_picker_api_client.js`.
  - updated all dependent imports in frontend entrypoint and orchestration runtime/flow/ui layers.
  - fixed post-move runtime imports in `web/orchestration/runtime/module_node_picker_runtime_setup.js`:
    - `createProcessUiController` import path,
    - `createModuleNodePickerRuntimeContext` import path.
  - updated module header paths (`Module:`) for moved files.
  - updated baseline contract test file-path markers in `tests/test_phase0_baseline.py` for new `core/ui/api` structure.
- Validation:
  - `conda run -n p313 node --check web/module_node_picker.js web/module_node_picker_tab_relay.js`,
  - `find web/orchestration -type f -name '*.js' | xargs conda run -n p313 node --check`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (40 passed).

## 0.16.22 — 2026-02-11
- Phase 2 orchestration structure cleanup continued:
  - introduced `web/orchestration/flow/` subfolder for pipeline-oriented modules,
  - moved flow modules:
    - `module_node_picker_data_flow.js`,
    - `module_node_picker_action_flows.js`,
    - `module_node_picker_actions.js`,
    - `module_node_picker_update_flow.js`,
    - `module_node_picker_resume_flow.js`,
    - `module_node_picker_resume_custom_refresh.js`,
    - `module_node_picker_resume_module_update.js`,
    - `module_node_picker_resume_comfy_refresh.js`,
    - `module_node_picker_flow_wiring.js`,
    - `module_node_picker_flow_stage.js`,
    - `module_node_picker_catalog_controller.js`,
    - `module_node_picker_polling_controller.js`,
    - `module_node_picker_module_panel_controller.js`.
  - updated imports to new `flow/` locations in composer and dependent modules.
  - updated module headers in moved files to reflect new paths.
  - updated baseline test file-path markers in `tests/test_phase0_baseline.py`.
- Docs path naming fix:
  - renamed docs folder `guedes/` to `guides/` (typo fix),
  - updated README and docs-check guide links to `guides/...`.
- Validation:
  - `conda run -n p313 node --check` for changed relay/runtime/flow/composer modules,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed, 40 tests).

## 0.16.21 — 2026-02-11
- Phase 2 orchestration structure cleanup:
  - introduced semantic subfolders in `web/orchestration`:
    - `web/orchestration/relay/` for tab-relay internals,
    - `web/orchestration/runtime/` for runtime/bootstrap/lifecycle/warmup internals.
  - moved relay modules:
    - `module_node_picker_tab_relay_helpers.js`,
    - `module_node_picker_tab_relay_runtime.js`,
    - `module_node_picker_tab_relay_tick.js`,
    - `module_node_picker_tab_relay_intent.js`.
  - moved runtime modules:
    - `module_node_picker_runtime_setup.js`,
    - `module_node_picker_runtime_bootstrap.js`,
    - `module_node_picker_runtime_bootstrap_bindings.js`,
    - `module_node_picker_runtime_projection.js`,
    - `module_node_picker_startup_flow.js`,
    - `module_node_picker_lifecycle.js`,
    - `module_node_picker_lifecycle_guard.js`,
    - `module_node_picker_warmup_controller.js`.
  - updated imports in composer, tab relay entrypoint, and dependent orchestration modules to new paths.
  - updated module header paths in moved files to reflect new locations.
  - updated baseline test path markers in `tests/test_phase0_baseline.py` for new folder structure.
- Validation:
  - `conda run -n p313 node --check` on changed relay/runtime/composer files,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed, 40 tests).

## 0.16.20 — 2026-02-11
- Phase 2 frontend decomposition continued:
  - extracted runtime-bootstrap callback adapters from composer into `web/orchestration/module_node_picker_runtime_bootstrap_bindings.js`,
  - `web/orchestration/module_node_picker_composer.js` now wires runtime bootstrap through prebuilt binding adapters (`createModuleNodePickerRuntimeBootstrapBindings`) to reduce inline lambda density and improve maintainability.
  - extracted runtime-setup projection/unpacking from composer into `web/orchestration/module_node_picker_runtime_projection.js` (`projectModuleNodePickerRuntimeSetup`) to reduce composer field-mapping density.
- Warmup UX reliability fix:
  - fixed stuck `warming up...` indicator by wiring warmup poller to catalog reload path in `web/orchestration/module_node_picker_catalog_controller.js` (`warmupController.setPoller(...)`),
  - added fail-safe warmup indicator reset paths in `web/orchestration/module_node_picker_warmup_controller.js` for retry budget exhaustion and poll errors.
- Baseline guardrail update:
  - extended `tests/test_phase0_baseline.py` marker checks for runtime-bootstrap binding module and warmup poller wiring.
- Validation:
  - `conda run -n p313 node --check web/orchestration/module_node_picker_runtime_bootstrap_bindings.js web/orchestration/module_node_picker_catalog_controller.js web/orchestration/module_node_picker_warmup_controller.js web/orchestration/module_node_picker_composer.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed, 40 tests).

## 0.16.19 — 2026-02-11
- Phase 2 frontend decomposition continued:
  - extracted deferred-stage bridge from composer into `web/orchestration/module_node_picker_stage_bridge.js`,
  - `web/orchestration/module_node_picker_composer.js` now uses `createModuleNodePickerStageBridge()` to keep flow-stage wiring and deferred handlers centralized with behavior parity.
- UI structure cleanup:
  - moved picker stylesheet module from orchestration layer to UI layer:
    - `web/orchestration/styles/module_node_picker_styles.js` -> `web/ui/styles/module_node_picker_styles.js`,
  - updated `web/ui/module_node_picker_shell.js` import path accordingly.
- Baseline guardrail update:
  - extended `tests/test_phase0_baseline.py` marker checks for `module_node_picker_stage_bridge.js` and composer stage-bridge wiring.
- Validation:
  - `conda run -n p313 node --check web/orchestration/module_node_picker_stage_bridge.js web/orchestration/module_node_picker_composer.js web/ui/styles/module_node_picker_styles.js web/ui/module_node_picker_shell.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed, 40 tests).

## 0.16.18 — 2026-02-11
- Phase 2 maintainability step for Module Node Picker UI styles:
  - extracted picker CSS from `web/ui/module_node_picker_shell.js` into a dedicated stylesheet source module:
    - `web/orchestration/styles/module_node_picker_styles.js`,
  - grouped styles into explicit sections (root layout, header/debug, structure, help/hints, status cards, module card, controls, node list, fallback button),
  - added detailed English comments in the style module describing which selectors control which widget area,
  - kept runtime behavior unchanged by preserving one-shot style injection in `injectModuleNodePickerStyles()` and switching style source to `getModuleNodePickerStyleText()`.
- Validation:
  - `conda run -n p313 node --check web/orchestration/styles/module_node_picker_styles.js web/ui/module_node_picker_shell.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed, 40 tests).

## 0.16.17 — 2026-02-11
- Module Node Picker first-open latency reduction:
  - moved runtime-state warmup to non-blocking background worker in `utils/module_node_browser_api.py` (`_start_runtime_state_warmup`),
  - `/alexz_tools/node_catalog` and `/alexz_tools/module_info` no longer block on synchronous `_ensure_runtime_state_ready()` during first UI open,
  - preserved existing warmup logic and caches; only startup execution mode changed from blocking to async kickoff.
- Runtime warmup UX follow-up:
  - added `runtime_warmup` state to `/alexz_tools/node_catalog` payload,
  - added silent catalog auto-reload loop in `web/orchestration/module_node_picker_catalog_controller.js` while warmup is not finished,
  - novelty/update markers now self-appear after warmup completion without tab switching,
  - added compact header hint (`warming up...`) near widget title while warmup is running.
  - extracted warmup polling orchestration to `web/orchestration/module_node_picker_warmup_controller.js` (catalog controller simplified, behavior preserved).
  - extracted relay passive tick-loop orchestration to `web/orchestration/module_node_picker_tab_relay_tick.js` (tab relay bindings simplified, behavior preserved).
  - extracted relay tab-intent/event handling orchestration to `web/orchestration/module_node_picker_tab_relay_intent.js` (pointer/mouse/keyup/visibility/pageshow handling centralized, behavior preserved).
- Baseline guardrails:
  - added warmup contract tests in `tests/test_phase0_baseline.py` for `_runtime_warmup_status()` and `_start_runtime_state_warmup()` no-op behavior after warmup completion.
  - updated frontend relay contract checks for new tick-loop and tab-intent module markers.
- Validation:
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed, 40 tests).

## 0.16.16 — 2026-02-11
- Phase 2 (tab-sync stabilization) continued:
  - split pending-resume internals into focused modules:
    - `web/orchestration/module_node_picker_resume_custom_refresh.js`,
    - `web/orchestration/module_node_picker_resume_module_update.js`,
    - `web/orchestration/module_node_picker_resume_comfy_refresh.js`,
  - kept stable public facade in `web/orchestration/module_node_picker_resume_flow.js` (same export names/behavior),
  - reduced composer stage-callback wiring churn by using deferred stage adapters in `web/orchestration/module_node_picker_composer.js`.
- Validation:
  - `conda run -n p313 node --check web/orchestration/module_node_picker_resume_flow.js web/orchestration/module_node_picker_resume_custom_refresh.js web/orchestration/module_node_picker_resume_module_update.js web/orchestration/module_node_picker_resume_comfy_refresh.js web/orchestration/module_node_picker_composer.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed).

## 0.16.15 — 2026-02-11
- Phase 2 (tab-sync stabilization) continued:
  - extracted large composer dependency maps into `web/orchestration/module_node_picker_context_builders.js`,
  - moved runtime-setup/ui-stage/flow-stage/runtime-bootstrap context assembly out of `web/orchestration/module_node_picker_composer.js`,
  - preserved behavior by keeping the same dependency wiring and lifecycle callbacks through builder adapters.
- Baseline guardrail update:
  - updated `tests/test_phase0_baseline.py` runtime marker aggregation to include `module_node_picker_context_builders.js`.
- Validation:
  - `conda run -n p313 node --check web/orchestration/module_node_picker_context_builders.js web/orchestration/module_node_picker_composer.js`,
  - `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py` (all passed).

## 0.16.14 — 2026-02-11
- Module Node Picker update logging modes:
  - added backend log mode switch for update jobs in `utils/module_node_browser_api.py` (`summary`/`verbose`),
  - update routes now accept `log_mode` and apply it to console output (`/alexz_tools/module_update`, `/alexz_tools/module_refresh`),
  - wired frontend API wrappers to pass `log_mode` in refresh/update requests (`web/api/module_node_picker_api.js`),
  - wired orchestration so current mode follows picker debug flag: `Debug=ON` => `verbose`, `Debug=OFF` => `summary`.
- Console output tuning:
  - `summary` now logs only key milestones (job start/finish, target scan summary, ComfyUI pull start/done),
  - `verbose` additionally logs per-module scan/pull steps and command timings.
- Validation:
  - `conda run -n p313 node --check` for changed JS modules,
  - `conda run -n p313 pytest -q tests/test_module_browser_tracker.py tests/test_phase0_baseline.py` (all passed).

## 0.16.13 — 2026-02-11
- Phase 2 (tab-sync stabilization) continued:
  - extracted UI-stage composition from `web/orchestration/module_node_picker_composer.js` into `web/orchestration/module_node_picker_ui_stage.js` (selector/busy/view/status controller assembly preserved),
  - extracted flow-stage composition from `web/orchestration/module_node_picker_composer.js` into `web/orchestration/module_node_picker_flow_stage.js` (polling/catalog/action/module-panel wiring preserved),
  - rewired composer to consume stage adapters with behavior parity and retained deferred callbacks (`loadCatalog`, `loadModuleInfo`, `renderNodeList`, `setExpandedModule`) to avoid startup regressions.
- Regression coverage:
  - updated `tests/test_phase0_baseline.py` contract bundle to include new stage modules in runtime marker scans.
  - verified with `conda run -n p313 node --check` for changed JS modules and `conda run -n p313 pytest -q tests/test_phase0_baseline.py tests/test_module_browser_tracker.py tests/test_smoke_nodes.py` (42 passed).

## 0.16.12 — 2026-02-11
- Phase 2 (tab-sync stabilization) continued:
  - extracted category/group/module dropdown orchestration from `web/module_node_picker.js` into `web/orchestration/module_node_picker_selection_controller.js`,
  - delegated module/group selector population and picker selection store sync to the new controller (behavior preserved),
  - removed direct selector-UI imports (`fillModuleSelectUi`, `fillGroupSelectUi`) from main picker composition module,
  - extracted refresh/update/resume action orchestration from `web/module_node_picker.js` into `web/orchestration/module_node_picker_action_flows.js`,
  - moved ComfyUI/custom refresh flows, module update flow, per-module refresh/install flows, requirements follow-up flow, and pending-job resume flows behind one flow-factory (behavior preserved),
  - extracted polling-token lifecycle (`refresh`/`update`) into `web/orchestration/module_node_picker_polling_controller.js`,
  - removed direct polling-token state from `web/module_node_picker.js`; picker dispose now invalidates poller via controller API.
  - extracted module-card/node-list rendering orchestration into `web/orchestration/module_node_picker_module_panel_controller.js` (expanded module state + render callbacks), reducing UI-state coupling inside `web/module_node_picker.js`.
  - extracted picker instance lifecycle/dispose orchestration into `web/orchestration/module_node_picker_lifecycle.js` (cleanup hooks for catalog tokens, polling invalidation, event unbind, startup cancel, debug/process dispose, API client dispose).
  - extracted extension registration/fallback mount orchestration into `web/orchestration/module_node_picker_registration.js`.
  - centralized picker constants (IDs, storage keys, labels, marks, defaults) in `web/constants/module_node_picker_constants.js`.
  - extracted node insertion helpers (LiteGraph node creation + canvas centering) into `web/ui/module_node_picker_node_factory.js`.
  - moved full picker composition (`renderPicker`) from `web/module_node_picker.js` into `web/orchestration/module_node_picker_composer.js`.
  - reduced `web/module_node_picker.js` to entrypoint-only registration and fallback wiring.
  - extracted selector/busy/view/status-card wiring into `web/orchestration/module_node_picker_ui_controllers.js`, reducing composition density in `module_node_picker_composer.js`.
  - extracted polling/catalog/update/module-panel orchestration bundle into `web/orchestration/module_node_picker_flow_wiring.js`; composer now wires this bundle via one runtime adapter.
  - extracted runtime bootstrap (event binding + ComfyUI card restore + startup coordinator wiring) into `web/orchestration/module_node_picker_runtime_bootstrap.js`.
  - extracted runtime setup (runtime context + lifecycle + API client + debug/process controllers) into `web/orchestration/module_node_picker_runtime_setup.js`.
- Startup regression safety:
  - kept deferred `loadModuleInfo`/`renderNodeList` binding so catalog controller and selection controller can initialize without TDZ/empty-dropdown regressions.
- Regression tests:
  - extended frontend contract markers in `tests/test_phase0_baseline.py` for the new selection-controller and action-flows modules/wiring.

## 0.16.11 — 2026-02-11
- Phase 2 (tab-sync stabilization) continued:
  - extracted relay runtime logic to `web/orchestration/module_node_picker_tab_relay_runtime.js`,
  - kept helper functions in `web/orchestration/module_node_picker_tab_relay_helpers.js`,
  - hardened relay click handling to avoid content-click detach regression,
  - simplified relay event sources (single pointer/mouse path, removed per-button listeners),
  - reduced passive relay tick pressure when picker tab is inactive,
  - added relay sync debounce and runtime `dispose()` cleanup on unbind,
  - avoided redundant attach/detach DOM operations in relay runtime,
  - tightened relay tab-button detection to sidebar-context tab controls only (reduced false positives from content clicks),
  - filtered `relay_keyup` handling to tab/navigation keys and excluded text-input targets to reduce noisy sync cycles,
  - added relay bind-token guards so stale timers/listeners from previous binds cannot apply visibility updates,
  - removed unused tab-id fallback helper from relay helpers to reduce dead heuristics.
- Module Node Picker UX cleanup:
  - fixed help-hint flicker when switching `ComfyUI check` mode by removing transient catalog-loading help overrides,
  - changed "no loaded nodes" message to compact hint style, removed module-name prefix, and highlighted it in red.
- Tab relay runtime tuning:
  - replaced fixed `setInterval` tick loop with adaptive `setTimeout` scheduling to reduce background relay pressure,
  - kept backward-compat cleanup for legacy `tickInterval` relay state shape during unbind.
- Tab relay click-candidate hardening:
  - narrowed sidebar tab candidate detection (removed generic sidebar-button fallback),
  - now accepts only explicit tab-like controls (`side-bar-button`, `*-tab-button`, `role=tab`, `aria-selected`, tab-like `aria-controls`).
- Catalog-load race protection:
  - added request token guard for node-catalog loading so stale async responses cannot override fresh selector/card state.
- Picker lifecycle and module-info race protection:
  - added per-render picker cleanup hook to invalidate stale async work and unbind relay before next render,
  - added request token guard for `loadModuleInfo` so stale module-info responses cannot overwrite current module card.
- Relay helper cleanup:
  - removed unused `hasSidebarTabId` helper,
  - centralized sidebar/tab candidate selectors as module constants for cleaner and safer maintenance.
- Baseline tests hardening:
  - added regression check to ensure legacy relay interval path (`setInterval`) and removed helper export are not reintroduced.
- Picker request/load flow hardening:
  - added early exit for `loadCatalog`/`loadModuleInfo` when picker instance is disposed (prevents unnecessary stale fetches),
  - added small debounce for `ComfyUI check` mode switch to avoid burst catalog reloads during fast toggles.
- Event lifecycle cleanup:
  - `bindModuleNodePickerEvents(...)` now returns unbind cleanup that clears handlers and debounce timer,
  - picker dispose now calls event unbind + debug-store unsubscribe to avoid stale listeners between re-renders.
- Long-running flow cancellation guards:
  - added shared `shouldContinue` guards across actions/update orchestration so refresh/update flows stop cleanly after picker dispose,
  - wired picker liveness check into polling/update/install/refresh contexts to block late UI writes from stale async tasks.
- Process UI lifecycle cleanup:
  - added `dispose()` to inline process controller and invoke it from picker dispose path,
  - ensures progress host/buttons are detached/cleared between re-renders and cannot leak stale state.
- Guard-logic deduplication:
  - extracted shared `shouldContinueContext(...)` into `web/orchestration/module_node_picker_lifecycle_guard.js`,
  - switched actions/update orchestration to import the shared guard instead of duplicating local implementations.
- Startup-open empty-state fix:
  - fixed first-open race where `loadCatalog` could skip due transient `root.isConnected` check during initial attach,
  - switched picker liveness guard to lifecycle (`!pickerDisposed`) and hide Custom alert card by default to avoid blank card flash.
- Startup catalog resilience:
  - added bounded startup retry loop for initial catalog load when backend returns temporary empty state,
  - startup retry timer is now explicitly canceled on picker dispose to prevent stale late reloads.
- Selector loading-state UX:
  - added explicit loading placeholders/disable-state for group/module selectors during catalog load,
  - prevents empty-looking dropdowns during startup/retry windows and re-enables controls after load completion.
- API timeout hardening:
  - introduced shared frontend API fetch wrapper with `AbortController` timeout handling,
  - applied bounded timeouts to catalog/module/status/update/install API calls to avoid indefinite UI hangs.
- API lifecycle-cancellation hardening:
  - picker render now owns a per-instance `AbortController` that is aborted on dispose/re-render,
  - all picker API calls now use lifecycle-bound `signal`, so in-flight requests are canceled immediately after dispose,
  - removed stale timer artifact from frontend API wrapper timeout path.
- Pending refresh resume:
  - Custom Nodes refresh now keeps a pending marker while operation is running,
  - when widget re-opens, it restores refresh progress/result in the Custom Nodes card,
  - completed pending refresh is finalized on return (catalog reload + novelty acknowledge path),
  - `Custom Nodes` status-checked state is now session-only (runtime memory), so refresh result survives widget switching but resets after full page/ComfyUI reload.
- Pending update resume:
  - module/comfy update jobs now keep pending markers while running,
  - when widget re-opens, update progress/result is restored from `/module_update_status`,
  - restore flow handles all scopes (`single`, `all`, `comfyui`) and keeps post-update actions (requirements prompt + catalog refresh),
  - pending markers are session-only (not kept after full page/ComfyUI reload).
- Pending ComfyUI info refresh resume:
  - interrupted `Refresh ComfyUI Info` now sets a session pending marker,
  - when widget re-opens, ComfyUI info refresh is resumed automatically and result is rendered back into ComfyUI card,
  - last explicitly refreshed ComfyUI status card is restored across widget switches in current session (and reset on full page/ComfyUI reload).
- Resume orchestration hardening:
  - pending restore flows (custom refresh, update jobs, ComfyUI info refresh) now run sequentially on widget startup,
  - avoids cross-flow UI races on `actionBusy`/process target and reduces startup flicker,
  - startup now uses a single coordinator: pending resumes first (if any), then catalog startup load, preventing parallel startup race between restore and catalog bootstrap.
  - added startup-settled callback contract in startup loader and unified `startupBusy` lock in UI controls, so action buttons stay disabled until startup bootstrap truly completes.
  - startup coordinator now starts catalog bootstrap immediately while restore flows run, so selectors/module cards are populated early and do not stay empty during long restore operations.
  - while startup/restore is active, picker UI is frozen (selectors/cards/buttons disabled) except progress/status rendering; state unfreezes only after startup settles.
  - extracted startup coordinator into dedicated orchestration module `web/orchestration/module_node_picker_startup_flow.js` (behavior preserved, lower complexity in main picker module).
  - extracted pending resume flows (Custom refresh, module update, ComfyUI info refresh) into dedicated orchestration module `web/orchestration/module_node_picker_resume_flow.js` with thin wrappers in picker (behavior preserved).
  - extracted runtime/session-state helpers into dedicated state module `web/state/module_node_picker_runtime_state.js` (pending/status markers + storage preferences), preserving existing behavior.
  - extracted busy/loading UI-state handling into dedicated orchestration module `web/orchestration/module_node_picker_busy_ui.js` (startup/action/catalog locks), keeping control-freeze behavior unchanged.
  - extracted static DOM layout construction into dedicated UI module `web/ui/module_node_picker_layout.js`, reducing main picker complexity while preserving element structure.
  - extracted debug/diagnostics panel orchestration into dedicated module `web/orchestration/module_node_picker_debug_ui.js` (toggle state, diag rendering, copy, cleanup), with picker behavior preserved.
  - extracted lifecycle-bound API wrapper creation into `web/orchestration/module_node_picker_api_client.js` (shared `AbortController` scope + dispose), keeping request-cancellation behavior unchanged.
  - extracted process/help/status text callbacks into dedicated module `web/orchestration/module_node_picker_view_helpers.js` to reduce picker composition complexity without UX changes.
  - extracted catalog/module loading controller into `web/orchestration/module_node_picker_catalog_controller.js` (request tokens, busy counters, option/diff caches), preserving existing loader behavior.
  - extracted top status-card orchestration into `web/orchestration/module_node_picker_status_cards.js` (Comfy/Custom card rendering + checked-state persistence), preserving card UX.
  - extracted picker runtime/store bootstrap into `web/state/module_node_picker_runtime_context.js` (store + diagnostics + runtime status accessors + mode storage wiring), behavior unchanged.
  - fixed startup regression after refactor: catalog controller now receives deferred `renderModuleInfo` callback to avoid TDZ access before function initialization.
- Canceled-request handling hardening:
  - added shared error-classifier `web/orchestration/module_node_picker_error_utils.js`,
  - resume/action/poll flows now suppress non-actionable warnings for intentionally canceled/aborted requests.
- Regression guardrails:
  - added frontend contract checks for pending/resume markers and session-runtime status in `tests/test_phase0_baseline.py`.
- ComfyUI mode-switch UX fix:
  - switching `ComfyUI check` now refreshes only ComfyUI status card (no full catalog reload),
  - prevents flicker/reset of group/module dropdowns during fast mode toggles.
- ComfyUI mode-switch behavior correction:
  - `ComfyUI check` selector now only persists mode choice,
  - backend refresh happens only on explicit `Refresh ComfyUI Info` action.
- Module UI/UX refinements:
  - module-card click now toggles node list (expand/collapse),
  - help/legend layout adjusted: hint near module card, legend shown between module card and node list,
  - ComfyUI status card is hidden on widget load and shown only after explicit ComfyUI refresh/actions.
- Strengthened frontend relay contract markers in `tests/test_phase0_baseline.py`.

## 0.16.10 — 2026-02-11
- Phase 2 start (tab-sync stabilization):
  - removed legacy container-ownership sync path from `web/module_node_picker.js`,
  - `Module Node Picker` now uses a single tab-sync mechanism via `web/module_node_picker_tab_relay.js`,
  - removed dual-path runtime switch and competing visibility controllers.
- Tab relay cleanup (same phase):
  - removed force-activation branch from relay (no forced tab switching),
  - removed duplicated `window` pointer/mouse/click listeners and kept document-level capture listeners,
  - removed dead `maybeForceRecentTab` logic and associated stale state,
  - added temporary `foreign tab intent` guard to keep picker detached right after foreign-tab clicks when sidebar state is stale.
- Updated refactoring plan status for Phase 2 to in-progress in EN/RU plan files.

## 0.16.9 — 2026-02-11
- Phase 1 continuation (frontend decomposition without UX changes):
  - added node/module card renderer module: `web/ui/module_node_picker_renderers.js`,
  - added top alert-card renderer module: `web/ui/module_node_picker_alerts.js`,
  - added inline process/progress controller module: `web/ui/module_node_picker_process.js`,
  - added catalog/group/module selector module: `web/ui/module_node_picker_catalog.js`,
  - added update-flow orchestration module: `web/orchestration/module_node_picker_update_flow.js`,
  - added catalog/module-info data-flow module: `web/orchestration/module_node_picker_data_flow.js`,
  - added module/card/top-refresh actions module: `web/orchestration/module_node_picker_actions.js`,
  - added event-binding/startup-load module: `web/orchestration/module_node_picker_bindings.js`.
- `web/module_node_picker.js` now delegates large UI blocks to modular helpers while preserving existing behavior and API calls.
- Verified with JS syntax checks (`node --check`) and regression tests (`34 passed`).

## 0.16.8 — 2026-02-10
- Module Node Picker: added backend console progress logs for `Refresh ComfyUI Info`.
  - On manual refresh request, ComfyUI console now prints refresh start parameters (`mode`, `acknowledge`).
  - After status collection, ComfyUI console prints refresh result summary (`update_status`, `update_available`, `local`, `remote`).

## 0.16.7 — 2026-02-10
- Phase 1 continuation (frontend decomposition without UX changes):
  - added API wrapper module `web/api/module_node_picker_api.js`,
  - added UI formatting helper module `web/ui/module_node_picker_formatters.js`,
  - added status-line helper module `web/ui/module_node_picker_status.js`,
  - added help-panel helper module `web/ui/module_node_picker_help.js`,
  - integrated both modules into `web/module_node_picker.js`.
- Widget refresh UX parity:
  - `Refresh Custom Nodes Info` now shows in-progress state directly in `Custom Nodes` card,
  - `Refresh ComfyUI Info` now shows in-progress state directly in `ComfyUI` card.
- `web/module_node_picker.js` reduced in size by moving API/display helpers out while preserving existing behavior.

## 0.16.6 — 2026-02-10
- Started Phase 1 (frontend foundation) for Module Node Picker.
- Added centralized picker state store: `web/state/store.js`.
  - persisted minimal keys: selected group, selected module, debug flag,
  - subscribe/unsubscribe API for incremental migration away from scattered local state.
- Added diagnostics logger module: `web/diagnostics/logger.js`.
  - levels: `info`, `warn`, `error`,
  - bounded in-memory entries,
  - runtime debug toggle support.
- Integrated Phase 1 foundation into `web/module_node_picker.js` without UX changes:
  - debug state now sourced from centralized store,
  - selected group/module now persisted through store,
  - diagnostics and sync logging routed through shared logger.
- Updated refactoring plan status:
  - `Phase 1` marked as **in progress** in RU/EN adopted plan documents.

## 0.16.5 — 2026-02-10
- Fixed dependency-update flow for ComfyUI and custom modules:
  - if `requirements.txt` changes after update, pending install state is now persisted in cache and survives restart;
  - pending state is cleared only after successful requirements installation.
- Module Node Picker UI:
  - ComfyUI status card now shows persistent `requirements.txt` pending state with `Install ComfyUI requirements` button;
  - custom module card now shows per-module `requirements.txt` pending state with `Install module requirements` button.
- This behavior now works for both update paths:
  - single module update (`Update module`),
  - bulk custom update (`Update Custom Nodes`).

## 0.16.4 — 2026-02-10
- Module novelty markers are now sticky across ComfyUI restarts:
  - updated modules stay marked (`✅`, green node frames),
  - new modules/nodes keep red/new markers,
  until explicit acknowledge.
- Added explicit acknowledge semantics:
  - `Обновить информацию о модуле` clears novelty markers for that module,
  - `Обновить информацию о модулях` clears novelty markers for all modules,
  - `Обновить информацию о ComfyUI` clears ComfyUI novelty marker after local update detection,
  - then shows current local/remote status (including red update square if upstream is newer).
- Startup behavior keeps cache-first policy, but now performs lightweight local change tracking to detect newly installed/updated modules between runs without forcing upstream refresh.
- Added local worktree change tracking for custom modules:
  - uncommitted git changes in module files are now detected between runs,
  - module receives persistent `✅` marker until explicit acknowledge.
- Added explicit local commit-change tracking:
  - any changed module commit SHA between runs marks the module as updated,
  - this does not depend on node-file changes and applies to module-wide updates.
- Module card rendering adjusted:
  - green border now appears not only for in-session updates but also for cached `updated_between_runs`,
  - card now always shows `Updated between runs` row for update markers (including local-change-only cases).
- Added tracker regression tests for persistent novelty markers and acknowledge flow (`tests/test_module_browser_tracker.py`), including global acknowledge and ComfyUI marker acknowledge.

## 0.16.3 — 2026-02-10
- Module Node Picker startup behavior changed to cache-first mode:
  - no automatic git/upstream status checks on widget load,
  - module and ComfyUI status at startup is read from cached state.
- Status checks are now triggered only by explicit widget actions:
  - `Обновить информацию о модулях`,
  - `Обновить информацию о модуле`,
  - `Обновить информацию о ComfyUI`.
- Frontend: removed automatic per-module badge probing on startup; badges are populated from `node_catalog` cache metadata.
- Backend:
  - `/alexz_tools/module_info` now supports `cache_only` mode (default on non-forced reads),
  - ComfyUI status now persists in module state cache and is reused without git calls unless force refresh is requested,
  - runtime-state initialization switched to lightweight cache bootstrap (no startup refresh job).
- Added regression tests for cache-only status behavior in `tests/test_phase0_baseline.py`.

## 0.16.2 — 2026-02-10
- Module Node Picker: diagnostics/debug output is now hidden by default.
- Added `Debug` checkbox in picker header to enable/disable diagnostics block on demand.
- Debug toggle also controls internal debug logging flag and persists between sessions via `localStorage`.

## 0.16.1 — 2026-02-10
- Phase 0 guardrails: expanded backend contract baseline in `tests/test_phase0_baseline.py`.
- Added regression checks for:
  - module query filtering behavior (`exact` priority over `partial`),
  - `module_nodes` payload shape built from catalog data,
  - frontend tab-relay marker contract (critical relay reasons + diagnostics keys).
- Extended `guides/GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md` with reproducible baseline steps and expected diagnostics fingerprints for the `Module Nodes -> NodesMap` empty-panel issue.
- Added manual Phase 0 smoke checklist to known-issues guide for repeatable pre/post-refactor verification.

## 0.16.0 — 2026-02-10
- Minor-version bump for large documentation/maintainability update across codebase.
- Standardized module headers in project Python/JS files with unified metadata block:
  - `Module`,
  - `Author`,
  - `Last updated`,
  - `Description`,
  - `Purpose`.
- Expanded JS code comments to informative JSDoc-style descriptions for key functions and internal helpers in:
  - `web/module_node_picker.js`,
  - `web/module_node_picker_tab_relay.js`,
  - `web/show_json.js`,
  - `web/video_cut_match_upload.js`.
- Updated package version in `pyproject.toml` and `README.md` to `0.16.0`.

## 0.15.11 — 2026-02-10
- Simplified `README.md` section `UI Tool: Module Node Picker`: replaced long inline block with short description and links.
- Added dedicated guide file `GUIde_Module_Node_Picker.md` with full widget usage and update workflow details.
- Added dedicated known-issues file `Known_issue_Module_Node_Picker.md` with `NodesMap` switching issue and workaround.
- Documented why inline HTML color in GitHub README may not render (style sanitization), switched to badge-based emphasis in known-issues file.

## 0.15.10 — 2026-02-10
- Refactored `Module Nodes` sidebar switching logic: moved Tab Relay implementation from `web/module_node_picker.js` into a dedicated module `web/module_node_picker_tab_relay.js`.
- Simplified relay lifecycle to explicit `bind/unbind` API and removed inlined high-risk reconnect logic from the main UI file.
- Kept diagnostics output format (`diag.*`) compatible while reducing coupling between rendering and sidebar event handling.
- Documented known `Module Nodes -> NodesMap` switching edge case and workaround in `README.md` (switch via another widget first).

## 0.15.9 — 2026-02-10
- Fixed sidebar content overlap for `Module Nodes` when switching to some third-party custom tabs (e.g. `NodesMap`, `PNG Info`) that append into shared sidebar container.
- Added passive visibility sync: `Module Nodes` root is auto-hidden whenever active sidebar tab is not `alexz-module-nodes`.
- This fix does not force tab switching and does not write to `activeSidebarTabId`; it only controls visibility of our own root.

## 0.15.8 — 2026-02-10
- Sidebar UX compatibility fix for `Module Nodes`:
  - added cleanup of stale fallback button instances on startup,
  - fallback button now has stable id (`alexz-module-nodes-fallback-btn`) and is deduplicated.
- Changed sidebar tab icon for `Module Nodes` from `pi pi-sitemap` to `pi pi-th-large` to avoid visual/behavioral conflicts with `NodesMap` tabs that use sitemap icon.

## 0.15.7 — 2026-02-10
- Fixed sidebar switching issue in `Module Nodes`: removed direct write to `activeSidebarTabId/activeSidebarTab` from fallback open flow.
- Added safe sidebar open call in fallback mode (`activateSidebarTab` only when API is provided by host).
- Added frontend singleton guard for `module_node_picker.js` to prevent duplicate extension registration in repeated script-init scenarios.

## 0.15.6 — 2026-02-10
- Module card now shows node-level change details as small lines:
  - `Обновлены ноды: ...`
  - `Добавлены ноды: ...`
- These lines are rendered inside module card (after `Updated between runs`), not in external help area.
- Changed node insert feedback text from `Добавлена: ...` to `Вставлена в граф: ...` to avoid confusion with update/change tracking.

## 0.15.5 — 2026-02-10
- `Module Nodes` update-flow UX fix: after `Update module`, selected module card is preserved (no reset to default module).
- Updated module card now gets a green border highlight in current session.
- After successful single-module update, node list for that module auto-expands.
- Catalog/group/module reload logic now supports preferred group/module preservation during refresh.

## 0.15.4 — 2026-02-10
- `Module Nodes` UX changed: selecting module now shows module card first; node list opens only when the module card is clicked.
- Added clickable module-card visual state and help hint for card-to-node-list interaction.
- Preserved module-card action buttons behavior (`Обновить информацию о модуле`, `Update module`) with click propagation blocked from opening node list unintentionally.

## 0.15.3 — 2026-02-10
- Fixed Module Nodes UI bug: module-card action buttons (`Обновить информацию о модуле`, `Update module`) no longer remain disabled after refresh/update action completes.
- `setActionBusy` now synchronizes disabled state for module-card action-row buttons after async operations.
- Module description sanitization improved: HTML wrapper lines (e.g. `<div align="center">`) are now removed from module card descriptions.

## 0.15.2 — 2026-02-10
- `Module Nodes` UI: added dedicated top button `Обновить информацию о ComfyUI` (refreshes ComfyUI status without full module refresh).
- Module card UI: added button `Обновить информацию о модуле` for targeted refresh of selected module info.
- Custom module status now explicitly shows red text `модуль требует обновления` when update is available; ComfyUI alert text updated to `ComfyUI требует обновления`.
- Backend: `/alexz_tools/module_info` now supports `refresh` and `sync_upstream` query flags for forced/targeted status refresh.
- Backend: added new endpoint `GET /alexz_tools/comfyui_info` for dedicated ComfyUI status refresh.
- Added regression test for forced module-info refresh with upstream sync (`tests/test_module_browser_tracker.py`).

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
