# ComfyUI_ALEXZ_tools — Adopted Refactoring Plan

Status: approved working plan for incremental implementation
Date: 2026-02-10
Source documents: `PROPOSAL_REFACTORING.md`, `PROPOSAL_VUE_INTEGRATION_FIX.md`, `PROPOSAL_PHASE_1_IMPLEMENTATION.md` (+ RU versions)

## 1) Goal

Stabilize `Module Node Picker` tab behavior and reduce maintenance risk without breaking ComfyUI compatibility.

Primary constraints:
- No breaking API changes for current routes and widget behavior.
- Incremental migration with rollback points after each phase.
- Keep production behavior stable while internal architecture improves.

## 2) What We Adopt / Postpone / Reject

### Adopt now (high value, low-to-medium risk)

1. Split frontend concerns gradually:
- state handling
- diagnostics
- API wrappers
- render logic

2. Introduce centralized state store (`web/state/store.js`) for picker state.

3. Introduce standalone diagnostics logger (`web/diagnostics/logger.js`) and remove ad-hoc scattered debug logic.

4. Simplify tab sync to one dominant mechanism and eliminate competing visibility controllers.

5. Begin backend decomposition of `utils/module_node_browser_api.py` into focused internal modules while preserving existing HTTP routes.

### Postpone (implement only after stabilization)

1. Deep async rewrite of all backend jobs.
2. Node base-class unification for every node file.
3. CLI tooling package for standalone usage.
4. Full E2E stack before core stabilization is complete.

### Reject as-is (needs redesign)

1. Vue heuristics tied to specific CSS class names (`vue-entering`, `vue-temp-wrapper`) as a core mechanism.
2. Proposed snippets that are not valid JS runtime in this project (e.g. TS-only syntax inside `.js`).
3. Broad concurrent `asyncio.gather` for git operations without concurrency limits.

## 3) Known Risks in Source Proposals (and Mitigation)

1. Invalid JS/TS mix in examples
- Risk: runtime failures.
- Mitigation: keep implementation in plain JS compatible with current ComfyUI web runtime.

2. Over-aggressive Vue coupling
- Risk: fragile behavior across ComfyUI builds.
- Mitigation: use container-ownership rules and deterministic DOM attachment, not Vue class heuristics.

3. Big-bang refactor
- Risk: regressions across module picker, updates, and status UI.
- Mitigation: phase-gated rollout with tests after each phase.

4. Backend async fan-out
- Risk: IO saturation and race conditions.
- Mitigation: bounded workers/queue and staged migration.

## 4) Execution Phases

## Phase 0 — Baseline and Guardrails

Status: ✅ completed (2026-02-10, commit `50a93f3`)

Deliverables:
- Freeze current behavior with regression checks for:
  - catalog loading
  - module info loading
  - update status polling
  - tab switching transitions
- Add/update smoke tests around critical API payload fields.

Exit criteria:
- Baseline tests pass.
- Current known issue reproduction is documented and repeatable.

## Phase 1 — Frontend Foundation (Store + Diagnostics)

Status: ✅ completed (2026-02-11)

Deliverables:
- Add `web/state/store.js`:
  - single source of truth for picker UI state
  - subscribe/unsubscribe APIs
  - minimal persisted keys only (`selectedGroup`, `selectedModule`, debug flag)
- Add `web/diagnostics/logger.js`:
  - leveled logs (`info`, `warn`, `error`)
  - bounded memory log buffer
  - opt-in debug mode
- Integrate these modules into `web/module_node_picker.js` without changing UI behavior.
- Extracted large picker sections into focused modules while preserving behavior:
  - UI: `web/ui/module_node_picker_renderers.js`, `web/ui/module_node_picker_alerts.js`,
    `web/ui/module_node_picker_process.js`, `web/ui/module_node_picker_catalog.js`
  - orchestration: `web/orchestration/flow/progress/module_node_picker_update_flow.js`,
    `web/orchestration/flow/catalog/module_node_picker_data_flow.js`,
    `web/orchestration/flow/actions/module_node_picker_actions.js`,
    `web/orchestration/core/infra/module_node_picker_bindings.js`

Exit criteria:
- No UX regressions.
- Diagnostics can be toggled at runtime.
- Existing picker workflows still work.

## Phase 2 — Tab-Sync Stabilization

Status: ✅ completed (2026-02-12)

Deliverables:
- Keep one tab synchronization mechanism as primary path.
- Remove or hard-disable competing sync paths that cause ownership conflicts.
- Ensure deterministic attach/detach for picker root when tab changes.
- Third-party `NodesMap` direct-switch issue is tracked as a known issue and scoped out of this phase.
- Progress checkpoint (2026-02-11):
  - removed legacy/competing sync paths and kept single relay path,
  - split relay internals into focused modules:
    - `web/orchestration/module_node_picker_tab_relay_helpers.js`
    - `web/orchestration/module_node_picker_tab_relay_runtime.js`
  - removed broad unknown-tab fallback detach logic from content clicks,
  - reduced relay event noise (single pointer/mouse path, no per-button listeners),
  - added relay runtime debounce and explicit dispose cleanup on unbind,
  - reduced passive tick pressure when picker tab is inactive.
  - constrained relay click detection to sidebar-context tab controls only (fewer false positives from content area clicks).
  - constrained keyup-driven relay sync to tab/navigation keys and non-input targets.
  - introduced bind-token guards so stale relay callbacks from prior binds cannot mutate current tab visibility.
  - removed unused relay fallback tab-id heuristic helper to simplify sync surface.
  - replaced fixed relay interval loop with adaptive timeout scheduling to lower background sync pressure.
  - narrowed relay tab-candidate matching to explicit tab-like sidebar controls (fewer false-positive sidebar button captures).
  - added catalog-load request token guard to prevent stale async catalog responses from overwriting active UI state.
  - added picker render-lifecycle cleanup hook and module-info request token guard to prevent stale async UI writes between re-renders.
  - cleaned relay helpers by removing unused API and centralizing tab/sidebar selector constants.
  - added disposed-instance early-exit guards for catalog/module-info loads plus debounced ComfyUI-check mode reload.
  - introduced explicit UI event lifecycle cleanup (unbind callbacks + timer cleanup + store unsubscribe on picker dispose).
  - propagated liveness-aware cancellation guards through refresh/update/install orchestration to prevent stale post-dispose UI mutations.
  - added explicit process-controller dispose lifecycle to fully clear/detach progress host between picker re-renders.
  - deduplicated lifecycle guard logic into a shared orchestration helper module.
  - corrected startup liveness semantics to lifecycle-based checks, preventing first-open catalog skip/empty-select regression.
  - added bounded startup catalog retry with explicit dispose cancellation to handle transient initial empty backend state.
  - added selector loading-state placeholders with disable/enable lifecycle during catalog load to avoid transient empty dropdown UX.
  - added frontend API timeout boundaries (AbortController) to prevent indefinite pending requests from blocking picker UX.
  - bound all picker API calls to per-render lifecycle `AbortController` and cancel in-flight requests on picker dispose/re-render.
  - decoupled ComfyUI-check mode switch from full catalog reload to stabilize selector UI during rapid mode toggles.
  - finalized ComfyUI-check selector semantics: mode switch is config-only; network refresh occurs only by explicit user action.
  - extracted category/group/module dropdown orchestration into `web/orchestration/ui/module_node_picker_selection_controller.js` (population/filtering/store sync), preserving existing UI behavior.
  - extracted long-running action orchestration (refresh/update/resume/requirements follow-up + per-module refresh/install) into `web/orchestration/flow/actions/module_node_picker_action_flows.js`, leaving `web/module_node_picker.js` with thin dependency wiring.
  - extracted token-based polling lifecycle (refresh/update progress loops) into `web/orchestration/module_node_picker_polling_controller.js`; picker dispose now invalidates polling via a dedicated controller API.
  - extracted module-card/node-list rendering orchestration (including expand/collapse UI state) into `web/orchestration/module_node_picker_module_panel_controller.js`.
  - extracted picker instance lifecycle/dispose controller into `web/orchestration/module_node_picker_lifecycle.js` for centralized token/polling/event/startup/debug/process/API cleanup.
  - extracted extension registration/fallback mount flow into `web/orchestration/core/infra/module_node_picker_registration.js`.
  - centralized picker constants (IDs/storage keys/group labels/marks/defaults) in `web/constants/module_node_picker_constants.js`.
  - extracted LiteGraph node create/place helpers into `web/ui/module_node_picker_node_factory.js`.
  - moved full picker composition from `web/module_node_picker.js` to `web/orchestration/core/composition/module_node_picker_composer.js`; main entry file now only handles extension registration/fallback wiring.
  - extracted selector/busy/view/status-card wiring into `web/orchestration/ui/module_node_picker_ui_controllers.js` to further thin composition code.
  - extracted polling/catalog/actions/module-panel composition into `web/orchestration/module_node_picker_flow_wiring.js` to keep runtime orchestration modular.
  - extracted runtime bootstrap (event binding, ComfyUI-card restore, startup coordinator wiring) into `web/orchestration/module_node_picker_runtime_bootstrap.js`.
  - extracted base runtime setup (runtime context, lifecycle, API client, debug/process controllers) into `web/orchestration/module_node_picker_runtime_setup.js`.
  - extracted UI-stage assembly into `web/orchestration/ui/module_node_picker_ui_stage.js` (selector/busy/view/status controller composition adapter).
  - extracted flow-stage assembly into `web/orchestration/module_node_picker_flow_stage.js` (polling/catalog/action/module-panel composition adapter).
  - extracted large composer dependency maps into `web/orchestration/core/composition/module_node_picker_context_builders.js` (runtime-setup/ui-stage/flow-stage/runtime-bootstrap context builders), reducing `module_node_picker_composer.js` size while preserving behavior.
  - split pending resume internals into focused modules (`module_node_picker_resume_custom_refresh.js`, `module_node_picker_resume_module_update.js`, `module_node_picker_resume_comfy_refresh.js`) while preserving stable facade exports in `module_node_picker_resume_flow.js`.
  - extracted runtime warmup polling orchestration into `web/orchestration/module_node_picker_warmup_controller.js` and kept first-open marker auto-hydration behavior in background.
  - extracted relay adaptive tick-loop into `web/orchestration/module_node_picker_tab_relay_tick.js` and kept binding/runtime behavior unchanged.
  - extracted relay tab-intent/event orchestration into `web/orchestration/module_node_picker_tab_relay_intent.js` and simplified `module_node_picker_tab_relay.js` listener wiring.
  - extracted Module Node Picker CSS into `web/orchestration/styles/module_node_picker_styles.js` with grouped, documented style sections to separate presentation tuning from orchestration logic.
  - moved Module Node Picker CSS module into UI layer (`web/ui/styles/module_node_picker_styles.js`) to align folder ownership with visual responsibilities.
  - extracted deferred-stage bridge wiring from composer into `web/orchestration/core/composition/module_node_picker_stage_bridge.js` so flow-stage handoff and adapter callbacks remain centralized and easier to evolve.
  - extracted runtime-bootstrap callback bindings from composer into `web/orchestration/module_node_picker_runtime_bootstrap_bindings.js` to reduce inline callback density in composition code.
  - extracted runtime-setup projection/unpacking into `web/orchestration/module_node_picker_runtime_projection.js`, reducing flat-field mapping noise in composer.
  - fixed warmup indicator stalling by wiring warmup poller to catalog reload path and adding fail-safe indicator reset on retry-budget exhaustion/poll errors.
  - introduced semantic orchestration subfolders:
    - `web/orchestration/relay/` for tab-relay internals,
    - `web/orchestration/runtime/` for runtime/bootstrap/lifecycle/warmup internals,
    and updated imports/tests accordingly.
  - introduced `web/orchestration/flow/` subfolder for pipeline/data/update/resume orchestration modules and moved related files out of flat root folder with updated imports/test markers.
  - introduced additional semantic orchestration subfolders:
    - `web/orchestration/core/` for composition/bootstrap adapters and shared orchestration helpers,
    - `web/orchestration/ui/` for picker UI controllers/views/status-card orchestration,
    - `web/orchestration/api/` for lifecycle-bound API client wrapper.
  - moved remaining flat `module_node_picker_*` orchestration modules into `core/ui/api` groups, updated all dependent imports, updated module headers, and updated frontend baseline contract test paths.
  - fixed post-move runtime import paths in `module_node_picker_runtime_setup.js` (`process` UI + runtime context imports) to preserve runtime behavior.
  - split flow orchestration by responsibility:
    - `web/orchestration/flow/progress/` for refresh/update polling and progress loops,
    - `web/orchestration/flow/resume/` for pending-refresh/update/comfy resume flows.
  - moved `module_node_picker_update_flow.js` and `module_node_picker_polling_controller.js` into `flow/progress/`, moved resume modules into `flow/resume/`, then updated all dependent imports, module headers, and baseline test path markers.
  - split runtime orchestration by responsibility:
    - `web/orchestration/runtime/bootstrap/` for runtime setup/bootstrap/startup/warmup modules,
    - `web/orchestration/runtime/lifecycle/` for lifecycle guards and picker instance disposal.
  - moved runtime modules into `bootstrap/lifecycle` groups, updated dependent imports in composer/flow/runtime modules, updated module headers, and refreshed baseline test path markers.
  - split remaining flow responsibilities into dedicated subfolders:
    - `web/orchestration/flow/actions/` for action handlers and composed action flows,
    - `web/orchestration/flow/catalog/` for catalog/module-info data controllers and loaders.
  - moved `module_node_picker_actions.js` + `module_node_picker_action_flows.js` into `flow/actions/`, moved `module_node_picker_catalog_controller.js` + `module_node_picker_data_flow.js` into `flow/catalog/`, then updated dependent imports, module headers, and baseline test path markers.
  - split remaining flat flow files by concern:
    - `web/orchestration/flow/stage/` for stage adapters (`flow_stage`, `flow_wiring`),
    - `web/orchestration/flow/panel/` for module panel rendering controller.
  - moved `module_node_picker_flow_stage.js` + `module_node_picker_flow_wiring.js` into `flow/stage/`, moved `module_node_picker_module_panel_controller.js` into `flow/panel/`, then updated dependent imports, module headers, and baseline test path markers.
  - split orchestration core internals by concern:
    - `web/orchestration/core/composition/` for composition/stage-bridge/context assembly,
    - `web/orchestration/core/infra/` for bindings/error/registration infrastructure.
  - moved `module_node_picker_composer.js` + `module_node_picker_context_builders.js` + `module_node_picker_stage_bridge.js` into `core/composition/`, moved `module_node_picker_bindings.js` + `module_node_picker_error_utils.js` + `module_node_picker_registration.js` into `core/infra/`, then updated dependent imports, module headers, plan path references, and baseline test path markers.
  - removed fragile deep `scripts/app.js` import from composer and switched to dependency injection from entrypoint (`renderModuleNodePicker(container, { appInstance: app })`) to reduce startup break risk after folder moves.
  - extracted relay bind/unbind implementation into `web/orchestration/relay/module_node_picker_tab_relay_facade.js` and converted `web/module_node_picker_tab_relay.js` into a stable re-export entrypoint, preserving existing import path while reducing root-level orchestration density.
  - centralized relay global state access in `web/orchestration/relay/module_node_picker_tab_relay_state.js` and switched relay facade to use shared state helpers for read/write/clear operations.
  - centralized relay reasons/timing thresholds in `web/orchestration/relay/module_node_picker_tab_relay_constants.js` and switched relay `runtime/intent/tick/facade` modules to constants-driven flow.
  - unified relay immediate-reason debounce bypass logic through shared `isImmediateRelayReason()` helper from relay constants module.
  - reduced relay diagnostics overhead by forwarding tab-relay diagnostics to debug panel only when debug mode is enabled.
  - added browser-context guards in relay facade bind/unbind path to prevent crashes in partial/non-browser initialization contexts.
  - added explicit `mountHost` relay wiring from composer to relay runtime so attach/detach recovery can rebind picker root against current sidebar render host when original parent changes.
  - normalized relay bind input (`root` element guard in facade) and improved runtime host preference so connected root is moved back under preferred mount host when container ownership drifts.
  - extracted relay DOM candidate/sidebar-context detection into `web/orchestration/relay/module_node_picker_tab_relay_dom.js` and switched relay helpers to shared DOM helper functions.
  - extracted relay bind-state lifecycle (state construction + deterministic dispose/unbind) into `web/orchestration/relay/module_node_picker_tab_relay_lifecycle.js` and switched facade to shared lifecycle helpers.
  - extracted relay diagnostics payload/deduplicated emit path into `web/orchestration/relay/module_node_picker_tab_relay_diagnostics.js` and switched relay runtime to diagnostics helpers.
  - extracted relay DOM ownership/host-recovery (attach/detach) into `web/orchestration/relay/module_node_picker_tab_relay_dom_ownership.js` and switched runtime visibility path to shared ownership controller.
  - extracted composer relay-bind wiring into `web/orchestration/core/composition/module_node_picker_relay_bridge.js` so composition body keeps relay boot logic in a focused bridge module.

Exit criteria:
- Repeated transitions for standard tabs are stable (`Module Nodes -> Workflows/PNG Info -> Module Nodes`).
- No duplicate roots and no empty panel regressions inside `Module Node Picker` itself.
- External direct-switch `NodesMap` issue is documented with workaround in `guides/GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md`.

## Slice 0 — Extensibility Foundation (nodes/widgets lifecycle)

Status: ✅ completed (2026-02-12)

Deliverables:
- Introduce a registry-first layer for module components (`nodes`, `widgets`, `api`) with auto-discovery.
- Introduce stable backend contracts (schema/versioned payload) and versioned runtime cache.
- Define single registration/unregistration points so adding/removing nodes/widgets does not require multi-file manual edits.
- Add manifest health-report and deterministic `manifest_signature` for quick component drift checks.
- Add minimal contract tests for:
  - new component added,
  - component removed,
  - route payload keys unchanged.

Exit criteria:
- Adding/removing a node or widget is done via registry with no cross-file manual wiring.
- API/UX behavior remains unchanged.
- Baseline tests pass.

## Phase 3 — Backend Modular Split (No API Changes)

Status: 🔄 in progress (2026-02-13)

Deliverables:
- Internally split `utils/module_node_browser_api.py` into focused modules:
  - catalog collection/building
  - git state/sync helpers
  - module info assembly
  - refresh/update job orchestration
- Keep route signatures and payload keys backward-compatible.
- Step 1 completed: refresh/update job helpers (status handling + update target
  resolution) moved to `utils/module_browser/jobs.py` with facade-compatible wrappers.
- Step 2 completed: catalog assembly (`collect/build/filter`) moved to
  `utils/module_browser/catalog.py` with compatibility wrappers preserved in API file.
- Step 3 completed: module-info text helpers (README summary + description sanitize)
  moved to `utils/module_browser/module_info_text.py`.
- Step 4 completed: module-info payload assembly and cached module-badge flags
  moved to `utils/module_browser/module_info.py` with facade-compatible wrappers
  preserved in `utils/module_node_browser_api.py`.
- Step 5 completed: git state/sync helper layer (`remote pick/resolve`, release-tag
  ref resolve, custom module git-state/sync, worktree signature) moved to
  `utils/module_browser/git_helpers.py` with API facade wrappers preserved.
- Step 6 completed: requirements-diff/install operations moved to
  `utils/module_browser/update_ops.py` (`requirements_changed_between`,
  module/comfy requirements install) with facade wrappers preserved.
- Step 7 completed: git pull/update orchestration helpers moved to
  `utils/module_browser/pull_ops.py` (`is_git_local_changes_block`,
  `pull_comfyui`, `pull_custom_module`) with facade wrappers preserved.
- Step 8 completed: batch requirements install aggregation moved to
  `utils/module_browser/update_ops.py` (`install_requirements_for_modules`)
  with API facade wrapper preserved.
- Step 9 completed: module-state file IO helpers moved to
  `utils/module_browser/state_store.py` (`load_state_file`,
  `save_state_file`) with API cache facade preserved.
- Step 10 completed: tracker/novelty operations moved to
  `utils/module_browser/tracker_ops.py` (`remember/apply/acknowledge`,
  `announce_tracked_module_updates`) with API facade wrappers preserved.
- Step 11 completed: ComfyUI startup novelty tracking helpers moved to
  `utils/module_browser/comfyui_tracking_ops.py`
  (`track_comfyui_local_update`, `acknowledge_comfyui_novelty`)
  with API facade wrappers preserved.
- Step 12 completed: node snapshot/path helpers moved to
  `utils/module_browser/node_snapshot_ops.py`
  (`node_source_file`, `relative_to_custom_roots`, `file_digest`,
  `build_node_snapshots`) with API facade wrappers preserved.
- Step 13 completed: runtime refresh phase orchestration moved to
  `utils/module_browser/runtime_refresh_ops.py`
  (`refresh_module_runtime_state`) with API facade wrapper preserved.
- Step 14 completed: module update job execution logic moved to
  `utils/module_browser/update_job_ops.py`
  (`run_module_update_job`) with API worker-thread wrapper preserved.
- Step 15 completed: refresh job execution logic moved to
  `utils/module_browser/refresh_job_ops.py`
  (`run_refresh_job`) with API worker-thread wrapper preserved.
- Step 16 completed: custom module identity helpers moved to
  `utils/module_browser/module_identity.py`
  (`discover/normalize/alias/canonical`) with API facade wrappers preserved.
- Step 17 completed: ComfyUI status/state merge helpers moved to
  `utils/module_browser/comfyui_state_ops.py`
  (template/cache resolve/pending merge/state persist) with API facade wrappers preserved.
- Step 18 completed: ComfyUI git-status orchestration moved to
  `utils/module_browser/comfyui_git_status_ops.py`
  (`collect_comfyui_git_status`) with API facade wrapper preserved.
- Step 19 completed: component-registry payload orchestration moved to
  `utils/module_browser/component_registry_payload_ops.py`
  (`collect_component_registry_payload`) with API facade wrapper preserved.
- Step 20 completed: ComfyUI-Manager metadata/statistics helpers moved to
  `utils/module_browser/manager_data_ops.py`
  (`load_manager_index`, `load_manager_github_stats`, `resolve/infer` helpers)
  with API facade wrappers preserved.
- Step 21 completed: subprocess/git command execution helpers moved to
  `utils/module_browser/command_ops.py`
  (`run_command`, `run_git`, `extract_git_repo_from_args`,
  `is_git_dubious_ownership_error`, `try_mark_git_safe_directory`, `tail_lines`)
  with API facade wrappers preserved.
- Step 22 completed: catalog-route payload builders moved to
  `utils/module_browser/catalog_payload_ops.py`
  (`build_group_payload`, `build_module_list_payload`,
  `build_module_nodes_payload`) with API facade wrappers preserved.
- Step 23 completed: widget-mode/log-mode helpers moved to
  `utils/module_browser/widget_mode_ops.py`
  (`custom_update_checked_flag`, `info_only_rejection_payload`,
  `set_custom_update_checked`, `normalize_log_mode`) with API facade wrappers preserved.
- Step 24 completed: pure value/date/repository helpers moved to
  `utils/module_browser/value_ops.py`
  (`short_commit`, `normalize_repo_url`, `github_id`, `repo_name`,
  `pick_repo_url`, `parse_datetime`, `to_iso`, `now_iso`,
  `normalize_comfyui_mode`) with API facade wrappers preserved.
- Step 25 completed: requirements-pending state mutation helpers moved to
  `utils/module_browser/requirements_pending_ops.py`
  (`set_comfyui_requirements_pending`, `set_module_requirements_pending`)
  with API facade wrappers preserved.
- Step 26 completed: path resolution helpers moved to
  `utils/module_browser/path_ops.py`
  (`custom_nodes_roots`, `manager_custom_db_path`,
  `manager_github_stats_path`, `module_dir`, `comfyui_root`)
  with API facade wrappers preserved.
- Step 27 completed: GitHub latest-release network helper moved to
  `utils/module_browser/release_ops.py`
  (`github_latest_release`) with API facade wrapper preserved.
- Step 28 completed: update-state decision helpers moved to
  `utils/module_browser/module_update_state_ops.py`
  (`module_needs_update_now`, `count_custom_modules_need_update`,
  `count_custom_modules_unknown_update`, `comfyui_needs_update_now`)
  with API facade wrappers preserved.
- Step 29 completed: repository bootstrap helpers moved to
  `utils/module_browser/repo_bootstrap_ops.py`
  (`comfyui_requirements_path`, `bootstrap_module_remote_from_manager`)
  with API facade wrappers preserved.
- Step 30 completed: node classification/annotation helpers moved to
  `utils/module_browser/node_classification_ops.py`
  (`module_root`, `classify_by_source_path`, `classify_by_relative_module`,
  `fallback_annotation`) with API facade wrappers preserved.

Exit criteria:
- Existing frontend works without API changes.
- Internal modules are easier to test in isolation.

## Phase 4 — Hardening and Coverage

Deliverables:
- Add integration tests for critical routes.
- Add frontend behavioral checks for tab transitions and refresh/update progress state.
- Add lightweight CI test workflow for Python tests and docs check.

Exit criteria:
- Coverage meaningfully improved on changed modules.
- No known blocking regression in picker workflows.

## 5) Engineering Rules for This Refactor

1. Preserve public behavior unless explicitly approved to change.
2. One migration axis per PR (state, sync, backend split, tests).
3. No new global `window[...]` state unless temporary and documented.
4. Feature-flag risky transitions where possible.
5. Keep rollback simple: each phase should be reversible independently.

## 6) Rollback Strategy

Per phase rollback:
- Phase 1: switch back to legacy local state paths.
- Phase 2: restore previous tab relay module if new sync path regresses.
- Phase 3: keep route handlers as stable facade to allow internal module fallback.

Repository practice:
- Small commits with explicit scope.
- No mixed refactor + feature changes in one commit.

## 7) Success Metrics

Functional:
- Stable repeated tab switching with no blank panel after second/third cycles.
- No regression in module refresh/update operations.

Maintainability:
- `web/module_node_picker.js` significantly reduced by moving infra logic out.
- `utils/module_node_browser_api.py` reduced to orchestration facade.

Quality:
- Increased automated checks for both frontend behavior (at least smoke-level) and backend route payloads.

## 8) Immediate Next Step

Start with **Slice 0** implementation, then proceed to Phase 3 while preserving API and UI behavior parity.
