# Changelog — ALEXZ_tools

## 0.16.20 — 2026-02-11
- Phase 2 frontend decomposition continued:
  - extracted runtime-bootstrap callback adapters from composer into `web/orchestration/module_node_picker_runtime_bootstrap_bindings.js`,
  - `web/orchestration/module_node_picker_composer.js` now wires runtime bootstrap through prebuilt binding adapters (`createModuleNodePickerRuntimeBootstrapBindings`) to reduce inline lambda density and improve maintainability.
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
- Extended `guedes/GUIDE_KNOWN_ISSUES_MODULE_NODE_PICKER.md` with reproducible baseline steps and expected diagnostics fingerprints for the `Module Nodes -> NodesMap` empty-panel issue.
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
