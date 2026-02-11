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
  - orchestration: `web/orchestration/module_node_picker_update_flow.js`,
    `web/orchestration/module_node_picker_data_flow.js`,
    `web/orchestration/module_node_picker_actions.js`,
    `web/orchestration/module_node_picker_bindings.js`

Exit criteria:
- No UX regressions.
- Diagnostics can be toggled at runtime.
- Existing picker workflows still work.

## Phase 2 — Tab-Sync Stabilization

Status: 🚧 in progress (started 2026-02-11)

Deliverables:
- Keep one tab synchronization mechanism as primary path.
- Remove or hard-disable competing sync paths that cause ownership conflicts.
- Ensure deterministic attach/detach for picker root when tab changes.
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
  - decoupled ComfyUI-check mode switch from full catalog reload to stabilize selector UI during rapid mode toggles.

Exit criteria:
- Repeated transitions `Module Nodes -> NodesMap -> Module Nodes` stay stable across multiple cycles.
- No duplicate roots and no empty panel regressions.

## Phase 3 — Backend Modular Split (No API Changes)

Deliverables:
- Internally split `utils/module_node_browser_api.py` into focused modules:
  - catalog collection/building
  - git state/sync helpers
  - module info assembly
  - refresh/update job orchestration
- Keep route signatures and payload keys backward-compatible.

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

Start Phase 1 implementation from this plan, but with corrected runtime-safe code (plain JS, no TS syntax), and with strict parity to current user-visible behavior.
