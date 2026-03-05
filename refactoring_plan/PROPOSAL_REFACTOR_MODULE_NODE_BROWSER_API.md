# Proposal: Refactor `utils/module_node_browser_api.py`

## Context
`utils/module_node_browser_api.py` currently contains:
- HTTP route wiring (aiohttp/PromptServer)
- long-running job orchestration and status state
- cache/TTL management and persisted state I/O
- domain operations glue across `utils/module_browser/*`

This creates a large "god module" that is hard to test, hard to change safely, and easy to break with small edits.

## Goals
- Keep all HTTP routes and payload schemas backward-compatible.
- Reduce file size and responsibility mixing by splitting the module into focused units.
- Make state/caches explicit and testable (no hidden globals leaking across tests).
- Prepare for future features (more status fields, more UI signals) without growing complexity.
- Preserve current runtime behavior (threading model, locks, TTLs, logging conventions).

## Non-Goals
- No user-facing behavior changes (texts, statuses, default values) unless explicitly noted.
- No redesign of the frontend widget orchestration.
- No large renaming of existing public functions inside `utils/module_browser/*`.
- No migration to async job execution model (threads remain as-is).

## Proposed Target Structure
Create a dedicated package:
- `utils/module_browser_api/`

Suggested modules:
- `utils/module_browser_api/state.py`
  - holds all runtime globals currently in `_REFRESH_STATUS`, `_UPDATE_STATUS`, caches, locks, TTL constants
  - exposes a `get_state()` singleton to avoid import-time side effects in tests
- `utils/module_browser_api/logging_ops.py`
  - `_update_console_log`, `_refresh_console_log`, log-mode normalization helpers
- `utils/module_browser_api/node_introspection.py`
  - `_node_mappings`, `_build_node_snapshots`, relative-path helpers
- `utils/module_browser_api/handlers_refresh.py`
  - `_start_refresh_job`, status snapshot routes, refresh progress callback wiring
- `utils/module_browser_api/handlers_update.py`
  - update-job start/status, update target resolution, requirements tracking integration
- `utils/module_browser_api/handlers_catalog.py`
  - `/alexz_tools/node_catalog` handler and payload assembly
- `utils/module_browser_api/routes.py`
  - PromptServer/aiohttp route registration only
- `utils/module_browser_api/__init__.py`
  - small compatibility re-export surface (see Compatibility Plan)

Keep `utils/module_node_browser_api.py` as a thin compatibility shim:
- imports and calls `utils.module_browser_api.routes.register_routes(...)`
- re-exports any symbols used elsewhere (if any)

## Phased Plan (Safe, Reviewable Steps)

### Phase 0: Inventory and Freeze
- Add a short developer note documenting:
  - existing routes and their payload keys
  - existing global state keys for refresh/update statuses
  - current persisted state path and schema versioning expectations
- Add a smoke test that imports the backend module without ComfyUI runtime (if missing).

Exit criteria:
- Tests pass unchanged.
- No runtime behavior change.

### Phase 1: Extract State and Console Logging
- Move all globals, locks, TTLs, and status templates into `utils/module_browser_api/state.py`.
- Move console logging helpers and log-mode normalization into `logging_ops.py`.
- Update existing code to reference `state.<...>` instead of module globals.
- Ensure tests that set caches to `None` still work (explicit `ensure_*` helpers stay or move).

Exit criteria:
- No route behavior change (payload keys, defaults).
- `conda run -n p313 pytest -q tests/test_module_browser_jobs.py` passes.
- `conda run -n p313 pytest -q tests/test_module_browser_runtime_refresh_ops.py` passes.

### Phase 2: Extract Route Handlers (Refresh/Update)
- Split refresh/update job handlers into `handlers_refresh.py` and `handlers_update.py`.
- Keep internal helper names stable where tests patch them (or provide aliases).
- Keep threading semantics identical (same locks, same thread names, same progress callbacks).

Exit criteria:
- `conda run -n p313 pytest -q tests/test_module_browser_refresh_job_ops.py` passes.
- `conda run -n p313 pytest -q tests/test_module_browser_update_job_ops.py` passes.

### Phase 3: Extract Catalog and Introspection
- Move node mapping and snapshot helpers into `node_introspection.py`.
- Move node-catalog route into `handlers_catalog.py`.
- Ensure any caching behavior stays identical (TTL, invalidation points).

Exit criteria:
- `conda run -n p313 pytest -q tests/test_module_browser_catalog.py` passes.
- `conda run -n p313 pytest -q tests/test_module_browser_catalog_payload_ops.py` passes.

### Phase 4: Routes Module and Compatibility Shim
- Add `routes.py` that registers all PromptServer routes.
- Convert `utils/module_node_browser_api.py` to a thin shim.
- Verify that import side effects remain acceptable (or explicitly minimal).

Exit criteria:
- `conda run -n p313 pytest -q` passes.
- ComfyUI loads the extension and widget works unchanged.

## Compatibility Plan
- Keep route paths identical (as declared in `utils/module_browser/catalog/api_manifest.py`).
- Keep top-level module name `utils.module_node_browser_api` importable.
- Maintain all response JSON key names.
- Keep the same default "monitor/info-only" mode behavior.

## Risks and Mitigations
- Risk: test monkeypatches rely on old symbol locations.
  - Mitigation: keep stable aliases in the shim for one release; update tests gradually.
- Risk: import-time side effects change route registration.
  - Mitigation: explicit `register_routes()` entry point in `routes.py`, called only when PromptServer is available.
- Risk: subtle global state differences after move.
  - Mitigation: state object with explicit initialization and snapshot helpers; add asserts in smoke tests.

## Follow-ups (Optional, After Refactor)
- Make persisted state path configurable (env var or `folder_paths` output/temp dir).
- Unify `_resolve_compute_device` across nodes into a shared utility.
- Add "contract tests" for API payload schemas (golden JSON minimal fixtures).

## Validation Commands (Conda env `p313`)
- `conda run -n p313 python utils/docs_check.py`
- `conda run -n p313 pytest -q tests/test_module_browser_jobs.py`
- `conda run -n p313 pytest -q tests/test_module_browser_refresh_job_ops.py`
- `conda run -n p313 pytest -q tests/test_module_browser_update_job_ops.py`
- `conda run -n p313 pytest -q`

