"""
Module: tests/test_phase0_baseline.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Phase 0 baseline guardrail tests for Module Node Picker backend.

Purpose:
    Freeze critical payload contracts for catalog/module-info/update-status and
    provide regression protection before deeper refactoring phases.
"""

import importlib
import os
from pathlib import Path
import sys
import types
import unittest


def _install_folder_paths_stub():
    """Install minimal folder_paths stub for non-Comfy test runtime."""
    if "folder_paths" in sys.modules:
        stub = sys.modules["folder_paths"]
        if not hasattr(stub, "get_folder_paths"):
            stub.get_folder_paths = lambda kind: [os.path.join(os.getcwd(), "custom_nodes")]
        return
    stub = types.SimpleNamespace(
        get_folder_paths=lambda kind: [os.path.join(os.getcwd(), "custom_nodes")]
    )
    sys.modules["folder_paths"] = stub


class Phase0BaselineContractsTests(unittest.TestCase):
    """Guardrail tests for baseline API contract and status payloads."""

    @classmethod
    def setUpClass(cls):
        """Prepare dynamic package import context for test runtime."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()

    def setUp(self):
        """Load backend module and capture patchable symbols."""
        self.api = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_node_browser_api")

        self._orig_collect_nodes = self.api._collect_nodes
        self._orig_discover_custom_modules = self.api._discover_custom_modules
        self._orig_module_git_state = self.api._module_git_state
        self._orig_sync_module_upstream = self.api._sync_module_upstream
        self._orig_manager_index = self.api._manager_index
        self._orig_manager_github_stats = self.api._manager_github_stats
        self._orig_module_repo_url = self.api._module_repo_url
        self._orig_module_local_readme_summary = self.api._module_local_readme_summary
        self._orig_remember_module_state = self.api._remember_module_state
        self._orig_apply_node_change_info = self.api._apply_node_change_info
        self._orig_announce_tracked_module_updates = self.api._announce_tracked_module_updates
        self._orig_comfyui_git_status = self.api._comfyui_git_status

        self._orig_module_info_cache = dict(self.api._MODULE_INFO_CACHE)
        self.api._MODULE_INFO_CACHE.clear()

    def tearDown(self):
        """Restore patched module symbols and caches."""
        self.api._collect_nodes = self._orig_collect_nodes
        self.api._discover_custom_modules = self._orig_discover_custom_modules
        self.api._module_git_state = self._orig_module_git_state
        self.api._sync_module_upstream = self._orig_sync_module_upstream
        self.api._manager_index = self._orig_manager_index
        self.api._manager_github_stats = self._orig_manager_github_stats
        self.api._module_repo_url = self._orig_module_repo_url
        self.api._module_local_readme_summary = self._orig_module_local_readme_summary
        self.api._remember_module_state = self._orig_remember_module_state
        self.api._apply_node_change_info = self._orig_apply_node_change_info
        self.api._announce_tracked_module_updates = self._orig_announce_tracked_module_updates
        self.api._comfyui_git_status = self._orig_comfyui_git_status

        self.api._MODULE_INFO_CACHE.clear()
        self.api._MODULE_INFO_CACHE.update(self._orig_module_info_cache)

    def test_node_catalog_group_payload_contract(self):
        """Freeze grouped catalog payload shape used by `/alexz_tools/node_catalog`."""
        self.api._collect_nodes = lambda: [
            {
                "node_name": "BuiltinNode",
                "display_name": "Builtin Node",
                "module": "nodes",
                "group": "core",
                "category": "image/utils",
                "annotation": "builtin",
            },
            {
                "node_name": "CustomNode",
                "display_name": "Custom Node",
                "module": "ComfyUI_ALEXZ_tools",
                "group": "custom",
                "category": "ALEXZ/test",
                "annotation": "custom",
            },
        ]
        self.api._discover_custom_modules = lambda: ["ComfyUI_ALEXZ_tools", "ComfyUI-EmptyModule"]

        grouped = self.api._build_group_catalog()
        modules_by_group = self.api._build_group_modules(grouped)

        groups = []
        for group_id, group_title in self.api._GROUP_ORDER:
            nodes = grouped.get(group_id, [])
            modules = modules_by_group.get(group_id, [])
            groups.append(
                {
                    "id": group_id,
                    "title": group_title,
                    "count": len(nodes),
                    "nodes": nodes,
                    "module_count": len(modules),
                    "modules": modules,
                }
            )

        self.assertEqual(len(groups), 4)
        required_group_keys = {"id", "title", "count", "nodes", "module_count", "modules"}
        required_node_keys = {"node_name", "display_name", "module", "group", "category", "annotation"}

        for entry in groups:
            self.assertTrue(required_group_keys.issubset(set(entry.keys())))
            self.assertIsInstance(entry["count"], int)
            self.assertIsInstance(entry["module_count"], int)
            self.assertIsInstance(entry["nodes"], list)
            self.assertIsInstance(entry["modules"], list)
            for node in entry["nodes"]:
                self.assertTrue(required_node_keys.issubset(set(node.keys())))

        custom_mods = {m["module"]: m["count"] for m in groups[3]["modules"]}
        self.assertIn("ComfyUI_ALEXZ_tools", custom_mods)
        self.assertIn("ComfyUI-EmptyModule", custom_mods)
        self.assertEqual(custom_mods["ComfyUI-EmptyModule"], 0)

    def test_module_info_custom_payload_contract(self):
        """Freeze key fields required by module card rendering for custom modules."""
        self.api._sync_module_upstream = lambda module_name: True
        self.api._module_git_state = lambda module_name: {
            "module_path": "/tmp/fake_mod",
            "repository": "https://github.com/alex/testmod",
            "installed_commit": "1234567890abcdef",
            "installed_updated_at": "2026-02-10T00:00:00+00:00",
            "remote_updated_at": "2026-02-10T01:00:00+00:00",
            "has_upstream": True,
            "ahead": 0,
            "behind": 2,
            "remote_head": "fedcba0987654321",
        }
        self.api._manager_index = lambda: {"by_github": {}, "by_id": {}, "by_repo_name": {}}
        self.api._manager_github_stats = lambda: {"by_url": {}, "by_github": {}}
        self.api._module_repo_url = lambda module_name: "https://github.com/alex/testmod"
        self.api._module_local_readme_summary = lambda module_name: "test module summary"
        self.api._remember_module_state = lambda module_name, info: None
        self.api._apply_node_change_info = lambda result, group, module_name: None

        info = self.api._resolve_module_info(
            "custom",
            "testmod",
            force_refresh=True,
            sync_upstream=True,
        )

        required = {
            "module",
            "group",
            "title",
            "description",
            "repository",
            "owner_url",
            "installed_commit",
            "installed_commit_short",
            "update_status",
            "git_has_upstream",
            "git_behind",
            "source",
        }
        self.assertTrue(required.issubset(set(info.keys())))
        self.assertEqual(info["group"], "custom")
        self.assertEqual(info["update_status"], "can_update")
        self.assertEqual(info["installed_commit_short"], "12345678")

    def test_module_info_builtin_payload_contract(self):
        """Freeze baseline payload for built-in groups (`core`, `api`, `core_extras`)."""
        info = self.api._resolve_module_info("core", "nodes", force_refresh=True)

        self.assertEqual(info.get("group"), "core")
        self.assertEqual(info.get("source"), "builtin")
        self.assertEqual(info.get("update_status"), "unknown")
        self.assertIn("new_nodes_between_runs", info)
        self.assertIn("updated_nodes_between_runs", info)

    def test_refresh_status_snapshot_contract(self):
        """Freeze refresh-status payload keys used by frontend polling loop."""
        snap = self.api._refresh_status_snapshot()
        required = {
            "running",
            "phase",
            "current",
            "total",
            "remaining",
            "modules_need_update",
            "module",
            "message",
            "error",
            "sync_upstreams",
            "started_at",
            "updated_at",
            "refreshed_at",
        }
        self.assertTrue(required.issubset(set(snap.keys())))

    def test_update_status_snapshot_contract(self):
        """Freeze update-status payload keys used by frontend polling loop."""
        snap = self.api._update_status_snapshot()
        required = {
            "running",
            "phase",
            "scope",
            "current",
            "total",
            "remaining",
            "module",
            "message",
            "error",
            "updated",
            "up_to_date",
            "failed",
            "requirements_changed",
            "requirements_modules",
            "results",
            "started_at",
            "updated_at",
            "finished_at",
        }
        self.assertTrue(required.issubset(set(snap.keys())))

    def test_refresh_runtime_state_result_contract(self):
        """Freeze return payload from runtime-state refresh orchestration."""
        events = []
        self.api._discover_custom_modules = lambda: []
        self.api._announce_tracked_module_updates = lambda: {"modules_need_update": 3}
        self.api._comfyui_git_status = lambda force_refresh=False: {"update_status": "unknown"}

        result = self.api._refresh_module_runtime_state(
            sync_upstreams=False,
            progress_cb=lambda **kw: events.append(dict(kw)),
        )

        self.assertEqual(result.get("status"), "ok")
        self.assertIn("refreshed_at", result)
        self.assertIn("comfyui", result)
        self.assertEqual(result.get("modules_need_update"), 3)

        phases = [ev.get("phase") for ev in events]
        self.assertIn("sync", phases)
        self.assertIn("snapshots", phases)
        self.assertIn("done", phases)

    def test_filter_modules_exact_priority_over_partial(self):
        """Ensure exact query hit has priority over fuzzy substring matches."""
        modules = ["ComfyUI_ALEXZ_tools", "ComfyUI_ALEXZ_tools_extra", "ComfyUI-Other"]
        exact = self.api._filter_modules("ComfyUI_ALEXZ_tools", modules)
        fuzzy = self.api._filter_modules("alexz", modules)

        self.assertEqual(exact, ["ComfyUI_ALEXZ_tools"])
        self.assertIn("ComfyUI_ALEXZ_tools", fuzzy)
        self.assertIn("ComfyUI_ALEXZ_tools_extra", fuzzy)

    def test_module_nodes_payload_shape_from_catalog(self):
        """Freeze payload shape expected by `/alexz_tools/module_nodes` consumers."""
        self.api._collect_nodes = lambda: [
            {
                "node_name": "AlphaNode",
                "display_name": "Alpha Node",
                "module": "ComfyUI_ALEXZ_tools",
                "group": "custom",
                "category": "ALEXZ/test",
                "annotation": "alpha",
            },
            {
                "node_name": "BetaNode",
                "display_name": "Beta Node",
                "module": "ComfyUI_ALEXZ_tools",
                "group": "custom",
                "category": "ALEXZ/test",
                "annotation": "beta",
            },
            {
                "node_name": "OtherNode",
                "display_name": "Other Node",
                "module": "ComfyUI-Other",
                "group": "custom",
                "category": "other/test",
                "annotation": "other",
            },
        ]

        query = "alexz"
        catalog = self.api._build_catalog()
        modules = list(catalog.keys())
        selected_modules = self.api._filter_modules(query, modules)
        results = [
            {
                "module": module_name,
                "count": len(catalog.get(module_name, [])),
                "nodes": catalog.get(module_name, []),
            }
            for module_name in selected_modules
        ]
        payload = {
            "query": query,
            "module_count": len(results),
            "results": results,
            "hint": "Введите имя python-модуля (например: ComfyUI_ALEXZ_tools).",
        }

        self.assertEqual(payload["module_count"], 1)
        self.assertEqual(payload["results"][0]["module"], "ComfyUI_ALEXZ_tools")
        self.assertEqual(payload["results"][0]["count"], 2)
        self.assertEqual(len(payload["results"][0]["nodes"]), 2)
        self.assertIn("hint", payload)

    def test_frontend_tab_relay_contract_markers_exist(self):
        """Freeze key tab-relay/diagnostic markers used in manual transition baseline checks."""
        repo_root = Path(__file__).resolve().parents[1]
        relay_text = (repo_root / "web" / "module_node_picker_tab_relay.js").read_text(
            encoding="utf-8"
        )
        relay_helpers_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_tab_relay_helpers.js"
        ).read_text(encoding="utf-8")
        relay_runtime_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_tab_relay_runtime.js"
        ).read_text(encoding="utf-8")
        picker_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("export function bindModuleNodesTabRelay", relay_text)
        self.assertIn("export function unbindModuleNodesTabRelay", relay_text)
        self.assertIn("easyuse_nodes_map", relay_helpers_text)
        self.assertIn("relay_init", relay_text)
        self.assertIn("relay_tick", relay_text)
        self.assertIn("MIN_SYNC_INTERVAL_MS", relay_runtime_text)
        self.assertIn("dispose()", relay_runtime_text)
        self.assertIn("hasPendingForeignIntent()", relay_runtime_text)
        self.assertIn("passiveTickBudget", relay_text)
        self.assertIn("relay_visibility", relay_text)
        self.assertIn("relay_pageshow", relay_text)
        self.assertIn("bindToken = Symbol", relay_text)
        self.assertIn("isCurrentBinding()", relay_text)
        self.assertIn("[contenteditable]", relay_text)

        self.assertIn("diag.active_tab=", picker_text)
        self.assertIn("diag.last_clicked_tab=", picker_text)
        self.assertIn("diag.child_nodes_short=", picker_text)
        self.assertIn("catalogLoadToken", picker_text)
        self.assertIn("moduleInfoLoadToken", picker_text)
        self.assertIn("catalogLoadBusyCount", picker_text)
        self.assertIn("setCatalogControlsLoading", picker_text)
        self.assertIn("Loading groups...", picker_text)
        self.assertIn("Loading modules...", picker_text)
        self.assertIn("PICKER_CLEANUP_KEY", picker_text)
        data_flow_text = (repo_root / "web" / "orchestration" / "module_node_picker_data_flow.js").read_text(encoding="utf-8")
        self.assertIn("isRequestActive", data_flow_text)
        self.assertIn("totalNodes", data_flow_text)

    def test_frontend_tab_relay_legacy_paths_are_not_reintroduced(self):
        """Keep removed relay legacy paths from silently returning in future edits."""
        repo_root = Path(__file__).resolve().parents[1]
        relay_text = (repo_root / "web" / "module_node_picker_tab_relay.js").read_text(
            encoding="utf-8"
        )
        helpers_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_tab_relay_helpers.js"
        ).read_text(encoding="utf-8")
        picker_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("setInterval(", relay_text)
        self.assertIn("setTimeout(runTick", relay_text)
        self.assertNotIn("export function hasSidebarTabId", helpers_text)
        bindings_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_bindings.js"
        ).read_text(encoding="utf-8")
        self.assertIn("saveComfyCheckMode?.(comfyModeSelect.value)", bindings_text)
        self.assertNotIn("comfyModeReloadTimer", bindings_text)
        self.assertIn("return () => {", bindings_text)
        self.assertIn("onchange = null", bindings_text)
        self.assertIn("unbindPickerEvents", picker_text)
        self.assertIn("startupRetryDelayMs", bindings_text)
        self.assertIn("cancelStartupLoad", picker_text)
        actions_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_actions.js"
        ).read_text(encoding="utf-8")
        update_flow_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_update_flow.js"
        ).read_text(encoding="utf-8")
        lifecycle_guard_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_lifecycle_guard.js"
        ).read_text(encoding="utf-8")
        api_text = (
            repo_root / "web" / "api" / "module_node_picker_api.js"
        ).read_text(encoding="utf-8")
        process_text = (
            repo_root / "web" / "ui" / "module_node_picker_process.js"
        ).read_text(encoding="utf-8")
        self.assertIn("import { shouldContinueContext }", actions_text)
        self.assertIn("import { shouldContinueContext }", update_flow_text)
        self.assertIn("export function shouldContinueContext", lifecycle_guard_text)
        self.assertIn("shouldContinue: isPickerAlive", picker_text)
        self.assertNotIn("refreshComfyUIModeInfoFlow", picker_text)
        self.assertIn("dispose", process_text)
        self.assertIn("processUi?.dispose?.()", picker_text)
        self.assertIn("AbortController", api_text)
        self.assertIn("API timeout after", api_text)
        self.assertNotIn("!pickerDisposed && root.isConnected", picker_text)

    def test_frontend_pending_resume_contract_markers_exist(self):
        """Freeze pending/resume markers for refresh/update lifecycle restoration."""
        repo_root = Path(__file__).resolve().parents[1]
        picker_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )
        actions_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_actions.js"
        ).read_text(encoding="utf-8")
        update_flow_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_update_flow.js"
        ).read_text(encoding="utf-8")
        bindings_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_bindings.js"
        ).read_text(encoding="utf-8")
        error_utils_text = (
            repo_root / "web" / "orchestration" / "module_node_picker_error_utils.js"
        ).read_text(encoding="utf-8")

        self.assertIn("MODULE_PICKER_RUNTIME_STATE_KEY", picker_text)
        self.assertIn("getRuntimePickerState", picker_text)
        self.assertIn("LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY", picker_text)
        self.assertIn("clearLegacyPersistentFlags", picker_text)
        self.assertIn("loadCustomStatusChecked", picker_text)
        self.assertIn("saveCustomStatusChecked", picker_text)
        self.assertIn("setCustomStatusChecked", picker_text)
        self.assertIn("loadComfyStatusChecked", picker_text)
        self.assertIn("saveComfyStatusChecked", picker_text)
        self.assertIn("loadComfyInfoSnapshot", picker_text)
        self.assertIn("saveComfyInfoSnapshot", picker_text)
        self.assertIn("setComfyStatusChecked", picker_text)
        self.assertIn("hasPendingCustomRefresh", picker_text)
        self.assertIn("setPendingCustomRefresh", picker_text)
        self.assertIn("clearPendingCustomRefresh", picker_text)
        self.assertIn("hasPendingUpdate", picker_text)
        self.assertIn("setPendingUpdate", picker_text)
        self.assertIn("clearPendingUpdate", picker_text)
        self.assertIn("hasPendingComfyInfoRefresh", picker_text)
        self.assertIn("setPendingComfyInfoRefresh", picker_text)
        self.assertIn("clearPendingComfyInfoRefresh", picker_text)
        self.assertIn("resumePendingCustomRefreshFlow", picker_text)
        self.assertIn("resumePendingModuleUpdateFlow", picker_text)
        self.assertIn("resumePendingComfyInfoRefreshFlow", picker_text)
        self.assertIn("runStartupSequence", picker_text)
        self.assertIn("hasPendingWork", picker_text)
        self.assertIn("runCatalogStartupLoad", picker_text)
        self.assertIn("shouldContinueStartup", picker_text)
        self.assertIn("setStartupBusy(true)", picker_text)
        self.assertIn("setStartupBusy(false)", picker_text)
        self.assertIn("isActionBusy: () => actionBusy || startupBusy", picker_text)
        self.assertIn("await resumePendingCustomRefreshFlow()", picker_text)
        self.assertIn("await resumePendingModuleUpdateFlow()", picker_text)
        self.assertIn("await resumePendingComfyInfoRefreshFlow()", picker_text)
        self.assertIn("cancelStartupLoad = runStartupSequence()", picker_text)
        self.assertIn("onSettled", bindings_text)
        self.assertIn("settle()", bindings_text)
        self.assertIn("import { isCanceledRequestError }", picker_text)
        self.assertIn("import { isCanceledRequestError }", actions_text)
        self.assertIn("import { isCanceledRequestError }", update_flow_text)
        self.assertIn("export function isCanceledRequestError", error_utils_text)
        self.assertIn("setPendingComfyInfoRefresh?.(true)", actions_text)
        self.assertIn("setComfyStatusChecked?.(true)", actions_text)
        self.assertIn("clearPendingComfyInfoRefresh?.()", actions_text)
        self.assertIn("setCustomStatusChecked?.(true)", actions_text)
        self.assertIn("setPendingCustomRefresh?.(true)", actions_text)
        self.assertIn("setPendingUpdate?.(true)", update_flow_text)
        self.assertIn("clearPendingUpdate?.()", update_flow_text)

    def test_comfyui_status_cache_only_skips_git_without_force_refresh(self):
        """Ensure non-forced ComfyUI status request returns cached/unknown data without git calls."""
        orig_run_git = self.api._run_git
        orig_comfy_cache = self.api._COMFYUI_STATUS_CACHE
        orig_state_cache = self.api._MODULE_STATE_CACHE
        self.api._COMFYUI_STATUS_CACHE = None
        self.api._MODULE_STATE_CACHE = {}

        def _fail_git(_args, timeout=2.0):
            raise AssertionError("git must not be called in cache-only mode")

        self.api._run_git = _fail_git
        try:
            status = self.api._comfyui_git_status(force_refresh=False)
        finally:
            self.api._run_git = orig_run_git
            self.api._COMFYUI_STATUS_CACHE = orig_comfy_cache
            self.api._MODULE_STATE_CACHE = orig_state_cache

        self.assertEqual(status.get("update_status"), "unknown")

    def test_module_info_cache_only_uses_state_snapshot(self):
        """Ensure cache-only module info path serves state cache without git probing."""
        orig_state_cache = self.api._MODULE_STATE_CACHE
        orig_module_git_state = self.api._module_git_state
        self.api._MODULE_STATE_CACHE = {
            "ComfyUI_Test": {
                "installed_commit": "abc12345ffff",
                "installed_updated_at": "2026-02-10T10:00:00+00:00",
                "remote_updated_at": "2026-02-10T11:00:00+00:00",
                "update_available": True,
                "repository": "https://github.com/example/repo",
                "module_path": "/tmp/ComfyUI_Test",
                "last_checked_at": "2026-02-10T12:00:00+00:00",
            }
        }

        def _fail_git_state(_module_name):
            raise AssertionError("git state must not be called in cache-only mode")

        self.api._module_git_state = _fail_git_state
        try:
            info = self.api._resolve_module_info(
                "custom",
                "ComfyUI_Test",
                force_refresh=True,
                cache_only=True,
            )
        finally:
            self.api._MODULE_STATE_CACHE = orig_state_cache
            self.api._module_git_state = orig_module_git_state

        self.assertEqual(info.get("installed_commit_short"), "abc12345")
        self.assertEqual(info.get("update_status"), "can_update")
        self.assertEqual(info.get("module_path"), "/tmp/ComfyUI_Test")


if __name__ == "__main__":
    unittest.main()
