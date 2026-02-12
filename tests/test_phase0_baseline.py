"""
Module: tests/test_phase0_baseline.py
Author: AlexZ1967
Last updated: 2026-02-11

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
        self.assertEqual(info.get("update_status"), "")
        self.assertEqual(info.get("update_available"), False)
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
            "modules_unknown_update",
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
        self.assertIn("modules_unknown_update", result)

        phases = [ev.get("phase") for ev in events]
        self.assertIn("sync", phases)
        self.assertIn("snapshots", phases)
        self.assertIn("done", phases)

    def test_runtime_warmup_status_contract(self):
        """Freeze runtime warmup status payload shape for first-open catalog UX."""
        orig_lazy = self.api._LAZY_REFRESH_DONE
        orig_thread = self.api._RUNTIME_WARMUP_THREAD
        try:
            self.api._LAZY_REFRESH_DONE = False
            self.api._RUNTIME_WARMUP_THREAD = None
            status = self.api._runtime_warmup_status()
            self.assertEqual(set(status.keys()), {"running", "done"})
            self.assertIs(status["running"], False)
            self.assertIs(status["done"], False)
        finally:
            self.api._LAZY_REFRESH_DONE = orig_lazy
            self.api._RUNTIME_WARMUP_THREAD = orig_thread

    def test_start_runtime_warmup_noop_when_already_ready(self):
        """Ensure warmup starter is a no-op once lazy runtime state is ready."""
        orig_lazy = self.api._LAZY_REFRESH_DONE
        orig_thread = self.api._RUNTIME_WARMUP_THREAD
        try:
            self.api._LAZY_REFRESH_DONE = True
            self.api._RUNTIME_WARMUP_THREAD = None
            started = self.api._start_runtime_state_warmup()
            self.assertIs(started, False)
        finally:
            self.api._LAZY_REFRESH_DONE = orig_lazy
            self.api._RUNTIME_WARMUP_THREAD = orig_thread

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
        relay_facade_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_facade.js"
        ).read_text(encoding="utf-8")
        relay_constants_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_constants.js"
        ).read_text(encoding="utf-8")
        relay_events_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_events.js"
        ).read_text(encoding="utf-8")
        relay_lifecycle_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_lifecycle.js"
        ).read_text(encoding="utf-8")
        relay_helpers_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_helpers.js"
        ).read_text(encoding="utf-8")
        relay_runtime_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_runtime.js"
        ).read_text(encoding="utf-8")
        relay_diagnostics_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_diagnostics.js"
        ).read_text(encoding="utf-8")
        relay_dom_ownership_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_dom_ownership.js"
        ).read_text(encoding="utf-8")
        relay_intent_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_intent.js"
        ).read_text(encoding="utf-8")
        relay_tick_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_tick.js"
        ).read_text(encoding="utf-8")
        busy_ui_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_busy_ui.js"
        ).read_text(encoding="utf-8")
        debug_ui_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_debug_ui.js"
        ).read_text(encoding="utf-8")
        catalog_controller_text = (
            repo_root / "web" / "orchestration" / "flow" / "catalog" / "module_node_picker_catalog_controller.js"
        ).read_text(encoding="utf-8")
        layout_text = (
            repo_root / "web" / "ui" / "module_node_picker_layout.js"
        ).read_text(encoding="utf-8")
        selection_controller_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_selection_controller.js"
        ).read_text(encoding="utf-8")
        ui_controllers_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_ui_controllers.js"
        ).read_text(encoding="utf-8")
        registration_text = (
            repo_root / "web" / "orchestration" / "core" / "infra" / "module_node_picker_registration.js"
        ).read_text(encoding="utf-8")
        constants_text = (
            repo_root / "web" / "constants" / "module_node_picker_constants.js"
        ).read_text(encoding="utf-8")
        node_factory_text = (
            repo_root / "web" / "ui" / "module_node_picker_node_factory.js"
        ).read_text(encoding="utf-8")
        picker_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )
        composer_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_composer.js"
        ).read_text(encoding="utf-8")
        context_builders_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_context_builders.js"
        ).read_text(encoding="utf-8")
        ui_stage_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_ui_stage.js"
        ).read_text(encoding="utf-8")
        flow_stage_text = (
            repo_root / "web" / "orchestration" / "flow" / "stage" / "module_node_picker_flow_stage.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_bindings_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap_bindings.js"
        ).read_text(encoding="utf-8")
        relay_bridge_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_relay_bridge.js"
        ).read_text(encoding="utf-8")
        runtime_projection_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_projection.js"
        ).read_text(encoding="utf-8")
        picker_runtime_text = (
            f"{picker_text}\n{composer_text}\n{context_builders_text}\n{ui_stage_text}\n{flow_stage_text}\n{runtime_bootstrap_bindings_text}\n{runtime_projection_text}\n{relay_bridge_text}"
        )

        self.assertIn("bindModuleNodesTabRelayFacade as bindModuleNodesTabRelay", relay_text)
        self.assertIn("unbindModuleNodesTabRelayFacade as unbindModuleNodesTabRelay", relay_text)
        self.assertIn("module_node_picker_tab_relay_facade.js", relay_text)
        self.assertIn("export function bindModuleNodesTabRelayFacade", relay_facade_text)
        self.assertIn("export function unbindModuleNodesTabRelayFacade", relay_facade_text)
        self.assertIn("mountHost", relay_facade_text)
        self.assertIn("mountHost", relay_runtime_text)
        self.assertIn("module_node_picker_tab_relay_diagnostics.js", relay_runtime_text)
        self.assertIn("module_node_picker_tab_relay_dom_ownership.js", relay_runtime_text)
        self.assertIn("module_node_picker_tab_relay_constants.js", relay_facade_text)
        self.assertIn("module_node_picker_tab_relay_events.js", relay_facade_text)
        self.assertIn("module_node_picker_tab_relay_lifecycle.js", relay_facade_text)
        self.assertIn("RELAY_REASON_INIT", relay_constants_text)
        self.assertIn("RELAY_REASON_TICK", relay_constants_text)
        self.assertIn("createRelayDomEventHandlers", relay_events_text)
        self.assertIn("bindRelayDomEvents", relay_events_text)
        self.assertIn("unbindRelayDomEvents", relay_events_text)
        self.assertIn("createRelayBindState", relay_lifecycle_text)
        self.assertIn("disposeRelayBindState", relay_lifecycle_text)
        self.assertIn("buildRelayDiagnosticsPayload", relay_diagnostics_text)
        self.assertIn("createRelayDiagnosticsEmitter", relay_diagnostics_text)
        self.assertIn("createRelayDomOwnershipController", relay_dom_ownership_text)
        self.assertIn("easyuse_nodes_map", relay_helpers_text)
        self.assertIn("relay_init", relay_constants_text)
        self.assertIn("relay_tick", relay_constants_text)
        self.assertIn("MIN_SYNC_INTERVAL_MS", relay_runtime_text)
        self.assertIn("dispose()", relay_runtime_text)
        self.assertIn("hasPendingForeignIntent()", relay_runtime_text)
        self.assertIn("startModuleNodePickerRelayTickLoop", relay_facade_text)
        self.assertIn("createModuleNodePickerRelayIntentController", relay_facade_text)
        self.assertIn("passiveTickBudget", relay_tick_text)
        self.assertIn("isOwnButtonSelected", relay_tick_text)
        self.assertIn("RELAY_REASON_PENDING_SWITCH", relay_intent_text)
        self.assertIn("RELAY_REASON_NATIVE_OK", relay_intent_text)
        self.assertIn("RELAY_REASON_VISIBILITY", relay_intent_text)
        self.assertIn("RELAY_REASON_PAGESHOW", relay_intent_text)
        self.assertIn("relay_pending_switch", relay_constants_text)
        self.assertIn("relay_native_ok", relay_constants_text)
        self.assertIn("relay_visibility", relay_constants_text)
        self.assertIn("relay_pageshow", relay_constants_text)
        self.assertIn("bindToken = Symbol", relay_facade_text)
        self.assertIn("isCurrentBinding", relay_facade_text)
        self.assertIn("[contenteditable]", relay_intent_text)

        self.assertIn("diag.active_tab=", debug_ui_text)
        self.assertIn("diag.last_clicked_tab=", debug_ui_text)
        self.assertIn("diag.child_nodes_short=", debug_ui_text)
        self.assertIn("catalogLoadToken", catalog_controller_text)
        self.assertIn("moduleInfoLoadToken", catalog_controller_text)
        self.assertIn("catalogLoadBusyCount", catalog_controller_text)
        self.assertIn("setCatalogControlsLoading", picker_runtime_text)
        self.assertIn("Loading groups...", busy_ui_text)
        self.assertIn("Loading modules...", busy_ui_text)
        self.assertIn("export function createModuleNodePickerLayout", layout_text)
        self.assertIn("Refresh ComfyUI Info", layout_text)
        self.assertIn("Refresh Custom Nodes Info", layout_text)
        self.assertIn("export function createModuleNodePickerSelectionController", selection_controller_text)
        self.assertIn("fillModuleSelectUi", selection_controller_text)
        self.assertIn("fillGroupSelectUi", selection_controller_text)
        self.assertIn("export function registerModuleNodePickerExtension", registration_text)
        self.assertIn("registerModuleNodePickerExtension({", picker_text)
        self.assertIn("injectStyles?.()", registration_text)
        self.assertIn("export const EXT_NAME", constants_text)
        self.assertIn("export const GROUP_LABELS", constants_text)
        self.assertIn("export const MODULE_MARK_UPDATED", constants_text)
        self.assertIn("export function centerNodeInCanvas", node_factory_text)
        self.assertIn("export function createNodeFromCatalogInfo", node_factory_text)
        self.assertIn("createNodeFromCatalogInfo(nodeInfo, LiteGraph)", picker_runtime_text)
        self.assertTrue(
            ("centerNodeInCanvas(node, app)" in picker_runtime_text)
            or ("centerNodeInCanvas(node, appInstance)" in picker_runtime_text)
        )
        self.assertIn("mountHost: container", picker_runtime_text)
        self.assertIn("MODULE_PICKER_GUARD_KEY", picker_text)
        self.assertIn("createModuleNodePickerSelectionController", ui_controllers_text)
        self.assertIn("fillModuleSelect: (options = {}) => selectionController.fillModuleSelect(options)", ui_controllers_text)
        self.assertIn("fillGroupSelect: (groups, options = {}) => selectionController.fillGroupSelect(groups, options)", ui_controllers_text)
        self.assertIn("createModuleNodePickerUiControllers", picker_runtime_text)
        self.assertIn("bindModuleNodePickerRelayBridge", composer_text)
        self.assertIn("export function bindModuleNodePickerRelayBridge", relay_bridge_text)
        self.assertNotIn("fillModuleSelectUi", picker_text)
        self.assertNotIn("fillGroupSelectUi", picker_text)
        self.assertIn("PICKER_CLEANUP_KEY", picker_runtime_text)
        data_flow_text = (repo_root / "web" / "orchestration" / "flow" / "catalog" / "module_node_picker_data_flow.js").read_text(encoding="utf-8")
        self.assertIn("isRequestActive", data_flow_text)
        self.assertIn("totalNodes", data_flow_text)

    def test_frontend_tab_relay_legacy_paths_are_not_reintroduced(self):
        """Keep removed relay legacy paths from silently returning in future edits."""
        repo_root = Path(__file__).resolve().parents[1]
        relay_text = (repo_root / "web" / "module_node_picker_tab_relay.js").read_text(
            encoding="utf-8"
        )
        relay_facade_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_facade.js"
        ).read_text(encoding="utf-8")
        relay_events_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_events.js"
        ).read_text(encoding="utf-8")
        relay_lifecycle_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_lifecycle.js"
        ).read_text(encoding="utf-8")
        helpers_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_helpers.js"
        ).read_text(encoding="utf-8")
        relay_intent_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_intent.js"
        ).read_text(encoding="utf-8")
        relay_tick_text = (
            repo_root / "web" / "orchestration" / "relay" / "module_node_picker_tab_relay_tick.js"
        ).read_text(encoding="utf-8")
        picker_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )
        composer_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_composer.js"
        ).read_text(encoding="utf-8")
        context_builders_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_context_builders.js"
        ).read_text(encoding="utf-8")
        ui_stage_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_ui_stage.js"
        ).read_text(encoding="utf-8")
        flow_stage_text = (
            repo_root / "web" / "orchestration" / "flow" / "stage" / "module_node_picker_flow_stage.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_bindings_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap_bindings.js"
        ).read_text(encoding="utf-8")
        relay_bridge_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_relay_bridge.js"
        ).read_text(encoding="utf-8")
        runtime_projection_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_projection.js"
        ).read_text(encoding="utf-8")
        picker_runtime_text = (
            f"{picker_text}\n{composer_text}\n{context_builders_text}\n{ui_stage_text}\n{flow_stage_text}\n{runtime_bootstrap_bindings_text}\n{runtime_projection_text}\n{relay_bridge_text}"
        )

        self.assertNotIn("setInterval(", relay_facade_text)
        self.assertNotIn("setTimeout(runTick", relay_facade_text)
        self.assertIn("startModuleNodePickerRelayTickLoop", relay_facade_text)
        self.assertIn("createModuleNodePickerRelayIntentController", relay_facade_text)
        self.assertIn("bindRelayDomEvents", relay_facade_text)
        self.assertIn("disposeRelayBindState", relay_facade_text)
        self.assertIn("disposeRelayBindState", relay_lifecycle_text)
        self.assertIn("module_node_picker_tab_relay_facade.js", relay_text)
        self.assertIn("document.addEventListener", relay_events_text)
        self.assertIn("document.removeEventListener", relay_events_text)
        self.assertIn("window.setTimeout(() =>", relay_intent_text)
        self.assertIn("window.setTimeout(runTick", relay_tick_text)
        self.assertNotIn("export function hasSidebarTabId", helpers_text)
        bindings_text = (
            repo_root / "web" / "orchestration" / "core" / "infra" / "module_node_picker_bindings.js"
        ).read_text(encoding="utf-8")
        self.assertIn("saveComfyCheckMode?.(comfyModeSelect.value)", bindings_text)
        self.assertNotIn("comfyModeReloadTimer", bindings_text)
        self.assertIn("return () => {", bindings_text)
        self.assertIn("onchange = null", bindings_text)
        self.assertIn("unbindPickerEvents", picker_runtime_text)
        self.assertIn("startupRetryDelayMs", bindings_text)
        self.assertIn("cancelStartupLoad", picker_runtime_text)
        actions_text = (
            repo_root / "web" / "orchestration" / "flow" / "actions" / "module_node_picker_actions.js"
        ).read_text(encoding="utf-8")
        update_flow_text = (
            repo_root / "web" / "orchestration" / "flow" / "progress" / "module_node_picker_update_flow.js"
        ).read_text(encoding="utf-8")
        lifecycle_guard_text = (
            repo_root / "web" / "orchestration" / "runtime" / "lifecycle" / "module_node_picker_lifecycle_guard.js"
        ).read_text(encoding="utf-8")
        lifecycle_text = (
            repo_root / "web" / "orchestration" / "runtime" / "lifecycle" / "module_node_picker_lifecycle.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap.js"
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
        self.assertIn("shouldContinue: isPickerAlive", picker_runtime_text)
        self.assertNotIn("refreshComfyUIModeInfoFlow", picker_runtime_text)
        self.assertIn("dispose", process_text)
        self.assertIn("getProcessUi()?.dispose?.()", lifecycle_text)
        self.assertIn("AbortController", api_text)
        self.assertIn("API timeout after", api_text)
        self.assertNotIn("!pickerDisposed && root.isConnected", picker_runtime_text)

    def test_frontend_core_subfolder_import_contract_markers_exist(self):
        """Ensure the core split keeps semantic composition/infra import paths."""
        repo_root = Path(__file__).resolve().parents[1]
        entry_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )
        runtime_bootstrap_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_bindings_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap_bindings.js"
        ).read_text(encoding="utf-8")
        flow_actions_text = (
            repo_root / "web" / "orchestration" / "flow" / "actions" / "module_node_picker_actions.js"
        ).read_text(encoding="utf-8")
        flow_update_text = (
            repo_root / "web" / "orchestration" / "flow" / "progress" / "module_node_picker_update_flow.js"
        ).read_text(encoding="utf-8")
        composer_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_composer.js"
        ).read_text(encoding="utf-8")

        self.assertIn("./orchestration/core/infra/module_node_picker_registration.js", entry_text)
        self.assertIn("./orchestration/core/composition/module_node_picker_composer.js", entry_text)
        self.assertIn("../../core/infra/module_node_picker_bindings.js", runtime_bootstrap_text)
        self.assertIn("../../core/infra/module_node_picker_bindings.js", runtime_bootstrap_bindings_text)
        self.assertIn("../../core/infra/module_node_picker_error_utils.js", flow_actions_text)
        self.assertIn("../../core/infra/module_node_picker_error_utils.js", flow_update_text)
        self.assertIn("../infra/module_node_picker_error_utils.js", composer_text)

        self.assertNotIn("./orchestration/core/module_node_picker_registration.js", entry_text)
        self.assertNotIn("./orchestration/core/module_node_picker_composer.js", entry_text)
        self.assertNotIn("../../core/module_node_picker_bindings.js", runtime_bootstrap_text)
        self.assertNotIn("../../core/module_node_picker_bindings.js", runtime_bootstrap_bindings_text)
        self.assertNotIn("../../core/module_node_picker_error_utils.js", flow_actions_text)
        self.assertNotIn("../../core/module_node_picker_error_utils.js", flow_update_text)
        self.assertNotIn("./module_node_picker_error_utils.js", composer_text)

    def test_frontend_pending_resume_contract_markers_exist(self):
        """Freeze pending/resume markers for refresh/update lifecycle restoration."""
        repo_root = Path(__file__).resolve().parents[1]
        picker_text = (repo_root / "web" / "module_node_picker.js").read_text(
            encoding="utf-8"
        )
        composer_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_composer.js"
        ).read_text(encoding="utf-8")
        context_builders_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_context_builders.js"
        ).read_text(encoding="utf-8")
        ui_stage_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_ui_stage.js"
        ).read_text(encoding="utf-8")
        flow_stage_text = (
            repo_root / "web" / "orchestration" / "flow" / "stage" / "module_node_picker_flow_stage.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_bindings_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap_bindings.js"
        ).read_text(encoding="utf-8")
        relay_bridge_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_relay_bridge.js"
        ).read_text(encoding="utf-8")
        runtime_projection_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_projection.js"
        ).read_text(encoding="utf-8")
        picker_runtime_text = (
            f"{picker_text}\n{composer_text}\n{context_builders_text}\n{ui_stage_text}\n{flow_stage_text}\n{runtime_bootstrap_bindings_text}\n{runtime_projection_text}\n{relay_bridge_text}"
        )
        actions_text = (
            repo_root / "web" / "orchestration" / "flow" / "actions" / "module_node_picker_actions.js"
        ).read_text(encoding="utf-8")
        update_flow_text = (
            repo_root / "web" / "orchestration" / "flow" / "progress" / "module_node_picker_update_flow.js"
        ).read_text(encoding="utf-8")
        bindings_text = (
            repo_root / "web" / "orchestration" / "core" / "infra" / "module_node_picker_bindings.js"
        ).read_text(encoding="utf-8")
        startup_flow_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_startup_flow.js"
        ).read_text(encoding="utf-8")
        api_client_text = (
            repo_root / "web" / "orchestration" / "api" / "module_node_picker_api_client.js"
        ).read_text(encoding="utf-8")
        catalog_controller_text = (
            repo_root / "web" / "orchestration" / "flow" / "catalog" / "module_node_picker_catalog_controller.js"
        ).read_text(encoding="utf-8")
        status_cards_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_status_cards.js"
        ).read_text(encoding="utf-8")
        runtime_context_text = (
            repo_root / "web" / "state" / "module_node_picker_runtime_context.js"
        ).read_text(encoding="utf-8")
        view_helpers_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_view_helpers.js"
        ).read_text(encoding="utf-8")
        action_flows_text = (
            repo_root / "web" / "orchestration" / "flow" / "actions" / "module_node_picker_action_flows.js"
        ).read_text(encoding="utf-8")
        flow_wiring_text = (
            repo_root / "web" / "orchestration" / "flow" / "stage" / "module_node_picker_flow_wiring.js"
        ).read_text(encoding="utf-8")
        ui_controllers_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_ui_controllers.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap.js"
        ).read_text(encoding="utf-8")
        polling_controller_text = (
            repo_root / "web" / "orchestration" / "flow" / "progress" / "module_node_picker_polling_controller.js"
        ).read_text(encoding="utf-8")
        module_panel_controller_text = (
            repo_root / "web" / "orchestration" / "flow" / "panel" / "module_node_picker_module_panel_controller.js"
        ).read_text(encoding="utf-8")
        warmup_controller_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_warmup_controller.js"
        ).read_text(encoding="utf-8")
        lifecycle_text = (
            repo_root / "web" / "orchestration" / "runtime" / "lifecycle" / "module_node_picker_lifecycle.js"
        ).read_text(encoding="utf-8")
        resume_flow_text = (
            repo_root / "web" / "orchestration" / "flow" / "resume" / "module_node_picker_resume_flow.js"
        ).read_text(encoding="utf-8")
        resume_custom_text = (
            repo_root / "web" / "orchestration" / "flow" / "resume" / "module_node_picker_resume_custom_refresh.js"
        ).read_text(encoding="utf-8")
        resume_update_text = (
            repo_root / "web" / "orchestration" / "flow" / "resume" / "module_node_picker_resume_module_update.js"
        ).read_text(encoding="utf-8")
        resume_comfy_text = (
            repo_root / "web" / "orchestration" / "flow" / "resume" / "module_node_picker_resume_comfy_refresh.js"
        ).read_text(encoding="utf-8")
        busy_ui_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_busy_ui.js"
        ).read_text(encoding="utf-8")
        debug_ui_text = (
            repo_root / "web" / "orchestration" / "ui" / "module_node_picker_debug_ui.js"
        ).read_text(encoding="utf-8")
        runtime_state_text = (
            repo_root / "web" / "state" / "module_node_picker_runtime_state.js"
        ).read_text(encoding="utf-8")
        error_utils_text = (
            repo_root / "web" / "orchestration" / "core" / "infra" / "module_node_picker_error_utils.js"
        ).read_text(encoding="utf-8")
        runtime_setup_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_setup.js"
        ).read_text(encoding="utf-8")
        stage_bridge_text = (
            repo_root / "web" / "orchestration" / "core" / "composition" / "module_node_picker_stage_bridge.js"
        ).read_text(encoding="utf-8")
        runtime_bootstrap_bindings_text = (
            repo_root / "web" / "orchestration" / "runtime" / "bootstrap" / "module_node_picker_runtime_bootstrap_bindings.js"
        ).read_text(encoding="utf-8")

        self.assertIn("MODULE_PICKER_RUNTIME_STATE_KEY", picker_runtime_text)
        self.assertIn("getRuntimePickerState", runtime_state_text)
        self.assertIn("LEGACY_CUSTOM_STATUS_CHECKED_STORAGE_KEY", picker_runtime_text)
        self.assertIn("clearLegacyPersistentFlags", runtime_state_text)
        self.assertIn("loadCustomStatusChecked", runtime_state_text)
        self.assertIn("saveCustomStatusChecked", runtime_state_text)
        self.assertIn("setCustomStatusChecked", picker_runtime_text)
        self.assertIn("loadComfyStatusChecked", runtime_state_text)
        self.assertIn("saveComfyStatusChecked", runtime_state_text)
        self.assertIn("loadComfyInfoSnapshot", picker_runtime_text)
        self.assertIn("saveComfyInfoSnapshot", picker_runtime_text)
        self.assertIn("setComfyStatusChecked", picker_runtime_text)
        self.assertIn("hasPendingCustomRefresh", picker_runtime_text)
        self.assertIn("setPendingCustomRefresh", picker_runtime_text)
        self.assertIn("clearPendingCustomRefresh", picker_runtime_text)
        self.assertIn("hasPendingUpdate", picker_runtime_text)
        self.assertIn("setPendingUpdate", picker_runtime_text)
        self.assertIn("clearPendingUpdate", picker_runtime_text)
        self.assertIn("hasPendingComfyInfoRefresh", picker_runtime_text)
        self.assertIn("setPendingComfyInfoRefresh", picker_runtime_text)
        self.assertIn("clearPendingComfyInfoRefresh", picker_runtime_text)
        self.assertIn("resumePendingCustomRefreshFlow", picker_runtime_text)
        self.assertIn("resumePendingModuleUpdateFlow", picker_runtime_text)
        self.assertIn("resumePendingComfyInfoRefreshFlow", picker_runtime_text)
        self.assertIn("initializeModuleNodePickerRuntime", picker_runtime_text)
        self.assertIn("runStartupCoordinator", runtime_bootstrap_text)
        self.assertIn("startCatalogStartupLoad", picker_runtime_text)
        self.assertTrue(
            ("isActionBusy: () => busyUi.isActionBusy()" in picker_runtime_text)
            or ("isActionBusy: () => context?.busyUi?.isActionBusy?.()" in picker_runtime_text)
        )
        self.assertIn("const cancelStartupLoad = runStartupCoordinator({", runtime_bootstrap_text)
        self.assertIn("onSettled", bindings_text)
        self.assertIn("settle()", bindings_text)
        self.assertIn("export function runStartupCoordinator", startup_flow_text)
        self.assertIn("setStartupBusy(true)", startup_flow_text)
        self.assertIn("setStartupBusy(false)", startup_flow_text)
        self.assertIn("shouldContinueStartup", startup_flow_text)
        self.assertIn("hasPendingWork", startup_flow_text)
        self.assertIn("runCatalogStartupLoad", startup_flow_text)
        self.assertIn("await resumePendingCustomRefreshFlow()", startup_flow_text)
        self.assertIn("await resumePendingModuleUpdateFlow()", startup_flow_text)
        self.assertIn("await resumePendingComfyInfoRefreshFlow()", startup_flow_text)
        self.assertIn("export function createModuleNodePickerActionFlows", action_flows_text)
        self.assertIn("runInstallComfyUIRequirementsFlow", action_flows_text)
        self.assertIn("maybeInstallChangedRequirementsFlow", action_flows_text)
        self.assertIn("runModuleUpdateFlow", action_flows_text)
        self.assertIn("runRefreshComfyUIInfoAction", action_flows_text)
        self.assertIn("runRefreshCustomNodesInfoAction", action_flows_text)
        self.assertIn("runRefreshModuleInfoAction", action_flows_text)
        self.assertIn("runInstallSingleModuleRequirementsAction", action_flows_text)
        self.assertIn("resumePendingCustomRefreshFlowImpl", action_flows_text)
        self.assertIn("resumePendingModuleUpdateFlowImpl", action_flows_text)
        self.assertIn("resumePendingComfyInfoRefreshFlowImpl", action_flows_text)
        self.assertNotIn("createModuleNodePickerActionFlows", picker_runtime_text)
        self.assertNotIn("runRefreshModuleInfoAction", picker_runtime_text)
        self.assertNotIn("runInstallSingleModuleRequirementsAction", picker_runtime_text)
        self.assertNotIn("createModuleNodePickerPollingController", picker_runtime_text)
        self.assertNotIn("createModuleNodePickerModulePanelController", picker_runtime_text)
        self.assertIn("createModuleNodePickerLifecycle", runtime_setup_text)
        self.assertIn("createModuleNodePickerUiControllers", picker_runtime_text)
        self.assertIn("createModuleNodePickerFlowWiring", picker_runtime_text)
        self.assertIn("createModuleNodePickerRuntimeSetup", picker_runtime_text)
        self.assertIn("createModuleNodePickerStageBridge", picker_runtime_text)
        self.assertIn("export function createModuleNodePickerFlowWiring", flow_wiring_text)
        self.assertIn("createModuleNodePickerPollingController", flow_wiring_text)
        self.assertIn("createModuleNodePickerCatalogController", flow_wiring_text)
        self.assertIn("createModuleNodePickerActionFlows", flow_wiring_text)
        self.assertIn("createModuleNodePickerModulePanelController", flow_wiring_text)
        self.assertIn("export function createModuleNodePickerUiControllers", ui_controllers_text)
        self.assertIn("createModuleNodePickerSelectionController", ui_controllers_text)
        self.assertIn("createModuleNodePickerViewHelpers", ui_controllers_text)
        self.assertIn("createModuleNodePickerStatusCards", ui_controllers_text)
        self.assertIn("createBusyUiController", ui_controllers_text)
        self.assertIn("export function initializeModuleNodePickerRuntime", runtime_bootstrap_text)
        self.assertIn("bindModuleNodePickerEvents", runtime_bootstrap_text)
        self.assertIn("export function createModuleNodePickerRuntimeSetup", runtime_setup_text)
        self.assertIn("createModuleNodePickerRuntimeContext", runtime_setup_text)
        self.assertIn("createModuleNodePickerLifecycle", runtime_setup_text)
        self.assertIn("createModuleNodePickerApiClient", runtime_setup_text)
        self.assertIn("createModuleNodePickerDebugUi", runtime_setup_text)
        self.assertIn("createProcessUiController", runtime_setup_text)
        self.assertIn("getPollingController()?.invalidate?.()", lifecycle_text)
        self.assertIn("export function createModuleNodePickerPollingController", polling_controller_text)
        self.assertIn("pollRefreshProgressLoop", polling_controller_text)
        self.assertIn("pollUpdateProgressLoop", polling_controller_text)
        self.assertIn("invalidate", polling_controller_text)
        self.assertIn("export function createModuleNodePickerModulePanelController", module_panel_controller_text)
        self.assertIn("renderNodeListPanel", module_panel_controller_text)
        self.assertIn("renderModuleInfoCard", module_panel_controller_text)
        self.assertIn("export function createModuleNodePickerWarmupController", warmup_controller_text)
        self.assertIn("handleCatalogResult", warmup_controller_text)
        self.assertIn("setWarmupIndicator", warmup_controller_text)
        self.assertIn("warmupPollAttempts >= maxAttempts", warmup_controller_text)
        self.assertIn("Promise.resolve(poller(nextOptions)).catch", warmup_controller_text)
        self.assertIn("export function createModuleNodePickerStageBridge", stage_bridge_text)
        self.assertIn("wireFlowStage", stage_bridge_text)
        self.assertIn("adapters", stage_bridge_text)
        self.assertIn("export function createModuleNodePickerRuntimeBootstrapBindings", runtime_bootstrap_bindings_text)
        self.assertIn("startCatalogStartupLoad", runtime_bootstrap_bindings_text)
        self.assertIn("export function projectModuleNodePickerRuntimeSetup", runtime_projection_text)
        self.assertIn("runtimeStatus", runtime_projection_text)
        self.assertIn("export function createModuleNodePickerLifecycle", lifecycle_text)
        self.assertIn("getCatalogController", lifecycle_text)
        self.assertIn("getPollingController", lifecycle_text)
        self.assertIn("dispose", lifecycle_text)
        self.assertIn("export async function resumePendingCustomRefreshFlow", resume_flow_text)
        self.assertIn("export async function resumePendingModuleUpdateFlow", resume_flow_text)
        self.assertIn("export async function resumePendingComfyInfoRefreshFlow", resume_flow_text)
        self.assertIn("resumePendingCustomRefreshFlowImpl", resume_flow_text)
        self.assertIn("resumePendingModuleUpdateFlowImpl", resume_flow_text)
        self.assertIn("resumePendingComfyInfoRefreshFlowImpl", resume_flow_text)
        self.assertIn("export async function resumePendingCustomRefreshFlowImpl", resume_custom_text)
        self.assertIn("export async function resumePendingModuleUpdateFlowImpl", resume_update_text)
        self.assertIn("export async function resumePendingComfyInfoRefreshFlowImpl", resume_comfy_text)
        self.assertIn("export function createBusyUiController", busy_ui_text)
        self.assertIn("setCatalogControlsLoading", busy_ui_text)
        self.assertIn("setActionBusy", busy_ui_text)
        self.assertIn("setStartupBusy", busy_ui_text)
        self.assertIn("export function createModuleNodePickerDebugUi", debug_ui_text)
        self.assertIn("onCopyStatus", debug_ui_text)
        self.assertIn("setDiagnosticText", debug_ui_text)
        self.assertIn("export function createModuleNodePickerApiClient", api_client_text)
        self.assertIn("AbortController", api_client_text)
        self.assertIn("dispose: () =>", api_client_text)
        self.assertIn("export function createModuleNodePickerCatalogController", catalog_controller_text)
        self.assertIn("loadCatalog", catalog_controller_text)
        self.assertIn("loadModuleInfo", catalog_controller_text)
        self.assertIn("bumpRequestTokens", catalog_controller_text)
        self.assertIn("warmupController.setPoller", catalog_controller_text)
        self.assertIn("renderModuleInfo: (...args) => renderModuleInfoImpl(...args)", flow_wiring_text)
        self.assertIn("export function createModuleNodePickerStatusCards", status_cards_text)
        self.assertIn("renderComfyAlertCard", status_cards_text)
        self.assertIn("renderCustomAlertCard", status_cards_text)
        self.assertIn("syncUpdateAllButton", status_cards_text)
        self.assertIn("export function createModuleNodePickerRuntimeContext", runtime_context_text)
        self.assertIn("createModuleNodePickerStore", runtime_context_text)
        self.assertIn("createModuleDiagnosticsLogger", runtime_context_text)
        self.assertIn("createRuntimeStatusAccessors", runtime_context_text)
        self.assertIn("__selection_initialized__", runtime_context_text)
        self.assertIn("export function createModuleNodePickerViewHelpers", view_helpers_text)
        self.assertIn("setProcessAction", view_helpers_text)
        self.assertIn("setRefreshLine", view_helpers_text)
        self.assertIn("setHelpModuleSummary", view_helpers_text)
        self.assertIn("export function getRuntimePickerState", runtime_state_text)
        self.assertIn("export function clearLegacyPersistentFlags", runtime_state_text)
        self.assertIn("export function createRuntimeStatusAccessors", runtime_state_text)
        self.assertIn("export function loadComfyCheckMode", runtime_state_text)
        self.assertIn("export function saveComfyCheckMode", runtime_state_text)
        self.assertIn("import { isCanceledRequestError }", picker_runtime_text)
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
