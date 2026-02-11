"""
Module: tests/test_slice0_registry.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Slice 0 regression tests for extensibility registry and state-schema helpers.

Purpose:
    Ensures add/remove lifecycle helpers for nodes/widgets/API registry and
    state-cache schema normalization remain stable while backend is refactored.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass


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


class Slice0RegistryTests(unittest.TestCase):
    """Validate Slice 0 extensibility and state-schema infrastructure."""

    @classmethod
    def setUpClass(cls):
        """Prepare dynamic package import context for test runtime."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()

    def test_node_registry_contains_expected_core_specs(self):
        """Node registry keeps canonical node manifest accessible via one source."""
        node_registry = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.node_registry")
        specs = list(node_registry.iter_node_specs())
        type_names = {item[0] for item in specs}
        self.assertIn("ImageColorMatchToReference", type_names)
        self.assertIn("VideoFrameMatch", type_names)
        self.assertIn("GenerateQRCode", type_names)

    def test_component_registry_supports_register_and_unregister(self):
        """Component registry supports add/remove lifecycle in-memory."""
        module = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.component_registry")
        registry = module.ComponentRegistry()

        entry = module.ComponentEntry(
            component_id="widget:test",
            kind="widget",
            name="Test Widget",
            module="web/test_widget.js",
            source="/tmp/test_widget.js",
        )
        registry.register(entry)
        self.assertEqual(registry.summary()["widget_count"], 1)
        self.assertEqual(registry.summary()["total"], 1)

        registry.unregister("widget:test")
        self.assertEqual(registry.summary()["widget_count"], 0)
        self.assertEqual(registry.summary()["total"], 0)

    def test_default_component_registry_has_nodes_widgets_and_api(self):
        """Default registry snapshot includes all three component categories."""
        module = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.component_registry")
        registry = module.build_default_component_registry()
        summary = registry.summary()
        self.assertGreater(summary["node_count"], 0)
        self.assertGreater(summary["widget_count"], 0)
        self.assertGreater(summary["api_count"], 0)

    def test_widget_manifest_contains_module_node_picker(self):
        """Widget manifest keeps Module Node Picker entrypoint in one canonical place."""
        module = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.widget_manifest")
        widget_ids = {item.widget_id for item in module.iter_widget_specs()}
        self.assertIn("module_node_picker", widget_ids)

    def test_api_manifest_exposes_component_registry_route(self):
        """API manifest keeps component-registry endpoint in published route list."""
        module = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.api_manifest")
        routes = set(module.iter_component_api_routes())
        all_routes = set(module.iter_all_api_routes())
        self.assertIn("/alexz_tools/component_registry", routes)
        self.assertIn("/alexz_tools/component_registry", all_routes)
        self.assertIn("/alexz_tools/module_update", all_routes)

    def test_module_state_schema_is_added_on_load(self):
        """Module state loader normalizes persisted cache to versioned schema."""
        api = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_node_browser_api")

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "module_state_cache.json"
            state_path.write_text("{}", encoding="utf-8")

            orig_path = api._MODULE_STATE_PATH
            orig_cache = api._MODULE_STATE_CACHE
            try:
                api._MODULE_STATE_PATH = state_path
                api._MODULE_STATE_CACHE = None
                state = api._load_module_state()
                meta = state.get("__meta__", {})
                self.assertEqual(meta.get("schema_name"), "alexz_module_state")
                self.assertEqual(meta.get("schema_version"), 1)
            finally:
                api._MODULE_STATE_PATH = orig_path
                api._MODULE_STATE_CACHE = orig_cache

    def test_component_registry_payload_contains_changes_block(self):
        """Component registry payload exposes deterministic change-tracking fields."""
        api = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_node_browser_api")
        contracts = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.contracts")
        payload = api._component_registry_payload(force_refresh=True)
        self.assertEqual(payload.get("schema_name"), contracts.COMPONENT_REGISTRY_SCHEMA_NAME)
        self.assertEqual(payload.get("schema_version"), contracts.COMPONENT_REGISTRY_SCHEMA_VERSION)
        self.assertIn("summary", payload)
        self.assertIn("changes", payload)
        self.assertIn("has_changes", payload)
        self.assertIn("manifest_signature", payload)
        self.assertIn("manifest_changed", payload)
        self.assertIn("node", payload["changes"])
        self.assertIn("widget", payload["changes"])
        self.assertIn("api", payload["changes"])
        self.assertIn("health", payload)
        self.assertIn("ok", payload["health"])
        self.assertIn("issue_count", payload["health"])
        self.assertIn("checked", payload["health"])
        self.assertIn("all_api_routes", payload["health"]["checked"])

    def test_component_registry_payload_detects_added_and_removed_components(self):
        """Change tracker reports added/removed component ids against previous snapshot."""
        api = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_node_browser_api")
        component_registry = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.component_registry")

        original_builder = api.build_default_component_registry
        original_state = api._MODULE_STATE_CACHE
        original_payload_cache = api._COMPONENT_REGISTRY_PAYLOAD_CACHE
        try:
            registry = component_registry.ComponentRegistry()
            registry.register(
                component_registry.ComponentEntry(
                    component_id="node:NewNode",
                    kind="node",
                    name="New Node",
                    module="nodes.new_node",
                    source="/tmp/new_node.py",
                )
            )
            registry.register(
                component_registry.ComponentEntry(
                    component_id="widget:module_node_picker",
                    kind="widget",
                    name="Module Node Picker",
                    module="web/module_node_picker.js",
                    source="/tmp/module_node_picker.js",
                )
            )
            registry.register(
                component_registry.ComponentEntry(
                    component_id="api:/alexz_tools/node_catalog",
                    kind="api",
                    name="/alexz_tools/node_catalog",
                    module="utils/module_node_browser_api.py",
                    source="/tmp/module_node_browser_api.py",
                )
            )
            api.build_default_component_registry = lambda: registry
            api._COMPONENT_REGISTRY_PAYLOAD_CACHE = None
            api._MODULE_STATE_CACHE = {
                "__component_registry__": {
                    "snapshot": {
                        "node": ["node:OldNode"],
                        "widget": ["widget:module_node_picker"],
                        "api": ["api:/alexz_tools/old_route"],
                    },
                    "manifest_signature": "deadbeef0000",
                    "updated_at": "2026-02-12T00:00:00+00:00",
                }
            }
            payload = api._component_registry_payload(force_refresh=True)
            node_changes = payload.get("changes", {}).get("node", {})
            api_changes = payload.get("changes", {}).get("api", {})
            self.assertIn("node:NewNode", node_changes.get("added", []))
            self.assertIn("node:OldNode", node_changes.get("removed", []))
            self.assertIn("api:/alexz_tools/old_route", api_changes.get("removed", []))
            self.assertTrue(bool(payload.get("has_changes")))
            self.assertTrue(bool(payload.get("manifest_changed")))
            self.assertTrue(str(payload.get("manifest_signature")))
            state_tracker = api._MODULE_STATE_CACHE.get("__component_registry__", {})
            self.assertEqual(state_tracker.get("schema_name"), "alexz_component_registry")
            self.assertEqual(state_tracker.get("schema_version"), 1)
            self.assertEqual(str(state_tracker.get("manifest_signature")), str(payload.get("manifest_signature")))
        finally:
            api.build_default_component_registry = original_builder
            api._MODULE_STATE_CACHE = original_state
            api._COMPONENT_REGISTRY_PAYLOAD_CACHE = original_payload_cache

    def test_component_health_report_detects_missing_widget_entrypoint(self):
        """Health report flags widget entries that point to missing frontend files."""
        health = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.health")

        @dataclass(frozen=True)
        class _FakeWidgetSpec:
            widget_id: str
            name: str
            entrypoint: str
            enabled: bool = True

        original_iter_widget_specs = health.iter_widget_specs
        try:
            health.iter_widget_specs = lambda: iter(
                [
                    _FakeWidgetSpec(
                        widget_id="missing_widget",
                        name="Missing Widget",
                        entrypoint="web/does_not_exist_widget.js",
                        enabled=True,
                    )
                ]
            )
            report = health.build_component_health_report()
            self.assertFalse(bool(report.get("ok")))
            self.assertGreater(int(report.get("issue_count", 0)), 0)
            issues = report.get("issues", [])
            self.assertTrue(any(item.get("code") == "entrypoint_not_found" for item in issues))
        finally:
            health.iter_widget_specs = original_iter_widget_specs


if __name__ == "__main__":
    unittest.main()
