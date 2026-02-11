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


if __name__ == "__main__":
    unittest.main()

