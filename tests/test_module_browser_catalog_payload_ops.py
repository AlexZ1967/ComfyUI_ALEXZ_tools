"""
Module: tests/test_module_browser_catalog_payload_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted catalog payload builder helpers.

Purpose:
    Verifies deterministic response assembly for catalog-related API routes
    after payload helper extraction from module_node_browser_api.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserCatalogPayloadOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.catalog_payload_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import payload helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import catalog_payload_ops

        self.ops = catalog_payload_ops

    def test_build_group_payload_respects_group_order(self):
        """Group payload should preserve provided order and include counters."""
        groups = self.ops.build_group_payload(
            group_order=(("custom", "Custom_Nodes"), ("core", "Core_Nodes")),
            grouped_nodes={"core": [{"node": "A"}], "custom": [{"node": "B"}, {"node": "C"}]},
            modules_by_group={"core": [{"module": "nodes"}], "custom": [{"module": "ComfyUI_ALEXZ_tools"}]},
        )
        self.assertEqual([item.get("id") for item in groups], ["custom", "core"])
        self.assertEqual(groups[0].get("count"), 2)
        self.assertEqual(groups[1].get("module_count"), 1)

    def test_build_module_list_payload_applies_lower_query_filter(self):
        """Module-list payload should filter modules case-insensitively."""
        payload = self.ops.build_module_list_payload(
            catalog={"ComfyUI_ALEXZ_tools": [{}, {}], "OtherPack": [{}]},
            query="alexz",
        )
        self.assertEqual(payload.get("query"), "alexz")
        modules = payload.get("modules") or []
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].get("module"), "ComfyUI_ALEXZ_tools")
        self.assertEqual(modules[0].get("count"), 2)

    def test_build_module_nodes_payload_uses_filter_and_preserves_hint(self):
        """Module-nodes payload should rely on injected filter function and keep hint text."""

        def _filter(query, module_names):
            self.assertEqual(query, "mod")
            self.assertEqual(module_names, ["modA", "modB"])
            return ["modB"]

        payload = self.ops.build_module_nodes_payload(
            catalog={"modA": [{"node": "A"}], "modB": [{"node": "B1"}, {"node": "B2"}]},
            query="mod",
            filter_modules_fn=_filter,
        )
        self.assertEqual(payload.get("module_count"), 1)
        results = payload.get("results") or []
        self.assertEqual(results[0].get("module"), "modB")
        self.assertEqual(results[0].get("count"), 2)
        self.assertIn("python-модуля", str(payload.get("hint") or ""))


if __name__ == "__main__":
    unittest.main()

