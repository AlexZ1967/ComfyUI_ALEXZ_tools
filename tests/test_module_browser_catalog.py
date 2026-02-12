"""
Module: tests/test_module_browser_catalog.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for pure catalog helpers used by Module Node Picker backend.

Purpose:
    Validates node collection, grouping, module summaries, and module filtering
    behavior after extracting catalog logic from API handlers.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class _NodeA:
    CATEGORY = "cat/A"


class _NodeB:
    CATEGORY = "cat/B"


class ModuleBrowserCatalogTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.catalog` pure helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import catalog helpers for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import catalog

        self.catalog = catalog

    def test_collect_nodes_uses_classifier_and_annotation(self):
        """Collector should map class/display/annotation fields into node entries."""
        class_map = {"NodeA": _NodeA, "NodeB": _NodeB}
        display_map = {"NodeA": "Node A", "NodeB": "Node B"}
        items = self.catalog.collect_nodes(
            class_map=class_map,
            display_map=display_map,
            annotation_resolver=lambda name, cls: f"anno:{name}",
            classifier=lambda cls: ("custom", "ComfyUI_ALEXZ_tools") if cls is _NodeA else ("core", "nodes"),
        )
        self.assertEqual(len(items), 2)
        by_name = {item["node_name"]: item for item in items}
        self.assertEqual(by_name["NodeA"]["annotation"], "anno:NodeA")
        self.assertEqual(by_name["NodeA"]["module"], "ComfyUI_ALEXZ_tools")
        self.assertEqual(by_name["NodeB"]["group"], "core")

    def test_build_catalog_sorts_modules_and_nodes(self):
        """Catalog builder should keep deterministic sorting for modules and node names."""
        items = [
            {"module": "z_mod", "display_name": "B"},
            {"module": "a_mod", "display_name": "Z"},
            {"module": "a_mod", "display_name": "A"},
        ]
        catalog = self.catalog.build_catalog(items)
        self.assertEqual(list(catalog.keys()), ["a_mod", "z_mod"])
        self.assertEqual([x["display_name"] for x in catalog["a_mod"]], ["A", "Z"])

    def test_build_group_modules_adds_custom_discovered_zero_count(self):
        """Group module summary should include discovered custom modules with zero nodes."""
        grouped_nodes = {
            "custom": [{"module": "modA"}, {"module": "modA"}],
            "core": [{"module": "nodes"}],
        }
        grouped = self.catalog.build_group_modules(
            grouped_nodes=grouped_nodes,
            discover_custom_modules=lambda: ["modA", "modB"],
            cached_module_flags=lambda group, mod: {"flag": f"{group}:{mod}"},
        )
        custom = {item["module"]: item for item in grouped["custom"]}
        self.assertEqual(custom["modA"]["count"], 2)
        self.assertEqual(custom["modB"]["count"], 0)
        self.assertEqual(custom["modB"]["flag"], "custom:modB")

    def test_filter_modules_prefers_exact_match(self):
        """Module filter should return exact case-insensitive match when available."""
        names = ["ComfyUI_ALEXZ_tools", "ComfyUI-Manager", "ComfyUI-RMBG"]
        exact = self.catalog.filter_modules("comfyui-manager", names)
        self.assertEqual(exact, ["ComfyUI-Manager"])
        partial = self.catalog.filter_modules("comfyui", names)
        self.assertEqual(partial, names)


if __name__ == "__main__":
    unittest.main()
