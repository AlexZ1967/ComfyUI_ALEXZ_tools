"""
Module: tests/test_module_browser_api_catalog_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Unit tests for extracted module-browser API catalog helpers.

Purpose:
    Verifies catalog orchestration seams moved out of
    `utils/module_node_browser_api.py` during Phase 3 stabilization.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class _NodeA:
    CATEGORY = "custom/demo"


class _NodeB:
    CATEGORY = "core/demo"


class ModuleBrowserApiCatalogOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser_api.catalog_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api import catalog_ops

        self.ops = catalog_ops

    def test_collect_nodes_builds_annotation_and_classifier_callbacks(self):
        """Collector helper should wire facade callbacks into catalog implementation."""
        captured = {}

        def _collect_nodes_impl(**kwargs):
            captured.update(kwargs)
            return [
                {
                    "node_name": "NodeA",
                    "module": "ComfyUI_ALEXZ_tools",
                    "group": "custom",
                }
            ]

        result = self.ops.collect_nodes(
            node_mappings=lambda: (
                {"NodeA": _NodeA, "NodeB": _NodeB},
                {"NodeA": "Node A", "NodeB": "Node B"},
            ),
            collect_nodes_impl=_collect_nodes_impl,
            annotations={"NodeA": "explicit"},
            fallback_annotation=lambda node_cls: f"fallback:{node_cls.__name__}",
            classify_by_relative_module=lambda node_cls: (
                ("custom", "ComfyUI_ALEXZ_tools") if node_cls is _NodeA else ("core", "nodes")
            ),
        )

        self.assertEqual(result[0]["node_name"], "NodeA")
        self.assertEqual(captured["class_map"]["NodeA"], _NodeA)
        self.assertEqual(captured["display_map"]["NodeB"], "Node B")
        self.assertEqual(captured["annotation_resolver"]("NodeA", _NodeA), "explicit")
        self.assertEqual(captured["annotation_resolver"]("NodeB", _NodeB), "fallback:_NodeB")
        self.assertEqual(captured["classifier"](_NodeA), ("custom", "ComfyUI_ALEXZ_tools"))

    def test_build_catalog_uses_collected_nodes(self):
        """Catalog builder helper should pass collected nodes to delegated implementation."""
        result = self.ops.build_catalog(
            collect_nodes_fn=lambda: [{"module": "ModA"}],
            build_catalog_impl=lambda items: {"count": len(items), "first": items[0]["module"]},
        )
        self.assertEqual(result, {"count": 1, "first": "ModA"})

    def test_build_group_catalog_uses_collected_nodes(self):
        """Group-catalog helper should pass collected nodes to delegated implementation."""
        result = self.ops.build_group_catalog(
            collect_nodes_fn=lambda: [{"group": "custom"}],
            build_group_catalog_impl=lambda items: {"custom": items},
        )
        self.assertEqual(result, {"custom": [{"group": "custom"}]})

    def test_build_group_modules_passes_runtime_dependencies(self):
        """Group-module helper should forward discovery and badge helpers unchanged."""
        grouped_nodes = {"custom": [{"module": "ModA"}]}

        def _build_group_modules_impl(**kwargs):
            self.assertIs(kwargs["grouped_nodes"], grouped_nodes)
            self.assertEqual(kwargs["discover_custom_modules"](), ["ModA", "ModB"])
            self.assertEqual(kwargs["cached_module_flags"]("custom", "ModB"), {"update_status": "unknown"})
            return {"custom": [{"module": "ModA", "count": 1}]}

        result = self.ops.build_group_modules(
            grouped_nodes,
            build_group_modules_impl=_build_group_modules_impl,
            discover_custom_modules=lambda: ["ModA", "ModB"],
            cached_module_flags=lambda group, module: {"update_status": "unknown"},
        )
        self.assertEqual(result, {"custom": [{"module": "ModA", "count": 1}]})

    def test_build_group_payload_forwards_group_order(self):
        """Group-payload helper should preserve explicit group ordering dependency."""
        result = self.ops.build_group_payload(
            grouped_nodes={"custom": [{"node_name": "NodeA"}]},
            modules_by_group={"custom": [{"module": "ModA"}]},
            build_group_payload_impl=lambda **kwargs: [
                kwargs["group_order"][0][0],
                kwargs["grouped_nodes"]["custom"][0]["node_name"],
                kwargs["modules_by_group"]["custom"][0]["module"],
            ],
            group_order=[("custom", "Custom_Nodes")],
        )
        self.assertEqual(result, ["custom", "NodeA", "ModA"])

    def test_build_module_list_payload_delegates_named_args(self):
        """Module-list helper should call delegated builder with stable named arguments."""
        captured = {}

        def _build_module_list_payload_impl(**kwargs):
            captured.update(kwargs)
            return {"modules": ["ModA"]}

        result = self.ops.build_module_list_payload(
            {"ModA": [{"node_name": "NodeA"}]},
            "mod",
            build_module_list_payload_impl=_build_module_list_payload_impl,
        )
        self.assertEqual(captured["query"], "mod")
        self.assertIn("ModA", captured["catalog"])
        self.assertEqual(result, {"modules": ["ModA"]})

    def test_build_module_nodes_payload_passes_filter_function(self):
        """Module-nodes helper should forward the filter seam to the delegated builder."""
        captured = {}

        def _build_module_nodes_payload_impl(**kwargs):
            captured.update(kwargs)
            return {"module_count": 1}

        def _filter_modules(query, module_names):
            return [name for name in module_names if query.lower() in name.lower()]

        result = self.ops.build_module_nodes_payload(
            {"ModA": [{"node_name": "NodeA"}]},
            "mod",
            build_module_nodes_payload_impl=_build_module_nodes_payload_impl,
            filter_modules_fn=_filter_modules,
        )
        self.assertEqual(captured["query"], "mod")
        self.assertIs(captured["filter_modules_fn"], _filter_modules)
        self.assertEqual(result, {"module_count": 1})


if __name__ == "__main__":
    unittest.main()
