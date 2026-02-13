"""
Module: tests/test_module_browser_node_classification_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted node classification and annotation helpers.

Purpose:
    Validates group classification and fallback annotation behavior moved from
    module_node_browser_api into node_classification_ops.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _DummyNode:
    """Simple dummy node class used for classification tests."""

    __module__ = "custom_nodes.ComfyUI_ALEXZ_tools.nodes.test"


class ModuleBrowserNodeClassificationOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.node_classification_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import node-classification helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import node_classification_ops

        self.ops = node_classification_ops

    def test_module_root_and_fallback_annotation(self):
        """Module root and fallback annotation should parse class metadata safely."""
        self.assertEqual(self.ops.module_root(_DummyNode), "custom_nodes")
        _DummyNode.CATEGORY = "test/cat"
        _DummyNode.RETURN_TYPES = ("IMAGE", "JSON")
        text = self.ops.fallback_annotation(_DummyNode)
        self.assertIn("test/cat", text)
        self.assertIn("IMAGE", text)

    def test_classify_by_relative_module_custom(self):
        """Relative module path should classify custom node module correctly."""
        _DummyNode.RELATIVE_PYTHON_MODULE = "custom_nodes.ComfyUI_ALEXZ_tools.nodes.image"
        group, module_name = self.ops.classify_by_relative_module(
            _DummyNode,
            canonical_custom_module_name_fn=lambda name: f"C::{name}",
            classify_by_source_path_fn=lambda node_cls: None,
            module_root_fn=self.ops.module_root,
        )
        self.assertEqual(group, "custom")
        self.assertEqual(module_name, "C::ComfyUI_ALEXZ_tools")

    def test_classify_by_source_path_custom_and_core_extras(self):
        """Source-path classification should detect custom and core_extras groups."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            custom_path = root / "ComfyUI_ALEXZ_tools" / "nodes" / "x.py"
            custom_path.parent.mkdir(parents=True, exist_ok=True)
            custom_path.write_text("# node", encoding="utf-8")

            result_custom = self.ops.classify_by_source_path(
                _DummyNode,
                node_source_file_fn=lambda node_cls: str(custom_path),
                custom_nodes_roots_fn=lambda: [root],
                canonical_custom_module_name_fn=lambda name: f"CANON::{name}",
                module_root_fn=self.ops.module_root,
            )
            self.assertEqual(result_custom, ("custom", "CANON::ComfyUI_ALEXZ_tools"))

            extras_path = root / "comfy_extras" / "nodes_foo.py"
            extras_path.parent.mkdir(parents=True, exist_ok=True)
            extras_path.write_text("# extra", encoding="utf-8")
            result_extras = self.ops.classify_by_source_path(
                _DummyNode,
                node_source_file_fn=lambda node_cls: str(extras_path),
                custom_nodes_roots_fn=lambda: [root / "not-custom"],
                canonical_custom_module_name_fn=lambda name: name,
                module_root_fn=self.ops.module_root,
            )
            self.assertEqual(result_extras, ("core_extras", "nodes_foo.py"))


if __name__ == "__main__":
    unittest.main()

