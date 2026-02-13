"""
Module: tests/test_module_browser_node_snapshot_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted node snapshot helper operations.

Purpose:
    Validates deterministic node snapshot and path helper behavior after
    extraction from backend API module.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _DummyNode:
    """Simple node-like class for snapshot tests."""


class ModuleBrowserNodeSnapshotOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.node_snapshot_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import node_snapshot_ops

        self.ops = node_snapshot_ops

    def test_relative_to_custom_roots_returns_relative_path(self):
        """Path helper should reduce absolute path to custom root relative path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "moduleA" / "file.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# test\n", encoding="utf-8")
            rel = self.ops.relative_to_custom_roots(
                str(target),
                custom_nodes_roots=lambda: [root],
            )
            self.assertEqual(rel, "moduleA/file.py")

    def test_build_node_snapshots_produces_grouped_payload(self):
        """Snapshot builder should return grouped/moduled node signature payload."""
        snapshots = self.ops.build_node_snapshots(
            class_map={"DummyNode": _DummyNode},
            classifier=lambda _cls: ("custom", "ComfyUI_ALEXZ_tools"),
            custom_nodes_roots=lambda: [Path.cwd()],
        )
        node = snapshots["custom"]["ComfyUI_ALEXZ_tools"]["DummyNode"]
        self.assertTrue(node["sig"].startswith("_DummyNode:"))
        self.assertIn("source", node)


if __name__ == "__main__":
    unittest.main()
