"""
Module: tests/test_module_browser_path_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted path resolution helper functions.

Purpose:
    Verifies custom roots, manager DB path lookup, module-dir resolution,
    and ComfyUI root detection moved into path_ops.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class ModuleBrowserPathOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.path_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import path helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import path_ops

        self.ops = path_ops

    def test_custom_nodes_roots_uses_folder_paths_when_available(self):
        """Custom roots should come from folder_paths provider when present."""

        class _FolderPaths:
            @staticmethod
            def get_folder_paths(_name):
                return ["/tmp/custom_a", "/tmp/custom_b"]

        roots = self.ops.custom_nodes_roots(
            folder_paths_module=_FolderPaths,
            fallback_root=Path("/fallback"),
        )
        self.assertEqual([str(x) for x in roots], ["/tmp/custom_a", "/tmp/custom_b"])

    def test_custom_nodes_roots_fallback(self):
        """Fallback root should be used when folder_paths is unavailable."""
        roots = self.ops.custom_nodes_roots(folder_paths_module=None, fallback_root=Path("/fallback"))
        self.assertEqual([str(x) for x in roots], ["/fallback"])

    def test_manager_paths_and_module_dir_resolution(self):
        """Manager DB paths and module dir should resolve from custom roots."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mgr_dir = root / "comfyui-manager"
            mgr_dir.mkdir(parents=True, exist_ok=True)
            custom_db = mgr_dir / "custom-node-list.json"
            stats_db = mgr_dir / "github-stats.json"
            custom_db.write_text("{}", encoding="utf-8")
            stats_db.write_text("{}", encoding="utf-8")
            mod_dir = root / "ComfyUI_ALEXZ_tools"
            mod_dir.mkdir(parents=True, exist_ok=True)

            roots_fn = lambda: [root]
            self.assertEqual(self.ops.manager_custom_db_path(custom_nodes_roots_fn=roots_fn), custom_db)
            self.assertEqual(self.ops.manager_github_stats_path(custom_nodes_roots_fn=roots_fn), stats_db)
            resolved = self.ops.module_dir(
                "alexz",
                canonical_custom_module_name_fn=lambda _name: "ComfyUI_ALEXZ_tools",
                custom_nodes_roots_fn=roots_fn,
            )
            self.assertEqual(resolved, mod_dir)

    def test_comfyui_root_detects_repo_marker(self):
        """ComfyUI root should be found by `nodes.py` and `.git` marker files."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ComfyUI"
            node_dir = root / "custom_nodes" / "ComfyUI_ALEXZ_tools" / "utils"
            node_dir.mkdir(parents=True, exist_ok=True)
            (root / "nodes.py").write_text("# marker", encoding="utf-8")
            (root / ".git").mkdir(parents=True, exist_ok=True)
            fake_api_file = node_dir / "module_node_browser_api.py"
            fake_api_file.write_text("# api", encoding="utf-8")
            detected = self.ops.comfyui_root(fake_api_file)
            self.assertEqual(detected, root)


if __name__ == "__main__":
    unittest.main()

