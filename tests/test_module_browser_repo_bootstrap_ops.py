"""
Module: tests/test_module_browser_repo_bootstrap_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted repository bootstrap helper functions.

Purpose:
    Verifies ComfyUI requirements-path lookup and manager-based remote bootstrap
    behavior moved from module_node_browser_api.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class ModuleBrowserRepoBootstrapOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.repo_bootstrap_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import repo-bootstrap helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import repo_bootstrap_ops

        self.ops = repo_bootstrap_ops

    def test_comfyui_requirements_path_found_or_missing(self):
        """Requirements path helper should return file path only when it exists."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = root / "requirements.txt"
            req.write_text("torch\n", encoding="utf-8")
            found = self.ops.comfyui_requirements_path(comfyui_root_fn=lambda: root)
            self.assertEqual(found, req)
            missing = self.ops.comfyui_requirements_path(comfyui_root_fn=lambda: root / "none")
            self.assertIsNone(missing)

    def test_bootstrap_module_remote_from_manager_success(self):
        """Remote bootstrap should add origin when no remotes and manager repo is known."""
        with tempfile.TemporaryDirectory() as tmp:
            module_dir = Path(tmp)
            calls: list[list[str]] = []

            def _run(args, timeout, disable_git_prompt):
                calls.append(list(args))
                return {"ok": True}

            result = self.ops.bootstrap_module_remote_from_manager(
                "modA",
                module_dir,
                git_remote_names_fn=lambda _dir: [],
                manager_meta_for_module_fn=lambda module, repository: {"repository": "https://github.com/example/modA.git"},
                normalize_repo_url_fn=lambda url: str(url or "").replace(".git", "") if url else None,
                run_command_fn=_run,
            )
            self.assertTrue(result)
            self.assertEqual(
                calls[-1],
                ["git", "-C", str(module_dir), "remote", "add", "origin", "https://github.com/example/modA"],
            )

    def test_bootstrap_module_remote_from_manager_skips_when_remote_exists(self):
        """Bootstrap should no-op when repository already has remotes configured."""
        result = self.ops.bootstrap_module_remote_from_manager(
            "modA",
            Path("/tmp/any"),
            git_remote_names_fn=lambda _dir: ["origin"],
            manager_meta_for_module_fn=lambda module, repository: {"repository": "https://github.com/example/modA.git"},
            normalize_repo_url_fn=lambda url: str(url or ""),
            run_command_fn=lambda args, timeout, disable_git_prompt: {"ok": False},
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

