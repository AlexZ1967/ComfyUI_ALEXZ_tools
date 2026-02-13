"""
Module: tests/test_module_browser_git_helpers.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted git helper functions.

Purpose:
    Validates remote selection/resolution and worktree signature behavior
    after Phase 3 git-helper extraction.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path


class ModuleBrowserGitHelpersTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.git_helpers` pure helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import git helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import git_helpers as gh

        self.gh = gh

    def test_git_pick_remote_prefers_upstream_prefix(self):
        """Remote name from explicit upstream token has top priority."""
        picked = self.gh.git_pick_remote(
            Path("/tmp/repo"),
            "origin/main",
            git_remote_names_fn=lambda _repo: ["upstream", "origin"],
        )
        self.assertEqual(picked, "origin")

    def test_git_pick_remote_prefers_origin_then_upstream(self):
        """Fallback order is origin -> upstream -> first remote."""
        self.assertEqual(
            self.gh.git_pick_remote(
                Path("/tmp/repo"),
                "",
                git_remote_names_fn=lambda _repo: ["foo", "origin", "upstream"],
            ),
            "origin",
        )
        self.assertEqual(
            self.gh.git_pick_remote(
                Path("/tmp/repo"),
                "",
                git_remote_names_fn=lambda _repo: ["foo", "upstream"],
            ),
            "upstream",
        )

    def test_git_resolve_remote_ref_uses_branch_when_present(self):
        """Resolver should choose remote/branch when that ref exists."""
        ref, branch = self.gh.git_resolve_remote_ref(
            Path("/tmp/repo"),
            "origin",
            "main",
            "",
            run_git=lambda _args, _timeout: "",
            git_ref_exists_fn=lambda _repo, ref_name: ref_name == "origin/main",
        )
        self.assertEqual(ref, "origin/main")
        self.assertEqual(branch, "main")

    def test_module_worktree_signature_hashes_status_lines(self):
        """Signature should be deterministic digest of sorted git status lines."""
        responses = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("--porcelain", "--untracked-files=no"): " M b.py\n M a.py\n",
        }

        def _run_git(args, _timeout):
            key = tuple(args[-2:])
            return responses.get(key, "")

        sig = self.gh.module_worktree_signature(
            "modA",
            module_dir_resolver=lambda _mod: Path("/tmp/repo"),
            run_git=_run_git,
        )
        self.assertTrue(bool(sig))
        self.assertEqual(len(sig), 12)


if __name__ == "__main__":
    unittest.main()
