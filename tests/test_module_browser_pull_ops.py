"""
Module: tests/test_module_browser_pull_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted git pull/update helper operations.

Purpose:
    Validates custom/ComfyUI pull orchestration helpers after Phase 3 split.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path


class ModuleBrowserPullOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.pull_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import pull ops helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import pull_ops

        self.ops = pull_ops

    def test_is_git_local_changes_block_detects_localized_output(self):
        """Localized git-merge conflict text should be recognized."""
        text = "Ваши локальные изменения в указанных файлах будут перезаписаны при слиянии."
        self.assertTrue(self.ops.is_git_local_changes_block(text))

    def test_pull_comfyui_returns_not_found_when_root_missing(self):
        """ComfyUI pull helper should stop early when root cannot be resolved."""
        result = self.ops.pull_comfyui(
            comfyui_root=lambda: None,
            update_console_log=lambda *_args, **_kwargs: None,
            run_git=lambda _args, _timeout: "",
            git_pick_remote=lambda _repo, _upstream: None,
            git_resolve_remote_ref=lambda _repo, _remote, _branch, _upstream: (None, None),
            run_command=lambda _args, _timeout, _disable: {"ok": True},
            requirements_changed_between=lambda _repo, _before, _after: False,
            set_comfyui_requirements_pending=lambda _pending, _before, _after: None,
            perf_counter=lambda: 0.0,
        )
        self.assertEqual(result.get("status"), "not_found")

    def test_pull_custom_module_uses_resolved_remote_ref_without_upstream(self):
        """When upstream is absent, helper should use resolved remote branch fallback."""
        run_command_calls = []
        pending_calls = []
        head_calls = [0]

        def _run_git(args, _timeout):
            if args[-2:] == ["rev-parse", "--is-inside-work-tree"]:
                return "true"
            if args[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
                return "main"
            if args[-4:] == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]:
                return ""
            if args[-2:] == ["rev-parse", "HEAD"]:
                head_calls[0] += 1
                return "aaa11111" if head_calls[0] == 1 else "bbb22222"
            return ""

        def _run_command(args, _timeout, _disable_git_prompt):
            run_command_calls.append(list(args))
            return {"ok": True, "stdout": "ok", "stderr": ""}

        result = self.ops.pull_custom_module(
            "modA",
            canonical_custom_module_name=lambda name: name,
            module_dir_resolver=lambda _name: Path("/tmp/modA"),
            update_console_log=lambda *_args, **_kwargs: None,
            run_git=_run_git,
            git_pick_remote=lambda _repo, _upstream: "origin",
            git_resolve_remote_ref=lambda _repo, _remote, _branch, _upstream: ("origin/main", "main"),
            bootstrap_module_remote_from_manager=lambda _name, _module_dir: False,
            run_command=_run_command,
            is_git_local_changes_block_fn=lambda _text: False,
            requirements_changed_between=lambda _repo, _before, _after: True,
            set_module_requirements_pending=lambda module, pending, before, after: pending_calls.append(
                (module, pending, before, after)
            ),
            perf_counter=lambda: 0.0,
        )

        self.assertEqual(result.get("status"), "updated")
        self.assertTrue(result.get("requirements_changed"))
        self.assertEqual(
            run_command_calls[-1],
            ["git", "-C", "/tmp/modA", "pull", "--ff-only", "origin", "main"],
        )
        self.assertEqual(
            pending_calls,
            [("modA", True, "aaa11111", "bbb22222")],
        )


if __name__ == "__main__":
    unittest.main()
