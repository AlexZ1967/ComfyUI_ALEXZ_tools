"""
Module: tests/test_module_browser_command_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted command execution helpers.

Purpose:
    Validates subprocess wrapper behavior (including git safe.directory retry)
    after command helper extraction from module_node_browser_api.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserCommandOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.command_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import command helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import command_ops

        self.ops = command_ops

    def test_extract_git_repo_from_args_resolves_c_path(self):
        """Git helper should resolve path from `git -C` command args."""
        repo = self.ops.extract_git_repo_from_args(["git", "-C", ".", "status"])
        self.assertTrue(bool(repo))
        self.assertNotIn("/./", str(repo))

    def test_run_git_returns_trimmed_stdout(self):
        """`run_git` should return trimmed stdout on successful command."""

        def _run_command(args, timeout=2.0, disable_git_prompt=False):
            self.assertEqual(args, ["git", "status"])
            self.assertEqual(timeout, 2.0)
            self.assertTrue(disable_git_prompt)
            return {"ok": True, "stdout": "  on branch main  "}

        out = self.ops.run_git(["git", "status"], run_command_fn=_run_command)
        self.assertEqual(out, "on branch main")

    def test_run_command_retries_after_safe_directory_fix(self):
        """Git command should retry after automatic safe.directory configuration."""
        fake_repo = "/tmp/fake_repo"
        calls: list[str] = []

        class _Proc:
            def __init__(self, returncode, stdout="", stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr

        def _fake_run(args, **_kwargs):
            cmd = " ".join(args)
            calls.append(cmd)
            if cmd == f"git -C {fake_repo} status":
                if calls.count(cmd) == 1:
                    return _Proc(
                        128,
                        "",
                        "detected dubious ownership in repository at "
                        f"'{fake_repo}'\n"
                        f"git config --global --add safe.directory {fake_repo}",
                    )
                return _Proc(0, "On branch main", "")
            if cmd == f"git config --global --add safe.directory {fake_repo}":
                return _Proc(0, "", "")
            return _Proc(0, "", "")

        result = self.ops.run_command(
            ["git", "-C", fake_repo, "status"],
            disable_git_prompt=True,
            subprocess_run=_fake_run,
        )

        self.assertTrue(bool(result.get("ok")))
        self.assertIn(f"git config --global --add safe.directory {fake_repo}", calls)
        self.assertGreaterEqual(calls.count(f"git -C {fake_repo} status"), 2)

    def test_tail_lines_keeps_last_lines_with_ellipsis(self):
        """Tail helper should keep only final non-empty lines after limit."""
        text = "\n".join([f"line-{idx}" for idx in range(6)])
        out = self.ops.tail_lines(text, max_lines=3)
        self.assertEqual(out.splitlines()[0], "...")
        self.assertEqual(out.splitlines()[-1], "line-5")


if __name__ == "__main__":
    unittest.main()

