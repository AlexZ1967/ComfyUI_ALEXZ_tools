"""
Module: tests/test_module_browser_update_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted update/install helper operations.

Purpose:
    Validates requirements diff and install helpers after Phase 3 extraction.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
import tempfile


class _NoopLogger:
    """Minimal logger stub used by helper tests."""

    def info(self, *_args, **_kwargs):
        """Ignore info logs in tests."""

    def warning(self, *_args, **_kwargs):
        """Ignore warning logs in tests."""

    def error(self, *_args, **_kwargs):
        """Ignore error logs in tests."""


class ModuleBrowserUpdateOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.update_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import update ops helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import update_ops

        self.ops = update_ops

    def test_requirements_changed_between_detects_change(self):
        """requirements diff helper returns true when requirements.txt is changed."""
        changed = self.ops.requirements_changed_between(
            Path("/tmp/repo"),
            "abc",
            "def",
            run_command=lambda args, timeout, disable_git_prompt: {"ok": True, "stdout": "requirements.txt\n"},
        )
        self.assertTrue(changed)

    def test_install_module_requirements_missing_module_dir(self):
        """Module requirements install should fail clearly when module directory is missing."""
        result = self.ops.install_module_requirements(
            "modA",
            canonical_custom_module_name=lambda name: name,
            module_dir_resolver=lambda _name: None,
            run_command=lambda args, timeout, disable_git_prompt: {"ok": True},
            python_executable=sys.executable,
            tail_lines=lambda value: str(value or ""),
            set_module_requirements_pending=lambda module, pending: None,
            logger=_NoopLogger(),
        )
        self.assertEqual(result.get("status"), "not_found")

    def test_install_module_requirements_success_clears_pending(self):
        """Successful module requirements install clears pending flag."""
        calls = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            module_dir = Path(tmp_dir)
            req = module_dir / "requirements.txt"
            req.write_text("# test\n", encoding="utf-8")
            result = self.ops.install_module_requirements(
                "modA",
                canonical_custom_module_name=lambda name: name,
                module_dir_resolver=lambda _name: module_dir,
                run_command=lambda args, timeout, disable_git_prompt: {"ok": True, "stdout": "ok", "stderr": ""},
                python_executable=sys.executable,
                tail_lines=lambda value: str(value or ""),
                set_module_requirements_pending=lambda module, pending: calls.append((module, pending)),
                logger=_NoopLogger(),
            )
            self.assertEqual(result.get("status"), "installed")
            self.assertIn(("modA", False), calls)

    def test_install_comfyui_requirements_missing_path(self):
        """ComfyUI requirements install should return missing state when file absent."""
        result = self.ops.install_comfyui_requirements(
            comfyui_requirements_path=lambda: None,
            run_command=lambda args, timeout, disable_git_prompt: {"ok": True},
            python_executable=sys.executable,
            tail_lines=lambda value: str(value or ""),
            set_comfyui_requirements_pending=lambda pending: None,
            logger=_NoopLogger(),
        )
        self.assertEqual(result.get("status"), "missing_requirements")

    def test_install_comfyui_requirements_success(self):
        """Successful ComfyUI requirements install clears pending flag."""
        calls = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            req = Path(tmp_dir) / "requirements.txt"
            req.write_text("# test\n", encoding="utf-8")
            result = self.ops.install_comfyui_requirements(
                comfyui_requirements_path=lambda: req,
                run_command=lambda args, timeout, disable_git_prompt: {"ok": True, "stdout": "ok", "stderr": ""},
                python_executable=sys.executable,
                tail_lines=lambda value: str(value or ""),
                set_comfyui_requirements_pending=lambda pending: calls.append(pending),
                logger=_NoopLogger(),
            )
            self.assertEqual(result.get("status"), "installed")
            self.assertIn(False, calls)

    def test_install_requirements_for_modules_rejects_non_list(self):
        """Batch installer should validate incoming modules payload type."""
        result = self.ops.install_requirements_for_modules(
            "modA",
            canonical_custom_module_name=lambda name: name,
            install_module_requirements_fn=lambda _name: {"status": "installed"},
            logger=_NoopLogger(),
        )
        self.assertEqual(result.get("status"), "error")

    def test_install_requirements_for_modules_dedupes_and_summarizes(self):
        """Batch installer should dedupe names and return installed/failed totals."""
        calls = []

        def _install(module_name):
            calls.append(module_name)
            if module_name == "ok_mod":
                return {"status": "installed"}
            return {"status": "error"}

        result = self.ops.install_requirements_for_modules(
            ["ok_mod", "bad_mod", "ok_mod", "   "],
            canonical_custom_module_name=lambda name: name.strip(),
            install_module_requirements_fn=_install,
            logger=_NoopLogger(),
        )
        self.assertEqual(calls, ["ok_mod", "bad_mod"])
        self.assertEqual(result.get("count"), 2)
        self.assertEqual(result.get("installed"), 1)
        self.assertEqual(result.get("failed"), 1)


if __name__ == "__main__":
    unittest.main()
