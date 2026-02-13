"""
Module: tests/test_module_browser_update_job_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted module-update job execution helper.

Purpose:
    Validates status/log sequencing for comfyui and custom-module update runs.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserUpdateJobOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.update_job_ops` helper."""

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
        from ComfyUI_ALEXZ_tools.utils.module_browser import update_job_ops

        self.ops = update_job_ops

    def test_run_module_update_job_comfyui_branch(self):
        """ComfyUI branch should mark done and invoke runtime refresh once."""
        logs = []
        statuses = []
        refresh_calls = []

        self.ops.run_module_update_job(
            scope_norm="comfyui",
            module_name="",
            normalized_log_mode="summary",
            update_console_log=lambda text, level: logs.append((level, text)),
            set_update_status=lambda **kwargs: statuses.append(kwargs),
            pull_comfyui=lambda: {"status": "updated", "requirements_changed": True},
            pull_custom_module=lambda _module: {"status": "up_to_date"},
            resolve_update_targets=lambda _scope, _module_name: [],
            refresh_module_runtime_state=lambda: refresh_calls.append(True),
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
            perf_counter=lambda: 0.0,
        )
        self.assertEqual(len(refresh_calls), 1)
        self.assertTrue(any(str(item.get("phase")) == "done" for item in statuses))
        self.assertTrue(any("scope=comfyui" in text for _level, text in logs))

    def test_run_module_update_job_custom_targets(self):
        """Custom-module branch should process all targets and finish with done status."""
        statuses = []
        pulled = []

        def _pull_custom(module):
            pulled.append(module)
            return {"status": "updated" if module == "m1" else "up_to_date"}

        self.ops.run_module_update_job(
            scope_norm="all",
            module_name="",
            normalized_log_mode="summary",
            update_console_log=lambda _text, _level: None,
            set_update_status=lambda **kwargs: statuses.append(kwargs),
            pull_comfyui=lambda: {"status": "up_to_date"},
            pull_custom_module=_pull_custom,
            resolve_update_targets=lambda _scope, _module_name: ["m1", "m2"],
            refresh_module_runtime_state=lambda: None,
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
            perf_counter=lambda: 0.0,
        )
        self.assertEqual(pulled, ["m1", "m2"])
        self.assertTrue(any(item.get("message") == "done" for item in statuses))


if __name__ == "__main__":
    unittest.main()
