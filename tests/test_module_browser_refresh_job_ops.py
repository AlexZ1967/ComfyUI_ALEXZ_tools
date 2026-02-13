"""
Module: tests/test_module_browser_refresh_job_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted refresh-job execution helper.

Purpose:
    Validates status/log behavior of synchronous refresh job helper.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserRefreshJobOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.refresh_job_ops` helper."""

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
        from ComfyUI_ALEXZ_tools.utils.module_browser import refresh_job_ops

        self.ops = refresh_job_ops

    def test_run_refresh_job_updates_status_and_logs_summary(self):
        """Refresh job helper should report done status and final summary line."""
        logs = []
        statuses = []

        result = self.ops.run_refresh_job(
            sync_upstreams=True,
            get_update_console_log_mode=lambda: "summary",
            refresh_console_log=lambda text, level: logs.append((level, text)),
            refresh_module_runtime_state=lambda _sync: {
                "refreshed_at": "2026-02-13T00:00:00+00:00",
                "modules_need_update": 3,
                "modules_unknown_update": 1,
            },
            set_refresh_status=lambda **kwargs: statuses.append(kwargs),
        )

        self.assertEqual(result["modules_need_update"], 3)
        self.assertEqual(result["modules_unknown_update"], 1)
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0].get("phase"), "done")
        self.assertTrue(any("job started" in text for _level, text in logs))
        self.assertTrue(any("job finished" in text for _level, text in logs))


if __name__ == "__main__":
    unittest.main()
