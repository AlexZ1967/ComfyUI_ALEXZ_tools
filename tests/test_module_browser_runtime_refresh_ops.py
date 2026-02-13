"""
Module: tests/test_module_browser_runtime_refresh_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted runtime refresh orchestration helper.

Purpose:
    Validates refresh phase flow and summary payload in isolated helper tests.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserRuntimeRefreshOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.runtime_refresh_ops` helpers."""

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
        from ComfyUI_ALEXZ_tools.utils.module_browser import runtime_refresh_ops

        self.ops = runtime_refresh_ops

    def test_refresh_runtime_fast_mode_returns_summary(self):
        """Fast mode should skip upstream sync and return announced counts."""
        progress_calls = []
        logs = []
        checked = []

        result = self.ops.refresh_module_runtime_state(
            sync_upstreams=False,
            progress_cb=lambda **kwargs: progress_calls.append(kwargs),
            module_info_cache_clear=lambda: checked.append("cache_clear"),
            reset_custom_alias_cache=lambda: checked.append("alias_reset"),
            clear_comfyui_status_cache=lambda: checked.append("comfy_cache_clear"),
            refresh_console_log=lambda text, level: logs.append((level, text)),
            get_update_console_log_mode=lambda: "summary",
            discover_custom_modules=lambda: ["m1", "m2"],
            sync_module_upstream=lambda _module: True,
            announce_tracked_module_updates=lambda: {
                "modules_need_update": 2,
                "modules_unknown_update": 1,
                "modules_checked": 3,
                "commit_change_modules": ["m1"],
                "local_change_modules": ["m2"],
                "node_changed_modules": ["m3"],
                "update_available_modules": ["m1", "m2"],
                "unknown_update_modules": ["m4"],
                "new_modules_between_runs": {"custom": ["m5"]},
            },
            comfyui_git_status=lambda: {
                "update_status": "up_to_date",
                "behind": 0,
                "ahead": 0,
                "installed_commit": "abc123456789",
                "remote_commit": "abc123456789",
            },
            short_commit=lambda value: str(value or "")[:8],
            set_custom_update_checked=lambda value: checked.append(f"status_checked={value}"),
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
            perf_counter=lambda: 0.0,
        )

        self.assertEqual(result["modules_need_update"], 2)
        self.assertEqual(result["modules_unknown_update"], 1)
        self.assertIn("cache_clear", checked)
        self.assertIn("alias_reset", checked)
        self.assertIn("comfy_cache_clear", checked)
        self.assertIn("status_checked=True", checked)
        phases = [str(item.get("phase")) for item in progress_calls]
        self.assertIn("sync", phases)
        self.assertIn("snapshots", phases)
        self.assertIn("done", phases)
        self.assertTrue(any("phase 1/3: upstream sync skipped" in line for _lvl, line in logs))

    def test_refresh_runtime_sync_mode_iterates_modules(self):
        """Sync mode should iterate all discovered modules in phase 1."""
        synced = []

        self.ops.refresh_module_runtime_state(
            sync_upstreams=True,
            progress_cb=lambda **_kwargs: None,
            module_info_cache_clear=lambda: None,
            reset_custom_alias_cache=lambda: None,
            clear_comfyui_status_cache=lambda: None,
            refresh_console_log=lambda _text, _level: None,
            get_update_console_log_mode=lambda: "summary",
            discover_custom_modules=lambda: ["m1", "m2", "m3"],
            sync_module_upstream=lambda module: synced.append(module) or True,
            announce_tracked_module_updates=lambda: {},
            comfyui_git_status=lambda: {"update_status": "unknown"},
            short_commit=lambda value: str(value or "")[:8],
            set_custom_update_checked=lambda _value: None,
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
            perf_counter=lambda: 0.0,
        )
        self.assertEqual(synced, ["m1", "m2", "m3"])


if __name__ == "__main__":
    unittest.main()
