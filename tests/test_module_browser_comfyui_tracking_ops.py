"""
Module: tests/test_module_browser_comfyui_tracking_ops.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted ComfyUI local-tracking helpers.

Purpose:
    Validates startup novelty tracking and acknowledge behavior for ComfyUI
    after Phase 3 extraction from backend API module.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path


class ModuleBrowserComfyuiTrackingOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.comfyui_tracking_ops` helpers."""

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
        from ComfyUI_ALEXZ_tools.utils.module_browser import comfyui_tracking_ops

        self.ops = comfyui_tracking_ops

    def test_acknowledge_comfyui_novelty_clears_pending_fields(self):
        """Acknowledge helper should clear pending/startup novelty fields."""
        state = {
            "__comfyui__": {
                "pending_prev_commit": "old1",
                "pending_new_commit": "new1",
                "pending_update_at": "2026-02-12T00:00:00Z",
                "startup_prev_commit": "old1",
                "startup_new_commit": "new1",
                "startup_update_at": "2026-02-12T00:00:00Z",
            }
        }
        saved = []
        cleared = []

        result = self.ops.acknowledge_comfyui_novelty(
            load_module_state=lambda: state,
            save_module_state=lambda payload: saved.append(payload),
            clear_comfyui_status_cache=lambda: cleared.append(True),
        )
        self.assertTrue(result.get("changed"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(cleared), 1)
        self.assertNotIn("pending_prev_commit", state["__comfyui__"])

    def test_track_comfyui_local_update_sets_pending_on_commit_change(self):
        """Tracker should mark pending update when local ComfyUI commit changed."""
        state = {"__comfyui__": {"installed_commit": "old11111"}}
        saved = []
        cleared = []

        def _run_git(args, _timeout):
            if args[-2:] == ["rev-parse", "--is-inside-work-tree"]:
                return "true"
            if args[-2:] == ["rev-parse", "HEAD"]:
                return "new22222"
            if args[-3:] == ["log", "-1", "--format=%cI"]:
                return "2026-02-12T00:00:00+00:00"
            return ""

        self.ops.track_comfyui_local_update(
            load_module_state=lambda: state,
            save_module_state=lambda payload: saved.append(payload),
            comfyui_root=lambda: Path("/tmp/ComfyUI"),
            run_git=_run_git,
            now_iso=lambda: "2026-02-12T01:00:00+00:00",
            short_commit=lambda value: str(value or "")[:8],
            clear_comfyui_status_cache=lambda: cleared.append(True),
        )
        entry = state.get("__comfyui__", {})
        self.assertEqual(entry.get("pending_prev_commit"), "old11111")
        self.assertEqual(entry.get("pending_new_commit"), "new22222")
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(cleared), 1)


if __name__ == "__main__":
    unittest.main()
