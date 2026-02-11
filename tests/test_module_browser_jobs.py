"""
Module: tests/test_module_browser_jobs.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for Module Node Picker backend job helpers.

Purpose:
    Verifies refresh/update status helpers and update target resolution contract
    after Phase 3 backend modular split.
"""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest


class ModuleBrowserJobsTests(unittest.TestCase):
    """Verify `utils.module_browser.jobs` helper behavior."""

    @classmethod
    def setUpClass(cls):
        """Install package path stubs used by direct module imports."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import jobs

        self.jobs = jobs

    def test_refresh_status_roundtrip(self):
        """Ensure refresh status update+snapshot keep expected fields."""
        lock = threading.Lock()
        state = {"running": False}
        self.jobs.set_refresh_status(lock=lock, status=state, now_iso=lambda: "2026-02-12T00:00:00+00:00", running=True)
        snap = self.jobs.refresh_status_snapshot(lock=lock, status=state)
        self.assertTrue(bool(snap.get("running")))
        self.assertEqual(snap.get("updated_at"), "2026-02-12T00:00:00+00:00")

    def test_update_status_format_line(self):
        """Ensure update status line contains stable counters and scope."""
        line = self.jobs.format_update_status_line(
            {
                "scope": "all",
                "phase": "update",
                "current": 2,
                "total": 5,
                "remaining": 3,
                "module": "demo",
                "updated": 1,
                "up_to_date": 1,
                "failed": 0,
                "message": "pull",
            }
        )
        self.assertIn("scope=all", line)
        self.assertIn("current=2/5", line)
        self.assertIn("module=demo", line)

    def test_emit_refresh_progress_returns_new_last_line(self):
        """Ensure refresh progress emits new line and updates status dict."""
        lock = threading.Lock()
        state: dict[str, object] = {}
        emitted: list[tuple[str, str]] = []
        line = self.jobs.emit_refresh_progress(
            lock=lock,
            status=state,
            now_iso=lambda: "2026-02-12T00:00:00+00:00",
            phase="snapshots",
            current=1,
            total=4,
            remaining=3,
            modules_need_update=2,
            message="scan",
            last_line="",
            logger_debug=lambda text: None,
            console_log=lambda text, level="summary": emitted.append((text, level)),
        )
        self.assertIn("phase=snapshots", line)
        self.assertEqual(state.get("message"), "scan")
        self.assertTrue(len(emitted) >= 1)

    def test_resolve_update_targets_all_scope(self):
        """Ensure all-scope resolver keeps order and deduplicates canonical names."""
        targets = self.jobs.resolve_update_targets(
            scope="all",
            module_name="",
            canonical_module_name=lambda name: {"m1": "M1", "m2": "M2"}.get(name, name),
            discover_modules=lambda: ["m1", "m2"],
            sync_module_upstream=lambda name: True,
            module_needs_update=lambda name: name == "m2",
            update_console_log=lambda text, level="summary": None,
            workers=2,
            warn=lambda text: None,
        )
        self.assertEqual(targets, ["M2"])


if __name__ == "__main__":
    unittest.main()
