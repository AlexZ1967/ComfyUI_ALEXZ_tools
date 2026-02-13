"""
Module: tests/test_module_browser_comfyui_state_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted ComfyUI state/status helper functions.

Purpose:
    Validates status template, cache resolution, pending merge, and persistence.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserComfyuiStateOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.comfyui_state_ops` helpers."""

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
        from ComfyUI_ALEXZ_tools.utils.module_browser import comfyui_state_ops

        self.ops = comfyui_state_ops

    def test_status_template_contains_required_keys(self):
        """Template helper should include stable keys used by frontend."""
        payload = self.ops.comfyui_status_template("releases")
        self.assertEqual(payload["check_mode"], "releases")
        self.assertIn("update_status", payload)
        self.assertIn("requirements_update_pending", payload)

    def test_resolve_cached_status_prefers_mode_specific(self):
        """Resolver should prefer status_by_mode over fallback status."""
        state = {
            "__comfyui__": {
                "status_by_mode": {"releases": {"check_mode": "releases", "marker": "mode"}},
                "status": {"check_mode": "commits", "marker": "fallback"},
            }
        }
        entry, status = self.ops.resolve_cached_status(state, "releases")
        self.assertIsNotNone(entry)
        self.assertEqual(status.get("marker"), "mode")

    def test_apply_cached_pending_fields_merges_pending_markers(self):
        """Pending markers from cached entry should be added to status payload."""
        result = {"check_mode": "releases"}
        merged = self.ops.apply_cached_pending_fields(
            result,
            {
                "pending_prev_commit": "old11111",
                "pending_new_commit": "new22222",
                "pending_update_at": "2026-02-13T00:00:00+00:00",
                "pending_requirements_update": True,
            },
            short_commit=lambda value: str(value or "")[:8],
        )
        self.assertTrue(merged.get("updated_between_runs"))
        self.assertEqual(merged.get("startup_prev_commit_short"), "old11111")
        self.assertTrue(merged.get("requirements_update_pending"))

    def test_persist_comfyui_status_updates_state_entry(self):
        """Persist helper should write status + status_by_mode + updated_at fields."""
        state = {}
        updated = self.ops.persist_comfyui_status(
            state,
            mode_norm="releases",
            result={"check_mode": "releases", "installed_commit": "abc123"},
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
        )
        entry = updated.get("__comfyui__", {})
        self.assertIn("status", entry)
        self.assertIn("status_by_mode", entry)
        self.assertEqual(entry.get("installed_commit"), "abc123")


if __name__ == "__main__":
    unittest.main()
