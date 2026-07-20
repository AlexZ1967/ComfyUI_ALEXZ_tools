"""
Module: tests/test_module_browser_widget_mode_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted widget-mode helper functions.

Purpose:
    Validates info-only payload, custom-update gate persistence, and log mode
    normalization moved out of module_node_browser_api.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserWidgetModeOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.widget_mode_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import widget-mode helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import widget_mode_ops

        self.ops = widget_mode_ops

    def test_info_only_rejection_payload_has_expected_shape(self):
        """Info-only payload should include standard status/feature/message keys."""
        payload = self.ops.info_only_rejection_payload("module_update")
        self.assertEqual(payload.get("status"), "disabled")
        self.assertEqual(payload.get("feature"), "module_update")
        self.assertIn("info-only mode", str(payload.get("message") or ""))

    def test_requirements_advisory_payload_builds_manual_commands(self):
        """Requirements advisory should expose deduplicated manual commands."""
        payload = self.ops.requirements_advisory_payload(
            feature="module_install_requirements",
            requirements_paths=["/tmp/a/requirements.txt", "/tmp/b/requirements.txt", "/tmp/a/requirements.txt"],
            headline="Manual follow-up required.",
        )
        self.assertEqual(payload.get("status"), "advisory")
        self.assertEqual(payload.get("feature"), "module_install_requirements")
        self.assertEqual(
            payload.get("commands"),
            [
                'python -m pip install -r "/tmp/a/requirements.txt"',
                'python -m pip install -r "/tmp/b/requirements.txt"',
            ],
        )
        self.assertEqual(payload.get("command"), 'python -m pip install -r "/tmp/a/requirements.txt"')
        self.assertEqual(payload.get("headline"), "Manual follow-up required.")

    def test_custom_update_checked_flag_reads_meta(self):
        """Gate flag should be read from cache `__meta__.custom_update_checked`."""
        self.assertFalse(self.ops.custom_update_checked_flag({}))
        self.assertTrue(self.ops.custom_update_checked_flag({"__meta__": {"custom_update_checked": 1}}))

    def test_set_custom_update_checked_persists_only_on_change(self):
        """Persist helper should update state and invoke callback only on value change."""
        state = {"__meta__": {"custom_update_checked": False}}
        saved = {"calls": 0}
        changed = {"calls": 0}

        def _load():
            return state

        def _save(new_state):
            saved["calls"] += 1
            snapshot = dict(new_state)
            state.clear()
            state.update(snapshot)

        def _changed():
            changed["calls"] += 1

        updated = self.ops.set_custom_update_checked(
            checked=True,
            load_state_fn=_load,
            save_state_fn=_save,
            now_iso_fn=lambda: "2026-02-13T00:00:00+00:00",
            on_changed=_changed,
        )
        self.assertTrue(updated)
        self.assertEqual(saved["calls"], 1)
        self.assertEqual(changed["calls"], 1)
        self.assertTrue(bool(state.get("__meta__", {}).get("custom_update_checked")))

        unchanged = self.ops.set_custom_update_checked(
            checked=True,
            load_state_fn=_load,
            save_state_fn=_save,
            now_iso_fn=lambda: "2026-02-13T00:00:01+00:00",
            on_changed=_changed,
        )
        self.assertFalse(unchanged)
        self.assertEqual(saved["calls"], 1)
        self.assertEqual(changed["calls"], 1)

    def test_normalize_log_mode_accepts_verbose_aliases(self):
        """Verbose aliases should normalize to `verbose`; others to `summary`."""
        self.assertEqual(self.ops.normalize_log_mode("verbose"), "verbose")
        self.assertEqual(self.ops.normalize_log_mode("debug"), "verbose")
        self.assertEqual(self.ops.normalize_log_mode("full"), "verbose")
        self.assertEqual(self.ops.normalize_log_mode("detailed"), "verbose")
        self.assertEqual(self.ops.normalize_log_mode("summary"), "summary")
        self.assertEqual(self.ops.normalize_log_mode(""), "summary")


if __name__ == "__main__":
    unittest.main()
