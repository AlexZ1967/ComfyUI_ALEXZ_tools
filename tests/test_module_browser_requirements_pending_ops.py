"""
Module: tests/test_module_browser_requirements_pending_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted requirements pending-state helper functions.

Purpose:
    Validates state mutation semantics for requirements follow-up markers moved
    from module_node_browser_api into requirements_pending_ops.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserRequirementsPendingOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.requirements_pending_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import requirements-pending helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import requirements_pending_ops

        self.ops = requirements_pending_ops

    def test_set_comfyui_requirements_pending_sets_and_clears_flags(self):
        """ComfyUI marker helper should set and clear pending fields deterministically."""
        state = {}
        saved = {"calls": 0}
        changed = {"calls": 0}

        def _load():
            return state

        def _save(new_state):
            saved["calls"] += 1
            snapshot = dict(new_state)
            state.clear()
            state.update(snapshot)

        updated = self.ops.set_comfyui_requirements_pending(
            pending=True,
            before_commit="aaaa1111",
            after_commit="bbbb2222",
            load_state_fn=_load,
            save_state_fn=_save,
            now_iso_fn=lambda: "2026-02-13T00:00:00+00:00",
            on_state_changed=lambda: changed.__setitem__("calls", changed["calls"] + 1),
        )
        self.assertTrue(updated)
        entry = state.get("__comfyui__", {})
        self.assertTrue(bool(entry.get("pending_requirements_update")))
        self.assertEqual(entry.get("pending_requirements_before_commit"), "aaaa1111")
        self.assertEqual(entry.get("pending_requirements_after_commit"), "bbbb2222")
        self.assertEqual(saved["calls"], 1)
        self.assertEqual(changed["calls"], 1)

        cleared = self.ops.set_comfyui_requirements_pending(
            pending=False,
            before_commit="",
            after_commit="",
            load_state_fn=_load,
            save_state_fn=_save,
            now_iso_fn=lambda: "2026-02-13T00:00:01+00:00",
            on_state_changed=lambda: changed.__setitem__("calls", changed["calls"] + 1),
        )
        self.assertTrue(cleared)
        entry_after = state.get("__comfyui__", {})
        self.assertNotIn("pending_requirements_update", entry_after)

    def test_set_module_requirements_pending_skips_unknown_module(self):
        """Module marker helper should ignore unknown/invalid canonical module names."""
        state = {}
        saved = {"calls": 0}

        result = self.ops.set_module_requirements_pending(
            module_name="m",
            pending=True,
            before_commit="a",
            after_commit="b",
            canonical_custom_module_name_fn=lambda _name: "unknown",
            load_state_fn=lambda: state,
            save_state_fn=lambda _new: saved.__setitem__("calls", saved["calls"] + 1),
            now_iso_fn=lambda: "2026-02-13T00:00:00+00:00",
        )
        self.assertFalse(result)
        self.assertEqual(saved["calls"], 0)

    def test_set_module_requirements_pending_sets_flags(self):
        """Module marker helper should persist pending flags for canonical module."""
        state = {}
        saved = {"calls": 0}
        changed = {"calls": 0}

        def _load():
            return state

        def _save(new_state):
            saved["calls"] += 1
            snapshot = dict(new_state)
            state.clear()
            state.update(snapshot)

        updated = self.ops.set_module_requirements_pending(
            module_name="ComfyUI_ALEXZ_tools",
            pending=True,
            before_commit="1111",
            after_commit="2222",
            canonical_custom_module_name_fn=lambda name: str(name),
            load_state_fn=_load,
            save_state_fn=_save,
            now_iso_fn=lambda: "2026-02-13T00:00:00+00:00",
            on_state_changed=lambda: changed.__setitem__("calls", changed["calls"] + 1),
        )
        self.assertTrue(updated)
        self.assertEqual(saved["calls"], 1)
        self.assertEqual(changed["calls"], 1)
        entry = state.get("ComfyUI_ALEXZ_tools", {})
        self.assertTrue(bool(entry.get("pending_requirements_update")))
        self.assertEqual(entry.get("pending_requirements_before_commit"), "1111")
        self.assertEqual(entry.get("pending_requirements_after_commit"), "2222")


if __name__ == "__main__":
    unittest.main()

