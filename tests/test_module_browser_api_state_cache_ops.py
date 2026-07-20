"""
Module: tests/test_module_browser_api_state_cache_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Unit tests for extracted module-browser API state/cache helpers.

Purpose:
    Verifies module-state persistence wrappers and runtime-warmup state syncing
    moved out of `utils/module_node_browser_api.py` during Phase 3.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserApiStateCacheOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser_api.state_cache_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.state import ModuleBrowserApiState
        from ComfyUI_ALEXZ_tools.utils.module_browser_api import state_cache_ops

        self.ModuleBrowserApiState = ModuleBrowserApiState
        self.ops = state_cache_ops

    def test_sync_runtime_warmup_roundtrip(self):
        """Legacy warmup flags should roundtrip through extracted sync helpers."""
        state = self.ModuleBrowserApiState()
        self.ops.sync_runtime_warmup_from_legacy(
            state,
            lazy_refresh_done=True,
            runtime_warmup_thread=None,
        )
        self.assertTrue(state.lazy_refresh_done)
        legacy = self.ops.sync_runtime_warmup_to_legacy(state)
        self.assertEqual(legacy, (True, None))

    def test_load_module_state_cache_keeps_existing_cache(self):
        """State loader should return preloaded cache without touching disk helper."""
        cached = {"__meta__": {"schema_version": 1}}
        result = self.ops.load_module_state_cache(
            cached,
            state_path=os.path.abspath("/tmp/unused.json"),
            ensure_schema=lambda payload: payload,
            load_state_file=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disk load not expected")),
        )
        self.assertIs(result, cached)

    def test_save_module_state_cache_returns_normalized_payload(self):
        """State saver should return normalized payload from delegated store helper."""
        normalized = self.ops.save_module_state_cache(
            {"modA": {"updated": True}},
            state_path=os.path.abspath("/tmp/module_state.json"),
            ensure_schema=lambda payload: payload,
            save_state_file=lambda *args, **kwargs: {"__meta__": {"schema_version": 1}},
            logger=None,
        )
        self.assertEqual(normalized, {"__meta__": {"schema_version": 1}})

    def test_runtime_warmup_status_reports_idle_state(self):
        """Warmup status helper should expose the stable frontend polling shape."""
        state = self.ModuleBrowserApiState()
        calls = {"count": 0}
        status = self.ops.runtime_warmup_status(
            state,
            sync_from_legacy=lambda: calls.__setitem__("count", calls["count"] + 1),
        )
        self.assertEqual(status, {"running": False, "done": False})
        self.assertEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
