"""
Module: tests/test_module_browser_update_handler.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Unit tests for extracted update-job handler orchestration.

Purpose:
    Verifies runtime update-job boundary behavior moved out of
    `utils/module_node_browser_api.py` during Phase 3 stabilization.
"""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest


class ModuleBrowserUpdateHandlerTests(unittest.TestCase):
    """Verify update handler thread lifecycle and validation behavior."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def test_start_module_update_job_rejects_invalid_scope(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.handlers_update import start_module_update_job
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.state import ModuleBrowserApiState

        state = ModuleBrowserApiState()
        result = start_module_update_job(
            state=state,
            scope_norm="invalid",
            module_name="",
            normalized_log_mode="summary",
            now_iso=lambda: "2026-07-20T00:00:00+00:00",
            comfyui_root_exists=True,
            single_module_exists=True,
            refresh_running=False,
            update_status_snapshot=lambda: dict(state.update_status),
            update_console_log=lambda _text, _level="summary": None,
            run_module_update_job=lambda **kwargs: None,
            resolve_update_targets=lambda _scope, _module: [],
            pull_comfyui=lambda **kwargs: {},
            pull_custom_module=lambda **kwargs: {},
            refresh_module_runtime_state=lambda: {},
            set_update_status=lambda **kwargs: state.update_status.update(kwargs),
        )
        self.assertEqual(result, {"status": "error", "error": "scope must be 'single', 'all' or 'comfyui'"})

    def test_start_module_update_job_marks_error_and_stops_running(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.handlers_update import start_module_update_job
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.state import ModuleBrowserApiState

        state = ModuleBrowserApiState()
        calls: list[dict[str, object]] = []
        started = threading.Event()
        release = threading.Event()

        def set_update_status(**kwargs):
            with state.update_lock:
                state.update_status.update(kwargs)
                calls.append(dict(kwargs))

        def run_module_update_job(**_kwargs):
            started.set()
            release.wait(timeout=2.0)
            raise RuntimeError("boom")

        result = start_module_update_job(
            state=state,
            scope_norm="all",
            module_name="",
            normalized_log_mode="summary",
            now_iso=lambda: "2026-07-20T00:00:00+00:00",
            comfyui_root_exists=True,
            single_module_exists=True,
            refresh_running=False,
            update_status_snapshot=lambda: dict(state.update_status),
            update_console_log=lambda _text, _level="summary": None,
            run_module_update_job=run_module_update_job,
            resolve_update_targets=lambda _scope, _module: ["ModA"],
            pull_comfyui=lambda **kwargs: {},
            pull_custom_module=lambda **kwargs: {},
            refresh_module_runtime_state=lambda: {},
            set_update_status=set_update_status,
        )

        self.assertEqual(result.get("status"), "started")
        self.assertTrue(started.wait(timeout=2.0))
        thread = state.update_thread
        self.assertIsNotNone(thread)
        assert isinstance(thread, threading.Thread)
        release.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(bool(state.update_status.get("running")))
        self.assertEqual(state.update_status.get("phase"), "error")
        self.assertEqual(state.update_status.get("message"), "error")
        self.assertEqual(state.update_status.get("error"), "boom")
        self.assertTrue(any(call.get("phase") == "error" for call in calls))


if __name__ == "__main__":
    unittest.main()
