from __future__ import annotations

import os
import sys
import threading
import types
import unittest


class ModuleBrowserRefreshHandlerTests(unittest.TestCase):
    """Verify refresh handler thread lifecycle updates shared status correctly."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def test_start_refresh_job_marks_done_and_stops_running(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.handlers_refresh import start_refresh_job
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.state import ModuleBrowserApiState

        state = ModuleBrowserApiState()
        calls: list[dict[str, object]] = []
        started = threading.Event()
        release = threading.Event()

        def set_refresh_status(**kwargs):
            with state.refresh_lock:
                state.refresh_status.update(kwargs)
                calls.append(dict(kwargs))

        def refresh_module_runtime_state(_sync_upstreams: bool):
            started.set()
            release.wait(timeout=2.0)
            state.refresh_status["phase"] = "done"
            return {
                "status": "ok",
                "refreshed_at": "2026-03-10T00:00:00+00:00",
                "modules_need_update": 1,
                "modules_unknown_update": 0,
                "unknown_update_modules": [],
            }

        result = start_refresh_job(
            state=state,
            sync_upstreams=True,
            now_iso=lambda: "2026-03-10T00:00:00+00:00",
            get_update_console_log_mode=lambda: "summary",
            refresh_console_log=lambda _text, _level="summary": None,
            refresh_module_runtime_state=refresh_module_runtime_state,
            set_refresh_status=set_refresh_status,
            refresh_status_snapshot=lambda: dict(state.refresh_status),
        )

        self.assertEqual(result.get("status"), "started")
        self.assertTrue(started.wait(timeout=2.0))
        thread = state.refresh_thread
        self.assertIsNotNone(thread)
        assert isinstance(thread, threading.Thread)
        release.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(bool(state.refresh_status.get("running")))
        self.assertEqual(state.refresh_status.get("phase"), "done")
        self.assertEqual(state.refresh_status.get("message"), "done")
        self.assertEqual(state.refresh_status.get("modules_need_update"), 1)
        self.assertTrue(any(call.get("running") is False for call in calls))


if __name__ == "__main__":
    unittest.main()
