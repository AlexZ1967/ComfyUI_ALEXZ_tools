"""Regression tests for Module Node Picker backend state tracking.

These tests validate module update detection, refresh status reporting, and
support utilities used by the Module Nodes widget API.
"""

import importlib
import os
import sys
import time
import types
import unittest


def _install_folder_paths_stub():
    """Internal helper: `_install_folder_paths_stub`."""
    if "folder_paths" in sys.modules:
        stub = sys.modules["folder_paths"]
        if not hasattr(stub, "get_folder_paths"):
            stub.get_folder_paths = lambda kind: [os.path.join(os.getcwd(), "custom_nodes")]
        return
    stub = types.SimpleNamespace(
        get_folder_paths=lambda kind: [os.path.join(os.getcwd(), "custom_nodes")]
    )
    sys.modules["folder_paths"] = stub


class ModuleBrowserTrackerTests(unittest.TestCase):
    """Verify module browser backend behavior for status and update flows."""
    @classmethod
    def setUpClass(cls):
        """Execute `setUpClass` routine."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()

    def setUp(self):
        """Load module-browser backend and capture patchable symbols."""
        self.api = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_node_browser_api")
        self._orig_state_cache = self.api._MODULE_STATE_CACHE
        self._orig_comfy_cache = self.api._COMFYUI_STATUS_CACHE
        self._orig_save_state = self.api._save_module_state
        self._orig_snapshots = self.api._build_node_snapshots
        self._orig_discover = self.api._discover_custom_modules
        self._orig_now_iso = self.api._now_iso
        self._orig_comfy_root = self.api._comfyui_root
        self._orig_run_git = self.api._run_git
        self._orig_module_git_state = self.api._module_git_state
        self._orig_sync_module_upstream = self.api._sync_module_upstream
        self._orig_announce_updates = self.api._announce_tracked_module_updates
        self._orig_comfy_status = self.api._comfyui_git_status
        self._orig_module_needs_update_now = self.api._module_needs_update_now
        self._orig_install_module_requirements = self.api._install_module_requirements
        self._orig_pull_comfyui = self.api._pull_comfyui
        self._orig_install_comfyui_requirements = self.api._install_comfyui_requirements
        self._orig_refresh_runtime_state = self.api._refresh_module_runtime_state
        self.api._MODULE_STATE_CACHE = {}
        self.api._COMFYUI_STATUS_CACHE = None
        self.api._save_module_state = lambda state: None
        self.api._MODULE_INFO_CACHE.clear()

    def tearDown(self):
        """Execute `tearDown` routine."""
        self.api._MODULE_STATE_CACHE = self._orig_state_cache
        self.api._COMFYUI_STATUS_CACHE = self._orig_comfy_cache
        self.api._save_module_state = self._orig_save_state
        self.api._build_node_snapshots = self._orig_snapshots
        self.api._discover_custom_modules = self._orig_discover
        self.api._now_iso = self._orig_now_iso
        self.api._comfyui_root = self._orig_comfy_root
        self.api._run_git = self._orig_run_git
        self.api._module_git_state = self._orig_module_git_state
        self.api._sync_module_upstream = self._orig_sync_module_upstream
        self.api._announce_tracked_module_updates = self._orig_announce_updates
        self.api._comfyui_git_status = self._orig_comfy_status
        self.api._module_needs_update_now = self._orig_module_needs_update_now
        self.api._install_module_requirements = self._orig_install_module_requirements
        self.api._pull_comfyui = self._orig_pull_comfyui
        self.api._install_comfyui_requirements = self._orig_install_comfyui_requirements
        self.api._refresh_module_runtime_state = self._orig_refresh_runtime_state
        self.api._MODULE_INFO_CACHE.clear()

    def test_new_module_marker_applies_without_node_diffs(self):
        """Validate `test_new_module_marker_applies_without_node_diffs` behavior."""
        self.api._MODULE_STATE_CACHE = {
            "__node_tracker__": {
                "startup_changes": {},
                "startup_new_modules": {"custom": ["ComfyUI-Inpaint-CropAndStitch"]},
            }
        }
        result = {
            "updated_between_runs": False,
            "new_nodes_between_runs": [],
            "updated_nodes_between_runs": [],
            "startup_node_update_at": "",
            "new_module_between_runs": False,
        }

        self.api._apply_node_change_info(result, "custom", "ComfyUI-Inpaint-CropAndStitch")

        self.assertTrue(result["new_module_between_runs"])
        self.assertTrue(result["updated_between_runs"])

    def test_startup_new_modules_detected_from_module_set_diff(self):
        """Validate `test_startup_new_modules_detected_from_module_set_diff` behavior."""
        self.api._now_iso = lambda: "2026-02-08T00:00:00+00:00"
        self.api._MODULE_STATE_CACHE = {
            "__node_tracker__": {
                "snapshots": {"custom": {"ExistingModule": {}}},
                "module_sets": {"custom": ["ExistingModule"]},
            }
        }
        self.api._build_node_snapshots = lambda: {"custom": {"ExistingModule": {}}}
        self.api._discover_custom_modules = lambda: ["ComfyUI-Inpaint-CropAndStitch", "ExistingModule"]

        self.api._announce_tracked_module_updates()

        tracker = self.api._MODULE_STATE_CACHE["__node_tracker__"]
        startup_new = tracker.get("startup_new_modules", {}).get("custom", [])
        self.assertIn("ComfyUI-Inpaint-CropAndStitch", startup_new)

    def test_comfyui_update_status_can_update(self):
        """Validate `test_comfyui_update_status_can_update` behavior."""
        self.api._comfyui_root = lambda: os.path.join(os.getcwd(), "fake_comfy")

        def fake_run_git(args, timeout=2.0):
            """Execute `fake_run_git` routine."""
            cmd = " ".join(args)
            table = {
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " rev-parse --is-inside-work-tree": "true",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " rev-parse --abbrev-ref HEAD": "master",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " rev-parse HEAD": "aaaaaaaa11111111",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " log -1 --format=%cI": "2026-02-08T01:00:00+00:00",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy")
                + " rev-parse --abbrev-ref --symbolic-full-name @{u}": "origin/master",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " fetch --quiet": "",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " rev-parse @{u}": "bbbbbbbb22222222",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy")
                + " log -1 --format=%cI @{u}": "2026-02-08T02:00:00+00:00",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " rev-list --left-right --count HEAD...@{u}": "0 3",
            }
            return table.get(cmd)

        self.api._run_git = fake_run_git
        status = self.api._comfyui_git_status(force_refresh=True)
        self.assertEqual(status.get("update_status"), "can_update")
        self.assertEqual(status.get("behind"), 3)

    def test_unseen_module_update_detected_between_runs(self):
        """Validate `test_unseen_module_update_detected_between_runs` behavior."""
        self.api._now_iso = lambda: "2026-02-08T00:00:00+00:00"
        self.api._build_node_snapshots = lambda: {"custom": {"comfyui-AGSoft": {}}}
        self.api._discover_custom_modules = lambda: ["comfyui-AGSoft"]

        states = [
            {"installed_commit": "old111", "installed_updated_at": "2026-02-01T00:00:00+00:00"},
            {"installed_commit": "new222", "installed_updated_at": "2026-02-08T00:00:00+00:00"},
        ]

        def fake_module_git_state(_module_name):
            """Execute `fake_module_git_state` routine."""
            return dict(states.pop(0))

        self.api._module_git_state = fake_module_git_state

        # First startup: baseline is recorded, no update marker yet.
        self.api._announce_tracked_module_updates()
        entry = self.api._MODULE_STATE_CACHE.get("comfyui-AGSoft", {})
        self.assertEqual(entry.get("installed_commit"), "old111")
        self.assertFalse(entry.get("startup_prev_commit"))
        self.assertFalse(entry.get("startup_new_commit"))

        # Second startup: changed commit is detected.
        self.api._announce_tracked_module_updates()
        entry = self.api._MODULE_STATE_CACHE.get("comfyui-AGSoft", {})
        self.assertEqual(entry.get("startup_prev_commit"), "old111")
        self.assertEqual(entry.get("startup_new_commit"), "new222")

    def test_refresh_syncs_custom_module_upstreams(self):
        """Validate `test_refresh_syncs_custom_module_upstreams` behavior."""
        called = []
        self.api._discover_custom_modules = lambda: ["modA", "modB"]
        self.api._sync_module_upstream = lambda module_name, timeout=15.0: called.append((module_name, timeout)) or True
        self.api._announce_tracked_module_updates = lambda: None
        self.api._comfyui_git_status = lambda force_refresh=False: {"update_status": "unknown"}

        self.api._refresh_module_runtime_state(sync_upstreams=True)

        self.assertEqual([x[0] for x in called], ["modA", "modB"])

    def test_initial_refresh_does_not_sync_upstreams_by_default(self):
        """Validate `test_initial_refresh_does_not_sync_upstreams_by_default` behavior."""
        called = []
        self.api._discover_custom_modules = lambda: ["modA"]
        self.api._sync_module_upstream = lambda module_name, timeout=15.0: called.append((module_name, timeout)) or True
        self.api._announce_tracked_module_updates = lambda: None
        self.api._comfyui_git_status = lambda force_refresh=False: {"update_status": "unknown"}

        self.api._refresh_module_runtime_state()

        self.assertEqual(called, [])

    def test_refresh_reports_progress_callback(self):
        """Validate `test_refresh_reports_progress_callback` behavior."""
        events = []
        self.api._discover_custom_modules = lambda: ["modA"]
        self.api._sync_module_upstream = lambda module_name, timeout=15.0: True
        self.api._announce_tracked_module_updates = lambda: None
        self.api._comfyui_git_status = lambda force_refresh=False: {"update_status": "unknown"}

        self.api._refresh_module_runtime_state(sync_upstreams=True, progress_cb=lambda **kw: events.append(dict(kw)))

        phases = [e.get("phase") for e in events]
        self.assertIn("sync", phases)
        self.assertIn("snapshots", phases)
        self.assertIn("done", phases)

    def test_refresh_reports_modules_need_update_count(self):
        """Validate `test_refresh_reports_modules_need_update_count` behavior."""
        events = []
        self.api._discover_custom_modules = lambda: []
        self.api._announce_tracked_module_updates = lambda: {"modules_need_update": 4}
        self.api._comfyui_git_status = lambda force_refresh=False: {"update_status": "unknown"}

        result = self.api._refresh_module_runtime_state(sync_upstreams=False, progress_cb=lambda **kw: events.append(dict(kw)))

        self.assertEqual(result.get("modules_need_update"), 4)
        done_events = [e for e in events if e.get("phase") == "done"]
        self.assertTrue(done_events)
        self.assertEqual(done_events[-1].get("modules_need_update"), 4)

    def test_resolve_update_targets_all_filters_modules(self):
        """Validate `test_resolve_update_targets_all_filters_modules` behavior."""
        self.api._discover_custom_modules = lambda: ["modA", "modB", "modC"]
        self.api._sync_module_upstream = lambda module_name, timeout=15.0: True
        self.api._module_needs_update_now = lambda module_name: module_name in {"modA", "modC"}

        targets = self.api._resolve_update_targets("all", "")

        self.assertEqual(targets, ["modA", "modC"])

    def test_install_requirements_for_modules_aggregates_results(self):
        """Validate `test_install_requirements_for_modules_aggregates_results` behavior."""
        def fake_install(module_name, timeout=1200.0):
            """Execute `fake_install` routine."""
            if module_name == "modA":
                return {"module": module_name, "status": "installed"}
            return {"module": module_name, "status": "error"}

        self.api._install_module_requirements = fake_install

        result = self.api._install_requirements_for_modules(["modA", "modA", "modB", ""])

        self.assertEqual(result.get("status"), "ok")
        self.assertEqual(result.get("count"), 2)
        self.assertEqual(result.get("installed"), 1)
        self.assertEqual(result.get("failed"), 1)

    def test_module_update_job_supports_comfyui_scope(self):
        """Validate `test_module_update_job_supports_comfyui_scope` behavior."""
        self.api._comfyui_root = lambda: os.getcwd()
        self.api._pull_comfyui = lambda timeout=240.0: {
            "module": "ComfyUI",
            "status": "updated",
            "requirements_changed": True,
        }
        self.api._refresh_module_runtime_state = lambda sync_upstreams=False, progress_cb=None: {"status": "ok"}

        started = self.api._start_module_update_job("comfyui", "")
        self.assertEqual(started.get("status"), "started")

        for _ in range(100):
            snap = self.api._update_status_snapshot()
            if not snap.get("running"):
                break
            time.sleep(0.01)

        done = self.api._update_status_snapshot()
        self.assertEqual(done.get("phase"), "done")
        self.assertEqual(done.get("updated"), 1)
        self.assertTrue(done.get("requirements_changed"))

    def test_install_comfyui_requirements_endpoint_helper(self):
        """Validate `test_install_comfyui_requirements_endpoint_helper` behavior."""
        self.api._install_comfyui_requirements = lambda timeout=1800.0: {"status": "installed"}
        result = self.api._install_comfyui_requirements()
        self.assertEqual(result.get("status"), "installed")


if __name__ == "__main__":
    unittest.main()
