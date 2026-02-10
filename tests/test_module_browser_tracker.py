"""
Module: tests/test_module_browser_tracker.py
Author: AlexZ1967
Last updated: 2026-02-10

Description:
    Regression tests for Module Node Picker backend tracking.

Purpose:
    Tests module update detection, state snapshot persistence, and refresh/update helper logic.
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
        self._orig_manager_index = self.api._manager_index
        self._orig_manager_github_stats = self.api._manager_github_stats
        self._orig_module_repo_url = self.api._module_repo_url
        self._orig_module_local_readme_summary = self.api._module_local_readme_summary
        self._orig_remember_module_state = self.api._remember_module_state
        self._orig_apply_node_change_info = self.api._apply_node_change_info
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
        self.api._manager_index = self._orig_manager_index
        self.api._manager_github_stats = self._orig_manager_github_stats
        self.api._module_repo_url = self._orig_module_repo_url
        self.api._module_local_readme_summary = self._orig_module_local_readme_summary
        self.api._remember_module_state = self._orig_remember_module_state
        self.api._apply_node_change_info = self._orig_apply_node_change_info
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
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " fetch --quiet origin": "",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy") + " rev-parse origin/master": "bbbbbbbb22222222",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy")
                + " log -1 --format=%cI origin/master": "2026-02-08T02:00:00+00:00",
                "git -C " + os.path.join(os.getcwd(), "fake_comfy")
                + " rev-list --left-right --count HEAD...origin/master": "0 3",
            }
            return table.get(cmd)

        self.api._run_git = fake_run_git
        status = self.api._comfyui_git_status(force_refresh=True)
        self.assertEqual(status.get("update_status"), "can_update")
        self.assertEqual(status.get("behind"), 3)

    def test_comfyui_update_status_detached_without_upstream(self):
        """Validate fallback update check via `origin/HEAD` when `@{u}` is missing."""
        fake_root = os.path.join(os.getcwd(), "fake_comfy_detached")
        self.api._comfyui_root = lambda: fake_root

        def fake_run_git(args, timeout=2.0):
            """Execute `fake_run_git` routine."""
            cmd = " ".join(args)
            table = {
                f"git -C {fake_root} rev-parse --is-inside-work-tree": "true",
                f"git -C {fake_root} rev-parse --abbrev-ref HEAD": "HEAD",
                f"git -C {fake_root} rev-parse HEAD": "aaaaaaaa11111111",
                f"git -C {fake_root} log -1 --format=%cI": "2026-02-08T01:00:00+00:00",
                f"git -C {fake_root} rev-parse --abbrev-ref --symbolic-full-name @{{u}}": None,
                f"git -C {fake_root} remote": "origin",
                f"git -C {fake_root} fetch --quiet origin": "",
                f"git -C {fake_root} symbolic-ref --quiet refs/remotes/origin/HEAD": "refs/remotes/origin/main",
                f"git -C {fake_root} rev-parse --verify origin/main": "bbbbbbbb22222222",
                f"git -C {fake_root} rev-parse origin/main": "bbbbbbbb22222222",
                f"git -C {fake_root} log -1 --format=%cI origin/main": "2026-02-08T02:00:00+00:00",
                f"git -C {fake_root} rev-list --left-right --count HEAD...origin/main": "0 2",
            }
            return table.get(cmd)

        self.api._run_git = fake_run_git
        status = self.api._comfyui_git_status(force_refresh=True)
        self.assertEqual(status.get("update_status"), "can_update")
        self.assertEqual(status.get("behind"), 2)
        self.assertEqual(status.get("remote_ref"), "origin/main")

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

    def test_pending_update_marker_persists_until_acknowledge(self):
        """Ensure local update marker stays sticky across restarts until explicit acknowledge."""
        self.api._now_iso = lambda: "2026-02-08T00:00:00+00:00"
        self.api._build_node_snapshots = lambda: {"custom": {"comfyui-AGSoft": {}}}
        self.api._discover_custom_modules = lambda: ["comfyui-AGSoft"]

        states = [
            {"installed_commit": "old111", "installed_updated_at": "2026-02-01T00:00:00+00:00"},
            {"installed_commit": "new222", "installed_updated_at": "2026-02-08T00:00:00+00:00"},
            {"installed_commit": "new222", "installed_updated_at": "2026-02-08T00:00:00+00:00"},
        ]

        def fake_module_git_state(_module_name):
            return dict(states.pop(0))

        self.api._module_git_state = fake_module_git_state

        # Baseline.
        self.api._announce_tracked_module_updates(local_only=True)
        # Change detected.
        self.api._announce_tracked_module_updates(local_only=True)
        # Next startup with same commit should keep marker.
        self.api._announce_tracked_module_updates(local_only=True)

        entry = self.api._MODULE_STATE_CACHE.get("comfyui-AGSoft", {})
        self.assertEqual(entry.get("pending_prev_commit"), "old111")
        self.assertEqual(entry.get("pending_new_commit"), "new222")

        self.api._acknowledge_module_novelty("custom", "comfyui-AGSoft")
        entry = self.api._MODULE_STATE_CACHE.get("comfyui-AGSoft", {})
        self.assertFalse(entry.get("pending_prev_commit"))
        self.assertFalse(entry.get("pending_new_commit"))

    def test_acknowledge_clears_pending_node_markers(self):
        """Ensure per-module acknowledge clears pending node/new-module markers."""
        self.api._MODULE_STATE_CACHE = {
            "__node_tracker__": {
                "pending_changes": {
                    "custom": {
                        "ComfyUI_Test": {
                            "new_nodes": ["NodeA"],
                            "updated_nodes": ["NodeB"],
                            "at": "2026-02-08T00:00:00+00:00",
                        }
                    }
                },
                "pending_new_modules": {
                    "custom": ["ComfyUI_Test"]
                },
            },
            "ComfyUI_Test": {
                "pending_prev_commit": "old111",
                "pending_new_commit": "new222",
            },
        }

        self.api._acknowledge_module_novelty("custom", "ComfyUI_Test")

        entry = self.api._MODULE_STATE_CACHE.get("ComfyUI_Test", {})
        self.assertFalse(entry.get("pending_prev_commit"))
        self.assertFalse(entry.get("pending_new_commit"))
        tracker = self.api._MODULE_STATE_CACHE.get("__node_tracker__", {})
        self.assertEqual(tracker.get("pending_changes", {}).get("custom", {}), {})
        self.assertEqual(tracker.get("pending_new_modules", {}).get("custom", []), [])

    def test_acknowledge_all_clears_all_novelty_markers(self):
        """Ensure global acknowledge clears novelty markers for every module."""
        self.api._MODULE_STATE_CACHE = {
            "__node_tracker__": {
                "pending_changes": {"custom": {"ComfyUI_A": {"new_nodes": ["N1"], "updated_nodes": [], "at": "t"}}},
                "pending_new_modules": {"custom": ["ComfyUI_B"]},
                "startup_changes": {"custom": {"ComfyUI_A": {"new_nodes": ["N1"], "updated_nodes": [], "at": "t"}}},
                "startup_new_modules": {"custom": ["ComfyUI_B"]},
            },
            "ComfyUI_A": {
                "pending_prev_commit": "oldA",
                "pending_new_commit": "newA",
                "startup_prev_commit": "oldA",
                "startup_new_commit": "newA",
            },
            "ComfyUI_B": {
                "pending_prev_commit": "oldB",
                "pending_new_commit": "newB",
            },
        }

        result = self.api._acknowledge_all_novelty()
        self.assertEqual(result.get("status"), "ok")
        self.assertTrue(bool(result.get("changed")))

        entry_a = self.api._MODULE_STATE_CACHE.get("ComfyUI_A", {})
        entry_b = self.api._MODULE_STATE_CACHE.get("ComfyUI_B", {})
        self.assertFalse(entry_a.get("pending_prev_commit"))
        self.assertFalse(entry_a.get("pending_new_commit"))
        self.assertFalse(entry_a.get("startup_prev_commit"))
        self.assertFalse(entry_a.get("startup_new_commit"))
        self.assertFalse(entry_b.get("pending_prev_commit"))
        self.assertFalse(entry_b.get("pending_new_commit"))

        tracker = self.api._MODULE_STATE_CACHE.get("__node_tracker__", {})
        self.assertEqual(tracker.get("pending_changes"), {})
        self.assertEqual(tracker.get("pending_new_modules"), {})
        self.assertEqual(tracker.get("startup_changes"), {})
        self.assertEqual(tracker.get("startup_new_modules"), {})

    def test_comfyui_local_update_marker_persists_until_acknowledge(self):
        """Ensure ComfyUI local update marker stays visible until explicit acknowledge."""
        fake_root = os.path.join(os.getcwd(), "fake_comfy_cache")
        self.api._MODULE_STATE_CACHE = {
            "__comfyui__": {
                "installed_commit": "old11111",
                "status": {
                    "installed_commit": "old11111",
                    "update_status": "unknown",
                },
            }
        }
        self.api._comfyui_root = lambda: fake_root

        def fake_run_git(args, timeout=2.0):
            cmd = " ".join(args)
            table = {
                f"git -C {fake_root} rev-parse --is-inside-work-tree": "true",
                f"git -C {fake_root} rev-parse HEAD": "new22222",
                f"git -C {fake_root} log -1 --format=%cI": "2026-02-10T11:00:00+00:00",
            }
            return table.get(cmd)

        self.api._run_git = fake_run_git
        self.api._track_comfyui_local_update()
        self.api._COMFYUI_STATUS_CACHE = None

        info = self.api._comfyui_git_status(force_refresh=False)
        self.assertTrue(bool(info.get("updated_between_runs")))
        self.assertEqual(info.get("startup_prev_commit_short"), "old11111")
        self.assertEqual(info.get("startup_new_commit_short"), "new22222")

        self.api._acknowledge_comfyui_novelty()
        self.api._COMFYUI_STATUS_CACHE = None
        info = self.api._comfyui_git_status(force_refresh=False)
        self.assertFalse(bool(info.get("updated_between_runs")))

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

    def test_sanitize_module_description_drops_html_lines(self):
        """Validate HTML wrapper lines are removed from module descriptions."""
        source = """
        <div align="center">
        My module does useful things.
        </div>
        """
        cleaned = self.api._sanitize_module_description(source)
        self.assertEqual(cleaned, "My module does useful things.")

    def test_force_refresh_module_info_syncs_upstream(self):
        """Validate `test_force_refresh_module_info_syncs_upstream` behavior."""
        calls = []
        self.api._sync_module_upstream = lambda module_name, timeout=15.0: calls.append(module_name) or True
        self.api._module_git_state = lambda module_name: {
            "module_path": "/tmp/fake_mod",
            "repository": "https://github.com/alex/testmod",
            "installed_commit": "1234567890abcdef",
            "installed_updated_at": "2026-02-10T00:00:00+00:00",
            "remote_updated_at": "2026-02-10T01:00:00+00:00",
            "has_upstream": True,
            "ahead": 0,
            "behind": 1,
            "remote_head": "fedcba0987654321",
        }
        self.api._manager_index = lambda: {"by_github": {}, "by_id": {}, "by_repo_name": {}}
        self.api._manager_github_stats = lambda: {"by_url": {}, "by_github": {}}
        self.api._module_repo_url = lambda module_name: "https://github.com/alex/testmod"
        self.api._module_local_readme_summary = lambda module_name: "test module summary"
        self.api._remember_module_state = lambda module_name, info: None
        self.api._apply_node_change_info = lambda result, group, module_name: None
        self.api._MODULE_INFO_CACHE[("custom", "testmod")] = (
            time.time(),
            {"module": "testmod", "group": "custom", "description": "cached"},
        )

        info = self.api._resolve_module_info("custom", "testmod", force_refresh=True, sync_upstream=True)

        self.assertEqual(calls, ["testmod"])
        self.assertEqual(info.get("installed_commit_short"), "12345678")
        self.assertEqual(info.get("update_status"), "can_update")


if __name__ == "__main__":
    unittest.main()
