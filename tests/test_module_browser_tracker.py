import importlib
import os
import sys
import types
import unittest


def _install_folder_paths_stub():
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
    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg
        _install_folder_paths_stub()

    def setUp(self):
        self.api = importlib.import_module("ComfyUI_ALEXZ_tools.module_node_browser_api")
        self._orig_state_cache = self.api._MODULE_STATE_CACHE
        self._orig_comfy_cache = self.api._COMFYUI_STATUS_CACHE
        self._orig_save_state = self.api._save_module_state
        self._orig_snapshots = self.api._build_node_snapshots
        self._orig_discover = self.api._discover_custom_modules
        self._orig_now_iso = self.api._now_iso
        self._orig_comfy_root = self.api._comfyui_root
        self._orig_run_git = self.api._run_git
        self.api._MODULE_STATE_CACHE = {}
        self.api._COMFYUI_STATUS_CACHE = None
        self.api._save_module_state = lambda state: None
        self.api._MODULE_INFO_CACHE.clear()

    def tearDown(self):
        self.api._MODULE_STATE_CACHE = self._orig_state_cache
        self.api._COMFYUI_STATUS_CACHE = self._orig_comfy_cache
        self.api._save_module_state = self._orig_save_state
        self.api._build_node_snapshots = self._orig_snapshots
        self.api._discover_custom_modules = self._orig_discover
        self.api._now_iso = self._orig_now_iso
        self.api._comfyui_root = self._orig_comfy_root
        self.api._run_git = self._orig_run_git
        self.api._MODULE_INFO_CACHE.clear()

    def test_new_module_marker_applies_without_node_diffs(self):
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
        self.api._comfyui_root = lambda: os.path.join(os.getcwd(), "fake_comfy")

        def fake_run_git(args, timeout=2.0):
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


if __name__ == "__main__":
    unittest.main()
