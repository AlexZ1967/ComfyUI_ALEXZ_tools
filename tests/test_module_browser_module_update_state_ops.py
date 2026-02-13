"""
Module: tests/test_module_browser_module_update_state_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted update-state evaluation helpers.

Purpose:
    Validates module/comfy update state decisions and counters moved from
    module_node_browser_api into module_update_state_ops.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserModuleUpdateStateOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.module_update_state_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import module-update-state helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import module_update_state_ops

        self.ops = module_update_state_ops

    def test_module_needs_update_from_git_state_behind_counter(self):
        """Behind counter from git state should drive update-availability decision."""
        result = self.ops.module_needs_update_now(
            "modA",
            canonical_custom_module_name=lambda name: name,
            load_module_state=lambda: {},
            module_git_state_fn=lambda module: {"behind": 2, "has_upstream": True},
            manager_meta_for_module_fn=lambda module, repo: None,
            infer_update_from_manager_stats_fn=lambda repo, installed: (None, ""),
        )
        self.assertTrue(result)

    def test_module_needs_update_fallback_to_manager_stats(self):
        """When git state is unavailable, manager-stats inference should be used."""
        result = self.ops.module_needs_update_now(
            "modA",
            canonical_custom_module_name=lambda name: name,
            load_module_state=lambda: {"modA": {"repository": "https://github.com/a/b", "installed_updated_at": "2026-02-10"}},
            module_git_state_fn=lambda module: {},
            manager_meta_for_module_fn=lambda module, repo: None,
            infer_update_from_manager_stats_fn=lambda repo, installed: (True, "2026-02-11T00:00:00+00:00"),
        )
        self.assertTrue(result)

    def test_count_custom_modules_need_update_and_unknown(self):
        """Need-update and unknown counters should follow stored `update_available` semantics."""
        state = {
            "modA": {"update_available": True},
            "modB": {"update_available": False},
            "modC": {"update_available": "unknown"},
        }
        modules = lambda: ["modA", "modB", "modC", "modD"]
        canonical = lambda name: name
        need = self.ops.count_custom_modules_need_update(
            load_module_state=lambda: state,
            discover_custom_modules=modules,
            canonical_custom_module_name=canonical,
        )
        unknown = self.ops.count_custom_modules_unknown_update(
            load_module_state=lambda: state,
            discover_custom_modules=modules,
            canonical_custom_module_name=canonical,
        )
        self.assertEqual(need, 2)  # truthy non-bool values are also counted by legacy behavior
        self.assertEqual(unknown, 2)  # modC non-bool + modD missing

    def test_comfyui_needs_update_now_uses_behind_or_status(self):
        """ComfyUI helper should prefer integer behind counter, then status string."""
        self.assertTrue(
            self.ops.comfyui_needs_update_now(
                comfyui_git_status_fn=lambda force_refresh=True, mode="releases": {"behind": 1}
            )
        )
        self.assertTrue(
            self.ops.comfyui_needs_update_now(
                comfyui_git_status_fn=lambda force_refresh=True, mode="releases": {"update_status": "can_update"}
            )
        )
        self.assertFalse(
            self.ops.comfyui_needs_update_now(
                comfyui_git_status_fn=lambda force_refresh=True, mode="releases": {"behind": 0}
            )
        )


if __name__ == "__main__":
    unittest.main()
