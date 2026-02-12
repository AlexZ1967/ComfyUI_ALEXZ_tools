"""
Module: tests/test_module_browser_module_info.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted module-info assembly helpers.

Purpose:
    Verifies cached badge and module-info payload behavior after Phase 3
    extraction to keep API facade behavior stable.
"""

import os
import sys
import types
import unittest


def _identity(name: str) -> str:
    """Return canonical module name as-is for test fixtures."""
    return str(name or "")


class ModuleBrowserModuleInfoTests(unittest.TestCase):
    """Validate extracted module-info helper behavior."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import module-info helpers for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser.module_info import (
            cached_module_flags,
            resolve_module_info_uncached,
        )

        self.cached_module_flags = cached_module_flags
        self.resolve_module_info_uncached = resolve_module_info_uncached

    def test_cached_flags_hide_unknown_before_checked(self):
        """Unknown update status is hidden until explicit custom refresh check."""
        state = {
            "__meta__": {"custom_update_checked": False},
            "modA": {"update_status": "unknown", "update_available": None},
        }
        flags = self.cached_module_flags(
            group_name="custom",
            module_name="modA",
            state=state,
            canonical_custom_module_name=_identity,
            custom_update_checked_flag=lambda s: bool((s or {}).get("__meta__", {}).get("custom_update_checked")),
        )
        self.assertEqual(flags.get("update_status"), "")
        self.assertFalse(bool(flags.get("update_available")))

    def test_cached_flags_pending_local_change_marks_updated(self):
        """Pending local-change marker keeps module marked as locally updated."""
        state = {
            "__meta__": {"custom_update_checked": False},
            "modA": {"pending_local_change": True},
        }
        flags = self.cached_module_flags(
            group_name="custom",
            module_name="modA",
            state=state,
            canonical_custom_module_name=_identity,
            custom_update_checked_flag=lambda s: bool((s or {}).get("__meta__", {}).get("custom_update_checked")),
        )
        self.assertTrue(bool(flags.get("updated_between_runs")))

    def test_resolve_builtin_module_info_contract(self):
        """Built-in groups produce non-updatable module payload."""
        info = self.resolve_module_info_uncached(
            group="core",
            module_name="nodes",
            sync_upstream=False,
            cache_only=True,
            canonical_custom_module_name=_identity,
            apply_node_change_info=lambda result, group, module: None,
            sync_module_upstream=lambda module: None,
            load_module_state=lambda: {},
            custom_update_checked_flag=lambda state: False,
            module_git_state=lambda module: {},
            module_repo_url=lambda module: "",
            manager_meta_for_module=lambda module, repo: None,
            module_local_readme_summary=lambda module: "",
            sanitize_module_description=lambda text: text,
            github_id=lambda repo: "",
            infer_update_from_manager_stats=lambda repo, installed_at: (None, None),
            short_commit=lambda commit: commit[:8],
            remember_module_state=lambda module, result: None,
        )
        self.assertEqual(info.get("group"), "core")
        self.assertEqual(info.get("update_status"), "")
        self.assertEqual(info.get("update_available"), False)

    def test_resolve_custom_cache_only_without_entry_hides_unknown(self):
        """Cache-only custom payload hides unknown when status was not explicitly checked."""
        info = self.resolve_module_info_uncached(
            group="custom",
            module_name="modMissing",
            sync_upstream=False,
            cache_only=True,
            canonical_custom_module_name=_identity,
            apply_node_change_info=lambda result, group, module: None,
            sync_module_upstream=lambda module: None,
            load_module_state=lambda: {"__meta__": {"custom_update_checked": False}},
            custom_update_checked_flag=lambda state: bool((state or {}).get("__meta__", {}).get("custom_update_checked")),
            module_git_state=lambda module: {},
            module_repo_url=lambda module: "",
            manager_meta_for_module=lambda module, repo: None,
            module_local_readme_summary=lambda module: "",
            sanitize_module_description=lambda text: text,
            github_id=lambda repo: "",
            infer_update_from_manager_stats=lambda repo, installed_at: (None, None),
            short_commit=lambda commit: commit[:8],
            remember_module_state=lambda module, result: None,
        )
        self.assertEqual(info.get("update_status"), "up_to_date")
        self.assertEqual(info.get("update_available"), False)


if __name__ == "__main__":
    unittest.main()
