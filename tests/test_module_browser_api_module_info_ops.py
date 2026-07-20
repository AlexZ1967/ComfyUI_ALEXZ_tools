"""
Module: tests/test_module_browser_api_module_info_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Unit tests for extracted module-browser API module-info helpers.

Purpose:
    Verifies module-info cache orchestration moved out of
    `utils/module_node_browser_api.py` during Phase 3 stabilization.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserApiModuleInfoOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser_api.module_info_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api import module_info_ops

        self.ops = module_info_ops

    def test_resolve_module_info_cached_returns_cached_copy_when_ttl_valid(self):
        """Module-info cache helper should reuse fresh cached payloads by copy."""
        cache = {("custom", "ModA", False): (100.0, {"module": "ModA", "cached": True})}
        result = self.ops.resolve_module_info_cached(
            group="custom",
            module_name="ModA",
            force_refresh=False,
            sync_upstream=False,
            cache_only=False,
            now_ts=105.0,
            module_info_cache=cache,
            ttl_sec=30.0,
            canonical_custom_module_name=lambda name: name,
            resolve_module_info_uncached=lambda **kwargs: (_ for _ in ()).throw(AssertionError("uncached resolver not expected")),
            apply_node_change_info_fn=lambda result, group, module: None,
            sync_module_upstream=lambda module: None,
            load_module_state=lambda: {},
            custom_update_checked_flag=lambda state=None: False,
            module_git_state=lambda module: {},
            module_repo_url=lambda module: "",
            manager_meta_for_module=lambda module, repo: None,
            module_local_readme_summary_fn=lambda module: "",
            sanitize_module_description_fn=lambda text: text,
            github_id=lambda repo: "",
            infer_update_from_manager_stats=lambda repo, installed: (None, ""),
            short_commit=lambda value: str(value or "")[:8],
            remember_module_state_fn=lambda module, result: None,
        )
        self.assertEqual(result, {"module": "ModA", "cached": True})
        self.assertIsNot(result, cache[("custom", "ModA", False)][1])

    def test_resolve_module_info_cached_canonicalizes_custom_and_writes_cache(self):
        """Custom-module info helper should canonicalize names and write refreshed cache entries."""
        cache: dict[tuple[str, str, bool], tuple[float, dict[str, object]]] = {}
        captured = {"module": None}

        def _resolve_uncached(**kwargs):
            captured["module"] = kwargs.get("module_name")
            return {"module": kwargs.get("module_name"), "group": kwargs.get("group")}

        result = self.ops.resolve_module_info_cached(
            group="custom",
            module_name="moda",
            force_refresh=True,
            sync_upstream=True,
            cache_only=True,
            now_ts=200.0,
            module_info_cache=cache,
            ttl_sec=30.0,
            canonical_custom_module_name=lambda name: "ModA" if str(name).lower() == "moda" else name,
            resolve_module_info_uncached=_resolve_uncached,
            apply_node_change_info_fn=lambda result, group, module: None,
            sync_module_upstream=lambda module: None,
            load_module_state=lambda: {},
            custom_update_checked_flag=lambda state=None: False,
            module_git_state=lambda module: {},
            module_repo_url=lambda module: "",
            manager_meta_for_module=lambda module, repo: None,
            module_local_readme_summary_fn=lambda module: "",
            sanitize_module_description_fn=lambda text: text,
            github_id=lambda repo: "",
            infer_update_from_manager_stats=lambda repo, installed: (None, ""),
            short_commit=lambda value: str(value or "")[:8],
            remember_module_state_fn=lambda module, result: None,
        )

        self.assertEqual(captured["module"], "ModA")
        self.assertEqual(result, {"module": "ModA", "group": "custom"})
        self.assertIn(("custom", "ModA", True), cache)

    def test_sanitize_module_description_delegates_regex_helper(self):
        """Description sanitizer helper should forward regex/context to implementation."""
        calls = {}

        def _sanitize(text, regex):
            calls["args"] = (text, regex)
            return "clean"

        cleaned = self.ops.sanitize_module_description(
            "<b>text</b>",
            sanitize_module_description_impl=_sanitize,
            html_tag_re="regex-marker",
        )
        self.assertEqual(calls["args"], ("<b>text</b>", "regex-marker"))
        self.assertEqual(cleaned, "clean")


if __name__ == "__main__":
    unittest.main()
