"""
Module: tests/test_module_browser_comfyui_git_status_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted ComfyUI git-status orchestration helper.

Purpose:
    Validates cache-path and no-root path behavior of status collector.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path


class ModuleBrowserComfyuiGitStatusOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.comfyui_git_status_ops` helper."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import comfyui_git_status_ops

        self.ops = comfyui_git_status_ops

    def test_collect_returns_cached_when_ttl_not_expired(self):
        """Collector should return cached payload without expensive callbacks."""
        cache = {"releases": (100.0, {"check_mode": "releases", "marker": "cached"})}
        result = self.ops.collect_comfyui_git_status(
            force_refresh=False,
            mode="releases",
            now_ts=100.1,
            cache=cache,
            ttl_sec=120.0,
            normalize_comfyui_mode=lambda mode: mode,
            comfyui_status_template=lambda mode: {"check_mode": mode},
            load_module_state=lambda: {},
            resolve_cached_status=lambda _state, _mode: (None, None),
            apply_cached_pending_fields=lambda result, _entry, _short: result,
            short_commit=lambda value: str(value or "")[:8],
            comfyui_root=lambda: None,
            run_git=lambda _args, _timeout: "",
            git_pick_remote=lambda _root, _upstream: None,
            github_latest_release=lambda _owner, _repo: {},
            resolve_release_ref=lambda _root, _remote, _tag: (None, ""),
            parse_datetime=lambda _text: None,
            to_iso=lambda _dt: "",
            git_resolve_remote_ref=lambda _root, _remote, _branch, _upstream: (None, None),
            persist_comfyui_status=lambda state, mode_norm, result, now_iso: state,
            save_module_state=lambda _state: None,
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
        )
        self.assertEqual(result.get("marker"), "cached")

    def test_collect_force_refresh_with_no_root_persists_template(self):
        """Force refresh with missing root should persist default status payload."""
        cache = {}
        state = {}
        saved = []

        def _persist(state_payload, mode_norm, result, now_iso):
            state_payload["__saved__"] = {"mode": mode_norm, "check_mode": result.get("check_mode")}
            return state_payload

        result = self.ops.collect_comfyui_git_status(
            force_refresh=True,
            mode="releases",
            now_ts=100.0,
            cache=cache,
            ttl_sec=120.0,
            normalize_comfyui_mode=lambda mode: mode,
            comfyui_status_template=lambda mode: {"check_mode": mode, "update_status": "unknown"},
            load_module_state=lambda: state,
            resolve_cached_status=lambda _state, _mode: (None, None),
            apply_cached_pending_fields=lambda result, _entry, _short: result,
            short_commit=lambda value: str(value or "")[:8],
            comfyui_root=lambda: None,
            run_git=lambda _args, _timeout: "",
            git_pick_remote=lambda _root, _upstream: None,
            github_latest_release=lambda _owner, _repo: {},
            resolve_release_ref=lambda _root, _remote, _tag: (None, ""),
            parse_datetime=lambda _text: None,
            to_iso=lambda _dt: "",
            git_resolve_remote_ref=lambda _root, _remote, _branch, _upstream: (None, None),
            persist_comfyui_status=_persist,
            save_module_state=lambda payload: saved.append(dict(payload)),
            now_iso=lambda: "2026-02-13T00:00:00+00:00",
        )
        self.assertEqual(result.get("check_mode"), "releases")
        self.assertTrue(saved)
        self.assertIn("releases", cache)


if __name__ == "__main__":
    unittest.main()
