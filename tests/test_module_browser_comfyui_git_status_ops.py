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

    def test_collect_release_mode_does_not_fallback_to_commits_when_tag_unresolved(self):
        """Release-mode refresh should stay in releases mode when tag resolution fails."""
        cache = {}
        state = {}
        saved = []

        def _run_git(args, _timeout):
            if args[-1] == "--is-inside-work-tree":
                return "true"
            if args[-2:] == ["--abbrev-ref", "HEAD"]:
                return "HEAD"
            if args[-2:] == ["rev-parse", "HEAD"]:
                return "306af3a8cafe"
            if args[-3:] == ["log", "-1", "--format=%cI"]:
                return "2026-07-19T00:00:00+00:00"
            if args[-2:] == ["--symbolic-full-name", "@{u}"]:
                return "origin/main"
            return ""

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
            apply_cached_pending_fields=lambda result, _entry, short_commit: result,
            short_commit=lambda value: str(value or "")[:8],
            comfyui_root=lambda: Path("/tmp/ComfyUI"),
            run_git=_run_git,
            git_pick_remote=lambda _root, _upstream: "origin",
            github_latest_release=lambda _owner, _repo: {"tag_name": "v0.3.40"},
            resolve_release_ref=lambda _root, _remote, _tag: (None, ""),
            parse_datetime=lambda _text: None,
            to_iso=lambda _dt: "",
            git_resolve_remote_ref=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("commit fallback not expected")),
            persist_comfyui_status=lambda state_payload, mode_norm, result, now_iso: state_payload,
            save_module_state=lambda payload: saved.append(dict(payload)),
            now_iso=lambda: "2026-07-20T00:00:00+00:00",
        )

        self.assertEqual(result.get("check_mode"), "releases")
        self.assertEqual(result.get("update_status"), "unknown")
        self.assertTrue(result.get("release_check_degraded"))
        self.assertEqual(result.get("release_check_reason"), "release_tag_not_resolved")
        self.assertIsNone(result.get("behind"))
        self.assertNotIn("commits", cache)
        self.assertIn("releases", cache)
        self.assertTrue(saved)

    def test_collect_release_mode_uses_local_latest_tag_when_github_release_unavailable(self):
        """Release-mode refresh should use newest local tag if GitHub release API is unavailable."""
        cache = {}
        state = {}
        saved = []

        def _run_git(args, _timeout):
            if args[-1] == "--is-inside-work-tree":
                return "true"
            if args[-2:] == ["--abbrev-ref", "HEAD"]:
                return "HEAD"
            if args[-2:] == ["rev-parse", "HEAD"]:
                return "306af3a8cafe"
            if args[-3:] == ["log", "-1", "--format=%cI"]:
                return "2026-07-19T00:00:00+00:00"
            if args[-2:] == ["--symbolic-full-name", "@{u}"]:
                return "origin/main"
            if "for-each-ref" in args:
                return "v0.28.2"
            if args[-2:] == ["--count", "HEAD...refs/tags/v0.28.2"]:
                return "0 0"
            return ""

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
            apply_cached_pending_fields=lambda result, _entry, short_commit: result,
            short_commit=lambda value: str(value or "")[:8],
            comfyui_root=lambda: Path("/tmp/ComfyUI"),
            run_git=_run_git,
            git_pick_remote=lambda _root, _upstream: "origin",
            github_latest_release=lambda _owner, _repo: {},
            resolve_release_ref=lambda _root, _remote, tag: (f"refs/tags/{tag}", "306af3a8cafe"),
            parse_datetime=lambda _text: None,
            to_iso=lambda _dt: "",
            git_resolve_remote_ref=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("commit fallback not expected")),
            persist_comfyui_status=lambda state_payload, mode_norm, result, now_iso: state_payload,
            save_module_state=lambda payload: saved.append(dict(payload)),
            now_iso=lambda: "2026-07-20T00:00:00+00:00",
        )

        self.assertEqual(result.get("check_mode"), "releases")
        self.assertEqual(result.get("release_tag"), "v0.28.2")
        self.assertEqual(result.get("update_status"), "up_to_date")
        self.assertFalse(result.get("release_check_degraded"))
        self.assertEqual(result.get("behind"), 0)
        self.assertEqual(result.get("ahead"), 0)
        self.assertTrue(saved)


if __name__ == "__main__":
    unittest.main()
