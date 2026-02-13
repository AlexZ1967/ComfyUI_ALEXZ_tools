"""
Module: tests/test_module_browser_release_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted GitHub release helper functions.

Purpose:
    Verifies latest-release fetch parsing and fallback behavior for release_ops.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserReleaseOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.release_ops` helper."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import release helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import release_ops

        self.ops = release_ops

    def test_github_latest_release_parses_payload(self):
        """Release helper should parse expected fields from GitHub payload."""

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"tag_name":"v1.2.3","published_at":"2026-02-13T00:00:00Z","html_url":"https://github.com/a/b/releases/tag/v1.2.3"}'

        result = self.ops.github_latest_release(
            "owner",
            "repo",
            request_factory=lambda url, headers=None: {"url": url, "headers": headers},
            urlopen_fn=lambda req, timeout=8.0: _Resp(),
        )
        self.assertEqual(result.get("tag_name"), "v1.2.3")
        self.assertEqual(result.get("published_at"), "2026-02-13T00:00:00Z")
        self.assertIn("html_url", result)

    def test_github_latest_release_returns_empty_on_missing_tag(self):
        """Helper should return empty payload if tag_name is missing."""

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"name":"release without tag"}'

        result = self.ops.github_latest_release(
            "owner",
            "repo",
            request_factory=lambda url, headers=None: {"url": url, "headers": headers},
            urlopen_fn=lambda req, timeout=8.0: _Resp(),
        )
        self.assertEqual(result, {})

    def test_github_latest_release_returns_empty_on_invalid_owner_repo(self):
        """Helper should short-circuit on empty owner/repo values."""
        self.assertEqual(self.ops.github_latest_release("", "repo"), {})
        self.assertEqual(self.ops.github_latest_release("owner", ""), {})


if __name__ == "__main__":
    unittest.main()

