"""
Module: tests/test_module_browser_value_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted pure value/repository/date helper functions.

Purpose:
    Validates behavior parity for helpers moved from module_node_browser_api
    into utils/module_browser/value_ops.py during Phase 3.
"""

from __future__ import annotations

import os
import re
import sys
import types
import unittest
from datetime import datetime, timezone


class ModuleBrowserValueOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.value_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import value helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import value_ops

        self.ops = value_ops
        self.github_re = re.compile(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)", re.IGNORECASE)

    def test_short_commit_returns_unknown_for_empty(self):
        """Short commit helper should return fallback for empty values."""
        self.assertEqual(self.ops.short_commit(""), "unknown")
        self.assertEqual(self.ops.short_commit("1234567890"), "12345678")

    def test_normalize_repo_url_handles_git_protocol_forms(self):
        """Repository URL normalization should convert git@/git:// to https."""
        self.assertEqual(
            self.ops.normalize_repo_url("git@github.com:owner/repo.git"),
            "https://github.com/owner/repo",
        )
        self.assertEqual(
            self.ops.normalize_repo_url("git://github.com/owner/repo.git"),
            "https://github.com/owner/repo",
        )

    def test_github_id_repo_name_and_pick_repo_url(self):
        """GitHub id/repo-name extraction and URL picking should be consistent."""
        gid = self.ops.github_id("https://github.com/Owner/Repo.git", github_re=self.github_re)
        self.assertEqual(gid, "Owner/Repo")
        repo_name = self.ops.repo_name(
            "https://github.com/Owner/Repo.git",
            github_id_fn=lambda url: self.ops.github_id(url, github_re=self.github_re),
        )
        self.assertEqual(repo_name, "Repo")

        picked = self.ops.pick_repo_url(
            {
                "repository": "https://example.com/nope.git",
                "reference": "https://github.com/owner/repo.git",
            },
            normalize_repo_url_fn=self.ops.normalize_repo_url,
        )
        self.assertEqual(picked, "https://github.com/owner/repo")

    def test_parse_datetime_and_to_iso(self):
        """Datetime parser should support iso/localized formats and UTC normalization."""
        dt_iso = self.ops.parse_datetime("2026-02-13T10:20:30Z")
        self.assertIsNotNone(dt_iso)
        self.assertEqual(self.ops.to_iso(dt_iso), "2026-02-13T10:20:30+00:00")

        dt_short = self.ops.parse_datetime("2026-02-13")
        self.assertIsNotNone(dt_short)
        self.assertEqual(self.ops.to_iso(dt_short), "2026-02-13T00:00:00+00:00")

        dt_ru = self.ops.parse_datetime("13.02.2026, 10:20:30")
        self.assertIsNotNone(dt_ru)
        self.assertEqual(self.ops.to_iso(dt_ru), "2026-02-13T10:20:30+00:00")

    def test_now_iso_and_normalize_comfyui_mode(self):
        """Now timestamp should be UTC-ISO; mode normalization should map commit aliases."""
        now_text = self.ops.now_iso()
        self.assertIn("+00:00", now_text)
        self.assertEqual(self.ops.normalize_comfyui_mode("commits"), "commits")
        self.assertEqual(self.ops.normalize_comfyui_mode("git"), "commits")
        self.assertEqual(self.ops.normalize_comfyui_mode("releases"), "releases")

    def test_to_iso_adds_utc_for_naive_datetime(self):
        """Naive datetime should be interpreted as UTC by `to_iso` helper."""
        naive = datetime(2026, 2, 13, 8, 0, 0)
        self.assertEqual(self.ops.to_iso(naive), "2026-02-13T08:00:00+00:00")
        aware = datetime(2026, 2, 13, 8, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(self.ops.to_iso(aware), "2026-02-13T08:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
