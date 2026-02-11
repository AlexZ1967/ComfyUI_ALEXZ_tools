"""
Module: tests/test_module_browser_manifest_check.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Tests for standalone component-manifest validator.

Purpose:
    Ensures `run_manifest_check` returns stable report shape and strict-mode
    behavior for healthy and unhealthy manifest states.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest


class ModuleBrowserManifestCheckTests(unittest.TestCase):
    """Validate manifest checker report contract and strict handling."""

    @classmethod
    def setUpClass(cls):
        """Install package path stubs used by direct module imports."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def test_manifest_check_reports_signature_and_summary(self):
        """Checker returns deterministic top-level fields for healthy state."""
        mod = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.manifest_check")
        report = mod.run_manifest_check(strict=True)
        self.assertIn(report.get("status"), {"ok", "failed"})
        self.assertIn("summary", report)
        self.assertIn("manifest_signature", report)
        self.assertTrue(str(report.get("manifest_signature")))
        self.assertIn("health", report)

    def test_manifest_check_strict_fails_when_health_not_ok(self):
        """Strict mode should convert health issues into failed status."""
        mod = importlib.import_module("ComfyUI_ALEXZ_tools.utils.module_browser.manifest_check")
        original_health = mod.build_component_health_report
        try:
            mod.build_component_health_report = lambda: {
                "ok": False,
                "issue_count": 1,
                "error_count": 0,
                "warning_count": 1,
                "issues": [{"code": "entrypoint_not_found"}],
                "checked": {},
            }
            report = mod.run_manifest_check(strict=True)
            self.assertEqual(report.get("status"), "failed")
            report_non_strict = mod.run_manifest_check(strict=False)
            self.assertEqual(report_non_strict.get("status"), "issues")
        finally:
            mod.build_component_health_report = original_health


if __name__ == "__main__":
    unittest.main()
