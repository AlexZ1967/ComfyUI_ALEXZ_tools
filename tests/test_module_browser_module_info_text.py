"""
Module: tests/test_module_browser_module_info_text.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for module-info text helpers.

Purpose:
    Verifies README summary extraction and HTML/noise cleanup used in module
    description cards after extraction from API layer.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest


class ModuleBrowserModuleInfoTextTests(unittest.TestCase):
    """Validate behavior of module-info text helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct helper imports."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import helper module for each test."""
        from ComfyUI_ALEXZ_tools.utils.module_browser.module import module_info_text as mod

        self.mod = mod

    def test_module_local_readme_summary_picks_first_meaningful_line(self):
        """Summary extractor should skip markdown headers/images and return text line."""
        with tempfile.TemporaryDirectory() as tmp:
            module_dir = Path(tmp) / "DemoModule"
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "README.md").write_text(
                "# Title\n\n![img](a.png)\n\nFirst meaningful summary line.\nSecond line\n",
                encoding="utf-8",
            )
            summary = self.mod.module_local_readme_summary(
                module_name="DemoModule",
                custom_nodes_roots=lambda: [Path(tmp)],
            )
            self.assertEqual(summary, "First meaningful summary line.")

    def test_sanitize_module_description_removes_html_noise(self):
        """Description sanitizer should drop html wrappers and keep plain summary line."""
        text = "<div align=\"center\"></div>\n\nSome <b>plain</b> summary.\n![](img.png)\n"
        cleaned = self.mod.sanitize_module_description(text, re.compile(r"<[^>]+>"))
        self.assertEqual(cleaned, "Some plain summary.")


if __name__ == "__main__":
    unittest.main()
