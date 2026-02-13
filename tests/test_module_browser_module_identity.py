"""
Module: tests/test_module_browser_module_identity.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Unit tests for extracted custom-module identity helpers.

Purpose:
    Validates discover/normalize/alias/canonical behavior for module names.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class ModuleBrowserModuleIdentityTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.module_identity` helpers."""

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
        from ComfyUI_ALEXZ_tools.utils.module_browser import module_identity

        self.mod = module_identity

    def test_normalize_module_token_strips_symbols(self):
        """Normalizer should lower-case and strip non-alnum characters."""
        self.assertEqual(self.mod.normalize_module_token("ComfyUI-RMBG"), "comfyuirmbg")

    def test_build_aliases_and_canonical_lookup(self):
        """Aliases should support direct, lowercase and normalized tokens."""
        aliases = self.mod.build_custom_module_aliases(
            discovered_modules=["ComfyUI-RMBG", "crt-nodes"],
        )
        self.assertEqual(
            self.mod.canonical_custom_module_name("comfyui_rmbg", aliases=aliases),
            "ComfyUI-RMBG",
        )
        self.assertEqual(
            self.mod.canonical_custom_module_name("CRT-NODES", aliases=aliases),
            "crt-nodes",
        )

    def test_discover_custom_modules_filters_non_modules(self):
        """Discovery should include module-like dirs and skip hidden/__pycache__."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "mod_a").mkdir()
            (root / "mod_a" / "__init__.py").write_text("# test\n", encoding="utf-8")
            (root / "mod_b").mkdir()
            (root / "mod_b" / "tool.py").write_text("# test\n", encoding="utf-8")
            (root / "__pycache__").mkdir()
            (root / ".hidden").mkdir()
            discovered = self.mod.discover_custom_modules(custom_nodes_roots=lambda: [root])
            self.assertIn("mod_a", discovered)
            self.assertIn("mod_b", discovered)
            self.assertNotIn("__pycache__", discovered)
            self.assertNotIn(".hidden", discovered)


if __name__ == "__main__":
    unittest.main()
