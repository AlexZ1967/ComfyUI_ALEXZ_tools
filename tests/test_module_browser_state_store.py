"""
Module: tests/test_module_browser_state_store.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Unit tests for extracted state-store JSON helpers.

Purpose:
    Validates disk load/save helpers used by Module Node Picker cache handling.
"""

from __future__ import annotations

import os
import sys
import types
import tempfile
import unittest
from pathlib import Path


class _NoopLogger:
    """Minimal logger stub used by helper tests."""

    def debug(self, *_args, **_kwargs):
        """Ignore debug logs in tests."""


class ModuleBrowserStateStoreTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser.state_store` helpers."""

    @classmethod
    def setUpClass(cls):
        """Install package-path stub for direct submodule imports in tests."""
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        """Import state-store helper module for each test case."""
        from ComfyUI_ALEXZ_tools.utils.module_browser import state_store

        self.store = state_store

    def test_load_state_file_missing_returns_schema_default(self):
        """Missing state file should return normalized empty schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "missing.json"
            result = self.store.load_state_file(
                state_path,
                ensure_schema=lambda value: {"__schema__": {"ok": bool(isinstance(value, dict))}},
            )
            self.assertTrue(result["__schema__"]["ok"])

    def test_load_state_file_invalid_json_returns_schema_default(self):
        """Corrupted state file should safely fall back to normalized empty schema."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "state.json"
            state_path.write_text("{not-json", encoding="utf-8")
            result = self.store.load_state_file(
                state_path,
                ensure_schema=lambda value: {"count": {"value": len(value)}},
            )
            self.assertEqual(result["count"]["value"], 0)

    def test_save_state_file_normalizes_and_persists(self):
        """Save helper should normalize and write state payload to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "state.json"
            normalized = self.store.save_state_file(
                state_path,
                {"modA": {"x": 1}},
                ensure_schema=lambda value: {"normalized": {"size": len(value)}},
                logger=_NoopLogger(),
            )
            self.assertEqual(normalized["normalized"]["size"], 1)
            loaded = self.store.load_state_file(
                state_path,
                ensure_schema=lambda value: value if isinstance(value, dict) else {},
            )
            self.assertIn("normalized", loaded)


if __name__ == "__main__":
    unittest.main()
