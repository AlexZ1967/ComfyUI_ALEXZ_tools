"""
Module: tests/test_module_browser_api_request_parsing.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Unit tests for extracted module-browser API request parsing helpers.

Purpose:
    Verifies malformed JSON fallback and boolean-flag normalization moved out
    of route glue during Phase 3 stabilization.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest


class _DummyRequest:
    def __init__(self, payload=None, *, json_raises=False):
        self._payload = payload
        self._json_raises = bool(json_raises)

    async def json(self):
        if self._json_raises:
            raise ValueError("invalid json")
        return self._payload


class ModuleBrowserApiRequestParsingTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser_api.request_parsing` helpers."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.request_parsing import (
            coerce_bool_flag,
            load_json_payload,
        )

        self.coerce_bool_flag = coerce_bool_flag
        self.load_json_payload = load_json_payload

    def test_coerce_bool_flag_supports_common_aliases(self):
        """Boolean flag coercion should preserve existing query alias semantics."""
        self.assertTrue(self.coerce_bool_flag("1", default=False))
        self.assertTrue(self.coerce_bool_flag("yes", default=False))
        self.assertFalse(self.coerce_bool_flag("0", default=True))
        self.assertFalse(self.coerce_bool_flag("off", default=True))
        self.assertTrue(self.coerce_bool_flag("", default=True))
        self.assertFalse(self.coerce_bool_flag("", default=False))

    def test_load_json_payload_returns_empty_on_parse_error(self):
        """Malformed JSON payloads should degrade to empty dict for route handlers."""
        payload = asyncio.run(self.load_json_payload(_DummyRequest(json_raises=True)))
        self.assertEqual(payload, {})

    def test_load_json_payload_rejects_non_mapping_payloads(self):
        """Non-dict JSON payloads should degrade to empty dict for route handlers."""
        payload = asyncio.run(self.load_json_payload(_DummyRequest(payload=["modA"])))
        self.assertEqual(payload, {})

    def test_load_json_payload_keeps_mapping_payload(self):
        """Mapping JSON payloads should be returned as regular dicts."""
        payload = asyncio.run(self.load_json_payload(_DummyRequest(payload={"modules": ["modA"]})))
        self.assertEqual(payload, {"modules": ["modA"]})


if __name__ == "__main__":
    unittest.main()
