"""Tests for deterministic Show/Save JSON helper behavior."""

from __future__ import annotations

import importlib
import json
import unittest


class JsonOutputOpsTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.json_output")

    def test_parse_json_string_preserves_invalid_text(self):
        self.assertEqual(self.module._parse_json_string("not-json"), "not-json")
        self.assertEqual(self.module._parse_json_string('"{\\"value\\": 3}"'), {"value": 3})

    def test_format_json_text_serializes_supported_values(self):
        payload = {"items": [1, 2], "enabled": True}
        self.assertEqual(json.loads(self.module._format_json_text(payload)), payload)

    def test_unexpected_decoder_failure_is_not_silenced(self):
        old_loads = self.module.json.loads
        try:
            self.module.json.loads = lambda _value: (_ for _ in ()).throw(RuntimeError("decoder failure"))
            with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                self.module._parse_json_string("{}")
        finally:
            self.module.json.loads = old_loads
