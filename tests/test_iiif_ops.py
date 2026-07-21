"""Unit tests for pure IIIF URL and identifier helpers."""

from __future__ import annotations

import os
import sys
import types
import unittest


class IiifOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            package = types.ModuleType("ComfyUI_ALEXZ_tools")
            package.__path__ = [root]
            sys.modules["ComfyUI_ALEXZ_tools"] = package

    def setUp(self):
        from ComfyUI_ALEXZ_tools.nodes import image_download_iiif_ops as ops

        self.ops = ops

    def test_normalizes_service_and_extracts_html_urls(self):
        self.assertEqual(
            self.ops.normalize_iiif_service_url("https://example.test/iiif/item/full/max/0/default.jpg"),
            "https://example.test/iiif/item",
        )
        self.assertEqual(
            self.ops.extract_first_generic_iiif_service_url('<img src="https://example.test/iiif/item/info.json">'),
            "https://example.test/iiif/item",
        )

    def test_nypl_override_parsing_and_injection(self):
        source = "https://digitalcollections.nypl.org/items/item-1?canvasIndex=2&imageid=old"
        self.assertEqual(self.ops.extract_forced_nypl_image_id_from_source_url(source), "old")
        updated = self.ops.inject_nypl_image_id_into_source_url(source, "57538105")
        self.assertIn("canvasIndex=2", updated)
        self.assertIn("image_id=57538105", updated)
        self.assertNotIn("imageid=old", updated)
        self.assertEqual(
            self.ops.extract_nypl_image_ids_from_json_payload({"x": [{"imageID": 57538105}]}),
            ["57538105"],
        )

    def test_gallica_service_url_defaults_and_page_selection(self):
        self.assertEqual(
            self.ops.extract_gallica_service_url_from_source_url("https://gallica.bnf.fr/ark:/12148/btv1b7002302c"),
            "https://gallica.bnf.fr/iiif/ark:/12148/btv1b7002302c/f1",
        )
        self.assertEqual(
            self.ops.extract_gallica_service_url_from_source_url("https://gallica.bnf.fr/ark:/12148/btv1b7002302c/f3.item"),
            "https://gallica.bnf.fr/iiif/ark:/12148/btv1b7002302c/f3",
        )


if __name__ == "__main__":
    unittest.main()
