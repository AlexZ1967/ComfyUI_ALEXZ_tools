"""Unit tests for pure IIIF URL and identifier helpers."""

from __future__ import annotations

import os
import sys
import tempfile
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

    def test_metadata_policy_helpers(self):
        self.assertEqual(self.ops.build_iiif_size_spec("width", 0), "1,")
        self.assertEqual(self.ops.iiif_source_dimensions({"width": 1000, "height": 500}), (1000, 500))
        limit = self.ops.iiif_limit_from_max_area({"width": 1000, "height": 500, "maxArea": 125000})
        self.assertEqual((limit["predicted_max_width"], limit["predicted_max_height"]), (500, 250))
        self.assertEqual(
            self.ops.iiif_tile_profile(
                {"tiles": [{"width": 1024, "height": 1024, "scaleFactors": [1, 2]}]},
                service_url="https://iiif.nypl.org/iiif/3/57538105",
            ),
            {"tile_width": 512, "tile_height": 512},
        )

    def test_cache_path_and_image_signature_helpers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            default_root = self.ops.Path(tmp_dir) / "default"
            custom_root = self.ops.resolve_iiif_cache_dir("~/iiif-test-cache", default_root=default_root)
            self.assertEqual(custom_root, self.ops.Path(os.path.expanduser("~/iiif-test-cache")).absolute())
            scope = self.ops.resolve_iiif_tile_cache_scope(
                None,
                "https://example.test/iiif/item/",
                {"width": 1000, "height": 500},
                default_root=default_root,
            )
            self.assertEqual(scope.parent, default_root)
            path = self.ops.iiif_tile_cache_path(scope, "https://example.test/tile/image.jpg?x=1")
            self.assertEqual(path.suffix, ".jpg")
            self.assertEqual(len(path.stem), 40)
        self.assertTrue(self.ops.looks_like_raster_image_bytes(b"\xff\xd8\xff\xe0" + b"x" * 8))
        self.assertTrue(self.ops.looks_like_raster_image_bytes(b"\x89PNG\r\n\x1a\n"))
        self.assertFalse(self.ops.looks_like_raster_image_bytes(b"<html>blocked</html>"))

    def test_title_and_filename_helpers(self):
        source = "https://gallica.bnf.fr/ark:/12148/btv1b7002302c.r=Anna%20Pavlova?rk=1"
        self.assertEqual(self.ops.extract_gallica_query_title(source), "Anna Pavlova")
        self.assertEqual(self.ops.extract_html_title("<title> Anna &amp; Pavlova </title>"), "Anna & Pavlova")
        self.assertEqual(self.ops.extract_html_title("<h1>Portrait</h1>", gallica=True), "Portrait")
        self.assertEqual(self.ops.extract_iiif_stable_id(source), "btv1b7002302c")
        self.assertEqual(self.ops.derive_output_stem_from_source_url(source), "btv1b7002302c")
        self.assertEqual(self.ops.sanitize_filename_component("Anna / Pavlova"), "Anna_Pavlova")


if __name__ == "__main__":
    unittest.main()
