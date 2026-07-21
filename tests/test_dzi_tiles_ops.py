"""
Module: tests/test_dzi_tiles_ops.py
Author: AlexZ1967
Last updated: 2026-07-21

Description:
    Unit tests for extracted pure DZI helper functions.

Purpose:
    Covers site-config normalization, URL building, request context, and
    filename policy separated from the Comfy node adapter during Phase 4.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class DziTilesOpsTests(unittest.TestCase):
    """Verify behavior of `nodes.image_download_dzi_tiles_ops` pure helpers."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.nodes import image_download_dzi_tiles_ops as ops

        self.ops = ops

    def test_normalize_dzi_site_config_filters_invalid_entries(self):
        """Config normalizer should keep only valid site rows and normalize defaults."""
        payload = self.ops.normalize_dzi_site_config(
            {
                "default_site": "Archive A",
                "sites": [
                    {"name": "Archive A", "base_url": "https://example.org/", "provider": "custom", "default_level": "9"},
                    {"name": "", "base_url": "https://broken.example", "provider": "custom"},
                    "invalid",
                ],
            }
        )
        self.assertEqual(payload["default_site"], "Archive A")
        self.assertEqual(len(payload["sites"]), 1)
        self.assertEqual(payload["sites"][0]["key"], "custom")
        self.assertEqual(payload["sites"][0]["base_url"], "https://example.org")
        self.assertEqual(payload["sites"][0]["default_level"], 9)

    def test_resolve_dzi_site_supports_custom_url_input(self):
        """Resolver should accept direct URL input as an ad-hoc custom site."""
        resolved = self.ops.resolve_dzi_site(
            "https://archive.example/viewer",
            "obj-42",
            sites=[],
            detect_provider_fn=lambda site, mw, provider: "custom",
            fallback_site={"name": "Fallback", "provider": "npg"},
        )
        self.assertEqual(resolved["base_url"], "https://archive.example/viewer")
        self.assertEqual(resolved["default_mw"], "obj-42")
        self.assertEqual(resolved["provider"], "custom")

    def test_build_dzi_source_urls_uses_templates(self):
        """Template-backed URL builder should preserve template mode and example URL."""
        source = self.ops.build_dzi_source_urls(
            "https://example.org",
            "obj-42",
            9,
            "custom",
            site_config={
                "provider": "custom",
                "base_url": "https://example.org",
                "object_url_template": "{base_url}/viewer/{mw}",
                "dzi_url_template": "{base_url}/iiif/{mw}/info.dzi",
                "tile_url_template": "{base_url}/iiif/{mw}/{level}/{x}-{y}.{ext}",
            },
            default_referer="https://ref.example/",
        )
        self.assertEqual(source["tile_url_mode"], "template")
        self.assertEqual(source["zoom_base"], "https://example.org/viewer/obj-42")
        self.assertEqual(source["dzi_url"], "https://example.org/iiif/obj-42/info.dzi")
        self.assertEqual(source["tile_example_url"], "https://example.org/iiif/obj-42/9/0-0.jpg")
        self.assertEqual(source["referer_root"], "https://example.org")

    def test_resolve_dzi_request_context_uses_default_level_and_prefix(self):
        """Request context should apply default id/level rules from resolved site config."""
        ctx = self.ops.resolve_dzi_request_context(
            "NLA",
            "138204672",
            -1,
            resolve_site_fn=lambda site, mw: {
                "name": "National Library of Australia",
                "base_url": "https://nla.gov.au",
                "provider": "nla",
                "mw_prefix": "nla.obj-",
                "default_mw": "nla.obj-138204672",
                "default_level": 11,
            },
            normalize_site_mw_fn=self.ops.normalize_site_mw,
        )
        self.assertEqual(ctx["provider_name"], "nla")
        self.assertEqual(ctx["effective_mw"], "nla.obj-138204672")
        self.assertEqual(ctx["effective_level"], 11)

    def test_parse_ids_and_render_filename_cover_title_fallback(self):
        """ID parsing and filename rendering should stay deterministic with bad templates."""
        parsed = self.ops.parse_dzi_ids_text("nla.obj-1, nla.obj-2\n# comment\nnla.obj-3; nla.obj-4")
        self.assertEqual(parsed, ["nla.obj-1", "nla.obj-2", "nla.obj-3", "nla.obj-4"])

        rendered = self.ops.render_dzi_filename(
            "{missing}",
            index=1,
            raw_id="nla.obj-1",
            effective_mw="nla.obj-1",
            site_config={"name": "National Library of Australia", "key": "nla"},
            effective_level=11,
            title_stem="Anna Pavlova",
        )
        self.assertEqual(rendered, "nla.obj-1")

        rendered_title = self.ops.render_dzi_filename(
            "{title}_{site_key}_{level}",
            index=2,
            raw_id="nla.obj-2",
            effective_mw="nla.obj-2",
            site_config={"name": "National Library of Australia", "key": "nla"},
            effective_level=9,
            title_stem="Anna Pavlova / The Dying Swan",
        )
        self.assertEqual(rendered_title, "Anna_Pavlova_The_Dying_Swan_nla_9")

    def test_extract_title_and_append_stable_id(self):
        """HTML title extraction and stable-id suffixing should remain readable."""
        title = self.ops.extract_html_title(
            '<html><head><meta property="og:title" content="Anna &amp; Pavlova"></head></html>'
        )
        self.assertEqual(title, "Anna & Pavlova")
        self.assertEqual(
            self.ops.append_dzi_stable_id_to_stem("Anna_Pavlova", "nla.obj-138204672"),
            "Anna_Pavlova_nla.obj-138204672",
        )
        self.assertEqual(
            self.ops.append_dzi_stable_id_to_stem("Anna_nla.obj-138204672", "nla.obj-138204672"),
            "Anna_nla.obj-138204672",
        )

    def test_resolve_unique_output_path_uses_numeric_suffix(self):
        """Unique-path resolver should suffix collisions without touching existing files."""
        existing = {
            "/tmp/out/item.png",
            "/tmp/out/item_2.png",
        }
        path, mode = self.ops.resolve_unique_output_path(
            "/tmp/out",
            "item",
            "png",
            "unique",
            exists_fn=lambda candidate: candidate in existing,
        )
        self.assertEqual(path, "/tmp/out/item_3.png")
        self.assertEqual(mode, "unique_suffix")


if __name__ == "__main__":
    unittest.main()
