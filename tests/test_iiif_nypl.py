"""
Test NYPL IIIF URL resolution in ImageDownloadIIIFImage node.
"""

import importlib
import tempfile
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

class TestNYPLResolution(unittest.TestCase):
    def test_nypl_resolve_uses_image_id_from_source_url_query_without_network(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = (
            "https://digitalcollections.nypl.org/items/"
            "3d9f41f0-c6bb-012f-b741-58d385a7bc34?canvasIndex=0&image_id=57538105"
        )
        old_http_get = iiif_mod._http_get
        try:
            iiif_mod._http_get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("must not call network"))
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/57538105")

    def test_nypl_resolve_uses_string_image_id_from_source_url_query_without_network(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = (
            "https://digitalcollections.nypl.org/items/"
            "3d9f41f0-c6bb-012f-b741-58d385a7bc34?canvasIndex=0&image_id=NIJINSKY_2032V"
        )
        old_http_get = iiif_mod._http_get
        try:
            iiif_mod._http_get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("must not call network"))
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/NIJINSKY_2032V")

    def test_nypl_resolve_uses_plain_source_url_image_id_without_network(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "NIJINSKY_2032V"
        old_http_get = iiif_mod._http_get
        try:
            iiif_mod._http_get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("must not call network"))
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/NIJINSKY_2032V")

    def test_nypl_session_uses_browser_like_image_headers(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        session = iiif_mod._new_http_session(
            "The New York Public Library (NYPL) Digital Collections",
            "NIJINSKY_2033V",
        )
        self.assertEqual(session.headers.get("Referer"), "https://digitalcollections.nypl.org/")
        self.assertEqual(session.headers.get("Accept-Language"), "en-US,en;q=0.9")
        self.assertIn("image/*", str(session.headers.get("Accept") or ""))

    def test_nypl_download_uses_plain_source_url_image_id_without_html_lookup(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_download = iiif_mod._download_iiif_image_bytes
        old_assemble = iiif_mod._assemble_iiif_full_image

        resolved_inputs = []

        def _fake_resolve(site, source_url, timeout=30.0, session=None):
            _ = (site, timeout, session)
            resolved_inputs.append(source_url)
            return "https://iiif.nypl.org/iiif/3/NIJINSKY_2032V"

        def _fake_info(*args, **kwargs):
            _ = (args, kwargs)
            return {
                "width": 4000,
                "height": 3000,
                "sizes": [{"width": 1200, "height": 900}],
            }

        def _fake_download(*args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("single-request path must not be used for NYPL")

        def _fake_assemble(service_url, info, *, output_format, timeout, session, cache_dir):
            _ = (service_url, info, output_format, timeout, session, cache_dir)
            arr = np.zeros((8, 10, 3), dtype=np.uint8)
            im = Image.fromarray(arr, mode="RGB")
            return im, {
                "mode": "tile_assemble_full",
                "tile_width": 512,
                "tile_height": 512,
                "tiles_x": 1,
                "tiles_y": 1,
                "tiles_total": 1,
                "tiles_downloaded": 1,
                "selected_format": "jpg",
                "last_tile_url": "https://iiif.nypl.org/tile.jpg",
                "cache_dir": "/tmp/mock_cache",
                "cache_hits": 0,
                "cache_misses": 1,
                "cache_stores": 1,
                "cache_cleared": False,
            }

        try:
            iiif_mod._resolve_iiif_service_url = _fake_resolve
            iiif_mod._fetch_iiif_info = _fake_info
            iiif_mod._download_iiif_image_bytes = _fake_download
            iiif_mod._assemble_iiif_full_image = _fake_assemble
            image, info_json = node.download(
                "The New York Public Library (NYPL) Digital Collections",
                "NIJINSKY_2032V",
            )
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._download_iiif_image_bytes = old_download
            iiif_mod._assemble_iiif_full_image = old_assemble

        self.assertEqual(tuple(image.shape), (1, 8, 10, 3))
        self.assertEqual(
            resolved_inputs,
            ["NIJINSKY_2032V"],
        )
        self.assertNotIn('"nypl_image_id":', info_json)

    def test_nypl_extract_image_id_from_real_item_html_block(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        html = (
            '<div class="css-0"><h2 class="chakra-heading css-bfy4z6" data-testid="ds-heading">'
            'Image ID</h2><p class="chakra-text css-1xdhyk6" data-testid="ds-text" id="image-id" '
            'aria-label="Image ID">57538105</p></div>'
        )
        self.assertEqual(iiif_mod._extract_nypl_image_id_from_html(html), "57538105")

    def test_nypl_extract_string_image_id_from_real_item_html_block(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        html = (
            '<div class="css-0"><h2 class="chakra-heading css-bfy4z6" data-testid="ds-heading">'
            'Image ID</h2><p class="chakra-text css-1xdhyk6" data-testid="ds-text" id="image-id" '
            'aria-label="Image ID">NIJINSKY_2032V</p></div>'
        )
        self.assertEqual(iiif_mod._extract_nypl_image_id_from_html(html), "NIJINSKY_2032V")

    def test_nypl_resolve_prefers_numeric_image_id_from_page(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0"
        old_http_get = iiif_mod._http_get

        class _Response:
            status_code = 200
            text = '<html><body>"imageId":57538105</body></html>'

        try:
            iiif_mod._http_get = lambda *args, **kwargs: _Response()
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/57538105")

    def test_nypl_resolve_fails_fast_without_image_id_when_page_is_blocked(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0"
        old_http_get = iiif_mod._http_get
        try:
            iiif_mod._http_get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("blocked"))
            with self.assertRaises(RuntimeError) as cm:
                iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertIn("Add `image_id=<nypl_image_id>`", str(cm.exception))

    def test_nypl_download_forces_tile_assembly_without_single_request_attempt(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        node = iiif_mod.ImageDownloadIIIFImage()
        old_resolve = iiif_mod._resolve_iiif_service_url
        old_info = iiif_mod._fetch_iiif_info
        old_download = iiif_mod._download_iiif_image_bytes
        old_assemble = iiif_mod._assemble_iiif_full_image

        calls = []
        assembled_calls = []

        def _fake_resolve(*args, **kwargs):
            _ = (args, kwargs)
            return "https://iiif.nypl.org/iiif/3/57538105"

        def _fake_info(*args, **kwargs):
            _ = (args, kwargs)
            return {
                "width": 4000,
                "height": 3000,
                "sizes": [{"width": 1200, "height": 900}],
            }

        def _fake_download(service_url, *, size_spec, output_format, timeout, session):
            _ = (service_url, output_format, timeout, session)
            calls.append(size_spec)
            raise RuntimeError("single-request path must not be used for NYPL")

        def _fake_assemble(service_url, info, *, output_format, timeout, session, cache_dir):
            _ = (service_url, info, output_format, timeout, session, cache_dir)
            assembled_calls.append("called")
            arr = np.zeros((8, 10, 3), dtype=np.uint8)
            arr[:, :, 0] = 64
            arr[:, :, 1] = 128
            arr[:, :, 2] = 192
            im = Image.fromarray(arr, mode="RGB")
            meta = {
                "mode": "tile_assemble_full",
                "tile_width": 512,
                "tile_height": 512,
                "tiles_x": 1,
                "tiles_y": 1,
                "tiles_total": 1,
                "tiles_downloaded": 1,
                "selected_format": "jpg",
                "last_tile_url": "https://iiif.nypl.org/tile.jpg",
                "cache_dir": "/tmp/mock_cache",
                "cache_hits": 0,
                "cache_misses": 1,
                "cache_stores": 1,
                "cache_cleared": False,
            }
            return im, meta

        try:
            iiif_mod._resolve_iiif_service_url = _fake_resolve
            iiif_mod._fetch_iiif_info = _fake_info
            iiif_mod._download_iiif_image_bytes = _fake_download
            iiif_mod._assemble_iiif_full_image = _fake_assemble
            image, info_json = node.download(
                "The New York Public Library (NYPL) Digital Collections",
                "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0",
                size_mode="max",
            )
        finally:
            iiif_mod._resolve_iiif_service_url = old_resolve
            iiif_mod._fetch_iiif_info = old_info
            iiif_mod._download_iiif_image_bytes = old_download
            iiif_mod._assemble_iiif_full_image = old_assemble

        self.assertEqual(calls, [])
        self.assertEqual(assembled_calls, ["called"])
        self.assertEqual(tuple(image.shape), (1, 8, 10, 3))
        self.assertIn('"mode": "tile_assemble_full"', info_json)

    def test_iiif_tile_download_uses_explicit_region_size_not_max(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        old_http_get = iiif_mod._http_get
        seen_urls = []

        class _Response:
            status_code = 200
            content = b"tile-bytes"

        try:
            def _fake_http_get(url, *, timeout, session=None, retries=3, retry_backoff=0.75):
                _ = (timeout, session, retries, retry_backoff)
                seen_urls.append(url)
                return _Response()

            iiif_mod._http_get = _fake_http_get
            with tempfile.TemporaryDirectory() as tmpdir:
                image_url, content, fmt = iiif_mod._download_iiif_tile_bytes(
                    "https://iiif.nypl.org/iiif/3/NIJINSKY_2033V",
                    region="0,0,462,512",
                    size_spec="462,512",
                    output_format="jpg",
                    timeout=1.0,
                    session=None,
                    cache_dir=tmpdir,
                    cache_stats=None,
                )
        finally:
            iiif_mod._http_get = old_http_get

        self.assertEqual(fmt, "jpg")
        self.assertEqual(content, b"tile-bytes")
        self.assertEqual(
            image_url,
            "https://iiif.nypl.org/iiif/3/NIJINSKY_2033V/0,0,462,512/462,512/0/default.jpg",
        )
        self.assertEqual(seen_urls, [image_url])

    def test_iiif_tile_download_rejects_html_body_before_decode(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        old_http_get = iiif_mod._http_get

        class _Response:
            status_code = 200
            content = b"<html><title>Forbidden</title></html>"
            headers = {"content-type": "text/html; charset=utf-8"}

        try:
            iiif_mod._http_get = lambda *args, **kwargs: _Response()
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(RuntimeError) as cm:
                    iiif_mod._download_iiif_tile_bytes(
                        "https://iiif.nypl.org/iiif/3/NIJINSKY_2033V",
                        region="0,0,462,512",
                        size_spec="462,512",
                        output_format="jpg",
                        timeout=1.0,
                        session=None,
                        cache_dir=tmpdir,
                        cache_stats=None,
                    )
        finally:
            iiif_mod._http_get = old_http_get

        self.assertIn("non-image content", str(cm.exception))
        self.assertIn("content-type=text/html", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
