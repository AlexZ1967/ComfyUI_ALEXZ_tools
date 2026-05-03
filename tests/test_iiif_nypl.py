"""
Test NYPL IIIF URL resolution in ImageDownloadIIIFImage node.
"""

import importlib
import unittest
from io import BytesIO

import numpy as np
from PIL import Image

class TestNYPLResolution(unittest.TestCase):
    def test_nypl_extract_image_id_from_real_item_html_block(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        html = (
            '<div class="css-0"><h2 class="chakra-heading css-bfy4z6" data-testid="ds-heading">'
            'Image ID</h2><p class="chakra-text css-1xdhyk6" data-testid="ds-text" id="image-id" '
            'aria-label="Image ID">57538105</p></div>'
        )
        self.assertEqual(iiif_mod._extract_nypl_image_id_from_html(html), "57538105")

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

    def test_nypl_resolve_prefers_local_override_without_page_access(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0"
        old_http_get = iiif_mod._http_get
        try:
            iiif_mod._http_get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("blocked"))
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(
            resolved,
            "https://iiif.nypl.org/iiif/3/57538105",
        )

    def test_nypl_resolve_uses_api_image_id_by_canvas_index_when_page_is_blocked(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=1"
        old_http_get = iiif_mod._http_get

        class _BlockedPageResponse:
            status_code = 403
            text = "blocked"

            @staticmethod
            def json():
                raise RuntimeError("no json")

        class _ApiResponse:
            status_code = 200
            text = '{"imageID":["57538105","57538106"]}'

            @staticmethod
            def json():
                return {"imageID": ["57538105", "57538106"]}

        def _fake_http_get(url, **kwargs):
            _ = kwargs
            if "digitalcollections.nypl.org/items/" in str(url):
                return _BlockedPageResponse()
            if "api.repo.nypl.org/api/v2/items/" in str(url):
                return _ApiResponse()
            raise RuntimeError(f"unexpected url: {url}")

        try:
            iiif_mod._http_get = _fake_http_get
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/57538106")

    def test_nypl_resolve_tries_rp_host_when_primary_page_is_blocked(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0"
        old_http_get = iiif_mod._http_get

        class _BlockedResponse:
            status_code = 403
            text = "blocked"

            @staticmethod
            def json():
                raise RuntimeError("no json")

        class _RpResponse:
            status_code = 200
            text = '<html><body><div>Image ID</div><div>57538105</div></body></html>'

            @staticmethod
            def json():
                raise RuntimeError("no json")

        def _fake_http_get(url, **kwargs):
            _ = kwargs
            url_text = str(url)
            if url_text.startswith("https://digitalcollections.nypl.org/items/"):
                return _BlockedResponse()
            if url_text.startswith("https://rp-digitalcollections.nypl.org/items/"):
                return _RpResponse()
            raise RuntimeError(f"unexpected url: {url_text}")

        try:
            iiif_mod._http_get = _fake_http_get
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/57538105")

    def test_nypl_resolve_uses_items_json_when_api_repo_is_unavailable(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0"
        old_http_get = iiif_mod._http_get

        class _BlockedResponse:
            status_code = 403
            text = "blocked"

            @staticmethod
            def json():
                raise RuntimeError("no json")

        class _RepoUnauthorized:
            status_code = 401
            text = "unauthorized"

            @staticmethod
            def json():
                raise RuntimeError("unauthorized")

        class _ItemsJsonResponse:
            status_code = 200
            text = '{"item":{"imageID":["57538105"]}}'

            @staticmethod
            def json():
                return {"item": {"imageID": ["57538105"]}}

        def _fake_http_get(url, **kwargs):
            _ = kwargs
            url_text = str(url)
            if url_text.startswith("https://digitalcollections.nypl.org/items/") and url_text.endswith(".json"):
                return _ItemsJsonResponse()
            if url_text.startswith("https://digitalcollections.nypl.org/items/"):
                return _BlockedResponse()
            if url_text.startswith("https://rp-digitalcollections.nypl.org/items/"):
                return _BlockedResponse()
            if url_text.startswith("https://api.repo.nypl.org/api/v2/items/"):
                return _RepoUnauthorized()
            raise RuntimeError(f"unexpected url: {url_text}")

        try:
            iiif_mod._http_get = _fake_http_get
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/57538105")

    def test_nypl_resolve_uses_local_override_when_all_network_paths_fail(self):
        iiif_mod = importlib.import_module("ComfyUI_ALEXZ_tools.nodes.image_download_iiif")
        site = "The New York Public Library (NYPL) Digital Collections"
        source_url = "https://digitalcollections.nypl.org/items/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e?canvasIndex=0"
        old_http_get = iiif_mod._http_get
        try:
            iiif_mod._http_get = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("blocked"))
            resolved = iiif_mod._resolve_iiif_service_url(site, source_url, timeout=1.0, session=None)
        finally:
            iiif_mod._http_get = old_http_get
        self.assertEqual(resolved, "https://iiif.nypl.org/iiif/3/57538105")

    def test_nypl_download_retries_with_tile_assembly_when_max_returns_403(self):
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
            if size_spec == "max":
                raise iiif_mod._IIIFImageRequestError("forbidden", last_status=403)
            raise RuntimeError("unexpected second single-request download")

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

        self.assertEqual(calls, ["max"])
        self.assertEqual(assembled_calls, ["called"])
        self.assertEqual(tuple(image.shape), (1, 8, 10, 3))
        self.assertIn('"mode": "tile_assemble_full"', info_json)

if __name__ == "__main__":
    unittest.main()
