"""
Test NYPL IIIF URL resolution in ImageDownloadIIIFImage node.
"""

import importlib
import unittest

class TestNYPLResolution(unittest.TestCase):
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

    def test_nypl_resolve_falls_back_to_item_token_without_page_access(self):
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
            "https://iiif.nypl.org/iiif/3/e4c3c3e0-71a8-0136-e6bf-134f659bcb2e",
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

if __name__ == "__main__":
    unittest.main()
