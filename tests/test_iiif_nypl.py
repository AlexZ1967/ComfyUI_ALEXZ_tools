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

if __name__ == "__main__":
    unittest.main()
