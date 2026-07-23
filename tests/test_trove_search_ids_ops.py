"""Unit tests for pure Trove search helper functions."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


class TroveSearchIdsOpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            package = types.ModuleType("ComfyUI_ALEXZ_tools")
            package.__path__ = [root]
            sys.modules["ComfyUI_ALEXZ_tools"] = package

    def setUp(self):
        from ComfyUI_ALEXZ_tools.nodes import trove_search_ids_ops as ops

        self.ops = ops

    def test_trove_api_params_normalize_images_category(self):
        params = self.ops.build_trove_api_params("Anna Pavlova", category="images", max_results=250)

        self.assertEqual(params["q"], "Anna Pavlova")
        self.assertEqual(params["category"], "image")
        self.assertEqual(params["encoding"], "json")
        self.assertEqual(params["n"], 100)
        self.assertEqual(params["l-availability"], "y/f")
        self.assertEqual(self.ops.normalize_trove_api_category("picture"), "image")

    def test_trove_api_key_prefers_input_over_environment(self):
        key, source = self.ops.resolve_trove_api_key(" explicit ", {"TROVE_API_KEY": "env"})
        self.assertEqual(key, "explicit")
        self.assertEqual(source, "input")

        key, source = self.ops.resolve_trove_api_key("", {"TROVE_API_KEY": " env "})
        self.assertEqual(key, "env")
        self.assertEqual(source, "env")

        key, source = self.ops.resolve_trove_api_key("", {})
        self.assertEqual(key, "")
        self.assertEqual(source, "anonymous")

    def test_trove_extracts_ids_from_nested_api_payload(self):
        payload = {
            "category": [
                {
                    "records": {
                        "work": [
                            {
                                "identifier": ["https://nla.gov.au/nla.obj-138204672"],
                                "troveUrl": "https://nla.gov.au/nla.obj-162204874",
                            },
                            {
                                "title": "Duplicate nla.obj-138204672",
                                "snippet": "extra NLA.OBJ-150139367",
                            },
                        ]
                    }
                }
            ]
        }

        self.assertEqual(
            self.ops.extract_nla_obj_ids_from_api_payload(payload),
            [
                "nla.obj-138204672",
                "nla.obj-162204874",
                "nla.obj-150139367",
            ],
        )

    def test_trove_result_sanitizer_redacts_api_key_headers(self):
        sanitized = self.ops.sanitize_trove_result(
            {
                "api_key": "secret",
                "request_headers": {"X-API-KEY": "secret", "User-Agent": "test"},
            }
        )

        self.assertNotIn("api_key", sanitized)
        self.assertEqual(sanitized["request_headers"]["X-API-KEY"], "***")
        self.assertEqual(sanitized["request_headers"]["User-Agent"], "test")

    def test_trove_api_search_uses_header_key_and_extracts_ids(self):
        from ComfyUI_ALEXZ_tools.nodes import trove_search_ids as node_mod

        class FakeResponse:
            status_code = 200
            reason = "OK"
            text = ""

            def json(self):
                return {
                    "category": [
                        {
                            "records": {
                                "total": 2,
                                "work": [
                                    {"troveUrl": "https://nla.gov.au/nla.obj-138204672"},
                                    {"identifier": ["https://nla.gov.au/nla.obj-162204874"]},
                                ],
                            }
                        }
                    ]
                }

        class FakeSession:
            def __init__(self):
                self.calls = []

            def get(self, url, *, params, headers, timeout):
                self.calls.append((url, params, headers, timeout))
                return FakeResponse()

        session = FakeSession()
        result = node_mod._search_trove_ids_via_api(
            "Pavlova",
            category="images",
            api_key="secret",
            max_results=5,
            session=session,
        )

        self.assertEqual(result["ids"], ["nla.obj-138204672", "nla.obj-162204874"])
        self.assertEqual(result["api_key_source"], "input")
        self.assertEqual(result["api_category"], "image")
        self.assertEqual(session.calls[0][0], self.ops.TROVE_API_RESULT_URL)
        self.assertEqual(session.calls[0][2], {"X-API-KEY": "secret"})
        self.assertNotIn("secret", self.ops.sanitize_trove_result(result).get("warning", ""))

    def test_trove_api_search_without_key_uses_anonymous_request(self):
        from ComfyUI_ALEXZ_tools.nodes import trove_search_ids as node_mod

        class FakeResponse:
            status_code = 200
            reason = "OK"
            text = ""

            def json(self):
                return {
                    "category": [
                        {
                            "records": {
                                "total": 1,
                                "work": [{"troveUrl": "https://nla.gov.au/nla.obj-138204672"}],
                            }
                        }
                    ]
                }

        class FakeSession:
            def __init__(self):
                self.calls = []

            def get(self, url, *, params, headers, timeout):
                self.calls.append((url, params, headers, timeout))
                return FakeResponse()

        session = FakeSession()
        result = node_mod._search_trove_ids_via_api("Pavlova", session=session)

        self.assertEqual(result["ids"], ["nla.obj-138204672"])
        self.assertEqual(result["api_key_source"], "anonymous")
        self.assertEqual(session.calls[0][2], {})

    def test_trove_anonymous_rate_limit_has_actionable_diagnostic(self):
        from ComfyUI_ALEXZ_tools.nodes import trove_search_ids as node_mod

        class FakeResponse:
            status_code = 429
            reason = "Too Many Requests"
            text = "rate limit"

        class FakeSession:
            def get(self, url, *, params, headers, timeout):
                return FakeResponse()

        result = node_mod._search_trove_ids_via_api("Pavlova", session=FakeSession())

        self.assertEqual(result["status_code"], 429)
        self.assertEqual(result["api_key_source"], "anonymous")
        self.assertIn("TROVE_API_KEY", result["diagnostic"]["hint"])

    def test_trove_anonymous_unauthorized_has_actionable_diagnostic(self):
        from ComfyUI_ALEXZ_tools.nodes import trove_search_ids as node_mod

        class FakeResponse:
            status_code = 401
            reason = "Unauthorized"
            text = '{"message":"No API key found in request"}'

        class FakeSession:
            def get(self, url, *, params, headers, timeout):
                self.headers = headers
                return FakeResponse()

        session = FakeSession()
        result = node_mod._search_trove_ids_via_api("Pavlova", session=session)

        self.assertEqual(session.headers, {})
        self.assertEqual(result["status_code"], 401)
        self.assertEqual(result["api_key_source"], "anonymous")
        self.assertIn("Anonymous access was rejected", result["diagnostic"]["hint"])
        self.assertIn("No API key found", result["diagnostic"]["detail"])

    def test_trove_browser_failure_keeps_json_compact_and_writes_verbose_log(self):
        from ComfyUI_ALEXZ_tools.nodes import trove_search_ids as node_mod

        class FakeProcess:
            returncode = 0
            stdout = "<!DOCTYPE html><html><head><title>Trove</title></head><body>App shell</body></html>"
            stderr = "ssl_client_socket_impl handshake failed"

        old_find_chrome = node_mod._find_chrome_binary
        old_run = node_mod.subprocess.run
        old_log_path = node_mod._TROVE_DIAGNOSTIC_LOG_PATH
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                node_mod._find_chrome_binary = lambda: "/usr/bin/google-chrome"
                node_mod.subprocess.run = lambda *args, **kwargs: FakeProcess()
                node_mod._TROVE_DIAGNOSTIC_LOG_PATH = Path(tmpdir) / "trove.log"

                result = node_mod._search_trove_ids_via_chrome("Pavlova")

                self.assertEqual(result["page_state"], "page_shell_only")
                self.assertIn("page shell loaded", result["warning"])
                self.assertEqual(result["diagnostic_log"], str(node_mod._TROVE_DIAGNOSTIC_LOG_PATH))
                self.assertNotIn("stdout_excerpt", result)
                self.assertNotIn("stderr_excerpt", result)
                contents = node_mod._TROVE_DIAGNOSTIC_LOG_PATH.read_text(encoding="utf-8")
                self.assertIn("=== STDOUT ===", contents)
                self.assertIn("App shell", contents)
                self.assertIn("=== STDERR ===", contents)
                self.assertIn("handshake failed", contents)
        finally:
            node_mod._find_chrome_binary = old_find_chrome
            node_mod.subprocess.run = old_run
            node_mod._TROVE_DIAGNOSTIC_LOG_PATH = old_log_path
