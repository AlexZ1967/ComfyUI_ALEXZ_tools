"""
Module: tests/test_module_browser_api_manager_cache_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Unit tests for extracted module-browser API manager/cache helpers.

Purpose:
    Verifies alias-cache reuse, PromptServer URL normalization, and manager
    update-override probe semantics moved out of `utils/module_node_browser_api.py`.
"""

from __future__ import annotations

import os
import sys
import types
import unittest


class ModuleBrowserApiManagerCacheOpsTests(unittest.TestCase):
    """Verify behavior of `utils.module_browser_api.manager_cache_ops` helpers."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api import manager_cache_ops

        self.ops = manager_cache_ops

    def test_custom_module_aliases_cache_reuses_existing_mapping(self):
        """Alias-cache helper should avoid rebuilding mappings when cache exists."""
        cached = {"moda": "ModA"}
        result = self.ops.custom_module_aliases_cache(
            cached,
            discover_custom_modules=lambda: (_ for _ in ()).throw(AssertionError("discover not expected")),
            normalize_token=lambda text: str(text).lower(),
            build_custom_module_aliases=lambda **kwargs: (_ for _ in ()).throw(AssertionError("build not expected")),
        )
        self.assertIs(result, cached)

    def test_promptserver_base_url_normalizes_wildcard_bind(self):
        """PromptServer URL helper should map wildcard bind to localhost."""
        server = types.SimpleNamespace(address="0.0.0.0", port=8188)
        prompt_server = types.SimpleNamespace(instance=server)
        self.assertEqual(
            self.ops.promptserver_base_url(prompt_server),
            "http://127.0.0.1:8188",
        )

    def test_manager_installed_update_overrides_respects_ttl_cache(self):
        """Override helper should reuse fresh cached payload without probing endpoints."""
        cached = (100.0, {"ModA": True})
        result, next_cache = self.ops.manager_installed_update_overrides(
            cache=cached,
            now_ts=105.0,
            ttl_sec=20.0,
            force_refresh=False,
            promptserver_base_url_fn=lambda: (_ for _ in ()).throw(AssertionError("base url probe not expected")),
            http_json_get_fn=lambda url, timeout: (_ for _ in ()).throw(AssertionError("http probe not expected")),
            normalize_repo_url=lambda value: value,
            github_id=lambda value: value,
            repo_name=lambda value: value,
            logger_debug=lambda *args, **kwargs: None,
        )
        self.assertEqual(result, {"ModA": True})
        self.assertEqual(next_cache, cached)

    def test_manager_installed_update_overrides_builds_override_map(self):
        """Override helper should derive module override flags from manager payloads."""
        payloads = {
            "http://127.0.0.1:8188/customnode/installed?mode=default": {
                "ModA": {"enabled": True, "cnr_id": "comfyui_alexz_tools", "aux_id": "alexz1967/ComfyUI_ALEXZ_tools"},
                "ModB": {"enabled": True, "cnr_id": "other", "aux_id": "example/other"},
            },
            "http://127.0.0.1:8188/customnode/getlist?mode=local&skip_update=false": {
                "node_packs": {
                    "ComfyUI_ALEXZ_tools": {
                        "id": "comfyui_alexz_tools",
                        "repository": "https://github.com/alexz1967/ComfyUI_ALEXZ_tools.git",
                        "update-state": "true",
                    },
                    "OtherNode": {
                        "id": "other",
                        "repository": "https://github.com/example/other.git",
                        "update-state": "false",
                    },
                }
            },
        }

        def _normalize_repo_url(url):
            text = str(url or "").strip()
            if text.endswith(".git"):
                text = text[:-4]
            return text or None

        def _github_id(url):
            text = str(url or "").strip().rstrip("/")
            marker = "github.com/"
            idx = text.lower().find(marker)
            if idx < 0:
                return None
            tail = text[idx + len(marker) :]
            parts = [part for part in tail.split("/") if part]
            if len(parts) < 2:
                return None
            return f"{parts[0].lower()}/{parts[1].lower()}"

        result, next_cache = self.ops.manager_installed_update_overrides(
            cache=None,
            now_ts=200.0,
            ttl_sec=20.0,
            force_refresh=True,
            promptserver_base_url_fn=lambda: "http://127.0.0.1:8188",
            http_json_get_fn=lambda url, timeout: payloads[url],
            normalize_repo_url=_normalize_repo_url,
            github_id=_github_id,
            repo_name=lambda url: str(_github_id(url) or "").split("/", 1)[1] if _github_id(url) else None,
            logger_debug=lambda *args, **kwargs: None,
        )

        self.assertEqual(result, {"ModA": True})
        self.assertEqual(next_cache, (200.0, {"ModA": True}))


if __name__ == "__main__":
    unittest.main()
