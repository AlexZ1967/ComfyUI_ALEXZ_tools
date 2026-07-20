"""
Module: tests/test_module_browser_api_routes.py
Author: AlexZ1967
Last updated: 2026-03-06

Description:
    Integration-style route wiring tests for Module Node Picker backend.

Purpose:
    Validates critical HTTP route behavior after routes-module extraction.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest


class _DummyResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = int(status)


class _DummyWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _DummyResponse(payload, status=status)


class _DummyRoutes:
    def __init__(self):
        self.handlers = {}

    def post(self, path):
        def _decorator(fn):
            self.handlers[("POST", str(path))] = fn
            return fn

        return _decorator

    def get(self, path):
        def _decorator(fn):
            self.handlers[("GET", str(path))] = fn
            return fn

        return _decorator


class _DummyPromptServer:
    class _Instance:
        def __init__(self):
            self.routes = _DummyRoutes()

    instance = _Instance()


class _DummyLogger:
    def __init__(self):
        self.info_lines = []
        self.error_lines = []

    def info(self, message, *args, **kwargs):
        text = str(message)
        if args:
            text = text % args
        self.info_lines.append(text)

    def error(self, message, *args, **kwargs):
        text = str(message)
        if args:
            text = text % args
        self.error_lines.append(text)


class _DummyRequest:
    def __init__(self, *, query=None, payload=None, json_raises=False):
        self.query = dict(query or {})
        self._payload = payload
        self._json_raises = bool(json_raises)

    async def json(self):
        if self._json_raises:
            raise ValueError("invalid json")
        return self._payload


def _build_api_stub():
    mod = types.ModuleType("api_stub")
    mod._ROUTES_REGISTERED = False

    mod.ROUTE_MODULE_REFRESH = "/alexz_tools/module_refresh"
    mod.ROUTE_MODULE_REFRESH_STATUS = "/alexz_tools/module_refresh_status"
    mod.ROUTE_MODULE_ACKNOWLEDGE_ALL = "/alexz_tools/module_acknowledge_all"
    mod.ROUTE_MODULE_UPDATE = "/alexz_tools/module_update"
    mod.ROUTE_MODULE_UPDATE_STATUS = "/alexz_tools/module_update_status"
    mod.ROUTE_MODULE_INSTALL_REQUIREMENTS = "/alexz_tools/module_install_requirements"
    mod.ROUTE_COMFYUI_INSTALL_REQUIREMENTS = "/alexz_tools/comfyui_install_requirements"
    mod.ROUTE_COMPONENT_REGISTRY = "/alexz_tools/component_registry"
    mod.ROUTE_NODE_CATALOG = "/alexz_tools/node_catalog"
    mod.ROUTE_MODULE_INFO = "/alexz_tools/module_info"
    mod.ROUTE_COMFYUI_INFO = "/alexz_tools/comfyui_info"
    mod.ROUTE_MODULE_LIST = "/alexz_tools/module_list"
    mod.ROUTE_MODULE_NODES = "/alexz_tools/module_nodes"

    mod._INFO_ONLY_WIDGET_MODE = False

    mod._normalize_log_mode = lambda value: str(value or "summary").strip().lower() or "summary"
    mod._normalize_comfyui_mode = lambda value: str(value or "fast").strip().lower() or "fast"
    mod._set_update_console_log_mode = lambda _mode: None
    mod._refresh_console_log = lambda _text, level="summary": None
    mod._start_refresh_job = lambda sync_upstreams=True: {"status": "started", "sync_upstreams": bool(sync_upstreams)}
    mod._refresh_status_snapshot = lambda: {"running": False}
    mod._acknowledge_all_novelty = lambda: {"status": "ok"}
    mod._start_module_update_job = (
        lambda scope="single", module_name="", log_mode="summary": {
            "status": "started",
            "scope": scope,
            "module": module_name,
            "log_mode": log_mode,
        }
    )
    mod._update_status_snapshot = lambda: {"running": False}
    mod._info_only_rejection_payload = lambda action: {"status": "info_only", "action": action}
    mod._requirements_advisory_for_modules = (
        lambda modules: {
            "status": "advisory",
            "modules": list(modules or []),
            "commands": ["python -m pip install -r /tmp/mod/requirements.txt"],
        }
    )
    mod._comfyui_requirements_advisory = (
        lambda: {
            "status": "advisory",
            "module": "ComfyUI",
            "commands": ["python -m pip install -r /tmp/ComfyUI/requirements.txt"],
        }
    )
    mod._component_registry_payload = lambda force_refresh=False: {"force_refresh": bool(force_refresh)}
    mod._start_runtime_state_warmup = lambda: None
    mod._build_group_catalog = lambda: []
    mod._build_group_modules = lambda _group: []
    mod._comfyui_git_status = lambda force_refresh=False, mode="fast": {
        "update_status": "unknown",
        "update_available": False,
        "check_mode": mode,
    }
    mod._custom_update_checked_flag = lambda _state=None: False
    mod._count_custom_modules_need_update = lambda: 0
    mod._count_custom_modules_unknown_update = lambda: 0
    mod._list_custom_modules_unknown_update = lambda: []
    mod._runtime_warmup_status = lambda: {"running": False}
    mod._build_group_payload = lambda **kwargs: kwargs
    mod.mb_api_build_node_catalog_payload = lambda **kwargs: {"status": "ok", "args": kwargs}
    mod._ensure_runtime_state_ready = lambda: None
    mod._resolve_module_info = (
        lambda group, module_name, force_refresh=False, sync_upstream=False, cache_only=True: {
            "group": group,
            "module": module_name,
            "force_refresh": bool(force_refresh),
            "sync_upstream": bool(sync_upstream),
            "cache_only": bool(cache_only),
        }
    )
    mod._acknowledge_module_novelty = lambda _group, _module_name: None
    mod._acknowledge_comfyui_novelty = lambda: None
    mod.mb_api_build_module_list_response = lambda **kwargs: {"status": "ok", "query": kwargs.get("query", "")}
    mod.mb_api_build_module_nodes_response = lambda **kwargs: {"status": "ok", "query": kwargs.get("query", "")}
    mod._build_catalog = lambda: {}
    mod._build_module_list_payload = lambda *args, **kwargs: {}
    mod._build_module_nodes_payload = lambda *args, **kwargs: {}
    return mod


class ModuleBrowserApiRoutesTests(unittest.TestCase):
    """Verify critical route behavior in extracted routes module."""

    @classmethod
    def setUpClass(cls):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        if "ComfyUI_ALEXZ_tools" not in sys.modules:
            pkg = types.ModuleType("ComfyUI_ALEXZ_tools")
            pkg.__path__ = [repo_root]
            sys.modules["ComfyUI_ALEXZ_tools"] = pkg

    def setUp(self):
        from ComfyUI_ALEXZ_tools.utils.module_browser_api.routes import register_routes

        self.register_routes = register_routes
        self.logger = _DummyLogger()
        _DummyPromptServer.instance = _DummyPromptServer._Instance()
        self.api = _build_api_stub()

    def _handler(self, method: str, path: str):
        return _DummyPromptServer.instance.routes.handlers[(method, path)]

    def test_register_routes_is_idempotent(self):
        """Routes should register once and skip repeated call."""
        first = self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        second = self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(bool(getattr(self.api, "_ROUTES_REGISTERED", False)))

    def test_register_routes_wires_expected_route_set(self):
        """Route registration should publish the full expected HTTP surface."""
        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        self.assertEqual(
            set(_DummyPromptServer.instance.routes.handlers.keys()),
            {
                ("POST", self.api.ROUTE_MODULE_REFRESH),
                ("GET", self.api.ROUTE_MODULE_REFRESH_STATUS),
                ("POST", self.api.ROUTE_MODULE_ACKNOWLEDGE_ALL),
                ("POST", self.api.ROUTE_MODULE_UPDATE),
                ("GET", self.api.ROUTE_MODULE_UPDATE_STATUS),
                ("POST", self.api.ROUTE_MODULE_INSTALL_REQUIREMENTS),
                ("POST", self.api.ROUTE_COMFYUI_INSTALL_REQUIREMENTS),
                ("GET", self.api.ROUTE_COMPONENT_REGISTRY),
                ("GET", self.api.ROUTE_NODE_CATALOG),
                ("GET", self.api.ROUTE_MODULE_INFO),
                ("GET", self.api.ROUTE_COMFYUI_INFO),
                ("GET", self.api.ROUTE_MODULE_LIST),
                ("GET", self.api.ROUTE_MODULE_NODES),
            },
        )

    def test_module_refresh_uses_payload_sync_and_log_mode(self):
        """Refresh route should parse payload sync flag and pass normalized log mode."""
        observed = {"mode": None, "sync": None}
        self.api._normalize_log_mode = lambda value: str(value or "summary").strip().lower()
        self.api._set_update_console_log_mode = lambda mode: observed.__setitem__("mode", mode)
        self.api._start_refresh_job = lambda sync_upstreams=True: {
            "status": "started",
            "sync_upstreams": bool(sync_upstreams),
        }

        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        req = _DummyRequest(payload={"sync_upstreams": "0", "log_mode": "VERBOSE"})
        handler = self._handler("POST", self.api.ROUTE_MODULE_REFRESH)
        response = asyncio.run(handler(req))

        self.assertEqual(response.status, 200)
        self.assertEqual(observed["mode"], "verbose")
        self.assertFalse(bool(response.payload.get("sync_upstreams")))

    def test_module_refresh_malformed_json_falls_back_to_query_flags(self):
        """Refresh route should stay operational when request JSON cannot be parsed."""
        observed = {"mode": None}
        self.api._normalize_log_mode = lambda value: str(value or "summary").strip().lower()
        self.api._set_update_console_log_mode = lambda mode: observed.__setitem__("mode", mode)

        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        req = _DummyRequest(query={"sync_upstreams": "0"}, json_raises=True)
        handler = self._handler("POST", self.api.ROUTE_MODULE_REFRESH)
        response = asyncio.run(handler(req))

        self.assertEqual(response.status, 200)
        self.assertFalse(bool(response.payload.get("sync_upstreams")))
        self.assertEqual(observed["mode"], "summary")

    def test_module_update_info_only_returns_403(self):
        """Info-only mode should reject module update route."""
        self.api._INFO_ONLY_WIDGET_MODE = True
        self.api._info_only_rejection_payload = lambda action: {"status": "info_only", "action": action}
        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        handler = self._handler("POST", self.api.ROUTE_MODULE_UPDATE)
        response = asyncio.run(handler(_DummyRequest(payload={"scope": "all"})))
        self.assertEqual(response.status, 403)
        self.assertEqual(response.payload.get("action"), "module_update")

    def test_module_update_malformed_json_uses_query_fallback(self):
        """Update route should use query params when JSON payload is malformed."""
        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        handler = self._handler("POST", self.api.ROUTE_MODULE_UPDATE)
        response = asyncio.run(
            handler(
                _DummyRequest(
                    query={"scope": "all", "module": "modA", "log_mode": "verbose"},
                    json_raises=True,
                )
            )
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("scope"), "all")
        self.assertEqual(response.payload.get("module"), "modA")
        self.assertEqual(response.payload.get("log_mode"), "verbose")

    def test_module_requirements_route_returns_manual_advisory(self):
        """Requirements route should return manual-install advisory without pip execution."""
        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        handler = self._handler("POST", self.api.ROUTE_MODULE_INSTALL_REQUIREMENTS)
        response = asyncio.run(handler(_DummyRequest(payload={"modules": ["modA"]})))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "advisory")
        self.assertEqual(response.payload.get("modules"), ["modA"])

    def test_comfyui_requirements_route_returns_manual_advisory(self):
        """ComfyUI requirements route should return manual-install advisory without pip execution."""
        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        handler = self._handler("POST", self.api.ROUTE_COMFYUI_INSTALL_REQUIREMENTS)
        response = asyncio.run(handler(_DummyRequest()))
        self.assertEqual(response.status, 200)
        self.assertEqual(response.payload.get("status"), "advisory")
        self.assertEqual(response.payload.get("module"), "ComfyUI")

    def test_module_info_requires_module_query(self):
        """Module info route should return 400 when module query is missing."""
        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        handler = self._handler("GET", self.api.ROUTE_MODULE_INFO)
        response = asyncio.run(handler(_DummyRequest(query={"group": "custom"})))
        self.assertEqual(response.status, 400)
        self.assertIn("error", response.payload)

    def test_comfyui_info_acknowledge_flow(self):
        """ComfyUI info route should call acknowledge when force refresh is requested."""
        observed = {"ack": 0, "mode": None, "log_mode": None}
        self.api._normalize_comfyui_mode = lambda value: f"mode:{value or 'fast'}"
        self.api._normalize_log_mode = lambda value: f"log:{value or 'summary'}"
        self.api._set_update_console_log_mode = lambda mode: observed.__setitem__("log_mode", mode)
        self.api._acknowledge_comfyui_novelty = lambda: observed.__setitem__("ack", observed["ack"] + 1)
        self.api._comfyui_git_status = lambda force_refresh=False, mode="fast": {
            "update_status": "can_update",
            "update_available": True,
            "check_mode": mode,
            "force_refresh": bool(force_refresh),
        }

        self.register_routes(
            PromptServer=_DummyPromptServer,
            web=_DummyWeb,
            api_module=self.api,
            logger=self.logger,
        )
        handler = self._handler("GET", self.api.ROUTE_COMFYUI_INFO)
        response = asyncio.run(
            handler(
                _DummyRequest(
                    query={
                        "refresh": "1",
                        "acknowledge": "1",
                        "mode": "smart",
                        "log_mode": "verbose",
                    }
                )
            )
        )

        self.assertEqual(response.status, 200)
        self.assertEqual(observed["ack"], 1)
        self.assertEqual(observed["log_mode"], "log:verbose")
        self.assertEqual(response.payload.get("status"), "ok")
        self.assertEqual(response.payload.get("comfyui", {}).get("check_mode"), "mode:smart")


if __name__ == "__main__":
    unittest.main()
