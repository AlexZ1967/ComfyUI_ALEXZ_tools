"""
Module: utils/module_browser_api/routes.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    PromptServer/aiohttp route registration for Module Node Picker backend.

Purpose:
    Centralize HTTP route wiring outside of
    `utils/module_node_browser_api.py` while preserving route contracts.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any


def register_routes(
    *,
    PromptServer: Any,
    web: Any,
    api_module: ModuleType,
    logger: Any,
) -> bool:
    """Register module-browser API routes on PromptServer instance once."""
    if PromptServer is None or web is None or getattr(PromptServer, "instance", None) is None:
        return False
    if bool(getattr(api_module, "_ROUTES_REGISTERED", False)):
        return False

    routes = PromptServer.instance.routes

    @routes.post(api_module.ROUTE_MODULE_REFRESH)
    async def alexz_tools_module_refresh(request):
        """API route that starts asynchronous module status refresh."""
        try:
            sync_raw = (request.query.get("sync_upstreams", "") or "").strip().lower()
            payload = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            if not sync_raw and isinstance(payload, dict):
                sync_raw = str(payload.get("sync_upstreams", "1") or "1").strip().lower()
            if not sync_raw:
                sync_raw = "1"
            do_sync = sync_raw not in {"0", "false", "no", "off"}
            requested_log_mode = api_module._normalize_log_mode(payload.get("log_mode") if isinstance(payload, dict) else None)
            api_module._set_update_console_log_mode(requested_log_mode)
            return web.json_response(api_module._start_refresh_job(sync_upstreams=do_sync))
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module refresh API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_MODULE_REFRESH_STATUS)
    async def alexz_tools_module_refresh_status(request):
        """API route that returns current module-refresh job status."""
        try:
            return web.json_response({"status": "ok", "refresh": api_module._refresh_status_snapshot()})
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module refresh status API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post(api_module.ROUTE_MODULE_ACKNOWLEDGE_ALL)
    async def alexz_tools_module_acknowledge_all(request):
        """API route that clears novelty markers for all modules."""
        try:
            result = api_module._acknowledge_all_novelty()
            return web.json_response(result)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module acknowledge-all API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post(api_module.ROUTE_MODULE_UPDATE)
    async def alexz_tools_module_update(request):
        """API route that starts asynchronous module update jobs."""
        try:
            if api_module._INFO_ONLY_WIDGET_MODE:
                return web.json_response(
                    api_module._info_only_rejection_payload("module_update"),
                    status=403,
                )
            payload = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            scope = str(payload.get("scope") or request.query.get("scope") or "single").strip().lower()
            module_name = str(payload.get("module") or request.query.get("module") or "").strip()
            requested_log_mode = api_module._normalize_log_mode(payload.get("log_mode") or request.query.get("log_mode") or "summary")
            started = api_module._start_module_update_job(scope=scope, module_name=module_name, log_mode=requested_log_mode)
            if started.get("status") == "error":
                return web.json_response(started, status=400)
            return web.json_response(started)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module update API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_MODULE_UPDATE_STATUS)
    async def alexz_tools_module_update_status(request):
        """API route that returns current module-update job status."""
        try:
            return web.json_response({"status": "ok", "update": api_module._update_status_snapshot()})
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module update status API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post(api_module.ROUTE_MODULE_INSTALL_REQUIREMENTS)
    async def alexz_tools_module_install_requirements(request):
        """API route that installs Python requirements for selected modules."""
        try:
            if api_module._INFO_ONLY_WIDGET_MODE:
                return web.json_response(
                    api_module._info_only_rejection_payload("module_install_requirements"),
                    status=403,
                )
            payload = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            modules = payload.get("modules")
            result = api_module._install_requirements_for_modules(modules if isinstance(modules, list) else [])
            status_code = 200 if result.get("status") == "ok" else 400
            return web.json_response(result, status=status_code)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module requirements install API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.post(api_module.ROUTE_COMFYUI_INSTALL_REQUIREMENTS)
    async def alexz_tools_comfyui_install_requirements(request):
        """API route that installs ComfyUI requirements in the active environment."""
        try:
            if api_module._INFO_ONLY_WIDGET_MODE:
                return web.json_response(
                    api_module._info_only_rejection_payload("comfyui_install_requirements"),
                    status=403,
                )
            result = api_module._install_comfyui_requirements()
            status_code = 200 if result.get("status") == "installed" else 400
            return web.json_response(result, status=status_code)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("ComfyUI requirements install API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_COMPONENT_REGISTRY)
    async def alexz_tools_component_registry(request):
        """API route that returns extensibility registry snapshot (nodes/widgets/api)."""
        try:
            refresh_raw = (request.query.get("refresh", "0") or "0").strip().lower()
            force_refresh = refresh_raw not in {"0", "false", "no", "off"}
            payload = api_module._component_registry_payload(force_refresh=force_refresh)
            return web.json_response({"status": "ok", "registry": payload})
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Component registry API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_NODE_CATALOG)
    async def alexz_tools_node_catalog(request):
        """API route that returns grouped module and node catalog data."""
        try:
            mode = api_module._normalize_comfyui_mode(request.query.get("comfyui_mode", "") or request.query.get("mode", ""))
            payload = api_module.mb_api_build_node_catalog_payload(
                mode=mode,
                start_runtime_state_warmup=api_module._start_runtime_state_warmup,
                build_group_catalog=api_module._build_group_catalog,
                build_group_modules=api_module._build_group_modules,
                comfyui_git_status=api_module._comfyui_git_status,
                custom_update_checked_flag=api_module._custom_update_checked_flag,
                count_custom_modules_need_update=api_module._count_custom_modules_need_update,
                count_custom_modules_unknown_update=api_module._count_custom_modules_unknown_update,
                list_custom_modules_unknown_update=api_module._list_custom_modules_unknown_update,
                runtime_warmup_status=api_module._runtime_warmup_status,
                build_group_payload=api_module._build_group_payload,
            )
            return web.json_response(payload)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Node catalog API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_MODULE_INFO)
    async def alexz_tools_module_info(request):
        """API route that returns detailed information for one module."""
        group = (request.query.get("group", "") or "").strip().lower()
        module_name = (request.query.get("module", "") or "").strip()
        refresh_raw = (request.query.get("refresh", "0") or "0").strip().lower()
        sync_raw = (request.query.get("sync_upstream", "0") or "0").strip().lower()
        cache_only_raw = (request.query.get("cache_only", "1") or "1").strip().lower()
        force_refresh = refresh_raw not in {"0", "false", "no", "off"}
        sync_upstream = sync_raw not in {"0", "false", "no", "off"}
        cache_only = cache_only_raw not in {"0", "false", "no", "off"}
        if force_refresh or sync_upstream:
            cache_only = False
        if not module_name:
            return web.json_response({"error": "module is required"}, status=400)
        try:
            api_module._ensure_runtime_state_ready()
            info = api_module._resolve_module_info(
                group,
                module_name,
                force_refresh=force_refresh,
                sync_upstream=sync_upstream,
                cache_only=cache_only,
            )
            if force_refresh:
                api_module._acknowledge_module_novelty(group, module_name)
                info = api_module._resolve_module_info(
                    group,
                    module_name,
                    force_refresh=True,
                    sync_upstream=False,
                    cache_only=True,
                )
            return web.json_response({"group": group, "module": module_name, "info": info})
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_COMFYUI_INFO)
    async def alexz_tools_comfyui_info(request):
        """API route that returns ComfyUI update and version status."""
        try:
            refresh_raw = (request.query.get("refresh", "1") or "1").strip().lower()
            force_refresh = refresh_raw not in {"0", "false", "no", "off"}
            ack_raw = (request.query.get("acknowledge", "1") or "1").strip().lower()
            acknowledge = ack_raw not in {"0", "false", "no", "off"}
            mode = api_module._normalize_comfyui_mode(request.query.get("mode", ""))
            log_mode = api_module._normalize_log_mode(request.query.get("log_mode", "summary"))
            api_module._set_update_console_log_mode(log_mode)
            if force_refresh:
                api_module._refresh_console_log(
                    "ComfyUI info refresh started (mode={mode}, acknowledge={ack}, log_mode={log})".format(
                        mode=mode,
                        ack="on" if acknowledge else "off",
                        log=log_mode,
                    )
                )
                logger.info(
                    "ComfyUI info refresh requested: mode=%s acknowledge=%s",
                    mode,
                    acknowledge,
                )
            if force_refresh and acknowledge:
                api_module._acknowledge_comfyui_novelty()
            comfyui = api_module._comfyui_git_status(force_refresh=force_refresh, mode=mode)
            if force_refresh:
                api_module._refresh_console_log(
                    "ComfyUI status: update_status={status}, update_available={avail}, local={local}, remote={remote}, "
                    "behind={behind}, ahead={ahead}, mode={mode}".format(
                        status=str(comfyui.get("update_status") or "unknown"),
                        avail=bool(comfyui.get("update_available")),
                        local=str(comfyui.get("installed_commit_short") or "unknown"),
                        remote=str(comfyui.get("remote_commit_short") or "unknown"),
                        behind=str(comfyui.get("behind") if comfyui.get("behind") is not None else "-"),
                        ahead=str(comfyui.get("ahead") if comfyui.get("ahead") is not None else "-"),
                        mode=str(comfyui.get("check_mode") or mode),
                    )
                )
                api_module._refresh_console_log(
                    "ComfyUI refs: path={path}, remote={remote_name}, branch={branch}, upstream={upstream}, remote_ref={remote_ref}".format(
                        path=str(comfyui.get("path") or "-"),
                        remote_name=str(comfyui.get("remote_name") or "-"),
                        branch=str(comfyui.get("branch") or "-"),
                        upstream=str(comfyui.get("upstream") or "-"),
                        remote_ref=str(comfyui.get("remote_ref") or "-"),
                    ),
                    level="verbose",
                )
                logger.info(
                    "ComfyUI info refresh finished: update_status=%s update_available=%s local=%s remote=%s mode=%s",
                    str(comfyui.get("update_status") or "unknown"),
                    bool(comfyui.get("update_available")),
                    str(comfyui.get("installed_commit_short") or "unknown"),
                    str(comfyui.get("remote_commit_short") or "unknown"),
                    str(comfyui.get("check_mode") or mode),
                )
                api_module._refresh_console_log("ComfyUI info refresh finished")
            return web.json_response({"status": "ok", "comfyui": comfyui})
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("ComfyUI info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_MODULE_LIST)
    async def alexz_tools_module_list(request):
        """API route that returns module list for the selected group."""
        query = (request.query.get("q", "") or "").strip().lower()
        try:
            payload = api_module.mb_api_build_module_list_response(
                query=query,
                build_catalog=api_module._build_catalog,
                build_module_list_payload=api_module._build_module_list_payload,
            )
            return web.json_response(payload)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module list API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @routes.get(api_module.ROUTE_MODULE_NODES)
    async def alexz_tools_module_nodes(request):
        """API route that returns node list for the selected module."""
        query = (request.query.get("module", "") or request.query.get("q", "")).strip()
        try:
            payload = api_module.mb_api_build_module_nodes_response(
                query=query,
                build_catalog=api_module._build_catalog,
                build_module_nodes_payload=api_module._build_module_nodes_payload,
            )
            return web.json_response(payload)
        except Exception as exc:  # pragma: no cover - diagnostic
            logger.error("Module browser API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    setattr(api_module, "_ROUTES_REGISTERED", True)
    logger.info("✅ Module Nodes widget backend loaded")
    return True
