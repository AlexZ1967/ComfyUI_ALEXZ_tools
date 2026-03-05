"""
Module: utils/module_browser_api/handlers_catalog.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Catalog/list/nodes payload handlers for module browser API facade.

Purpose:
    Move route payload assembly for catalog endpoints out of
    `utils/module_node_browser_api.py` while preserving response contracts.
"""

from __future__ import annotations

from typing import Any, Callable


def build_node_catalog_payload(
    *,
    mode: str,
    start_runtime_state_warmup: Callable[[], None],
    build_group_catalog: Callable[[], dict[str, list[dict[str, Any]]]],
    build_group_modules: Callable[[dict[str, list[dict[str, Any]]]], dict[str, list[dict[str, Any]]]],
    comfyui_git_status: Callable[..., dict[str, Any]],
    custom_update_checked_flag: Callable[[], bool],
    count_custom_modules_need_update: Callable[[], int],
    count_custom_modules_unknown_update: Callable[[], int],
    list_custom_modules_unknown_update: Callable[[], list[str]],
    runtime_warmup_status: Callable[[], dict[str, Any]],
    build_group_payload: Callable[[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]], list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble `/node_catalog` response payload with stable key contract."""
    start_runtime_state_warmup()
    grouped = build_group_catalog()
    modules_by_group = build_group_modules(grouped)
    comfyui = comfyui_git_status(mode=mode)
    show_custom_update_status = bool(custom_update_checked_flag())
    custom_modules_need_update = count_custom_modules_need_update() if show_custom_update_status else 0
    custom_modules_unknown_update = count_custom_modules_unknown_update() if show_custom_update_status else 0
    custom_modules_unknown_update_modules = (
        list_custom_modules_unknown_update() if show_custom_update_status else []
    )
    runtime_warmup = runtime_warmup_status()
    groups = build_group_payload(grouped, modules_by_group)
    return {
        "groups": groups,
        "comfyui": comfyui,
        "custom_modules_need_update": custom_modules_need_update,
        "custom_modules_unknown_update": custom_modules_unknown_update,
        "custom_modules_unknown_update_modules": custom_modules_unknown_update_modules,
        "runtime_warmup": runtime_warmup,
    }


def build_module_list_response(
    *,
    query: str,
    build_catalog: Callable[[], dict[str, list[dict[str, Any]]]],
    build_module_list_payload: Callable[[dict[str, list[dict[str, Any]]], str], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble `/module_list` response payload."""
    catalog = build_catalog()
    return build_module_list_payload(catalog, query)


def build_module_nodes_response(
    *,
    query: str,
    build_catalog: Callable[[], dict[str, list[dict[str, Any]]]],
    build_module_nodes_payload: Callable[[dict[str, list[dict[str, Any]]], str], dict[str, Any]],
) -> dict[str, Any]:
    """Assemble `/module_nodes` response payload."""
    catalog = build_catalog()
    return build_module_nodes_payload(catalog, query)
