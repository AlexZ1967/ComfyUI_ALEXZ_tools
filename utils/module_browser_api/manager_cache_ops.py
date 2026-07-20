"""
Module: utils/module_browser_api/manager_cache_ops.py
Author: AlexZ1967
Last updated: 2026-07-20

Description:
    Manager metadata/cache helpers for module browser API facade.

Purpose:
    Keeps manager-backed cache loading, PromptServer probing, and installed
    update-override collection out of `utils/module_node_browser_api.py`.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from typing import Any, Callable


def custom_module_aliases_cache(
    cache: dict[str, str] | None,
    *,
    discover_custom_modules: Callable[[], list[str]],
    normalize_token: Callable[[str], str],
    build_custom_module_aliases: Callable[..., dict[str, str]],
) -> dict[str, str]:
    """Build custom-module alias cache once and reuse it across facade calls."""
    if cache is not None:
        return cache
    return build_custom_module_aliases(
        discovered_modules=discover_custom_modules(),
        normalize_token=normalize_token,
    )


def manager_github_stats_cache(
    cache: dict[str, dict[str, dict[str, Any]]] | None,
    *,
    load_manager_github_stats: Callable[..., dict[str, dict[str, dict[str, Any]]]],
    manager_github_stats_path: Callable[[], Any],
    normalize_repo_url: Callable[[str | None], str | None],
    github_id: Callable[[str | None], str | None],
    logger_warning: Callable[..., None],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load manager GitHub stats once and return the normalized cache payload."""
    return load_manager_github_stats(
        cache=cache,
        manager_github_stats_path=manager_github_stats_path,
        normalize_repo_url=normalize_repo_url,
        github_id=github_id,
        logger_warning=logger_warning,
    )


def manager_index_cache(
    cache: dict[str, dict[str, dict[str, Any]]] | None,
    *,
    load_manager_index: Callable[..., dict[str, dict[str, dict[str, Any]]]],
    manager_custom_db_path: Callable[[], Any],
    pick_repo_url: Callable[[dict[str, Any]], str | None],
    github_id: Callable[[str | None], str | None],
    repo_name: Callable[[str | None], str | None],
    logger_warning: Callable[..., None],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load manager custom-node index once and return the normalized cache payload."""
    return load_manager_index(
        cache=cache,
        manager_custom_db_path=manager_custom_db_path,
        pick_repo_url=pick_repo_url,
        github_id=github_id,
        repo_name=repo_name,
        logger_warning=logger_warning,
    )


def promptserver_base_url(PromptServer: Any) -> str | None:
    """Resolve PromptServer base URL for local in-process API probing."""
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return None
    server = PromptServer.instance
    address = str(getattr(server, "address", "127.0.0.1") or "127.0.0.1").strip()
    if address in {"", "0.0.0.0", "::"}:
        address = "127.0.0.1"
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    port = int(getattr(server, "port", 8188) or 8188)
    return f"http://{address}:{port}"


def http_json_get(
    url: str,
    timeout: float = 20.0,
    *,
    urlopen_fn: Callable[..., Any] = urlopen,
    json_loads: Callable[[str], Any] = json.loads,
) -> dict[str, Any]:
    """Load JSON payload from local HTTP endpoint with strict timeouts."""
    with urlopen_fn(url, timeout=max(1.0, float(timeout))) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json_loads(raw)
    return payload if isinstance(payload, dict) else {}


def manager_installed_update_overrides(
    *,
    cache: tuple[float, dict[str, bool]] | None,
    now_ts: float,
    ttl_sec: float,
    force_refresh: bool,
    promptserver_base_url_fn: Callable[[], str | None],
    http_json_get_fn: Callable[[str, float], dict[str, Any]],
    normalize_repo_url: Callable[[str | None], str | None],
    github_id: Callable[[str | None], str | None],
    repo_name: Callable[[str | None], str | None],
    logger_debug: Callable[..., None],
) -> tuple[dict[str, bool], tuple[float, dict[str, bool]] | None]:
    """Collect installed-module update overrides reported by ComfyUI-Manager."""
    if not force_refresh and cache is not None:
        cached_ts, cached_payload = cache
        if (now_ts - cached_ts) < ttl_sec:
            return (dict(cached_payload), cache)

    base_url = promptserver_base_url_fn()
    if not base_url:
        return ({}, cache)

    try:
        installed_payload = http_json_get_fn(f"{base_url}/customnode/installed?mode=default", 20.0)
        list_payload = http_json_get_fn(f"{base_url}/customnode/getlist?mode=local&skip_update=false", 90.0)
    except (TimeoutError, URLError, HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger_debug("ComfyUI-Manager update override probe failed: %s", exc)
        updated_cache = (now_ts, {})
        return ({}, updated_cache)

    installed = installed_payload if isinstance(installed_payload, dict) else {}
    node_packs = list_payload.get("node_packs") if isinstance(list_payload, dict) else {}
    node_packs = node_packs if isinstance(node_packs, dict) else {}

    by_id: dict[str, dict[str, Any]] = {}
    by_github: dict[str, dict[str, Any]] = {}
    by_repo_name: dict[str, dict[str, Any]] = {}
    for pack_key, raw_meta in node_packs.items():
        if not isinstance(raw_meta, dict):
            continue
        meta = raw_meta
        id_candidates = {
            str(meta.get("id") or "").strip().lower(),
            str(pack_key or "").strip().lower(),
        }
        for candidate in id_candidates:
            if candidate:
                by_id[candidate] = meta

        repo_sources = [
            str(meta.get("repository") or "").strip(),
            str(meta.get("reference") or "").strip(),
        ]
        files = meta.get("files")
        if isinstance(files, list):
            for item in files:
                text = str(item or "").strip()
                if text:
                    repo_sources.append(text)
        for source in repo_sources:
            repo_norm = normalize_repo_url(source)
            if not repo_norm:
                continue
            gid = str(github_id(repo_norm) or "").lower()
            if gid:
                by_github[gid] = meta
            repo_short = str(repo_name(repo_norm) or "").lower()
            if repo_short:
                by_repo_name[repo_short] = meta

    overrides: dict[str, bool] = {}
    for module_name, raw_meta in installed.items():
        if not isinstance(raw_meta, dict):
            continue
        if not bool(raw_meta.get("enabled")):
            continue
        cnr_id = str(raw_meta.get("cnr_id") or "").strip().lower()
        aux_id = str(raw_meta.get("aux_id") or "").strip().lower().strip("/")
        module_l = str(module_name or "").strip().lower()

        matched_meta = None
        for candidate in (cnr_id, aux_id, module_l):
            if candidate and candidate in by_id:
                matched_meta = by_id[candidate]
                break
        if matched_meta is None and "/" in aux_id and aux_id in by_github:
            matched_meta = by_github[aux_id]
        if matched_meta is None and "/" in aux_id:
            aux_repo = aux_id.split("/", 1)[1].strip().lower()
            if aux_repo and aux_repo in by_repo_name:
                matched_meta = by_repo_name[aux_repo]

        if not isinstance(matched_meta, dict):
            continue
        update_state = str(matched_meta.get("update-state") or "").strip().lower()
        if update_state == "true":
            overrides[str(module_name)] = True

    updated_cache = (now_ts, dict(overrides))
    return (overrides, updated_cache)
