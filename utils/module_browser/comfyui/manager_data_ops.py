"""
Module: utils/module_browser/manager_data_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    ComfyUI-Manager metadata/statistics helpers for Module Node Picker backend.

Purpose:
    Extracts manager index/stats loading and update-inference helpers from API
    facade while preserving cache semantics and payload behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def load_manager_github_stats(
    *,
    cache: dict[str, dict[str, dict[str, Any]]] | None,
    manager_github_stats_path: Callable[[], Path | None],
    normalize_repo_url: Callable[[str | None], str],
    github_id: Callable[[str | None], str],
    logger_warning: Callable[[str, Exception], None],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load and cache module update timestamps from manager stats file."""
    if cache is not None:
        return cache

    stats: dict[str, dict[str, dict[str, Any]]] = {"by_url": {}, "by_github": {}}
    db_path = manager_github_stats_path()
    if db_path is None:
        return stats
    try:
        with db_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger_warning("Failed to load ComfyUI-Manager github stats: %s", exc)
        return stats

    if not isinstance(payload, dict):
        return stats
    for raw_url, raw_meta in payload.items():
        if not isinstance(raw_meta, dict):
            continue
        url_text = str(raw_url).strip().replace("htps://", "https://")
        norm_url = normalize_repo_url(url_text)
        if not norm_url:
            continue
        stats["by_url"][norm_url] = raw_meta
        gid = github_id(norm_url)
        if gid:
            stats["by_github"][gid] = raw_meta
    return stats


def load_manager_index(
    *,
    cache: dict[str, dict[str, dict[str, Any]]] | None,
    manager_custom_db_path: Callable[[], Path | None],
    pick_repo_url: Callable[[dict[str, Any]], str | None],
    github_id: Callable[[str | None], str],
    repo_name: Callable[[str | None], str],
    logger_warning: Callable[[str, Exception], None],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load and cache manager metadata index for custom modules."""
    if cache is not None:
        return cache

    index: dict[str, dict[str, dict[str, Any]]] = {"by_id": {}, "by_github": {}, "by_repo_name": {}}
    db_path = manager_custom_db_path()
    if db_path is None:
        return index

    try:
        with db_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger_warning("Failed to load ComfyUI-Manager DB: %s", exc)
        return index

    entries = payload.get("custom_nodes", []) if isinstance(payload, dict) else []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        author = str(raw.get("author") or "").strip()
        description = str(raw.get("description") or "").strip()
        node_id = str(raw.get("id") or "").strip().lower()
        repo_url = pick_repo_url(raw)

        meta = {
            "title": title,
            "author": author,
            "description": description,
            "repository": repo_url,
        }
        if node_id:
            index["by_id"][node_id] = meta
        gid = github_id(repo_url)
        if gid:
            index["by_github"][gid] = meta
        repo = repo_name(repo_url)
        if repo:
            index["by_repo_name"][repo.lower()] = meta
    return index


def resolve_manager_meta_for_module(
    *,
    module_name: str,
    repository_url: str | None,
    canonical_custom_module_name: Callable[[str], str],
    normalize_repo_url: Callable[[str | None], str],
    github_id: Callable[[str | None], str],
    repo_name: Callable[[str | None], str],
    manager_index: Callable[[], dict[str, dict[str, dict[str, Any]]]],
) -> dict[str, Any] | None:
    """Resolve ComfyUI-Manager metadata record for module by id/repository aliases."""
    module_l = canonical_custom_module_name(module_name).lower()
    repo_norm = normalize_repo_url(repository_url)
    repo_gid = github_id(repo_norm)
    repo_text = repo_name(repo_norm)
    data = manager_index()
    if repo_gid:
        meta = data["by_github"].get(repo_gid)
        if isinstance(meta, dict):
            return meta
    if module_l:
        meta = data["by_id"].get(module_l)
        if isinstance(meta, dict):
            return meta
    if repo_text:
        meta = data["by_repo_name"].get(repo_text.lower())
        if isinstance(meta, dict):
            return meta
    return None


def manager_stats_last_update(
    *,
    repository_url: str | None,
    manager_github_stats: Callable[[], dict[str, dict[str, dict[str, Any]]]],
    normalize_repo_url: Callable[[str | None], str],
    github_id: Callable[[str | None], str],
    parse_datetime: Callable[[str | None], Any],
    to_iso: Callable[[Any], str | None],
) -> str:
    """Return normalized last-update timestamp from Manager GitHub stats for repository URL."""
    norm_repo = normalize_repo_url(repository_url)
    if not norm_repo:
        return ""
    stats = manager_github_stats()
    stats_meta = stats["by_url"].get(norm_repo)
    if stats_meta is None:
        repo_gid = github_id(norm_repo)
        if repo_gid:
            stats_meta = stats["by_github"].get(repo_gid)
    if not isinstance(stats_meta, dict):
        return ""
    remote_dt = parse_datetime(stats_meta.get("last_update"))
    return to_iso(remote_dt) or ""


def infer_update_from_manager_stats(
    *,
    repository_url: str | None,
    installed_updated_at: str | None,
    manager_stats_last_update_fn: Callable[[str | None], str],
    parse_datetime: Callable[[str | None], Any],
) -> tuple[bool | None, str]:
    """Infer update availability from manager stats when git upstream is unavailable."""
    remote_updated_at = manager_stats_last_update_fn(repository_url)
    if not remote_updated_at:
        return (None, "")
    local_dt = parse_datetime(installed_updated_at)
    remote_dt = parse_datetime(remote_updated_at)
    if local_dt is None or remote_dt is None:
        return (None, remote_updated_at)
    # Keep a small tolerance for second-level timestamp differences.
    needs_update = (remote_dt - local_dt).total_seconds() > 60.0
    return (needs_update, remote_updated_at)

