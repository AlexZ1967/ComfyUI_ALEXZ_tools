"""
Module: utils/module_node_browser_api.py
Author: AlexZ1967
Last updated: 2026-03-05

Description:
    Backend facade for Module Node Picker widget.

Purpose:
    Preserves compatibility helpers/state while delegating route registration
    and focused operations to extracted `utils/module_browser_api/*` modules.
"""

from __future__ import annotations


import json
import logging
import re
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from datetime import datetime
from pathlib import Path
from typing import Any

from .module_browser import (
    build_default_component_registry,
    build_registry_snapshot,
    build_component_health_report,
    compute_snapshot_signature,
    ensure_module_state_schema,
)
from .module_browser.catalog.api_manifest import (
    ROUTE_COMFYUI_INFO,
    ROUTE_COMFYUI_INSTALL_REQUIREMENTS,
    ROUTE_COMPONENT_REGISTRY,
    ROUTE_MODULE_ACKNOWLEDGE_ALL,
    ROUTE_MODULE_INFO,
    ROUTE_MODULE_INSTALL_REQUIREMENTS,
    ROUTE_MODULE_LIST,
    ROUTE_MODULE_NODES,
    ROUTE_MODULE_REFRESH,
    ROUTE_MODULE_REFRESH_STATUS,
    ROUTE_MODULE_UPDATE,
    ROUTE_MODULE_UPDATE_STATUS,
    ROUTE_NODE_CATALOG,
)
from .module_browser.contracts import (
    COMPONENT_REGISTRY_SCHEMA_NAME,
    COMPONENT_REGISTRY_SCHEMA_VERSION,
)
from .module_browser.jobs import (
    emit_refresh_progress,
    format_update_status_line,
    refresh_status_snapshot,
    resolve_update_targets,
    set_refresh_status,
    set_update_status,
    update_status_snapshot,
)
from .module_browser.catalog import (
    build_catalog as catalog_build_catalog,
    build_group_catalog as catalog_build_group_catalog,
    build_group_modules as catalog_build_group_modules,
    collect_nodes as catalog_collect_nodes,
    filter_modules as catalog_filter_modules,
)
from .module_browser.module.module_info_text import (
    module_local_readme_summary as mb_module_local_readme_summary,
    sanitize_module_description as mb_sanitize_module_description,
)
from .module_browser.module.module_info import (
    cached_module_flags as mb_cached_module_flags,
    resolve_module_info_uncached as mb_resolve_module_info_uncached,
)
from .module_browser.git.git_helpers import (
    git_pick_remote as mb_git_pick_remote,
    git_ref_exists as mb_git_ref_exists,
    git_remote_names as mb_git_remote_names,
    git_resolve_remote_ref as mb_git_resolve_remote_ref,
    module_git_state as mb_module_git_state,
    module_repo_url as mb_module_repo_url,
    module_worktree_signature as mb_module_worktree_signature,
    resolve_release_ref as mb_resolve_release_ref,
    sync_module_upstream as mb_sync_module_upstream,
)
from .module_browser.jobs.update_ops import (
    install_comfyui_requirements as mb_install_comfyui_requirements,
    install_requirements_for_modules as mb_install_requirements_for_modules,
    install_module_requirements as mb_install_module_requirements,
    requirements_changed_between as mb_requirements_changed_between,
)
from .module_browser.state.state_store import (
    load_state_file as mb_load_state_file,
    save_state_file as mb_save_state_file,
)
from .module_browser.tracking.tracker_ops import (
    acknowledge_all_novelty as mb_acknowledge_all_novelty,
    acknowledge_module_novelty as mb_acknowledge_module_novelty,
    announce_tracked_module_updates as mb_announce_tracked_module_updates,
    apply_node_change_info as mb_apply_node_change_info,
    remember_module_state as mb_remember_module_state,
)
from .module_browser.comfyui.comfyui_tracking_ops import (
    acknowledge_comfyui_novelty as mb_acknowledge_comfyui_novelty,
    track_comfyui_local_update as mb_track_comfyui_local_update,
)
from .module_browser.module.node_snapshot_ops import (
    file_digest as mb_file_digest,
    node_source_file as mb_node_source_file,
    relative_to_custom_roots as mb_relative_to_custom_roots,
)
from .module_browser.state.runtime_refresh_ops import (
    refresh_module_runtime_state as mb_refresh_module_runtime_state,
)
from .module_browser.jobs.update_job_ops import (
    run_module_update_job as mb_run_module_update_job,
)
from .module_browser.jobs.refresh_job_ops import (
    run_refresh_job as mb_run_refresh_job,
)
from .module_browser.module.module_identity import (
    build_custom_module_aliases as mb_build_custom_module_aliases,
    canonical_custom_module_name as mb_canonical_custom_module_name,
    discover_custom_modules as mb_discover_custom_modules,
    normalize_module_token as mb_normalize_module_token,
)
from .module_browser.comfyui.comfyui_state_ops import (
    apply_cached_pending_fields as mb_apply_cached_pending_fields,
    comfyui_status_template as mb_comfyui_status_template,
    persist_comfyui_status as mb_persist_comfyui_status,
    resolve_cached_status as mb_resolve_cached_comfyui_status,
)
from .module_browser.comfyui.comfyui_git_status_ops import (
    collect_comfyui_git_status as mb_collect_comfyui_git_status,
)
from .module_browser.catalog.component_registry_payload_ops import (
    collect_component_registry_payload as mb_collect_component_registry_payload,
)
from .module_browser.comfyui.manager_data_ops import (
    infer_update_from_manager_stats as mb_infer_update_from_manager_stats,
    load_manager_github_stats as mb_load_manager_github_stats,
    load_manager_index as mb_load_manager_index,
    manager_stats_last_update as mb_manager_stats_last_update,
    resolve_manager_meta_for_module as mb_resolve_manager_meta_for_module,
)
from .module_browser.git.pull_ops import (
    is_git_local_changes_block as mb_is_git_local_changes_block,
    pull_comfyui as mb_pull_comfyui,
    pull_custom_module as mb_pull_custom_module,
)
from .module_browser.git.command_ops import (
    extract_git_repo_from_args as mb_extract_git_repo_from_args,
    is_git_dubious_ownership_error as mb_is_git_dubious_ownership_error,
    run_command as mb_run_command,
    run_git as mb_run_git,
    tail_lines as mb_tail_lines,
    try_mark_git_safe_directory as mb_try_mark_git_safe_directory,
)
from .module_browser.catalog.catalog_payload_ops import (
    build_group_payload as mb_build_group_payload,
    build_module_list_payload as mb_build_module_list_payload,
    build_module_nodes_payload as mb_build_module_nodes_payload,
)
from .module_browser.core.widget_mode_ops import (
    custom_update_checked_flag as mb_custom_update_checked_flag,
    info_only_rejection_payload as mb_info_only_rejection_payload,
    requirements_advisory_payload as mb_requirements_advisory_payload,
    normalize_log_mode as mb_normalize_log_mode,
    set_custom_update_checked as mb_set_custom_update_checked,
)
from .module_browser.core.value_ops import (
    github_id as mb_github_id,
    normalize_comfyui_mode as mb_normalize_comfyui_mode,
    normalize_repo_url as mb_normalize_repo_url,
    now_iso as mb_now_iso,
    parse_datetime as mb_parse_datetime,
    pick_repo_url as mb_pick_repo_url,
    repo_name as mb_repo_name,
    short_commit as mb_short_commit,
    to_iso as mb_to_iso,
)
from .module_browser.state.requirements_pending_ops import (
    set_comfyui_requirements_pending as mb_set_comfyui_requirements_pending,
    set_module_requirements_pending as mb_set_module_requirements_pending,
)
from .module_browser.core.path_ops import (
    comfyui_root as mb_comfyui_root,
    custom_nodes_roots as mb_custom_nodes_roots,
    manager_custom_db_path as mb_manager_custom_db_path,
    manager_github_stats_path as mb_manager_github_stats_path,
    module_dir as mb_module_dir,
)
from .module_browser.core.release_ops import (
    github_latest_release as mb_github_latest_release,
)
from .module_browser.module.module_update_state_ops import (
    comfyui_needs_update_now as mb_comfyui_needs_update_now,
    count_custom_modules_need_update as mb_count_custom_modules_need_update,
    count_custom_modules_unknown_update as mb_count_custom_modules_unknown_update,
    list_custom_modules_unknown_update as mb_list_custom_modules_unknown_update,
    module_needs_update_now as mb_module_needs_update_now,
)
from .module_browser.bootstrap.repo_bootstrap_ops import (
    bootstrap_module_remote_from_manager as mb_bootstrap_module_remote_from_manager,
    comfyui_requirements_path as mb_comfyui_requirements_path,
)
from .module_browser.module.node_classification_ops import (
    classify_by_relative_module as mb_classify_by_relative_module,
    classify_by_source_path as mb_classify_by_source_path,
    fallback_annotation as mb_fallback_annotation,
    module_root as mb_module_root,
)
from .module_browser_api.logging_ops import (
    get_update_console_log_mode as mb_api_get_update_console_log_mode,
    refresh_console_log as mb_api_refresh_console_log,
    set_update_console_log_mode as mb_api_set_update_console_log_mode,
    update_console_log as mb_api_update_console_log,
)
from .module_browser_api.handlers_refresh import (
    start_refresh_job as mb_api_start_refresh_job,
)
from .module_browser_api.handlers_update import (
    start_module_update_job as mb_api_start_module_update_job,
)
from .module_browser_api.handlers_catalog import (
    build_module_list_response as mb_api_build_module_list_response,
    build_module_nodes_response as mb_api_build_module_nodes_response,
    build_node_catalog_payload as mb_api_build_node_catalog_payload,
)
from .module_browser_api.node_introspection import (
    build_node_snapshots as mb_api_build_node_snapshots,
    node_mappings as mb_api_node_mappings,
)
from .module_browser_api.state import (
    get_state as mb_api_get_state,
)
from .module_browser_api.routes import (
    register_routes as mb_api_register_routes,
)

try:
    import folder_paths
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - non-Comfy environment
    folder_paths = None
    PromptServer = None
    web = None


_LOGGER = logging.getLogger("ALEXZ_tools.ModuleBrowser")
_MODULE_INFO_CACHE: dict[tuple[str, str, bool], tuple[float, dict[str, Any]]] = {}
_MODULE_INFO_TTL_SEC = 30.0
_MANAGER_INDEX_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None
_MANAGER_GITHUB_STATS_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None
_MANAGER_UPDATE_OVERRIDE_CACHE: tuple[float, dict[str, bool]] | None = None
_MODULE_STATE_CACHE: dict[str, dict[str, Any]] | None = None
_CUSTOM_MODULE_ALIAS_CACHE: dict[str, str] | None = None
_COMFYUI_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_COMFYUI_STATUS_TTL_SEC = 120.0
_MANAGER_UPDATE_OVERRIDE_TTL_SEC = 20.0
_API_STATE = mb_api_get_state()
# Legacy compatibility mirrors kept for phase0 baseline tests.
_LAZY_REFRESH_DONE = _API_STATE.lazy_refresh_done
_RUNTIME_WARMUP_THREAD: threading.Thread | None = _API_STATE.runtime_warmup_thread
_GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MODULE_STATE_PATH = Path(__file__).resolve().parents[1] / "module_state_cache.json"
_GROUP_ORDER = (
    ("core", "Core_Nodes"),
    ("core_extras", "Core_Extras_Nodes"),
    ("api", "API_Nodes"),
    ("custom", "Custom_Nodes"),
)
_UPDATE_TARGET_SCAN_WORKERS = 4
_COMPONENT_REGISTRY_PAYLOAD_CACHE: tuple[float, dict[str, Any]] | None = None
_COMPONENT_REGISTRY_TTL_SEC = 15.0
_INFO_ONLY_WIDGET_MODE = True


def _sync_runtime_warmup_from_legacy() -> None:
    """Apply legacy runtime warmup globals to extracted state container."""
    _API_STATE.lazy_refresh_done = bool(_LAZY_REFRESH_DONE)
    _API_STATE.runtime_warmup_thread = _RUNTIME_WARMUP_THREAD


def _sync_runtime_warmup_to_legacy() -> None:
    """Mirror extracted runtime warmup state back to legacy globals."""
    global _LAZY_REFRESH_DONE
    global _RUNTIME_WARMUP_THREAD
    _LAZY_REFRESH_DONE = bool(_API_STATE.lazy_refresh_done)
    _RUNTIME_WARMUP_THREAD = _API_STATE.runtime_warmup_thread


def _custom_update_checked_flag(state: dict[str, Any] | None = None) -> bool:
    """Return whether custom-module remote update check was explicitly run in current session."""
    cache = state if isinstance(state, dict) else _load_module_state()
    return mb_custom_update_checked_flag(cache)


def _info_only_rejection_payload(feature: str) -> dict[str, Any]:
    """Build a consistent rejection payload for disabled mutate operations."""
    return mb_info_only_rejection_payload(feature)


def _requirements_advisory_for_modules(modules: list[str]) -> dict[str, Any]:
    """Return manual-install advisory payload for changed custom-module requirements."""
    module_tokens = [str(item).strip() for item in (modules if isinstance(modules, list) else []) if str(item).strip()]
    canonical = [_canonical_custom_module_name(name) for name in module_tokens]
    canonical = [name for name in dict.fromkeys(canonical) if name and name != "unknown"]
    requirement_paths: list[str] = []
    for module_name in canonical:
        module_dir = _module_dir(module_name)
        if module_dir is None:
            continue
        requirements_path = module_dir / "requirements.txt"
        if requirements_path.exists():
            requirement_paths.append(str(requirements_path))
    headline = "Manual dependency follow-up required for custom modules."
    if canonical:
        headline = f"Manual dependency follow-up required for: {', '.join(canonical)}."
    payload = mb_requirements_advisory_payload(
        feature="module_install_requirements",
        requirements_paths=requirement_paths,
        headline=headline,
    )
    payload["modules"] = canonical
    return payload


def _comfyui_requirements_advisory() -> dict[str, Any]:
    """Return manual-install advisory payload for changed ComfyUI requirements."""
    req = _comfyui_requirements_path()
    payload = mb_requirements_advisory_payload(
        feature="comfyui_install_requirements",
        requirements_paths=[str(req)] if req is not None else [],
        headline="Manual dependency follow-up required for ComfyUI.",
    )
    payload["module"] = "ComfyUI"
    return payload


def _set_custom_update_checked(checked: bool) -> None:
    """Persist custom-module update-check visibility gate for initial widget state."""
    mb_set_custom_update_checked(
        checked=bool(checked),
        load_state_fn=_load_module_state,
        save_state_fn=_save_module_state,
        now_iso_fn=_now_iso,
        on_changed=_MODULE_INFO_CACHE.clear,
    )


def _normalize_log_mode(value: str | None) -> str:
    """Normalize console log mode for update jobs."""
    return mb_normalize_log_mode(value)


def _set_update_console_log_mode(mode: str | None) -> str:
    """Set active console log mode for update jobs and return normalized value."""
    return mb_api_set_update_console_log_mode(
        state=_API_STATE,
        mode=mode,
        normalize_log_mode=_normalize_log_mode,
    )


def _get_update_console_log_mode() -> str:
    """Read active console log mode for update jobs."""
    return mb_api_get_update_console_log_mode(_API_STATE)


def _update_console_log(message: str, level: str = "summary") -> None:
    """Print update-progress line to ComfyUI console according to selected log mode."""
    mb_api_update_console_log(
        state=_API_STATE,
        message=message,
        level=level,
        normalize_log_mode=_normalize_log_mode,
    )


def _refresh_console_log(message: str, level: str = "summary") -> None:
    """Print refresh-progress line to ComfyUI console according to selected log mode."""
    mb_api_refresh_console_log(
        state=_API_STATE,
        message=message,
        level=level,
        normalize_log_mode=_normalize_log_mode,
    )


_ALEXZ_ANNOTATIONS = {
    "ImagePrepare_for_QwenEdit_outpaint": "Подготавливает изображение и latent под QwenEdit Outpaint.",
    "ImageAlignOverlayToBackground": "Выравнивает оверлей относительно фона по ключевым точкам.",
    "JsonDisplayAndSave": "Показывает и сохраняет JSON в читаемом виде.",
    "VideoInpaintWatermark": "Удаляет статический вотермарк/объект из видео.",
    "ImageColorMatchToReference": "Подгоняет цвет и тон изображения под референс.",
    "VideoFrameMatch": "Ищет наиболее похожий кадр в видео для входной картинки.",
    "VideoCutMatch": "Подбирает оптимальную пару кадров для склейки двух видео.",
    "ImageDifference": "Строит абсолютную разницу двух изображений.",
    "ImageWaveformScope": "Строит waveform/parade scope для анализа яркости и каналов.",
    "ImageHistogramScope": "Строит RGB/Luma гистограмму изображения.",
    "GenerateQRCode": "Генерирует QR-код из ссылки или текста.",
    "ALEXZTestNode": "Тестовая нода для проверки загрузки/обновления и работы Module Nodes.",
}


def _short_commit(commit: str | None) -> str:
    """Return short 8-character representation of a git commit hash."""
    return mb_short_commit(commit)


def _ensure_comfyui_status_cache() -> dict[str, tuple[float, dict[str, Any]]]:
    """Return ComfyUI status cache dict, reinitializing it if tests set it to None."""
    global _COMFYUI_STATUS_CACHE
    if not isinstance(_COMFYUI_STATUS_CACHE, dict):
        _COMFYUI_STATUS_CACHE = {}
    return _COMFYUI_STATUS_CACHE


def _clear_comfyui_status_cache() -> None:
    """Clear ComfyUI status cache safely even if it was replaced with None."""
    cache = _ensure_comfyui_status_cache()
    cache.clear()


def _component_registry_payload(force_refresh: bool = False) -> dict[str, Any]:
    """Return cached component-registry snapshot used for Slice 0 diagnostics."""
    global _COMPONENT_REGISTRY_PAYLOAD_CACHE
    now_ts = time.time()
    payload, _COMPONENT_REGISTRY_PAYLOAD_CACHE = mb_collect_component_registry_payload(
        force_refresh=force_refresh,
        now_ts=now_ts,
        cache_payload=_COMPONENT_REGISTRY_PAYLOAD_CACHE,
        ttl_sec=_COMPONENT_REGISTRY_TTL_SEC,
        build_default_component_registry=build_default_component_registry,
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        build_registry_snapshot=build_registry_snapshot,
        compute_snapshot_signature=compute_snapshot_signature,
        build_component_health_report=build_component_health_report,
        schema_name=COMPONENT_REGISTRY_SCHEMA_NAME,
        schema_version=COMPONENT_REGISTRY_SCHEMA_VERSION,
        now_iso=_now_iso,
    )
    return payload


def _node_mappings() -> tuple[dict[str, Any], dict[str, str]]:
    """Return NODE_CLASS_MAPPINGS from loaded extension modules."""
    return mb_api_node_mappings()


def _node_source_file(node_cls: Any) -> str:
    """Resolve source file path for a node class object."""
    return mb_node_source_file(node_cls)


def _relative_to_custom_roots(path_text: str) -> str:
    """Resolve path relative to known custom_nodes roots when possible."""
    return mb_relative_to_custom_roots(path_text, custom_nodes_roots=_custom_nodes_roots)


def _file_digest(path_text: str) -> str:
    """Compute SHA1 digest for file content used in node-change tracking."""
    return mb_file_digest(path_text)


def _build_node_snapshots() -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Build stable per-node file snapshots used to detect node additions/changes."""
    return mb_api_build_node_snapshots(
        classifier=_classify_by_relative_module,
        custom_nodes_roots=_custom_nodes_roots,
        node_mappings_fn=_node_mappings,
    )


def _module_root(node_cls: Any) -> str:
    """Resolve module root directory for a file path inside the extension."""
    return mb_module_root(node_cls)


def _classify_by_relative_module(node_cls: Any) -> tuple[str, str]:
    """Classify node group and module name using path relative to ComfyUI roots."""
    return mb_classify_by_relative_module(
        node_cls,
        canonical_custom_module_name_fn=_canonical_custom_module_name,
        classify_by_source_path_fn=_classify_by_source_path,
        module_root_fn=_module_root,
    )


def _fallback_annotation(node_cls: Any) -> str:
    """Build fallback node annotation from class metadata when no static annotation exists."""
    return mb_fallback_annotation(node_cls)


def _custom_nodes_roots() -> list[Path]:
    """Return existing custom_nodes root directories."""
    return mb_custom_nodes_roots(
        folder_paths_module=folder_paths,
        fallback_root=Path(__file__).resolve().parents[1],
    )


def _discover_custom_modules() -> list[str]:
    """Discover installed custom module directories under custom_nodes roots."""
    return mb_discover_custom_modules(custom_nodes_roots=_custom_nodes_roots)


def _normalize_module_token(name: str) -> str:
    """Normalize module token for case-insensitive matching and aliases."""
    return mb_normalize_module_token(name)


def _custom_module_aliases() -> dict[str, str]:
    """Build alias map for custom module names and normalized tokens."""
    global _CUSTOM_MODULE_ALIAS_CACHE
    if _CUSTOM_MODULE_ALIAS_CACHE is not None:
        return _CUSTOM_MODULE_ALIAS_CACHE

    aliases = mb_build_custom_module_aliases(
        discovered_modules=_discover_custom_modules(),
        normalize_token=_normalize_module_token,
    )
    _CUSTOM_MODULE_ALIAS_CACHE = aliases
    return aliases


def _canonical_custom_module_name(module_name: str) -> str:
    """Resolve user-provided module token to canonical custom module name."""
    return mb_canonical_custom_module_name(
        module_name,
        aliases=_custom_module_aliases(),
        normalize_token=_normalize_module_token,
    )


def _classify_by_source_path(node_cls: Any) -> tuple[str, str] | None:
    """Classify node into core/extras/api/custom groups from source path."""
    return mb_classify_by_source_path(
        node_cls,
        node_source_file_fn=_node_source_file,
        custom_nodes_roots_fn=_custom_nodes_roots,
        canonical_custom_module_name_fn=_canonical_custom_module_name,
        module_root_fn=_module_root,
    )


def _normalize_repo_url(url: str | None) -> str | None:
    """Normalize repository URL to canonical HTTPS GitHub form."""
    return mb_normalize_repo_url(url)


def _github_id(url: str | None) -> str | None:
    """Extract owner/repository identifier from normalized GitHub URL."""
    value = mb_github_id(url, github_re=_GITHUB_RE)
    return value.lower() if value else None


def _repo_name(url: str | None) -> str | None:
    """Return repository name parsed from module URL."""
    return mb_repo_name(url, github_id_fn=_github_id)


def _pick_repo_url(entry: dict[str, Any]) -> str | None:
    """Choose best repository URL from module metadata candidates."""
    value = mb_pick_repo_url(entry, normalize_repo_url_fn=_normalize_repo_url)
    if not value:
        return None
    return value


def _manager_custom_db_path() -> Path | None:
    """Return path to ComfyUI-Manager custom-node database file."""
    return mb_manager_custom_db_path(custom_nodes_roots_fn=_custom_nodes_roots)


def _manager_github_stats_path() -> Path | None:
    """Return path to cached GitHub-stats file maintained by ComfyUI-Manager."""
    return mb_manager_github_stats_path(custom_nodes_roots_fn=_custom_nodes_roots)


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime text from manager metadata into timezone-aware object."""
    return mb_parse_datetime(value)


def _to_iso(dt: datetime | None) -> str | None:
    """Convert datetime value to ISO-8601 string in UTC."""
    return mb_to_iso(dt)


def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return mb_now_iso()


def _set_comfyui_requirements_pending(pending: bool, before_commit: str = "", after_commit: str = "") -> None:
    """Persist pending ComfyUI requirements-install marker in module state cache."""
    mb_set_comfyui_requirements_pending(
        pending=bool(pending),
        before_commit=before_commit or "",
        after_commit=after_commit or "",
        load_state_fn=_load_module_state,
        save_state_fn=_save_module_state,
        now_iso_fn=_now_iso,
        on_state_changed=_clear_comfyui_status_cache,
    )


def _set_module_requirements_pending(
    module_name: str, pending: bool, before_commit: str = "", after_commit: str = ""
) -> None:
    """Persist pending requirements-install marker for one custom module."""
    mb_set_module_requirements_pending(
        module_name=module_name,
        pending=bool(pending),
        before_commit=before_commit or "",
        after_commit=after_commit or "",
        canonical_custom_module_name_fn=_canonical_custom_module_name,
        load_state_fn=_load_module_state,
        save_state_fn=_save_module_state,
        now_iso_fn=_now_iso,
        on_state_changed=_MODULE_INFO_CACHE.clear,
    )


def _normalize_comfyui_mode(value: str | None) -> str:
    """Normalize ComfyUI update-check mode to supported values."""
    return mb_normalize_comfyui_mode(value)


def _github_latest_release(owner: str, repo: str, timeout: float = 8.0) -> dict[str, Any]:
    """Fetch latest GitHub release metadata for a repository."""
    return mb_github_latest_release(owner, repo, timeout=timeout)


def _manager_github_stats() -> dict[str, dict[str, dict[str, Any]]]:
    """Load and cache module update timestamps from manager stats file."""
    global _MANAGER_GITHUB_STATS_CACHE
    _MANAGER_GITHUB_STATS_CACHE = mb_load_manager_github_stats(
        cache=_MANAGER_GITHUB_STATS_CACHE,
        manager_github_stats_path=_manager_github_stats_path,
        normalize_repo_url=_normalize_repo_url,
        github_id=_github_id,
        logger_warning=_LOGGER.warning,
    )
    return _MANAGER_GITHUB_STATS_CACHE


def _manager_index() -> dict[str, dict[str, dict[str, Any]]]:
    """Load and cache manager metadata index for custom modules."""
    global _MANAGER_INDEX_CACHE
    _MANAGER_INDEX_CACHE = mb_load_manager_index(
        cache=_MANAGER_INDEX_CACHE,
        manager_custom_db_path=_manager_custom_db_path,
        pick_repo_url=_pick_repo_url,
        github_id=_github_id,
        repo_name=_repo_name,
        logger_warning=_LOGGER.warning,
    )
    return _MANAGER_INDEX_CACHE


def _manager_meta_for_module(module_name: str, repository_url: str | None = None) -> dict[str, Any] | None:
    """Resolve ComfyUI-Manager metadata record for module by id/repository aliases."""
    return mb_resolve_manager_meta_for_module(
        module_name=module_name,
        repository_url=repository_url,
        canonical_custom_module_name=_canonical_custom_module_name,
        normalize_repo_url=_normalize_repo_url,
        github_id=_github_id,
        repo_name=_repo_name,
        manager_index=_manager_index,
    )


def _manager_stats_last_update(repository_url: str | None) -> str:
    """Return normalized last-update timestamp from Manager GitHub stats for repository URL."""
    return mb_manager_stats_last_update(
        repository_url=repository_url,
        manager_github_stats=_manager_github_stats,
        normalize_repo_url=_normalize_repo_url,
        github_id=_github_id,
        parse_datetime=_parse_datetime,
        to_iso=_to_iso,
    )


def _infer_update_from_manager_stats(
    repository_url: str | None,
    installed_updated_at: str | None,
) -> tuple[bool | None, str]:
    """Infer update availability from Manager GitHub stats when git upstream is unavailable."""
    return mb_infer_update_from_manager_stats(
        repository_url=repository_url,
        installed_updated_at=installed_updated_at,
        manager_stats_last_update_fn=_manager_stats_last_update,
        parse_datetime=_parse_datetime,
    )


def _promptserver_base_url() -> str | None:
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


def _http_json_get(url: str, timeout: float = 20.0) -> dict[str, Any]:
    """Load JSON payload from local HTTP endpoint with strict timeouts."""
    with urlopen(url, timeout=max(1.0, float(timeout))) as response:
        raw = response.read().decode("utf-8", errors="replace")
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else {}


def _manager_installed_update_overrides(force_refresh: bool = False) -> dict[str, bool]:
    """Return installed-module update overrides derived from ComfyUI-Manager."""
    global _MANAGER_UPDATE_OVERRIDE_CACHE
    now_ts = time.time()
    if not force_refresh and _MANAGER_UPDATE_OVERRIDE_CACHE is not None:
        cached_ts, cached_payload = _MANAGER_UPDATE_OVERRIDE_CACHE
        if (now_ts - cached_ts) < _MANAGER_UPDATE_OVERRIDE_TTL_SEC:
            return dict(cached_payload)

    base_url = _promptserver_base_url()
    if not base_url:
        return {}

    try:
        installed_payload = _http_json_get(f"{base_url}/customnode/installed?mode=default", timeout=20.0)
        list_payload = _http_json_get(f"{base_url}/customnode/getlist?mode=local&skip_update=false", timeout=90.0)
    except (TimeoutError, URLError, HTTPError, ValueError, json.JSONDecodeError) as exc:
        _LOGGER.debug("ComfyUI-Manager update override probe failed: %s", exc)
        _MANAGER_UPDATE_OVERRIDE_CACHE = (now_ts, {})
        return {}

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
            repo_norm = _normalize_repo_url(source)
            if not repo_norm:
                continue
            gid = _github_id(repo_norm).lower()
            if gid:
                by_github[gid] = meta
            repo_short = _repo_name(repo_norm).lower()
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

    _MANAGER_UPDATE_OVERRIDE_CACHE = (now_ts, dict(overrides))
    return overrides


def _run_git(args: list[str], timeout: float = 2.0) -> str | None:
    """Run git command in non-interactive mode and return trimmed stdout on success."""
    return mb_run_git(args, timeout=timeout, run_command_fn=_run_command)


def _extract_git_repo_from_args(args: list[str]) -> str | None:
    """Extract normalized git working directory from `git -C <path>` argument list."""
    return mb_extract_git_repo_from_args(args)


def _is_git_dubious_ownership_error(text: str) -> bool:
    """Check whether git stderr/stdout indicates `safe.directory` ownership protection."""
    return mb_is_git_dubious_ownership_error(text)


def _try_mark_git_safe_directory(repo_dir: str, env: dict[str, str], timeout: float = 15.0) -> bool:
    """Attempt to add repository path to git safe.directory list."""
    return mb_try_mark_git_safe_directory(
        repo_dir,
        env,
        timeout=timeout,
        subprocess_run=subprocess.run,
        logger=_LOGGER,
    )


def _run_command(args: list[str], timeout: float = 120.0, disable_git_prompt: bool = False) -> dict[str, Any]:
    """Run a subprocess command and return exit code plus output text."""
    return mb_run_command(
        args,
        timeout=timeout,
        disable_git_prompt=disable_git_prompt,
        subprocess_run=subprocess.run,
        logger=_LOGGER,
    )


def _tail_lines(text: str | None, max_lines: int = 80) -> str:
    """Return tail lines from command output for concise console diagnostics."""
    return mb_tail_lines(text, max_lines=max_lines)


def _is_git_local_changes_block(text: str | None) -> bool:
    """Detect pull errors caused by local-changes merge conflicts."""
    return mb_is_git_local_changes_block(text)


def _module_dir(module_name: str) -> Path | None:
    """Resolve filesystem directory for a custom module by name."""
    return mb_module_dir(
        module_name,
        canonical_custom_module_name_fn=_canonical_custom_module_name,
        custom_nodes_roots_fn=_custom_nodes_roots,
    )


def _requirements_changed_between(module_dir: Path, before_commit: str, after_commit: str) -> bool:
    """Check whether requirements.txt changed between two git revisions."""
    return mb_requirements_changed_between(
        module_dir,
        before_commit,
        after_commit,
        run_command=lambda args, timeout, disable_git_prompt: _run_command(
            args,
            timeout=timeout,
            disable_git_prompt=disable_git_prompt,
        ),
    )


def _module_needs_update_now(module_name: str) -> bool:
    """Check whether local module commit differs from tracked remote commit."""
    return mb_module_needs_update_now(
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        load_module_state=_load_module_state,
        module_git_state_fn=_module_git_state,
        manager_meta_for_module_fn=_manager_meta_for_module,
        infer_update_from_manager_stats_fn=_infer_update_from_manager_stats,
    )


def _module_worktree_signature(module_name: str) -> str:
    """Return short signature of local uncommitted changes for module worktree."""
    return mb_module_worktree_signature(
        module_name,
        module_dir_resolver=_module_dir,
        run_git=_run_git,
    )


def _count_custom_modules_need_update() -> int:
    """Count custom modules that currently report available updates."""
    return mb_count_custom_modules_need_update(
        load_module_state=_load_module_state,
        discover_custom_modules=_discover_custom_modules,
        canonical_custom_module_name=_canonical_custom_module_name,
    )


def _count_custom_modules_unknown_update() -> int:
    """Count custom modules whose remote update status is unknown/uncheckable."""
    return mb_count_custom_modules_unknown_update(
        load_module_state=_load_module_state,
        discover_custom_modules=_discover_custom_modules,
        canonical_custom_module_name=_canonical_custom_module_name,
    )


def _list_custom_modules_unknown_update() -> list[str]:
    """List custom modules whose remote update status is unknown/uncheckable."""
    return mb_list_custom_modules_unknown_update(
        load_module_state=_load_module_state,
        discover_custom_modules=_discover_custom_modules,
        canonical_custom_module_name=_canonical_custom_module_name,
    )


def _cached_module_flags(group: str, module_name: str) -> dict[str, Any]:
    """Return lightweight cached update flags for module dropdown badges."""
    return mb_cached_module_flags(
        group_name=group,
        module_name=module_name,
        state=_load_module_state(),
        canonical_custom_module_name=_canonical_custom_module_name,
        custom_update_checked_flag=_custom_update_checked_flag,
    )


def _comfyui_requirements_path() -> Path | None:
    """Resolve requirements.txt path for the main ComfyUI repository."""
    return mb_comfyui_requirements_path(comfyui_root_fn=_comfyui_root)


def _comfyui_needs_update_now() -> bool:
    """Check whether local ComfyUI commit is behind remote tracking commit."""
    return mb_comfyui_needs_update_now(comfyui_git_status_fn=_comfyui_git_status)


def _git_remote_names(repo_root: Path) -> list[str]:
    """Return list of configured git remote names for repository."""
    return mb_git_remote_names(repo_root, run_git=_run_git)


def _git_pick_remote(repo_root: Path, upstream: str | None) -> str | None:
    """Choose preferred git remote (upstream, origin, or first available)."""
    return mb_git_pick_remote(
        repo_root,
        upstream,
        git_remote_names_fn=_git_remote_names,
    )


def _git_ref_exists(repo_root: Path, ref_name: str) -> bool:
    """Check whether a local or remote git reference exists."""
    return mb_git_ref_exists(repo_root, ref_name, run_git=_run_git)


def _git_resolve_remote_ref(
    repo_root: Path,
    remote_name: str,
    branch_name: str | None,
    upstream: str | None,
) -> tuple[str | None, str | None]:
    """Resolve remote tracking reference to compare local and upstream revisions."""
    return mb_git_resolve_remote_ref(
        repo_root,
        remote_name,
        branch_name,
        upstream,
        run_git=_run_git,
        git_ref_exists_fn=_git_ref_exists,
    )


def _resolve_release_ref(repo_root: Path, remote_name: str, tag_name: str) -> tuple[str | None, str]:
    """Resolve git reference for a release tag and ensure tag exists locally."""
    return mb_resolve_release_ref(
        repo_root,
        remote_name,
        tag_name,
        run_git=_run_git,
        git_ref_exists_fn=_git_ref_exists,
    )


def _pull_comfyui(timeout: float = 240.0) -> dict[str, Any]:
    """Pull latest ComfyUI changes from selected remote with fast-forward strategy."""
    return mb_pull_comfyui(
        comfyui_root=_comfyui_root,
        update_console_log=lambda message, level: _update_console_log(message, level=level),
        run_git=_run_git,
        git_pick_remote=_git_pick_remote,
        git_resolve_remote_ref=_git_resolve_remote_ref,
        run_command=lambda args, run_timeout, disable_git_prompt: _run_command(
            args,
            timeout=run_timeout,
            disable_git_prompt=disable_git_prompt,
        ),
        requirements_changed_between=_requirements_changed_between,
        set_comfyui_requirements_pending=lambda pending, before, after: _set_comfyui_requirements_pending(
            pending,
            before,
            after,
        ),
        perf_counter=time.perf_counter,
        timeout=timeout,
    )


def _pull_custom_module(module_name: str, timeout: float = 180.0) -> dict[str, Any]:
    """Pull latest changes for one custom module from its git remote."""
    return mb_pull_custom_module(
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        module_dir_resolver=_module_dir,
        update_console_log=lambda message, level: _update_console_log(message, level=level),
        run_git=_run_git,
        git_pick_remote=_git_pick_remote,
        git_resolve_remote_ref=_git_resolve_remote_ref,
        bootstrap_module_remote_from_manager=_bootstrap_module_remote_from_manager,
        run_command=lambda args, run_timeout, disable_git_prompt: _run_command(
            args,
            timeout=run_timeout,
            disable_git_prompt=disable_git_prompt,
        ),
        is_git_local_changes_block_fn=_is_git_local_changes_block,
        requirements_changed_between=_requirements_changed_between,
        set_module_requirements_pending=lambda module, pending, before, after: _set_module_requirements_pending(
            module,
            pending,
            before,
            after,
        ),
        perf_counter=time.perf_counter,
        timeout=timeout,
    )


def _install_module_requirements(module_name: str, timeout: float = 1200.0) -> dict[str, Any]:
    """Install Python dependencies from module requirements.txt in active runtime environment."""
    return mb_install_module_requirements(
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        module_dir_resolver=_module_dir,
        run_command=lambda args, run_timeout, disable_git_prompt: _run_command(
            args,
            timeout=run_timeout,
            disable_git_prompt=disable_git_prompt,
        ),
        python_executable=sys.executable,
        tail_lines=_tail_lines,
        set_module_requirements_pending=lambda module, pending: _set_module_requirements_pending(module, pending),
        logger=_LOGGER,
        timeout=timeout,
    )


def _install_comfyui_requirements(timeout: float = 1800.0) -> dict[str, Any]:
    """Install Python dependencies from ComfyUI requirements.txt in active runtime environment."""
    return mb_install_comfyui_requirements(
        comfyui_requirements_path=_comfyui_requirements_path,
        run_command=lambda args, run_timeout, disable_git_prompt: _run_command(
            args,
            timeout=run_timeout,
            disable_git_prompt=disable_git_prompt,
        ),
        python_executable=sys.executable,
        tail_lines=_tail_lines,
        set_comfyui_requirements_pending=lambda pending: _set_comfyui_requirements_pending(pending),
        logger=_LOGGER,
        timeout=timeout,
    )

def _module_repo_url(module_name: str) -> str | None:
    """Resolve module repository URL using manager metadata and git remotes."""
    return mb_module_repo_url(
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        custom_nodes_roots=_custom_nodes_roots,
        run_git=_run_git,
        normalize_repo_url=_normalize_repo_url,
    )


def _bootstrap_module_remote_from_manager(module_name: str, module_dir: Path) -> bool:
    """Configure `origin` remote from ComfyUI-Manager metadata for repos without remotes."""
    return mb_bootstrap_module_remote_from_manager(
        module_name,
        module_dir,
        git_remote_names_fn=_git_remote_names,
        manager_meta_for_module_fn=_manager_meta_for_module,
        normalize_repo_url_fn=_normalize_repo_url,
        run_command_fn=lambda args, timeout, disable_git_prompt: _run_command(
            args,
            timeout=timeout,
            disable_git_prompt=disable_git_prompt,
        ),
        logger_info=_LOGGER.info,
        timeout=20.0,
    )


def _module_git_state(module_name: str) -> dict[str, Any]:
    """Collect local/remote git commit and timestamp state for one module."""
    return mb_module_git_state(
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        custom_nodes_roots=_custom_nodes_roots,
        run_git=_run_git,
        normalize_repo_url=_normalize_repo_url,
        git_pick_remote_fn=_git_pick_remote,
        git_resolve_remote_ref_fn=_git_resolve_remote_ref,
    )


def _sync_module_upstream(module_name: str, timeout: float = 15.0) -> bool:
    """Fetch module remotes and refresh local view of upstream references."""
    return mb_sync_module_upstream(
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        custom_nodes_roots=_custom_nodes_roots,
        run_git=_run_git,
        git_pick_remote_fn=_git_pick_remote,
        bootstrap_module_remote_fn=_bootstrap_module_remote_from_manager,
        timeout=timeout,
    )


def _comfyui_root() -> Path | None:
    """Resolve root path of the currently running ComfyUI installation."""
    return mb_comfyui_root(__file__)


def _comfyui_git_status(force_refresh: bool = False, mode: str = "releases") -> dict[str, Any]:
    """Collect local/remote git status summary for ComfyUI repository."""
    global _COMFYUI_STATUS_CACHE
    now_ts = time.time()
    cache = _ensure_comfyui_status_cache()
    return mb_collect_comfyui_git_status(
        force_refresh=force_refresh,
        mode=mode,
        now_ts=now_ts,
        cache=cache,
        ttl_sec=_COMFYUI_STATUS_TTL_SEC,
        normalize_comfyui_mode=_normalize_comfyui_mode,
        comfyui_status_template=mb_comfyui_status_template,
        load_module_state=_load_module_state,
        resolve_cached_status=mb_resolve_cached_comfyui_status,
        apply_cached_pending_fields=mb_apply_cached_pending_fields,
        short_commit=_short_commit,
        comfyui_root=_comfyui_root,
        run_git=_run_git,
        git_pick_remote=_git_pick_remote,
        github_latest_release=_github_latest_release,
        resolve_release_ref=_resolve_release_ref,
        parse_datetime=_parse_datetime,
        to_iso=_to_iso,
        git_resolve_remote_ref=_git_resolve_remote_ref,
        persist_comfyui_status=mb_persist_comfyui_status,
        save_module_state=_save_module_state,
        now_iso=_now_iso,
    )


def _track_comfyui_local_update() -> None:
    """Track local ComfyUI commit changes between restarts without upstream sync."""
    mb_track_comfyui_local_update(
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        comfyui_root=_comfyui_root,
        run_git=_run_git,
        now_iso=_now_iso,
        short_commit=_short_commit,
        clear_comfyui_status_cache=_clear_comfyui_status_cache,
    )


def _acknowledge_comfyui_novelty() -> dict[str, Any]:
    """Clear pending ComfyUI novelty markers after explicit user refresh action."""
    return mb_acknowledge_comfyui_novelty(
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        clear_comfyui_status_cache=_clear_comfyui_status_cache,
    )


def _load_module_state() -> dict[str, dict[str, Any]]:
    """Load persisted module snapshot state from extension cache file."""
    global _MODULE_STATE_CACHE
    if _MODULE_STATE_CACHE is not None:
        return _MODULE_STATE_CACHE
    _MODULE_STATE_CACHE = mb_load_state_file(
        _MODULE_STATE_PATH,
        ensure_schema=ensure_module_state_schema,
    )
    return _MODULE_STATE_CACHE


def _save_module_state(state: dict[str, dict[str, Any]]) -> None:
    """Persist module snapshot state to extension cache file."""
    normalized = mb_save_state_file(
        _MODULE_STATE_PATH,
        state,
        ensure_schema=ensure_module_state_schema,
        logger=_LOGGER,
    )
    global _MODULE_STATE_CACHE
    _MODULE_STATE_CACHE = normalized


def _remember_module_state(module_name: str, result: dict[str, Any]) -> None:
    """Capture current module/node snapshot as baseline for next ComfyUI start."""
    mb_remember_module_state(
        module_name,
        result,
        canonical_custom_module_name=_canonical_custom_module_name,
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        now_iso=_now_iso,
        short_commit=_short_commit,
    )


def _apply_node_change_info(result: dict[str, Any], group: str, module_name: str) -> None:
    """Attach node-level change markers to module info payload for UI rendering."""
    mb_apply_node_change_info(
        result,
        group,
        module_name,
        load_module_state=_load_module_state,
    )


def _acknowledge_module_novelty(group: str, module_name: str) -> None:
    """Clear pending novelty markers for one module after explicit user refresh."""
    mb_acknowledge_module_novelty(
        group,
        module_name,
        canonical_custom_module_name=_canonical_custom_module_name,
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        clear_module_info_cache=_MODULE_INFO_CACHE.clear,
    )


def _acknowledge_all_novelty() -> dict[str, Any]:
    """Clear pending novelty markers for all modules after explicit global refresh."""
    return mb_acknowledge_all_novelty(
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        clear_module_info_cache=_MODULE_INFO_CACHE.clear,
    )


def _announce_tracked_module_updates(local_only: bool = False) -> dict[str, Any]:
    """Build per-module node-change info by comparing saved and current snapshots."""
    return mb_announce_tracked_module_updates(
        local_only=local_only,
        load_module_state=_load_module_state,
        save_module_state=_save_module_state,
        now_iso=_now_iso,
        discover_custom_modules=_discover_custom_modules,
        canonical_custom_module_name=_canonical_custom_module_name,
        module_git_state=_module_git_state,
        manager_meta_for_module=_manager_meta_for_module,
        infer_update_from_manager_stats=_infer_update_from_manager_stats,
        manager_update_overrides=lambda: _manager_installed_update_overrides(force_refresh=not local_only),
        module_worktree_signature=_module_worktree_signature,
        build_node_snapshots=_build_node_snapshots,
    )


def _module_local_readme_summary(module_name: str) -> str | None:
    """Read and extract short description snippet from module README file."""
    return mb_module_local_readme_summary(
        module_name=module_name,
        custom_nodes_roots=_custom_nodes_roots,
    )


def _sanitize_module_description(text: str) -> str:
    """Normalize module description text for UI card rendering."""
    return mb_sanitize_module_description(text, _HTML_TAG_RE)


def _resolve_module_info(
    group: str,
    module_name: str,
    *,
    force_refresh: bool = False,
    sync_upstream: bool = False,
    cache_only: bool = False,
) -> dict[str, Any]:
    """Build complete module info payload with metadata, git state, and change markers."""
    group = (group or "").strip().lower()
    module_name = (module_name or "").strip()
    if group == "custom":
        module_name = _canonical_custom_module_name(module_name)

    key = (group or "", module_name or "", bool(cache_only))
    if force_refresh:
        _MODULE_INFO_CACHE.pop(key, None)
    now_ts = time.time()
    cached = _MODULE_INFO_CACHE.get(key)
    if cached is not None and (now_ts - cached[0]) < _MODULE_INFO_TTL_SEC:
        return dict(cached[1])
    result = mb_resolve_module_info_uncached(
        group=group,
        module_name=module_name,
        sync_upstream=sync_upstream,
        cache_only=cache_only,
        canonical_custom_module_name=_canonical_custom_module_name,
        apply_node_change_info=_apply_node_change_info,
        sync_module_upstream=_sync_module_upstream,
        load_module_state=_load_module_state,
        custom_update_checked_flag=_custom_update_checked_flag,
        module_git_state=_module_git_state,
        module_repo_url=_module_repo_url,
        manager_meta_for_module=_manager_meta_for_module,
        module_local_readme_summary=_module_local_readme_summary,
        sanitize_module_description=_sanitize_module_description,
        github_id=_github_id,
        infer_update_from_manager_stats=_infer_update_from_manager_stats,
        short_commit=_short_commit,
        remember_module_state=_remember_module_state,
    )
    _MODULE_INFO_CACHE[key] = (now_ts, dict(result))
    return result


def _collect_nodes() -> list[dict[str, Any]]:
    """Collect node definitions from registered ComfyUI mappings."""
    class_map, display_map = _node_mappings()
    return catalog_collect_nodes(
        class_map=class_map,
        display_map=display_map,
        annotation_resolver=lambda node_name, node_cls: _ALEXZ_ANNOTATIONS.get(node_name) or _fallback_annotation(node_cls),
        classifier=_classify_by_relative_module,
    )


def _build_catalog() -> dict[str, list[dict[str, Any]]]:
    """Build cached module-to-node catalog from discovered nodes."""
    return catalog_build_catalog(_collect_nodes())


def _build_group_catalog() -> dict[str, list[dict[str, Any]]]:
    """Build grouped node catalog for one category."""
    return catalog_build_group_catalog(_collect_nodes())


def _build_group_modules(grouped_nodes: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Build grouped module summaries for one category."""
    return catalog_build_group_modules(
        grouped_nodes=grouped_nodes,
        discover_custom_modules=_discover_custom_modules,
        cached_module_flags=_cached_module_flags,
    )


def _filter_modules(query: str, module_names: list[str]) -> list[str]:
    """Filter module list by case-insensitive text query over module names."""
    return catalog_filter_modules(query, module_names)


def _build_group_payload(
    grouped_nodes: dict[str, list[dict[str, Any]]],
    modules_by_group: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build ordered group payload for node-catalog API route."""
    return mb_build_group_payload(
        group_order=_GROUP_ORDER,
        grouped_nodes=grouped_nodes,
        modules_by_group=modules_by_group,
    )


def _build_module_list_payload(catalog: dict[str, list[dict[str, Any]]], query: str) -> dict[str, Any]:
    """Build module-list payload for module-list API route."""
    return mb_build_module_list_payload(catalog=catalog, query=query)


def _build_module_nodes_payload(catalog: dict[str, list[dict[str, Any]]], query: str) -> dict[str, Any]:
    """Build module-nodes payload for module-nodes API route."""
    return mb_build_module_nodes_payload(
        catalog=catalog,
        query=query,
        filter_modules_fn=_filter_modules,
    )


def _set_refresh_status(**kwargs: Any) -> None:
    """Set shared refresh job status fields in a thread-safe way."""
    set_refresh_status(
        lock=_API_STATE.refresh_lock,
        status=_API_STATE.refresh_status,
        now_iso=_now_iso,
        **kwargs,
    )


def _refresh_status_snapshot() -> dict[str, Any]:
    """Return thread-safe snapshot of refresh-job status."""
    return refresh_status_snapshot(lock=_API_STATE.refresh_lock, status=_API_STATE.refresh_status)


def _refresh_progress(
    *,
    phase: str,
    current: int = 0,
    total: int = 0,
    remaining: int = 0,
    modules_need_update: int = 0,
    modules_unknown_update: int = 0,
    unknown_update_modules: list[str] | None = None,
    module: str = "",
    message: str = "",
) -> None:
    """Update refresh-job progress counters and status text."""
    _API_STATE.refresh_log_last = emit_refresh_progress(
        lock=_API_STATE.refresh_lock,
        status=_API_STATE.refresh_status,
        now_iso=_now_iso,
        phase=phase,
        current=current,
        total=total,
        remaining=remaining,
        modules_need_update=modules_need_update,
        modules_unknown_update=modules_unknown_update,
        unknown_update_modules=unknown_update_modules,
        module=module,
        message=message,
        last_line=_API_STATE.refresh_log_last,
        logger_debug=lambda line: _LOGGER.debug("Module refresh: %s", line),
        console_log=lambda text, level="summary": _refresh_console_log(text, level=level),
    )


def _refresh_module_runtime_state(sync_upstreams: bool = False, progress_cb: Any | None = None) -> dict[str, Any]:
    """Recompute module snapshots and update persisted runtime tracking state."""
    _sync_runtime_warmup_from_legacy()

    def _reset_custom_alias_cache() -> None:
        """Reset custom module alias cache before recomputing runtime state."""
        global _CUSTOM_MODULE_ALIAS_CACHE
        global _MANAGER_UPDATE_OVERRIDE_CACHE
        _CUSTOM_MODULE_ALIAS_CACHE = None
        _MANAGER_UPDATE_OVERRIDE_CACHE = None

    if progress_cb is None:
        progress_cb = _refresh_progress
    result = mb_refresh_module_runtime_state(
        sync_upstreams=sync_upstreams,
        progress_cb=progress_cb,
        module_info_cache_clear=_MODULE_INFO_CACHE.clear,
        reset_custom_alias_cache=_reset_custom_alias_cache,
        clear_comfyui_status_cache=_clear_comfyui_status_cache,
        refresh_console_log=lambda text, level="summary": _refresh_console_log(text, level=level),
        get_update_console_log_mode=_get_update_console_log_mode,
        discover_custom_modules=_discover_custom_modules,
        sync_module_upstream=_sync_module_upstream,
        announce_tracked_module_updates=_announce_tracked_module_updates,
        comfyui_git_status=lambda: _comfyui_git_status(force_refresh=True),
        short_commit=_short_commit,
        set_custom_update_checked=_set_custom_update_checked,
        now_iso=_now_iso,
        perf_counter=time.perf_counter,
    )
    _API_STATE.lazy_refresh_done = True
    _sync_runtime_warmup_to_legacy()
    return result


def _ensure_runtime_state_ready() -> None:
    """Ensure runtime snapshot cache is initialized before serving API requests."""
    _sync_runtime_warmup_from_legacy()
    if _API_STATE.lazy_refresh_done:
        return
    _load_module_state()
    # On process start, hide stale remote-update status until user runs explicit refresh.
    _set_custom_update_checked(False)
    _announce_tracked_module_updates(local_only=True)
    _track_comfyui_local_update()
    _API_STATE.lazy_refresh_done = True
    _sync_runtime_warmup_to_legacy()


def _start_runtime_state_warmup() -> bool:
    """Start non-blocking runtime-state warmup once and return start status."""
    _sync_runtime_warmup_from_legacy()
    if _API_STATE.lazy_refresh_done:
        return False

    with _API_STATE.runtime_warmup_lock:
        _sync_runtime_warmup_from_legacy()
        if _API_STATE.lazy_refresh_done:
            return False
        existing = _API_STATE.runtime_warmup_thread
        if existing is not None and existing.is_alive():
            return False
        # Reset stale remote-update visibility immediately on first access.
        _set_custom_update_checked(False)

        def _runner() -> None:
            """Warmup worker for runtime state cache used by first-open UI paths."""
            try:
                _ensure_runtime_state_ready()
            except Exception as exc:  # pragma: no cover - diagnostic
                _LOGGER.warning("Runtime warmup failed: %s", exc, exc_info=True)
            finally:
                with _API_STATE.runtime_warmup_lock:
                    _API_STATE.runtime_warmup_thread = None
                    _sync_runtime_warmup_to_legacy()

        _API_STATE.runtime_warmup_thread = threading.Thread(
            target=_runner,
            name="ALEXZ_tools_RuntimeWarmup",
            daemon=True,
        )
        _sync_runtime_warmup_to_legacy()
        _API_STATE.runtime_warmup_thread.start()
        return True


def _runtime_warmup_status() -> dict[str, Any]:
    """Return lightweight runtime warmup state for frontend polling hints."""
    _sync_runtime_warmup_from_legacy()
    with _API_STATE.runtime_warmup_lock:
        thread = _API_STATE.runtime_warmup_thread
        running = bool(thread is not None and thread.is_alive())
    done = bool(_API_STATE.lazy_refresh_done)
    return {
        "running": running,
        "done": done,
    }


def _start_refresh_job(sync_upstreams: bool) -> dict[str, Any]:
    """Start background module refresh job if one is not already running."""
    return mb_api_start_refresh_job(
        state=_API_STATE,
        sync_upstreams=bool(sync_upstreams),
        now_iso=_now_iso,
        get_update_console_log_mode=_get_update_console_log_mode,
        refresh_console_log=lambda text, level="summary": _refresh_console_log(text, level=level),
        refresh_module_runtime_state=lambda do_sync: _refresh_module_runtime_state(
            sync_upstreams=do_sync,
            progress_cb=_refresh_progress,
        ),
        set_refresh_status=_set_refresh_status,
        refresh_status_snapshot=_refresh_status_snapshot,
    )


def _set_update_status(**kwargs: Any) -> None:
    """Set shared module-update job status fields in a thread-safe way."""
    set_update_status(
        lock=_API_STATE.update_lock,
        status=_API_STATE.update_status,
        now_iso=_now_iso,
        **kwargs,
    )
    line = format_update_status_line(_API_STATE.update_status)
    if line != _API_STATE.update_log_last:
        _API_STATE.update_log_last = line
        _LOGGER.info("Module update: %s", line)


def _update_status_snapshot() -> dict[str, Any]:
    """Return thread-safe snapshot of module-update job status."""
    return update_status_snapshot(lock=_API_STATE.update_lock, status=_API_STATE.update_status)


def _resolve_update_targets(scope: str, module_name: str) -> list[str]:
    """Resolve concrete module names targeted by update request payload."""
    return resolve_update_targets(
        scope=scope,
        module_name=module_name,
        canonical_module_name=_canonical_custom_module_name,
        discover_modules=_discover_custom_modules,
        sync_module_upstream=_sync_module_upstream,
        module_needs_update=_module_needs_update_now,
        update_console_log=lambda text, level="summary": _update_console_log(text, level=level),
        workers=_UPDATE_TARGET_SCAN_WORKERS,
        warn=lambda text: _LOGGER.warning("%s", text),
    )


def _start_module_update_job(scope: str, module_name: str, log_mode: str = "summary") -> dict[str, Any]:
    """Start background module update job for selected custom modules."""
    scope_norm = (scope or "").strip().lower()
    normalized_log_mode = _set_update_console_log_mode(log_mode)
    module_exists = True
    if scope_norm == "single":
        canonical = _canonical_custom_module_name(module_name)
        module_exists = _module_dir(canonical) is not None
    comfyui_root_exists = _comfyui_root() is not None
    with _API_STATE.refresh_lock:
        refresh_running = bool(
            _API_STATE.refresh_thread is not None and _API_STATE.refresh_thread.is_alive()
        )
    return mb_api_start_module_update_job(
        state=_API_STATE,
        scope_norm=scope_norm,
        module_name=module_name,
        normalized_log_mode=normalized_log_mode,
        now_iso=_now_iso,
        comfyui_root_exists=comfyui_root_exists,
        single_module_exists=module_exists,
        refresh_running=refresh_running,
        update_status_snapshot=_update_status_snapshot,
        update_console_log=lambda text, level="summary": _update_console_log(text, level=level),
        run_module_update_job=lambda **kwargs: mb_run_module_update_job(
            **kwargs,
            perf_counter=time.perf_counter,
        ),
        resolve_update_targets=_resolve_update_targets,
        pull_comfyui=_pull_comfyui,
        pull_custom_module=_pull_custom_module,
        refresh_module_runtime_state=lambda: _refresh_module_runtime_state(
            sync_upstreams=False,
            progress_cb=lambda **kwargs: None,
        ),
        set_update_status=_set_update_status,
    )


def _install_requirements_for_modules(modules: list[str]) -> dict[str, Any]:
    """Install requirements.txt for a list of modules after update confirmation."""
    return mb_install_requirements_for_modules(
        modules,
        canonical_custom_module_name=_canonical_custom_module_name,
        install_module_requirements_fn=_install_module_requirements,
        logger=_LOGGER,
    )


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None):
    mb_api_register_routes(
        PromptServer=PromptServer,
        web=web,
        api_module=sys.modules[__name__],
        logger=_LOGGER,
    )
