"""
Module: utils/module_node_browser_api.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Backend API for Module Node Picker widget.

Purpose:
    Implements catalog, module info, refresh/update jobs, git status tracking,
    requirements installation routes, and Slice 0 component-registry endpoints.
"""

from __future__ import annotations


import importlib
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from hashlib import sha1
from collections import defaultdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

from .module_browser import (
    build_default_component_registry,
    build_registry_snapshot,
    build_component_health_report,
    compute_snapshot_signature,
    ensure_module_state_schema,
)
from .module_browser.api_manifest import (
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
from .module_browser.module_info_text import (
    module_local_readme_summary as mb_module_local_readme_summary,
    sanitize_module_description as mb_sanitize_module_description,
)
from .module_browser.module_info import (
    cached_module_flags as mb_cached_module_flags,
    resolve_module_info_uncached as mb_resolve_module_info_uncached,
)
from .module_browser.git_helpers import (
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
from .module_browser.update_ops import (
    install_comfyui_requirements as mb_install_comfyui_requirements,
    install_module_requirements as mb_install_module_requirements,
    requirements_changed_between as mb_requirements_changed_between,
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
_MODULE_STATE_CACHE: dict[str, dict[str, Any]] | None = None
_CUSTOM_MODULE_ALIAS_CACHE: dict[str, str] | None = None
_COMFYUI_STATUS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_COMFYUI_STATUS_TTL_SEC = 120.0
_LAZY_REFRESH_DONE = False
_RUNTIME_WARMUP_LOCK = threading.Lock()
_RUNTIME_WARMUP_THREAD: threading.Thread | None = None
_REFRESH_LOCK = threading.Lock()
_REFRESH_THREAD: threading.Thread | None = None
_REFRESH_LOG_LAST = ""
_REFRESH_CONSOLE_LOG_LAST = ""
_REFRESH_STATUS: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "current": 0,
    "total": 0,
    "remaining": 0,
    "modules_need_update": 0,
    "modules_unknown_update": 0,
    "module": "",
    "message": "",
    "error": "",
    "sync_upstreams": False,
    "started_at": "",
    "updated_at": "",
    "refreshed_at": "",
}
_UPDATE_LOCK = threading.Lock()
_UPDATE_THREAD: threading.Thread | None = None
_UPDATE_LOG_LAST = ""
_UPDATE_CONSOLE_LOG_MODE = "summary"
_UPDATE_STATUS: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "scope": "",
    "current": 0,
    "total": 0,
    "remaining": 0,
    "module": "",
    "message": "",
    "error": "",
    "updated": 0,
    "up_to_date": 0,
    "failed": 0,
    "requirements_changed": False,
    "requirements_modules": [],
    "results": [],
    "started_at": "",
    "updated_at": "",
    "finished_at": "",
}
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


def _custom_update_checked_flag(state: dict[str, Any] | None = None) -> bool:
    """Return whether custom-module remote update check was explicitly run in current session."""
    cache = state if isinstance(state, dict) else _load_module_state()
    meta = cache.get("__meta__") if isinstance(cache, dict) else None
    if not isinstance(meta, dict):
        return False
    return bool(meta.get("custom_update_checked"))


def _info_only_rejection_payload(feature: str) -> dict[str, Any]:
    """Build a consistent rejection payload for disabled mutate operations."""
    return {
        "status": "disabled",
        "feature": feature,
        "message": "This widget runs in info-only mode. Use ComfyUI-Manager for install/update actions.",
    }


def _set_custom_update_checked(checked: bool) -> None:
    """Persist custom-module update-check visibility gate for initial widget state."""
    global _MODULE_INFO_CACHE
    state = _load_module_state()
    if not isinstance(state, dict):
        return
    meta_raw = state.get("__meta__")
    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
    value = bool(checked)
    if bool(meta.get("custom_update_checked")) == value:
        return
    meta["custom_update_checked"] = value
    meta["custom_update_checked_at"] = _now_iso()
    state["__meta__"] = meta
    _save_module_state(state)
    _MODULE_INFO_CACHE.clear()


def _normalize_log_mode(value: str | None) -> str:
    """Normalize console log mode for update jobs."""
    text = str(value or "").strip().lower()
    return "verbose" if text in {"verbose", "debug", "full", "detailed"} else "summary"


def _set_update_console_log_mode(mode: str | None) -> str:
    """Set active console log mode for update jobs and return normalized value."""
    global _UPDATE_CONSOLE_LOG_MODE
    normalized = _normalize_log_mode(mode)
    with _UPDATE_LOCK:
        _UPDATE_CONSOLE_LOG_MODE = normalized
    return normalized


def _get_update_console_log_mode() -> str:
    """Read active console log mode for update jobs."""
    with _UPDATE_LOCK:
        return _UPDATE_CONSOLE_LOG_MODE


def _update_console_log(message: str, level: str = "summary") -> None:
    """Print update-progress line to ComfyUI console according to selected log mode."""
    if _normalize_log_mode(level) == "verbose" and _get_update_console_log_mode() != "verbose":
        return
    text = str(message or "").strip()
    if not text:
        return
    try:
        print(f"ALEXZ_tools Module update: {text}", flush=True)
    except Exception:
        pass


def _refresh_console_log(message: str, level: str = "summary") -> None:
    """Print refresh-progress line to ComfyUI console according to selected log mode."""
    global _REFRESH_CONSOLE_LOG_LAST
    if _normalize_log_mode(level) == "verbose" and _get_update_console_log_mode() != "verbose":
        return
    text = str(message or "").strip()
    if not text:
        return
    if text == _REFRESH_CONSOLE_LOG_LAST:
        return
    _REFRESH_CONSOLE_LOG_LAST = text
    try:
        print(f"ALEXZ_tools Module refresh: {text}", flush=True)
    except Exception:
        pass


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
    value = (commit or "").strip()
    if not value:
        return "unknown"
    return value[:8]


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
    if (
        not force_refresh
        and isinstance(_COMPONENT_REGISTRY_PAYLOAD_CACHE, tuple)
        and len(_COMPONENT_REGISTRY_PAYLOAD_CACHE) == 2
        and (now_ts - float(_COMPONENT_REGISTRY_PAYLOAD_CACHE[0])) < _COMPONENT_REGISTRY_TTL_SEC
    ):
        return dict(_COMPONENT_REGISTRY_PAYLOAD_CACHE[1])

    registry = build_default_component_registry()
    state = _load_module_state()
    tracker_raw = state.get("__component_registry__") if isinstance(state, dict) else None
    tracker = dict(tracker_raw) if isinstance(tracker_raw, dict) else {}
    prev_snapshot_raw = tracker.get("snapshot")
    prev_snapshot = dict(prev_snapshot_raw) if isinstance(prev_snapshot_raw, dict) else {}

    node_entries = [entry.to_dict() for entry in registry.list("node")]
    widget_entries = [entry.to_dict() for entry in registry.list("widget")]
    api_entries = [entry.to_dict() for entry in registry.list("api")]
    current_snapshot = build_registry_snapshot(registry)
    current_signature = compute_snapshot_signature(current_snapshot)
    previous_signature = str(tracker.get("manifest_signature") or "")

    changes: dict[str, dict[str, list[str]]] = {}
    has_changes = False
    for kind in ("node", "widget", "api"):
        prev_ids = {str(x) for x in (prev_snapshot.get(kind) or []) if str(x)}
        curr_ids = {str(x) for x in (current_snapshot.get(kind) or []) if str(x)}
        added = sorted(curr_ids - prev_ids, key=str.lower)
        removed = sorted(prev_ids - curr_ids, key=str.lower)
        if added or removed:
            has_changes = True
        changes[kind] = {"added": added, "removed": removed}

    payload = {
        "schema_name": COMPONENT_REGISTRY_SCHEMA_NAME,
        "schema_version": COMPONENT_REGISTRY_SCHEMA_VERSION,
        "summary": registry.summary(),
        "health": build_component_health_report(),
        "nodes": node_entries,
        "widgets": widget_entries,
        "apis": api_entries,
        "changes": changes,
        "has_changes": has_changes,
        "manifest_signature": current_signature,
        "manifest_changed": bool(previous_signature and previous_signature != current_signature),
        "previous_snapshot_at": str(tracker.get("updated_at") or ""),
        "refreshed_at": _now_iso(),
    }

    if (
        not isinstance(tracker_raw, dict)
        or tracker.get("snapshot") != current_snapshot
        or str(tracker.get("manifest_signature") or "") != current_signature
    ):
        state["__component_registry__"] = {
            "schema_name": COMPONENT_REGISTRY_SCHEMA_NAME,
            "schema_version": COMPONENT_REGISTRY_SCHEMA_VERSION,
            "snapshot": current_snapshot,
            "manifest_signature": current_signature,
            "summary": dict(payload["summary"]),
            "updated_at": payload["refreshed_at"],
        }
        _save_module_state(state)

    _COMPONENT_REGISTRY_PAYLOAD_CACHE = (now_ts, dict(payload))
    return payload


def _node_mappings() -> tuple[dict[str, Any], dict[str, str]]:
    """Return NODE_CLASS_MAPPINGS from loaded extension modules."""
    comfy_nodes = importlib.import_module("nodes")
    class_map = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    display_map = getattr(comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
    return class_map, display_map


def _node_source_file(node_cls: Any) -> str:
    """Resolve source file path for a node class object."""
    source_file = ""
    try:
        source_file = inspect.getsourcefile(node_cls) or ""
    except Exception:
        source_file = ""
    if source_file:
        try:
            return str(Path(source_file).resolve())
        except Exception:
            return source_file

    module_name = getattr(node_cls, "__module__", "") or ""
    module_obj = sys.modules.get(module_name)
    module_file = getattr(module_obj, "__file__", "") if module_obj is not None else ""
    if not module_file:
        return ""
    try:
        return str(Path(module_file).resolve())
    except Exception:
        return module_file


def _relative_to_custom_roots(path_text: str) -> str:
    """Resolve path relative to known custom_nodes roots when possible."""
    if not path_text:
        return ""
    try:
        path_obj = Path(path_text).resolve()
    except Exception:
        return path_text
    for root in _custom_nodes_roots():
        try:
            return str(path_obj.relative_to(root.resolve()))
        except Exception:
            continue
    return str(path_obj)


def _file_digest(path_text: str) -> str:
    """Compute SHA1 digest for file content used in node-change tracking."""
    if not path_text:
        return ""
    try:
        data = Path(path_text).read_bytes()
        return sha1(data).hexdigest()[:12]
    except Exception:
        return ""


def _build_node_snapshots() -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Build stable per-node file snapshots used to detect node additions/changes."""
    class_map, _ = _node_mappings()
    snapshots: dict[str, dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    digest_cache: dict[str, str] = {}
    for node_name, node_cls in class_map.items():
        group, module_bucket = _classify_by_relative_module(node_cls)
        source_file = _node_source_file(node_cls)
        digest = digest_cache.get(source_file)
        if digest is None:
            digest = _file_digest(source_file)
            digest_cache[source_file] = digest
        snapshots[group][module_bucket][node_name] = {
            "sig": f"{getattr(node_cls, '__name__', '')}:{digest}",
            "source": _relative_to_custom_roots(source_file),
        }

    out: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for group, modules in snapshots.items():
        out[group] = {}
        for module_name, nodes in modules.items():
            out[group][module_name] = dict(sorted(nodes.items(), key=lambda kv: kv[0].lower()))
    return out


def _module_root(node_cls: Any) -> str:
    """Resolve module root directory for a file path inside the extension."""
    module_name = getattr(node_cls, "__module__", "") or ""
    if not module_name:
        return "unknown"
    return module_name.split(".", 1)[0]


def _classify_by_relative_module(node_cls: Any) -> tuple[str, str]:
    """Classify node group and module name using path relative to ComfyUI roots."""
    rel = getattr(node_cls, "RELATIVE_PYTHON_MODULE", None)
    if isinstance(rel, str) and rel:
        parts = [p for p in rel.split(".") if p]
        if len(parts) >= 2:
            root, module_name = parts[0], parts[1]
        elif len(parts) == 1:
            root, module_name = parts[0], parts[0]
        else:
            root, module_name = "", ""

        if root == "custom_nodes":
            return ("custom", _canonical_custom_module_name(module_name))
        if root == "comfy_extras":
            return ("core_extras", module_name)
        if root == "comfy_api_nodes":
            return ("api", module_name)

    source_hit = _classify_by_source_path(node_cls)
    if source_hit is not None:
        return source_hit

    module_name = getattr(node_cls, "__module__", "") or ""
    module_l = module_name.lower()
    if module_l.startswith("comfy_extras."):
        parts = module_name.split(".")
        return ("core_extras", parts[1] if len(parts) > 1 else module_name)
    if module_l.startswith("comfy_api_nodes."):
        parts = module_name.split(".")
        return ("api", parts[1] if len(parts) > 1 else module_name)
    return ("core", _module_root(node_cls))


def _fallback_annotation(node_cls: Any) -> str:
    """Build fallback node annotation from class metadata when no static annotation exists."""
    category = getattr(node_cls, "CATEGORY", "") or "unknown"
    return_names = getattr(node_cls, "RETURN_NAMES", None)
    if not return_names:
        return_types = getattr(node_cls, "RETURN_TYPES", ())
        return_names = return_types

    if return_names is None:
        output_items = []
    elif isinstance(return_names, (str, bytes)):
        output_items = [str(return_names)]
    else:
        try:
            output_items = [str(x) for x in islice(iter(return_names), 3)]
        except Exception:
            output_items = [str(return_names)]

    outputs = ", ".join(output_items) or "unknown"
    return f"Категория: {category}. Выходы: {outputs}."


def _custom_nodes_roots() -> list[Path]:
    """Return existing custom_nodes root directories."""
    if folder_paths is not None and hasattr(folder_paths, "get_folder_paths"):
        try:
            roots = [Path(x) for x in folder_paths.get_folder_paths("custom_nodes") if x]
            if roots:
                return roots
        except Exception:
            pass
    return [Path(__file__).resolve().parents[1]]


def _discover_custom_modules() -> list[str]:
    """Discover installed custom module directories under custom_nodes roots."""
    names: set[str] = set()
    for root in _custom_nodes_roots():
        if not root.exists():
            continue
        try:
            entries = list(root.iterdir())
        except Exception:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            name = entry.name
            if not name or name.startswith(".") or name == "__pycache__":
                continue
            has_markers = (
                (entry / "__init__.py").exists()
                or (entry / "pyproject.toml").exists()
                or any(entry.glob("*.py"))
            )
            if has_markers:
                names.add(name)
    return sorted(names, key=str.lower)


def _normalize_module_token(name: str) -> str:
    """Normalize module token for case-insensitive matching and aliases."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _custom_module_aliases() -> dict[str, str]:
    """Build alias map for custom module names and normalized tokens."""
    global _CUSTOM_MODULE_ALIAS_CACHE
    if _CUSTOM_MODULE_ALIAS_CACHE is not None:
        return _CUSTOM_MODULE_ALIAS_CACHE

    aliases: dict[str, str] = {}
    for module_name in _discover_custom_modules():
        aliases[module_name] = module_name
        aliases[module_name.lower()] = module_name
        norm = _normalize_module_token(module_name)
        if norm and norm not in aliases:
            aliases[norm] = module_name

    _CUSTOM_MODULE_ALIAS_CACHE = aliases
    return aliases


def _canonical_custom_module_name(module_name: str) -> str:
    """Resolve user-provided module token to canonical custom module name."""
    name = (module_name or "").strip()
    if not name:
        return "unknown"

    aliases = _custom_module_aliases()
    direct = aliases.get(name) or aliases.get(name.lower())
    if direct:
        return direct

    norm = _normalize_module_token(name)
    if norm:
        matched = aliases.get(norm)
        if matched:
            return matched
    return name


def _classify_by_source_path(node_cls: Any) -> tuple[str, str] | None:
    """Classify node into core/extras/api/custom groups from source path."""
    source = _node_source_file(node_cls)
    if not source:
        return None

    try:
        src_path = Path(source).resolve()
    except Exception:
        return None

    for root in _custom_nodes_roots():
        try:
            rel = src_path.relative_to(root.resolve())
        except Exception:
            continue
        if rel.parts:
            return ("custom", _canonical_custom_module_name(rel.parts[0]))

    parts_l = [p.lower() for p in src_path.parts]
    if "comfy_extras" in parts_l:
        idx = parts_l.index("comfy_extras")
        module_name = src_path.parts[idx + 1] if (idx + 1) < len(src_path.parts) else _module_root(node_cls)
        return ("core_extras", module_name)
    if "comfy_api_nodes" in parts_l:
        idx = parts_l.index("comfy_api_nodes")
        module_name = src_path.parts[idx + 1] if (idx + 1) < len(src_path.parts) else _module_root(node_cls)
        return ("api", module_name)
    return None


def _normalize_repo_url(url: str | None) -> str | None:
    """Normalize repository URL to canonical HTTPS GitHub form."""
    if not isinstance(url, str):
        return None
    value = url.strip()
    if not value:
        return None
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value[len("git@github.com:") :]
    elif value.startswith("git://github.com/"):
        value = "https://github.com/" + value[len("git://github.com/") :]
    if value.endswith(".git"):
        value = value[:-4]
    return value.rstrip("/")


def _github_id(url: str | None) -> str | None:
    """Extract owner/repository identifier from normalized GitHub URL."""
    norm = _normalize_repo_url(url)
    if not norm:
        return None
    match = _GITHUB_RE.search(norm)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}".lower()


def _repo_name(url: str | None) -> str | None:
    """Return repository name parsed from module URL."""
    gid = _github_id(url)
    if not gid:
        return None
    return gid.split("/", 1)[1]


def _pick_repo_url(entry: dict[str, Any]) -> str | None:
    """Choose best repository URL from module metadata candidates."""
    candidates: list[str] = []
    for key in ("repository", "reference"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
    files = entry.get("files")
    if isinstance(files, list):
        candidates.extend(x for x in files if isinstance(x, str))
    for candidate in candidates:
        norm = _normalize_repo_url(candidate)
        if norm and "github.com/" in norm.lower():
            return norm
    return _normalize_repo_url(candidates[0]) if candidates else None


def _manager_custom_db_path() -> Path | None:
    """Return path to ComfyUI-Manager custom-node database file."""
    for root in _custom_nodes_roots():
        db_path = root / "comfyui-manager" / "custom-node-list.json"
        if db_path.exists():
            return db_path
    return None


def _manager_github_stats_path() -> Path | None:
    """Return path to cached GitHub-stats file maintained by ComfyUI-Manager."""
    for root in _custom_nodes_roots():
        db_path = root / "comfyui-manager" / "github-stats.json"
        if db_path.exists():
            return db_path
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime text from manager metadata into timezone-aware object."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _to_iso(dt: datetime | None) -> str | None:
    """Convert datetime value to ISO-8601 string in UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _set_comfyui_requirements_pending(pending: bool, before_commit: str = "", after_commit: str = "") -> None:
    """Persist pending ComfyUI requirements-install marker in module state cache."""
    global _COMFYUI_STATUS_CACHE
    state = _load_module_state()
    if not isinstance(state, dict):
        return
    entry_raw = state.get("__comfyui__")
    before_entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    if pending:
        entry["pending_requirements_update"] = True
        if before_commit:
            entry["pending_requirements_before_commit"] = before_commit
        if after_commit:
            entry["pending_requirements_after_commit"] = after_commit
        entry["pending_requirements_updated_at"] = _now_iso()
    else:
        entry.pop("pending_requirements_update", None)
        entry.pop("pending_requirements_before_commit", None)
        entry.pop("pending_requirements_after_commit", None)
        entry.pop("pending_requirements_updated_at", None)
    if entry == before_entry:
        return
    state["__comfyui__"] = entry
    _clear_comfyui_status_cache()
    _save_module_state(state)


def _set_module_requirements_pending(
    module_name: str, pending: bool, before_commit: str = "", after_commit: str = ""
) -> None:
    """Persist pending requirements-install marker for one custom module."""
    module = _canonical_custom_module_name(module_name)
    if not module or module == "unknown":
        return
    state = _load_module_state()
    if not isinstance(state, dict):
        return
    entry_raw = state.get(module)
    before_entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    if pending:
        entry["pending_requirements_update"] = True
        if before_commit:
            entry["pending_requirements_before_commit"] = before_commit
        if after_commit:
            entry["pending_requirements_after_commit"] = after_commit
        entry["pending_requirements_updated_at"] = _now_iso()
    else:
        entry.pop("pending_requirements_update", None)
        entry.pop("pending_requirements_before_commit", None)
        entry.pop("pending_requirements_after_commit", None)
        entry.pop("pending_requirements_updated_at", None)
    if entry == before_entry:
        return
    state[module] = entry
    _MODULE_INFO_CACHE.clear()
    _save_module_state(state)


def _normalize_comfyui_mode(value: str | None) -> str:
    """Normalize ComfyUI update-check mode to supported values."""
    text = (value or "").strip().lower()
    if text in {"commit", "commits", "git"}:
        return "commits"
    return "releases"


def _github_latest_release(owner: str, repo: str, timeout: float = 8.0) -> dict[str, Any]:
    """Fetch latest GitHub release metadata for a repository."""
    owner_text = (owner or "").strip()
    repo_text = (repo or "").strip()
    if not owner_text or not repo_text:
        return {}
    url = f"https://api.github.com/repos/{owner_text}/{repo_text}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ALEXZ_tools-module-picker",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 429}:
            return {}
        return {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return {}
    return {
        "tag_name": tag,
        "published_at": str(payload.get("published_at") or "").strip(),
        "created_at": str(payload.get("created_at") or "").strip(),
        "name": str(payload.get("name") or "").strip(),
        "html_url": str(payload.get("html_url") or "").strip(),
    }


def _manager_github_stats() -> dict[str, dict[str, dict[str, Any]]]:
    """Load and cache module update timestamps from manager stats file."""
    global _MANAGER_GITHUB_STATS_CACHE
    if _MANAGER_GITHUB_STATS_CACHE is not None:
        return _MANAGER_GITHUB_STATS_CACHE

    stats = {"by_url": {}, "by_github": {}}
    db_path = _manager_github_stats_path()
    if db_path is None:
        _MANAGER_GITHUB_STATS_CACHE = stats
        return stats
    try:
        with db_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        _LOGGER.warning("Failed to load ComfyUI-Manager github stats: %s", exc)
        _MANAGER_GITHUB_STATS_CACHE = stats
        return stats

    if not isinstance(payload, dict):
        _MANAGER_GITHUB_STATS_CACHE = stats
        return stats
    for raw_url, raw_meta in payload.items():
        if not isinstance(raw_meta, dict):
            continue
        url_text = str(raw_url).strip().replace("htps://", "https://")
        norm_url = _normalize_repo_url(url_text)
        if not norm_url:
            continue
        stats["by_url"][norm_url] = raw_meta
        gid = _github_id(norm_url)
        if gid:
            stats["by_github"][gid] = raw_meta
    _MANAGER_GITHUB_STATS_CACHE = stats
    return stats


def _manager_index() -> dict[str, dict[str, dict[str, Any]]]:
    """Load and cache manager metadata index for custom modules."""
    global _MANAGER_INDEX_CACHE
    if _MANAGER_INDEX_CACHE is not None:
        return _MANAGER_INDEX_CACHE

    index = {
        "by_id": {},
        "by_github": {},
        "by_repo_name": {},
    }
    db_path = _manager_custom_db_path()
    if db_path is None:
        _MANAGER_INDEX_CACHE = index
        return index

    try:
        with db_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        _LOGGER.warning("Failed to load ComfyUI-Manager DB: %s", exc)
        _MANAGER_INDEX_CACHE = index
        return index

    entries = payload.get("custom_nodes", []) if isinstance(payload, dict) else []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        title = (raw.get("title") or "").strip()
        author = (raw.get("author") or "").strip()
        description = (raw.get("description") or "").strip()
        node_id = (raw.get("id") or "").strip().lower()
        repo_url = _pick_repo_url(raw)

        meta = {
            "title": title,
            "author": author,
            "description": description,
            "repository": repo_url,
        }
        if node_id:
            index["by_id"][node_id] = meta
        gid = _github_id(repo_url)
        if gid:
            index["by_github"][gid] = meta
        repo = _repo_name(repo_url)
        if repo:
            index["by_repo_name"][repo.lower()] = meta

    _MANAGER_INDEX_CACHE = index
    return index


def _manager_meta_for_module(module_name: str, repository_url: str | None = None) -> dict[str, Any] | None:
    """Resolve ComfyUI-Manager metadata record for module by id/repository aliases."""
    module_l = _canonical_custom_module_name(module_name).lower()
    repo_norm = _normalize_repo_url(repository_url)
    repo_gid = _github_id(repo_norm)
    repo_name = _repo_name(repo_norm)
    manager_data = _manager_index()
    if repo_gid:
        meta = manager_data["by_github"].get(repo_gid)
        if isinstance(meta, dict):
            return meta
    if module_l:
        meta = manager_data["by_id"].get(module_l)
        if isinstance(meta, dict):
            return meta
    if repo_name:
        meta = manager_data["by_repo_name"].get(repo_name.lower())
        if isinstance(meta, dict):
            return meta
    return None


def _manager_stats_last_update(repository_url: str | None) -> str:
    """Return normalized last-update timestamp from Manager GitHub stats for repository URL."""
    norm_repo = _normalize_repo_url(repository_url)
    if not norm_repo:
        return ""
    stats = _manager_github_stats()
    stats_meta = stats["by_url"].get(norm_repo)
    if stats_meta is None:
        repo_gid = _github_id(norm_repo)
        if repo_gid:
            stats_meta = stats["by_github"].get(repo_gid)
    if not isinstance(stats_meta, dict):
        return ""
    remote_raw = stats_meta.get("last_update")
    remote_dt = _parse_datetime(remote_raw)
    return _to_iso(remote_dt) or ""


def _infer_update_from_manager_stats(
    repository_url: str | None,
    installed_updated_at: str | None,
) -> tuple[bool | None, str]:
    """Infer update availability from Manager GitHub stats when git upstream is unavailable."""
    remote_updated_at = _manager_stats_last_update(repository_url)
    if not remote_updated_at:
        return (None, "")
    local_dt = _parse_datetime(installed_updated_at)
    remote_dt = _parse_datetime(remote_updated_at)
    if local_dt is None or remote_dt is None:
        return (None, remote_updated_at)
    # Keep a small tolerance for second-level timestamp differences.
    needs_update = (remote_dt - local_dt).total_seconds() > 60.0
    return (needs_update, remote_updated_at)


def _run_git(args: list[str], timeout: float = 2.0) -> str | None:
    """Run git command in non-interactive mode and return trimmed stdout on success."""
    result = _run_command(args, timeout=timeout, disable_git_prompt=True)
    if not result.get("ok"):
        return None
    out = str(result.get("stdout") or "").strip()
    return out or None


def _extract_git_repo_from_args(args: list[str]) -> str | None:
    """Extract normalized git working directory from `git -C <path>` argument list."""
    if not args or str(args[0]).strip() != "git":
        return None
    try:
        idx = args.index("-C")
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    try:
        return str(Path(str(args[idx + 1])).resolve())
    except Exception:
        return str(args[idx + 1])


def _is_git_dubious_ownership_error(text: str) -> bool:
    """Check whether git stderr/stdout indicates `safe.directory` ownership protection."""
    lower = (text or "").strip().lower()
    return "detected dubious ownership in repository" in lower and "safe.directory" in lower


def _try_mark_git_safe_directory(repo_dir: str, env: dict[str, str], timeout: float = 15.0) -> bool:
    """Attempt to add repository path to git safe.directory list."""
    repo = str(repo_dir or "").strip()
    if not repo:
        return False
    try:
        proc = subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", repo],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        _LOGGER.warning("Failed to add safe.directory for %s: %s", repo, exc)
        return False
    if proc.returncode == 0:
        _LOGGER.info("Added git safe.directory: %s", repo)
        return True
    _LOGGER.warning(
        "Unable to add safe.directory for %s: %s",
        repo,
        (proc.stderr or proc.stdout or "unknown error").strip(),
    )
    return False


def _run_command(args: list[str], timeout: float = 120.0, disable_git_prompt: bool = False) -> dict[str, Any]:
    """Run a subprocess command and return exit code plus output text."""
    env = os.environ.copy()
    if disable_git_prompt:
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_ASKPASS", "echo")
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}
    result = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }
    if result["ok"] or not args or str(args[0]).strip() != "git":
        return result

    repo_dir = _extract_git_repo_from_args(args)
    if not repo_dir:
        return result

    error_text = f"{result.get('stderr', '')}\n{result.get('stdout', '')}"
    if not _is_git_dubious_ownership_error(error_text):
        return result

    if not _try_mark_git_safe_directory(repo_dir, env):
        return result

    try:
        proc_retry = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(exc)}
    return {
        "ok": proc_retry.returncode == 0,
        "returncode": proc_retry.returncode,
        "stdout": (proc_retry.stdout or "").strip(),
        "stderr": (proc_retry.stderr or "").strip(),
    }


def _tail_lines(text: str | None, max_lines: int = 80) -> str:
    """Return tail lines from command output for concise console diagnostics."""
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(["...", *lines[-max_lines:]])


def _is_git_local_changes_block(text: str | None) -> bool:
    """Check whether git pull failed because local changes must be stashed/committed first."""
    lower = str(text or "").strip().lower()
    if not lower:
        return False
    markers = (
        # English git messages.
        "please commit your changes or stash them before you merge",
        "your local changes to the following files would be overwritten by merge",
        "local changes would be overwritten by merge",
        "the following untracked working tree files would be overwritten by merge",
        "please move or remove them before you merge",
        "cannot pull with rebase: you have unstaged changes",
        "cannot pull with rebase: your index contains uncommitted changes",
        # Russian git messages (localized output).
        "сделайте коммит или спрячьте ваши изменения перед слиянием веток",
        "ваши локальные изменения в указанных файлах будут перезаписаны при слиянии",
        "указанные неотслеживаемые файлы в рабочем каталоге будут перезаписаны при слиянии",
        "переместите эти файлы или удалите их перед переключением веток",
    )
    return any(marker in lower for marker in markers)


def _module_dir(module_name: str) -> Path | None:
    """Resolve filesystem directory for a custom module by name."""
    module_name = _canonical_custom_module_name((module_name or "").strip())
    if not module_name:
        return None
    for root in _custom_nodes_roots():
        module_dir = root / module_name
        if module_dir.exists() and module_dir.is_dir():
            return module_dir
    return None


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
    module = _canonical_custom_module_name(module_name)
    state = _load_module_state()
    entry = state.get(module) if isinstance(state, dict) else None
    cached_update: bool | None = None
    if isinstance(entry, dict):
        value = entry.get("update_available")
        if isinstance(value, bool):
            cached_update = value

    git_state = _module_git_state(module)
    if not git_state:
        repository = ""
        installed_updated_at = ""
        if isinstance(entry, dict):
            repository = str(entry.get("repository") or "")
            installed_updated_at = str(entry.get("installed_updated_at") or "")
        if not repository:
            meta = _manager_meta_for_module(module, repository)
            if isinstance(meta, dict):
                repository = str(meta.get("repository") or "")
        inferred, _ = _infer_update_from_manager_stats(repository, installed_updated_at)
        if isinstance(inferred, bool):
            return inferred
        return bool(cached_update)
    behind = git_state.get("behind")
    if isinstance(behind, int):
        return behind > 0
    remote_head = (git_state.get("remote_head") or "").strip()
    installed = (git_state.get("installed_commit") or "").strip()
    if bool(git_state.get("has_upstream") and remote_head and installed):
        return remote_head != installed
    repository = str(git_state.get("repository") or "")
    installed_updated_at = str(git_state.get("installed_updated_at") or "")
    if not repository and isinstance(entry, dict):
        repository = str(entry.get("repository") or "")
    if not installed_updated_at and isinstance(entry, dict):
        installed_updated_at = str(entry.get("installed_updated_at") or "")
    if not repository:
        meta = _manager_meta_for_module(module, repository)
        if isinstance(meta, dict):
            repository = str(meta.get("repository") or "")
    inferred, _ = _infer_update_from_manager_stats(repository, installed_updated_at)
    if isinstance(inferred, bool):
        return inferred
    return bool(cached_update)


def _module_worktree_signature(module_name: str) -> str:
    """Return short signature of local uncommitted changes for module worktree."""
    return mb_module_worktree_signature(
        module_name,
        module_dir_resolver=_module_dir,
        run_git=_run_git,
    )


def _count_custom_modules_need_update() -> int:
    """Count custom modules that currently report available updates."""
    state = _load_module_state()
    if not isinstance(state, dict):
        return 0
    count = 0
    for module_name in _discover_custom_modules():
        entry = state.get(_canonical_custom_module_name(module_name))
        if isinstance(entry, dict) and bool(entry.get("update_available")):
            count += 1
    return count


def _count_custom_modules_unknown_update() -> int:
    """Count custom modules whose remote update status is unknown/uncheckable."""
    state = _load_module_state()
    if not isinstance(state, dict):
        return 0
    count = 0
    for module_name in _discover_custom_modules():
        entry = state.get(_canonical_custom_module_name(module_name))
        if not isinstance(entry, dict):
            count += 1
            continue
        if not isinstance(entry.get("update_available"), bool):
            count += 1
    return count


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
    root = _comfyui_root()
    if root is None:
        return None
    req = root / "requirements.txt"
    return req if req.exists() else None


def _comfyui_needs_update_now() -> bool:
    """Check whether local ComfyUI commit is behind remote tracking commit."""
    status = _comfyui_git_status(force_refresh=True, mode="releases")
    behind = status.get("behind")
    if isinstance(behind, int):
        return behind > 0
    return bool(status.get("update_status") == "can_update")


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
    root = _comfyui_root()
    result: dict[str, Any] = {
        "module": "ComfyUI",
        "status": "error",
        "message": "",
        "updated": False,
        "requirements_changed": False,
        "before_commit": "",
        "after_commit": "",
    }
    if root is None:
        result["status"] = "not_found"
        result["message"] = "ComfyUI root not found"
        return result
    root_str = str(root)
    _update_console_log(f"ComfyUI pull: repo={root_str}", level="verbose")
    is_git = _run_git(["git", "-C", root_str, "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        result["status"] = "no_git"
        result["message"] = "ComfyUI is not a git repository"
        return result
    branch = _run_git(["git", "-C", root_str, "rev-parse", "--abbrev-ref", "HEAD"]) or ""
    upstream = _run_git(["git", "-C", root_str, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    remote_name = _git_pick_remote(root, upstream)
    if not remote_name:
        result["status"] = "no_remote"
        result["message"] = "ComfyUI remote is not configured"
        return result

    _update_console_log(f"ComfyUI pull: fetch {remote_name}...", level="verbose")
    _run_git(["git", "-C", root_str, "fetch", "--quiet", remote_name], timeout=20.0)
    remote_ref, remote_branch = _git_resolve_remote_ref(root, remote_name, branch, upstream)
    if not remote_ref:
        result["status"] = "no_upstream"
        result["message"] = "ComfyUI upstream/default branch is not configured"
        return result

    if branch == "HEAD" and remote_branch:
        checkout = _run_command(
            ["git", "-C", root_str, "checkout", remote_branch],
            timeout=timeout,
            disable_git_prompt=True,
        )
        if not checkout.get("ok"):
            checkout = _run_command(
                ["git", "-C", root_str, "checkout", "-B", remote_branch, remote_ref],
                timeout=timeout,
                disable_git_prompt=True,
            )
        if not checkout.get("ok"):
            result["status"] = "error"
            result["message"] = str(checkout.get("stderr") or checkout.get("stdout") or "git checkout failed")
            return result

    before_commit = _run_git(["git", "-C", root_str, "rev-parse", "HEAD"]) or ""
    result["before_commit"] = before_commit
    if upstream:
        pull_cmd = ["git", "-C", root_str, "pull", "--ff-only"]
    else:
        pull_cmd = ["git", "-C", root_str, "pull", "--ff-only", remote_name]
        if remote_branch:
            pull_cmd.append(remote_branch)
    _update_console_log(f"ComfyUI pull: running {' '.join(pull_cmd)}", level="verbose")
    pull_started = time.perf_counter()
    pull = _run_command(pull_cmd, timeout=timeout, disable_git_prompt=True)
    _update_console_log(
        f"ComfyUI pull: command finished in {time.perf_counter() - pull_started:.2f}s",
        level="verbose",
    )
    if not pull.get("ok"):
        result["status"] = "error"
        result["message"] = str(pull.get("stderr") or pull.get("stdout") or "git pull failed")
        return result

    after_commit = _run_git(["git", "-C", root_str, "rev-parse", "HEAD"]) or ""
    result["after_commit"] = after_commit
    updated = bool(before_commit and after_commit and before_commit != after_commit)
    result["updated"] = updated
    if updated:
        result["status"] = "updated"
        result["message"] = "ComfyUI updated"
        requirements_changed = _requirements_changed_between(root, before_commit, after_commit)
        result["requirements_changed"] = requirements_changed
        if requirements_changed:
            _set_comfyui_requirements_pending(True, before_commit, after_commit)
    else:
        result["status"] = "up_to_date"
        result["message"] = "already up to date"
    return result


def _pull_custom_module(module_name: str, timeout: float = 180.0) -> dict[str, Any]:
    """Pull latest changes for one custom module from its git remote."""
    module = _canonical_custom_module_name(module_name)
    module_dir = _module_dir(module)
    result: dict[str, Any] = {
        "module": module,
        "status": "error",
        "message": "",
        "updated": False,
        "requirements_changed": False,
        "stashed_local_changes": False,
        "stash_ref": "",
        "before_commit": "",
        "after_commit": "",
    }
    if module_dir is None:
        result["status"] = "not_found"
        result["message"] = "module directory not found"
        return result

    _update_console_log(f"{module}: repo={module_dir}", level="verbose")
    is_git = _run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        result["status"] = "no_git"
        result["message"] = "not a git repository"
        return result

    _update_console_log(f"{module}: resolving upstream...", level="verbose")
    branch = _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "HEAD"]) or ""
    upstream = _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    remote_name = _git_pick_remote(module_dir, upstream)
    if not remote_name and _bootstrap_module_remote_from_manager(module, module_dir):
        upstream = _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        remote_name = _git_pick_remote(module_dir, upstream)
    if not remote_name:
        result["status"] = "no_remote"
        result["message"] = "remote is not configured and manager metadata did not provide repository URL"
        return result

    _run_git(["git", "-C", str(module_dir), "fetch", "--quiet", remote_name], timeout=20.0)
    remote_ref, remote_branch = _git_resolve_remote_ref(module_dir, remote_name, branch, upstream)
    if not remote_ref:
        result["status"] = "no_upstream"
        result["message"] = "upstream/default branch is not configured"
        return result

    if branch == "HEAD" and remote_branch:
        checkout = _run_command(
            ["git", "-C", str(module_dir), "checkout", remote_branch],
            timeout=timeout,
            disable_git_prompt=True,
        )
        if not checkout.get("ok"):
            checkout = _run_command(
                ["git", "-C", str(module_dir), "checkout", "-B", remote_branch, remote_ref],
                timeout=timeout,
                disable_git_prompt=True,
            )
        if not checkout.get("ok"):
            result["status"] = "error"
            result["message"] = str(checkout.get("stderr") or checkout.get("stdout") or "git checkout failed")
            return result

    before_commit = _run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"]) or ""
    result["before_commit"] = before_commit
    if upstream:
        pull_cmd = ["git", "-C", str(module_dir), "pull", "--ff-only"]
    else:
        pull_cmd = ["git", "-C", str(module_dir), "pull", "--ff-only", remote_name]
        if remote_branch:
            pull_cmd.append(remote_branch)
    _update_console_log(f"{module}: running {' '.join(pull_cmd)}", level="verbose")
    pull_started = time.perf_counter()
    pull = _run_command(pull_cmd, timeout=timeout, disable_git_prompt=True)
    pull_elapsed = time.perf_counter() - pull_started
    _update_console_log(f"{module}: pull command finished in {pull_elapsed:.2f}s", level="verbose")

    if not pull.get("ok"):
        error_text = "{stderr}\n{stdout}".format(
            stderr=str(pull.get("stderr") or ""),
            stdout=str(pull.get("stdout") or ""),
        )
        if _is_git_local_changes_block(error_text):
            _update_console_log(f"{module}: local changes detected, trying auto-stash before update")
            stash = _run_command(
                [
                    "git",
                    "-C",
                    str(module_dir),
                    "stash",
                    "push",
                    "-u",
                    "-m",
                    "ALEXZ_tools auto-stash before module update",
                ],
                timeout=60.0,
                disable_git_prompt=True,
            )
            if not stash.get("ok"):
                result["status"] = "error"
                result["message"] = str(stash.get("stderr") or stash.get("stdout") or "git stash failed")
                return result
            stash_out = str(stash.get("stdout") or stash.get("stderr") or "").strip()
            result["stashed_local_changes"] = True
            result["stash_ref"] = stash_out
            _update_console_log(f"{module}: auto-stash created; retrying pull", level="verbose")
            pull_started = time.perf_counter()
            pull = _run_command(pull_cmd, timeout=timeout, disable_git_prompt=True)
            pull_elapsed = time.perf_counter() - pull_started
            _update_console_log(f"{module}: retry pull finished in {pull_elapsed:.2f}s", level="verbose")

    if not pull.get("ok"):
        result["status"] = "error"
        result["message"] = str(pull.get("stderr") or pull.get("stdout") or "git pull failed")
        return result

    after_commit = _run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"]) or ""
    result["after_commit"] = after_commit
    updated = bool(before_commit and after_commit and before_commit != after_commit)
    result["updated"] = updated
    if updated:
        result["status"] = "updated"
        if result.get("stashed_local_changes"):
            result["message"] = "module updated (local changes were stashed)"
        else:
            result["message"] = "module updated"
        requirements_changed = _requirements_changed_between(module_dir, before_commit, after_commit)
        result["requirements_changed"] = requirements_changed
        if requirements_changed:
            _set_module_requirements_pending(module, True, before_commit, after_commit)
    else:
        result["status"] = "up_to_date"
        if result.get("stashed_local_changes"):
            result["message"] = "already up to date (local changes were stashed)"
        else:
            result["message"] = "already up to date"
    return result


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
    remotes = _git_remote_names(module_dir)
    if remotes:
        return True
    meta = _manager_meta_for_module(module_name, None)
    repo_url = _normalize_repo_url(meta.get("repository")) if isinstance(meta, dict) else None
    if not repo_url:
        return False
    add = _run_command(
        ["git", "-C", str(module_dir), "remote", "add", "origin", repo_url],
        timeout=20.0,
        disable_git_prompt=True,
    )
    if not add.get("ok"):
        return False
    _LOGGER.info("Configured origin remote from manager metadata for module %s: %s", module_name, repo_url)
    return True


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
    base = Path(__file__).resolve()
    for candidate in (base.parents[2], *base.parents):
        try:
            if (candidate / "nodes.py").exists() and (candidate / ".git").exists():
                return candidate
        except Exception:
            continue
    return None


def _comfyui_git_status(force_refresh: bool = False, mode: str = "releases") -> dict[str, Any]:
    """Collect local/remote git status summary for ComfyUI repository."""
    global _COMFYUI_STATUS_CACHE
    mode_norm = _normalize_comfyui_mode(mode)
    now_ts = time.time()
    cache = _ensure_comfyui_status_cache()
    cached_mode = cache.get(mode_norm)
    if (
        not force_refresh
        and cached_mode is not None
        and (now_ts - cached_mode[0]) < _COMFYUI_STATUS_TTL_SEC
    ):
        return dict(cached_mode[1])

    result: dict[str, Any] = {
        "path": "",
        "repository": "https://github.com/comfyanonymous/ComfyUI",
        "check_mode": mode_norm,
        "remote_name": "",
        "remote_ref": "",
        "branch": "",
        "upstream": "",
        "installed_commit": "",
        "installed_commit_short": "",
        "installed_updated_at": "",
        "remote_commit": "",
        "remote_commit_short": "",
        "remote_updated_at": "",
        "release_tag": "",
        "release_name": "",
        "release_url": "",
        "ahead": None,
        "behind": None,
        "update_available": None,
        "update_status": "unknown",
        "requirements_update_pending": False,
        "requirements_pending_before_commit": "",
        "requirements_pending_after_commit": "",
        "requirements_pending_updated_at": "",
    }

    if not force_refresh:
        state = _load_module_state()
        cached_entry = state.get("__comfyui__") if isinstance(state, dict) else None
        status_by_mode = cached_entry.get("status_by_mode") if isinstance(cached_entry, dict) else None
        cached_status: dict[str, Any] | None = None
        if isinstance(status_by_mode, dict):
            candidate = status_by_mode.get(mode_norm)
            if isinstance(candidate, dict):
                cached_status = candidate
        if cached_status is None and isinstance(cached_entry, dict):
            candidate = cached_entry.get("status")
            if isinstance(candidate, dict):
                cached_status = candidate
        if isinstance(cached_status, dict) and cached_status:
            merged = dict(cached_status)
            merged["check_mode"] = str(merged.get("check_mode") or mode_norm)
            pending_prev = (
                (cached_entry.get("pending_prev_commit") or cached_entry.get("startup_prev_commit") or "").strip()
                if isinstance(cached_entry, dict)
                else ""
            )
            pending_new = (
                (cached_entry.get("pending_new_commit") or cached_entry.get("startup_new_commit") or "").strip()
                if isinstance(cached_entry, dict)
                else ""
            )
            pending_at = (
                (cached_entry.get("pending_update_at") or cached_entry.get("startup_update_at") or "").strip()
                if isinstance(cached_entry, dict)
                else ""
            )
            merged["updated_between_runs"] = bool(pending_prev and pending_new)
            merged["startup_prev_commit_short"] = _short_commit(pending_prev) if pending_prev else ""
            merged["startup_new_commit_short"] = _short_commit(pending_new) if pending_new else ""
            merged["startup_update_at"] = pending_at
            merged["requirements_update_pending"] = bool(cached_entry.get("pending_requirements_update"))
            merged["requirements_pending_before_commit"] = str(cached_entry.get("pending_requirements_before_commit") or "")
            merged["requirements_pending_after_commit"] = str(cached_entry.get("pending_requirements_after_commit") or "")
            merged["requirements_pending_updated_at"] = str(cached_entry.get("pending_requirements_updated_at") or "")
            cache[mode_norm] = (now_ts, dict(merged))
            return merged
        cache[mode_norm] = (now_ts, dict(result))
        return result

    root = _comfyui_root()
    if root is None:
        cache[mode_norm] = (now_ts, dict(result))
        state = _load_module_state()
        if isinstance(state, dict):
            entry_raw = state.get("__comfyui__")
            entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
            by_mode = dict(entry.get("status_by_mode")) if isinstance(entry.get("status_by_mode"), dict) else {}
            by_mode[mode_norm] = dict(result)
            entry["status_by_mode"] = by_mode
            entry["status"] = dict(result)
            entry["updated_at"] = _now_iso()
            state["__comfyui__"] = entry
            _save_module_state(state)
        return result

    result["path"] = str(root)
    is_git = _run_git(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        cache[mode_norm] = (now_ts, dict(result))
        state = _load_module_state()
        if isinstance(state, dict):
            entry_raw = state.get("__comfyui__")
            entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
            by_mode = dict(entry.get("status_by_mode")) if isinstance(entry.get("status_by_mode"), dict) else {}
            by_mode[mode_norm] = dict(result)
            entry["status_by_mode"] = by_mode
            entry["status"] = dict(result)
            entry["updated_at"] = _now_iso()
            state["__comfyui__"] = entry
            _save_module_state(state)
        return result

    result["branch"] = _run_git(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]) or ""
    result["installed_commit"] = _run_git(["git", "-C", str(root), "rev-parse", "HEAD"]) or ""
    result["installed_commit_short"] = _short_commit(result["installed_commit"]) if result["installed_commit"] else ""
    result["installed_updated_at"] = _run_git(["git", "-C", str(root), "log", "-1", "--format=%cI"]) or ""

    upstream = _run_git(
        ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
    )
    result["upstream"] = upstream or ""
    remote_name = _git_pick_remote(root, upstream)
    result["remote_name"] = remote_name or ""
    if not remote_name:
        cache[mode_norm] = (now_ts, dict(result))
        return result

    # Keep remote refs fresh to reflect actual GitHub state.
    _run_git(["git", "-C", str(root), "fetch", "--quiet", remote_name], timeout=20.0)

    remote_ref = ""
    if mode_norm == "releases":
        release = _github_latest_release("comfyanonymous", "ComfyUI")
        tag_name = str(release.get("tag_name") or "").strip()
        tag_ref, release_commit = _resolve_release_ref(root, remote_name, tag_name)
        if tag_ref and release_commit:
            remote_ref = tag_ref
            result["remote_ref"] = tag_ref
            result["release_tag"] = tag_name
            result["release_name"] = str(release.get("name") or "").strip()
            result["release_url"] = str(release.get("html_url") or "").strip()
            result["remote_commit"] = release_commit
            result["remote_commit_short"] = _short_commit(release_commit)
            published = _parse_datetime(str(release.get("published_at") or release.get("created_at") or ""))
            if published is not None:
                result["remote_updated_at"] = _to_iso(published) or ""
            if not result["remote_updated_at"]:
                result["remote_updated_at"] = _run_git(["git", "-C", str(root), "log", "-1", "--format=%cI", tag_ref]) or ""
        else:
            mode_norm = "commits"
            result["check_mode"] = "commits"

    if mode_norm == "commits":
        remote_ref, _remote_branch = _git_resolve_remote_ref(root, remote_name, result["branch"], upstream)
        result["remote_ref"] = remote_ref or ""
        if remote_ref:
            result["remote_commit"] = _run_git(["git", "-C", str(root), "rev-parse", remote_ref]) or ""
            result["remote_commit_short"] = _short_commit(result["remote_commit"]) if result["remote_commit"] else ""
            result["remote_updated_at"] = _run_git(["git", "-C", str(root), "log", "-1", "--format=%cI", remote_ref]) or ""

    if result["remote_ref"] and result["remote_commit"]:
        counts = _run_git(["git", "-C", str(root), "rev-list", "--left-right", "--count", f"HEAD...{result['remote_ref']}"])
        if counts:
            parts = counts.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                result["ahead"] = int(parts[0])
                result["behind"] = int(parts[1])
                result["update_available"] = result["behind"] > 0
                result["update_status"] = "can_update" if result["behind"] > 0 else "up_to_date"
        elif result["installed_commit"] and result["remote_commit"]:
            if result["installed_commit"] == result["remote_commit"]:
                result["update_available"] = False
                result["update_status"] = "up_to_date"
            else:
                # If exact counters are unavailable, assume remote difference requires update.
                result["update_available"] = True
                result["update_status"] = "can_update"
                result["behind"] = 1

    state = _load_module_state()
    cached_entry = state.get("__comfyui__") if isinstance(state, dict) else None
    if isinstance(cached_entry, dict):
        pending_prev = (cached_entry.get("pending_prev_commit") or cached_entry.get("startup_prev_commit") or "").strip()
        pending_new = (cached_entry.get("pending_new_commit") or cached_entry.get("startup_new_commit") or "").strip()
        pending_at = (cached_entry.get("pending_update_at") or cached_entry.get("startup_update_at") or "").strip()
        result["updated_between_runs"] = bool(pending_prev and pending_new)
        result["startup_prev_commit_short"] = _short_commit(pending_prev) if pending_prev else ""
        result["startup_new_commit_short"] = _short_commit(pending_new) if pending_new else ""
        result["startup_update_at"] = pending_at
        result["requirements_update_pending"] = bool(cached_entry.get("pending_requirements_update"))
        result["requirements_pending_before_commit"] = str(cached_entry.get("pending_requirements_before_commit") or "")
        result["requirements_pending_after_commit"] = str(cached_entry.get("pending_requirements_after_commit") or "")
        result["requirements_pending_updated_at"] = str(cached_entry.get("pending_requirements_updated_at") or "")

    cache[result["check_mode"]] = (now_ts, dict(result))
    state = _load_module_state()
    if isinstance(state, dict):
        prev_entry = state.get("__comfyui__")
        entry = dict(prev_entry) if isinstance(prev_entry, dict) else {}
        by_mode = dict(entry.get("status_by_mode")) if isinstance(entry.get("status_by_mode"), dict) else {}
        by_mode[result["check_mode"]] = dict(result)
        entry["status_by_mode"] = by_mode
        entry["status"] = dict(result)
        entry["updated_at"] = _now_iso()
        entry["installed_commit"] = result.get("installed_commit") or entry.get("installed_commit")
        entry["installed_updated_at"] = result.get("installed_updated_at") or entry.get("installed_updated_at")
        state["__comfyui__"] = entry
        _save_module_state(state)
    return result


def _track_comfyui_local_update() -> None:
    """Track local ComfyUI commit changes between restarts without upstream sync."""
    global _COMFYUI_STATUS_CACHE
    state = _load_module_state()
    if not isinstance(state, dict):
        return
    root = _comfyui_root()
    if root is None:
        return
    is_git = _run_git(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        return

    current_commit = _run_git(["git", "-C", str(root), "rev-parse", "HEAD"]) or ""
    current_updated_at = _run_git(["git", "-C", str(root), "log", "-1", "--format=%cI"]) or ""
    if not current_commit:
        return

    entry_raw = state.get("__comfyui__")
    entry = dict(entry_raw) if isinstance(entry_raw, dict) else {}
    status_raw = entry.get("status")
    status = dict(status_raw) if isinstance(status_raw, dict) else {}
    status_by_mode_raw = entry.get("status_by_mode")
    status_by_mode = dict(status_by_mode_raw) if isinstance(status_by_mode_raw, dict) else {}
    prev_commit = (
        (entry.get("installed_commit") or status.get("installed_commit") or "").strip()
    )
    now = _now_iso()
    changed = False

    if prev_commit and prev_commit != current_commit:
        entry["pending_prev_commit"] = prev_commit
        entry["pending_new_commit"] = current_commit
        entry["pending_update_at"] = now
        entry["startup_prev_commit"] = prev_commit
        entry["startup_new_commit"] = current_commit
        entry["startup_update_at"] = now
        changed = True

    if entry.get("installed_commit") != current_commit:
        entry["installed_commit"] = current_commit
        changed = True
    if entry.get("installed_updated_at") != current_updated_at:
        entry["installed_updated_at"] = current_updated_at
        changed = True

    status.setdefault("repository", "https://github.com/comfyanonymous/ComfyUI")
    status["path"] = str(root)
    status["installed_commit"] = current_commit
    status["installed_commit_short"] = _short_commit(current_commit)
    status["installed_updated_at"] = current_updated_at
    status.setdefault("update_status", "unknown")
    entry["status"] = status
    for mode_name, mode_status_raw in status_by_mode.items():
        if not isinstance(mode_status_raw, dict):
            continue
        mode_status = dict(mode_status_raw)
        mode_status.setdefault("repository", "https://github.com/comfyanonymous/ComfyUI")
        mode_status["path"] = str(root)
        mode_status["installed_commit"] = current_commit
        mode_status["installed_commit_short"] = _short_commit(current_commit)
        mode_status["installed_updated_at"] = current_updated_at
        mode_status.setdefault("update_status", "unknown")
        status_by_mode[mode_name] = mode_status
    if status_by_mode:
        entry["status_by_mode"] = status_by_mode
    entry["updated_at"] = now
    state["__comfyui__"] = entry

    if changed:
        _clear_comfyui_status_cache()
        _save_module_state(state)


def _acknowledge_comfyui_novelty() -> dict[str, Any]:
    """Clear pending ComfyUI novelty markers after explicit user refresh action."""
    global _COMFYUI_STATUS_CACHE
    state = _load_module_state()
    if not isinstance(state, dict):
        return {"status": "ok", "changed": False}
    entry_raw = state.get("__comfyui__")
    if not isinstance(entry_raw, dict):
        return {"status": "ok", "changed": False}

    entry = dict(entry_raw)
    before = dict(entry)
    for key in (
        "pending_prev_commit",
        "pending_new_commit",
        "pending_update_at",
        "startup_prev_commit",
        "startup_new_commit",
        "startup_update_at",
    ):
        entry.pop(key, None)

    changed = entry != before
    if changed:
        _clear_comfyui_status_cache()
        state["__comfyui__"] = entry
        _save_module_state(state)
    return {"status": "ok", "changed": changed}


def _load_module_state() -> dict[str, dict[str, Any]]:
    """Load persisted module snapshot state from extension cache file."""
    global _MODULE_STATE_CACHE
    if _MODULE_STATE_CACHE is not None:
        return _MODULE_STATE_CACHE
    if not _MODULE_STATE_PATH.exists():
        _MODULE_STATE_CACHE = ensure_module_state_schema({})
        return _MODULE_STATE_CACHE
    try:
        with _MODULE_STATE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        _MODULE_STATE_CACHE = ensure_module_state_schema(data if isinstance(data, dict) else {})
    except Exception:
        _MODULE_STATE_CACHE = ensure_module_state_schema({})
    return _MODULE_STATE_CACHE


def _save_module_state(state: dict[str, dict[str, Any]]) -> None:
    """Persist module snapshot state to extension cache file."""
    normalized = ensure_module_state_schema(state)
    try:
        with _MODULE_STATE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=True, indent=2, sort_keys=True)
        global _MODULE_STATE_CACHE
        _MODULE_STATE_CACHE = normalized
    except Exception as exc:
        _LOGGER.debug("Failed to save module state cache: %s", exc)


def _remember_module_state(module_name: str, result: dict[str, Any]) -> None:
    """Capture current module/node snapshot as baseline for next ComfyUI start."""
    module_name = _canonical_custom_module_name(module_name)
    state = _load_module_state()
    now = _now_iso()
    entry = state.get(module_name, {})
    prev_commit = entry.get("installed_commit")
    current_commit = result.get("installed_commit")
    if not entry.get("first_seen_at"):
        entry["first_seen_at"] = now
    if current_commit and current_commit != prev_commit:
        entry["last_local_change_at"] = now
    entry["last_checked_at"] = now
    entry["installed_commit"] = current_commit
    entry["installed_updated_at"] = result.get("installed_updated_at")
    entry["remote_updated_at"] = result.get("remote_updated_at")
    entry["update_available"] = result.get("update_available")
    entry["update_status"] = result.get("update_status")
    entry["module_path"] = result.get("module_path")
    entry["repository"] = result.get("repository")
    state[module_name] = entry
    result["last_checked_at"] = entry.get("last_checked_at")
    result["last_local_change_at"] = entry.get("last_local_change_at")
    startup_prev = (entry.get("pending_prev_commit") or entry.get("startup_prev_commit") or "").strip()
    startup_new = (entry.get("pending_new_commit") or entry.get("startup_new_commit") or "").strip()
    result["updated_between_runs"] = (
        bool(startup_prev and startup_new)
        or bool(entry.get("pending_commit_change"))
        or bool(entry.get("pending_local_change"))
    )
    result["startup_prev_commit_short"] = _short_commit(startup_prev) if startup_prev else ""
    result["startup_new_commit_short"] = _short_commit(startup_new) if startup_new else ""
    result["startup_update_at"] = entry.get("pending_update_at") or entry.get("startup_update_at") or ""
    _save_module_state(state)


def _apply_node_change_info(result: dict[str, Any], group: str, module_name: str) -> None:
    """Attach node-level change markers to module info payload for UI rendering."""
    state = _load_module_state()
    tracker = state.get("__node_tracker__")
    if not isinstance(tracker, dict):
        return
    startup_changes = tracker.get("pending_changes") or tracker.get("startup_changes")
    if isinstance(startup_changes, dict):
        group_changes = startup_changes.get(group)
        if isinstance(group_changes, dict):
            entry = group_changes.get(module_name)
            if isinstance(entry, dict):
                new_nodes = entry.get("new_nodes")
                upd_nodes = entry.get("updated_nodes")
                result["new_nodes_between_runs"] = new_nodes if isinstance(new_nodes, list) else []
                result["updated_nodes_between_runs"] = upd_nodes if isinstance(upd_nodes, list) else []
                result["startup_node_update_at"] = entry.get("at") or ""
                if result["new_nodes_between_runs"] or result["updated_nodes_between_runs"]:
                    result["updated_between_runs"] = True

    startup_new_modules = tracker.get("pending_new_modules") or tracker.get("startup_new_modules")
    if isinstance(startup_new_modules, dict):
        group_new = startup_new_modules.get(group)
        if isinstance(group_new, list) and module_name in group_new:
            result["new_module_between_runs"] = True
            result["updated_between_runs"] = True


def _acknowledge_module_novelty(group: str, module_name: str) -> None:
    """Clear pending novelty markers for one module after explicit user refresh."""
    group_name = (group or "").strip().lower()
    module = (module_name or "").strip()
    if not module:
        return
    if group_name == "custom":
        module = _canonical_custom_module_name(module)

    state = _load_module_state()
    if not isinstance(state, dict):
        return

    changed = False
    entry = state.get(module)
    if isinstance(entry, dict):
        for key in (
            "pending_prev_commit",
            "pending_new_commit",
            "pending_update_at",
            "pending_commit_change",
            "pending_local_change",
            "startup_prev_commit",
            "startup_new_commit",
            "startup_update_at",
        ):
            if key in entry:
                entry.pop(key, None)
                changed = True
        state[module] = entry

    tracker = state.get("__node_tracker__")
    if isinstance(tracker, dict):
        for pending_key, legacy_key in (("pending_changes", "startup_changes"), ("pending_new_modules", "startup_new_modules")):
            bucket = tracker.get(pending_key)
            if not isinstance(bucket, dict):
                bucket = tracker.get(legacy_key)
            if not isinstance(bucket, dict):
                continue
            group_bucket = bucket.get(group_name)
            if isinstance(group_bucket, dict) and module in group_bucket:
                group_bucket.pop(module, None)
                changed = True
            elif isinstance(group_bucket, list) and module in group_bucket:
                bucket[group_name] = [x for x in group_bucket if x != module]
                changed = True
            updated_group_bucket = bucket.get(group_name)
            if isinstance(updated_group_bucket, dict) and not updated_group_bucket:
                bucket.pop(group_name, None)
            if isinstance(updated_group_bucket, list) and not updated_group_bucket:
                bucket.pop(group_name, None)
            tracker[pending_key] = bucket
        state["__node_tracker__"] = tracker

    if changed:
        _MODULE_INFO_CACHE.clear()
        _save_module_state(state)


def _acknowledge_all_novelty() -> dict[str, Any]:
    """Clear pending novelty markers for all modules after explicit global refresh."""
    state = _load_module_state()
    if not isinstance(state, dict):
        return {"status": "ok", "changed": False}

    changed = False
    cleared_modules = 0
    for module_name, entry in list(state.items()):
        if str(module_name).startswith("__") or not isinstance(entry, dict):
            continue
        before = dict(entry)
        for key in (
            "pending_prev_commit",
            "pending_new_commit",
            "pending_update_at",
            "pending_commit_change",
            "pending_local_change",
            "startup_prev_commit",
            "startup_new_commit",
            "startup_update_at",
        ):
            entry.pop(key, None)
        if entry != before:
            state[module_name] = entry
            cleared_modules += 1
            changed = True

    tracker = state.get("__node_tracker__")
    if isinstance(tracker, dict):
        for key in ("pending_changes", "pending_new_modules", "startup_changes", "startup_new_modules"):
            value = tracker.get(key)
            if isinstance(value, dict) and value:
                tracker[key] = {}
                changed = True
        state["__node_tracker__"] = tracker

    if changed:
        _MODULE_INFO_CACHE.clear()
        _save_module_state(state)
    return {"status": "ok", "changed": changed, "cleared_modules": cleared_modules}


def _announce_tracked_module_updates(local_only: bool = False) -> dict[str, Any]:
    """Build per-module node-change info by comparing saved and current snapshots."""
    state = _load_module_state()
    if not isinstance(state, dict):
        return {"modules_need_update": 0}

    now = _now_iso()
    changed = False
    modules_need_update = 0
    modules_unknown_update = 0
    modules_checked = 0
    update_available_modules: list[str] = []
    unknown_update_modules: list[str] = []
    local_change_modules: list[str] = []
    commit_change_modules: list[str] = []

    known_modules = set(_discover_custom_modules())
    for key in list(state.keys()):
        if key.startswith("__"):
            continue
        canonical = _canonical_custom_module_name(key)
        known_modules.add(canonical)
        if canonical != key:
            src = state.get(key)
            dst = state.get(canonical)
            if isinstance(src, dict):
                if not isinstance(dst, dict):
                    state[canonical] = dict(src)
                else:
                    merged = dict(dst)
                    for mk, mv in src.items():
                        if mk not in merged or not merged.get(mk):
                            merged[mk] = mv
                    state[canonical] = merged
                state.pop(key, None)
                changed = True

    for module_name in sorted(known_modules, key=str.lower):
        modules_checked += 1
        entry = state.get(module_name, {})
        if not isinstance(entry, dict):
            entry = {}
        prev_commit = (entry.get("installed_commit") or "").strip()
        git_state = _module_git_state(module_name)
        current_commit = (git_state.get("installed_commit") or "").strip()
        before = dict(entry)
        manager_meta = _manager_meta_for_module(module_name, entry.get("repository"))
        manager_repo = ""
        if isinstance(manager_meta, dict):
            manager_repo = str(manager_meta.get("repository") or "").strip()

        entry["last_checked_at"] = now
        needs_update: bool | None = None
        prev_worktree_sig = str(entry.get("worktree_signature") or "")
        curr_worktree_sig = _module_worktree_signature(module_name)
        if curr_worktree_sig != prev_worktree_sig:
            if prev_worktree_sig:
                entry["pending_local_change"] = True
                entry["pending_update_at"] = now
            entry["worktree_signature"] = curr_worktree_sig
        if git_state:
            entry["module_path"] = git_state.get("module_path") or entry.get("module_path")
            entry["repository"] = git_state.get("repository") or entry.get("repository")
            entry["installed_updated_at"] = git_state.get("installed_updated_at") or entry.get("installed_updated_at")
            if not local_only:
                entry["remote_updated_at"] = git_state.get("remote_updated_at") or entry.get("remote_updated_at")
                behind = git_state.get("behind")
                remote_head = (git_state.get("remote_head") or "").strip()
                if isinstance(behind, int):
                    needs_update = behind > 0
                elif git_state.get("has_upstream") and remote_head and current_commit:
                    needs_update = remote_head != current_commit
        if manager_repo and not entry.get("repository"):
            entry["repository"] = manager_repo
        if not local_only:
            inferred_update, inferred_remote_updated_at = _infer_update_from_manager_stats(
                entry.get("repository"),
                entry.get("installed_updated_at"),
            )
            if inferred_remote_updated_at and not entry.get("remote_updated_at"):
                entry["remote_updated_at"] = inferred_remote_updated_at
            if not isinstance(needs_update, bool) and isinstance(inferred_update, bool):
                needs_update = inferred_update

        if isinstance(needs_update, bool):
            entry["update_available"] = needs_update
            entry["update_status"] = "can_update" if needs_update else "up_to_date"
        else:
            entry["update_available"] = None
            entry["update_status"] = "unknown"

        if current_commit:
            if prev_commit and current_commit != prev_commit:
                entry["installed_commit"] = current_commit
                entry["last_local_change_at"] = now
                entry["pending_prev_commit"] = prev_commit
                entry["pending_new_commit"] = current_commit
                entry["pending_update_at"] = now
                entry["pending_commit_change"] = True
                entry.pop("pending_local_change", None)
                entry["startup_prev_commit"] = prev_commit
                entry["startup_new_commit"] = current_commit
                entry["startup_update_at"] = now
                commit_change_modules.append(module_name)
            else:
                entry["installed_commit"] = current_commit

        if bool(entry.get("pending_local_change")):
            local_change_modules.append(module_name)

        if isinstance(needs_update, bool) and needs_update:
            modules_need_update += 1
            update_available_modules.append(module_name)
        elif not isinstance(needs_update, bool):
            modules_unknown_update += 1
            unknown_update_modules.append(module_name)

        state[module_name] = entry
        if entry != before:
            changed = True

    tracker = state.get("__node_tracker__")
    if not isinstance(tracker, dict):
        tracker = {}
    prev_snapshots_raw = tracker.get("snapshots")
    prev_snapshots = prev_snapshots_raw if isinstance(prev_snapshots_raw, dict) else {}
    prev_module_sets_raw = tracker.get("module_sets")
    prev_module_sets = prev_module_sets_raw if isinstance(prev_module_sets_raw, dict) else {}
    current_snapshots = _build_node_snapshots()
    startup_changes: dict[str, dict[str, dict[str, Any]]] = {}
    startup_new_modules: dict[str, list[str]] = {}
    pending_changes_raw = tracker.get("pending_changes")
    pending_changes: dict[str, dict[str, dict[str, Any]]] = (
        pending_changes_raw if isinstance(pending_changes_raw, dict) else {}
    )
    pending_new_modules_raw = tracker.get("pending_new_modules")
    pending_new_modules: dict[str, list[str]] = (
        pending_new_modules_raw if isinstance(pending_new_modules_raw, dict) else {}
    )

    current_module_sets: dict[str, list[str]] = {}
    for group_name, modules in current_snapshots.items():
        if isinstance(modules, dict):
            current_module_sets[group_name] = sorted(modules.keys(), key=str.lower)
    custom_from_fs = _discover_custom_modules()
    if custom_from_fs:
        existing = set(current_module_sets.get("custom", []))
        current_module_sets["custom"] = sorted(existing.union(custom_from_fs), key=str.lower)

    for group_name, modules in current_snapshots.items():
        if not isinstance(modules, dict):
            continue
        group_prev = prev_snapshots.get(group_name)
        group_prev = group_prev if isinstance(group_prev, dict) else {}
        for module_name, current_snapshot in modules.items():
            if not isinstance(current_snapshot, dict):
                continue
            prev_snapshot_raw = group_prev.get(module_name)
            prev_snapshot = prev_snapshot_raw if isinstance(prev_snapshot_raw, dict) else {}
            prev_names = {k for k in prev_snapshot if isinstance(k, str)}
            curr_names = {k for k in current_snapshot if isinstance(k, str)}

            new_nodes: list[str] = []
            updated_nodes: list[str] = []
            if prev_snapshot:
                new_nodes = sorted(curr_names - prev_names)
                for node_name in sorted(curr_names & prev_names):
                    prev_node = prev_snapshot.get(node_name, {})
                    prev_sig = prev_node.get("sig") if isinstance(prev_node, dict) else None
                    curr_sig = current_snapshot.get(node_name, {}).get("sig")
                    if prev_sig != curr_sig:
                        updated_nodes.append(node_name)

            if new_nodes or updated_nodes:
                startup_changes.setdefault(group_name, {})[module_name] = {
                    "new_nodes": new_nodes,
                    "updated_nodes": updated_nodes,
                    "at": now,
                }
                existing_entry = pending_changes.setdefault(group_name, {}).get(module_name, {})
                prev_new = existing_entry.get("new_nodes") if isinstance(existing_entry, dict) else []
                prev_updated = existing_entry.get("updated_nodes") if isinstance(existing_entry, dict) else []
                merged_new = sorted(set(prev_new if isinstance(prev_new, list) else []).union(new_nodes))
                merged_updated = sorted(
                    set(prev_updated if isinstance(prev_updated, list) else []).union(updated_nodes)
                )
                pending_changes.setdefault(group_name, {})[module_name] = {
                    "new_nodes": merged_new,
                    "updated_nodes": merged_updated,
                    "at": now,
                }

    for group_name, current_list in current_module_sets.items():
        prev_list_raw = prev_module_sets.get(group_name)
        if not isinstance(prev_list_raw, list):
            continue
        prev_set = {x for x in prev_list_raw if isinstance(x, str)}
        curr_set = {x for x in current_list if isinstance(x, str)}
        new_modules = sorted(curr_set - prev_set, key=str.lower)
        if new_modules:
            startup_new_modules[group_name] = new_modules
            existing = pending_new_modules.get(group_name)
            existing_list = existing if isinstance(existing, list) else []
            pending_new_modules[group_name] = sorted(set(existing_list).union(new_modules), key=str.lower)

    if prev_snapshots != current_snapshots:
        changed = True
    if prev_module_sets != current_module_sets:
        changed = True
    tracker["snapshots"] = current_snapshots
    tracker["startup_changes"] = startup_changes
    tracker["module_sets"] = current_module_sets
    tracker["startup_new_modules"] = startup_new_modules
    tracker["pending_changes"] = pending_changes
    tracker["pending_new_modules"] = pending_new_modules
    tracker["updated_at"] = now
    state["__node_tracker__"] = tracker

    if changed:
        _save_module_state(state)
    node_changed_modules = sorted(
        {
            module_name
            for modules in startup_changes.values()
            if isinstance(modules, dict)
            for module_name in modules.keys()
            if isinstance(module_name, str)
        },
        key=str.lower,
    )
    return {
        "modules_need_update": modules_need_update,
        "modules_unknown_update": modules_unknown_update,
        "modules_checked": modules_checked,
        "update_available_modules": sorted(update_available_modules, key=str.lower),
        "unknown_update_modules": sorted(unknown_update_modules, key=str.lower),
        "local_change_modules": sorted(set(local_change_modules), key=str.lower),
        "commit_change_modules": sorted(set(commit_change_modules), key=str.lower),
        "node_changed_modules": node_changed_modules,
        "new_modules_between_runs": startup_new_modules,
    }


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


def _set_refresh_status(**kwargs: Any) -> None:
    """Set shared refresh job status fields in a thread-safe way."""
    set_refresh_status(
        lock=_REFRESH_LOCK,
        status=_REFRESH_STATUS,
        now_iso=_now_iso,
        **kwargs,
    )


def _refresh_status_snapshot() -> dict[str, Any]:
    """Return thread-safe snapshot of refresh-job status."""
    return refresh_status_snapshot(lock=_REFRESH_LOCK, status=_REFRESH_STATUS)


def _refresh_progress(
    *,
    phase: str,
    current: int = 0,
    total: int = 0,
    remaining: int = 0,
    modules_need_update: int = 0,
    modules_unknown_update: int = 0,
    module: str = "",
    message: str = "",
) -> None:
    """Update refresh-job progress counters and status text."""
    global _REFRESH_LOG_LAST
    _REFRESH_LOG_LAST = emit_refresh_progress(
        lock=_REFRESH_LOCK,
        status=_REFRESH_STATUS,
        now_iso=_now_iso,
        phase=phase,
        current=current,
        total=total,
        remaining=remaining,
        modules_need_update=modules_need_update,
        modules_unknown_update=modules_unknown_update,
        module=module,
        message=message,
        last_line=_REFRESH_LOG_LAST,
        logger_debug=lambda line: _LOGGER.debug("Module refresh: %s", line),
        console_log=lambda text, level="summary": _refresh_console_log(text, level=level),
    )


def _refresh_module_runtime_state(sync_upstreams: bool = False, progress_cb: Any | None = None) -> dict[str, Any]:
    """Recompute module snapshots and update persisted runtime tracking state."""
    global _LAZY_REFRESH_DONE
    global _CUSTOM_MODULE_ALIAS_CACHE
    global _COMFYUI_STATUS_CACHE
    _MODULE_INFO_CACHE.clear()
    _CUSTOM_MODULE_ALIAS_CACHE = None
    _clear_comfyui_status_cache()
    refresh_started = time.perf_counter()
    _refresh_console_log(
        "runtime refresh started (sync_upstreams={sync}, log_mode={mode})".format(
            sync="on" if sync_upstreams else "off",
            mode=_get_update_console_log_mode(),
        )
    )
    if progress_cb is None:
        progress_cb = _refresh_progress
    if sync_upstreams:
        module_names = _discover_custom_modules()
        total = len(module_names)
        _refresh_console_log(f"phase 1/3: upstream sync enabled for {total} custom module(s)")
        progress_cb(phase="sync", current=0, total=total, remaining=total, message="sync_upstreams")
        for idx, module_name in enumerate(module_names, start=1):
            sync_started = time.perf_counter()
            synced = _sync_module_upstream(module_name)
            elapsed = time.perf_counter() - sync_started
            status = "synced" if synced else "skip"
            progress_cb(
                phase="sync",
                current=idx,
                total=total,
                remaining=total - idx,
                module=module_name,
                message=f"{status} ({elapsed:.2f}s)",
            )
    else:
        _refresh_console_log("phase 1/3: upstream sync skipped (fast mode)")
        progress_cb(phase="sync", current=0, total=0, remaining=0, message="fast_mode")

    scan_started = time.perf_counter()
    _refresh_console_log("phase 2/3: recomputing module snapshots and local git deltas...")
    progress_cb(phase="snapshots", current=0, total=0, remaining=0, message="recompute_snapshots")
    announce_summary = _announce_tracked_module_updates()
    modules_need_update = 0
    modules_unknown_update = 0
    if isinstance(announce_summary, dict):
        modules_need_update = max(0, int(announce_summary.get("modules_need_update", 0)))
        modules_unknown_update = max(0, int(announce_summary.get("modules_unknown_update", 0)))
    modules_checked = max(0, int((announce_summary or {}).get("modules_checked", 0)))
    commit_changed = list((announce_summary or {}).get("commit_change_modules") or [])
    local_changed = list((announce_summary or {}).get("local_change_modules") or [])
    node_changed = list((announce_summary or {}).get("node_changed_modules") or [])
    new_modules_map = (announce_summary or {}).get("new_modules_between_runs") or {}
    new_modules_count = 0
    if isinstance(new_modules_map, dict):
        for value in new_modules_map.values():
            if isinstance(value, list):
                new_modules_count += len(value)
    scan_elapsed = time.perf_counter() - scan_started
    _refresh_console_log(
        "phase 2/3 done in {elapsed:.2f}s: checked={checked}, need_update={need}, unknown_update={unknown}, "
        "commit_changed={commit}, local_changed={local}, node_changed={node}, new_modules={new}".format(
            elapsed=scan_elapsed,
            checked=modules_checked,
            need=modules_need_update,
            unknown=modules_unknown_update,
            commit=len(commit_changed),
            local=len(local_changed),
            node=len(node_changed),
            new=new_modules_count,
        )
    )
    if commit_changed:
        _refresh_console_log(f"commit changed modules: {', '.join(commit_changed)}", level="verbose")
    if local_changed:
        _refresh_console_log(f"locally changed modules: {', '.join(local_changed)}", level="verbose")
    if node_changed:
        _refresh_console_log(f"node changed modules: {', '.join(node_changed)}", level="verbose")
    update_available_modules = list((announce_summary or {}).get("update_available_modules") or [])
    if update_available_modules:
        _refresh_console_log(f"update available modules: {', '.join(update_available_modules)}", level="verbose")
    unknown_update_modules = list((announce_summary or {}).get("unknown_update_modules") or [])
    if unknown_update_modules:
        _refresh_console_log(f"unknown update status modules: {', '.join(unknown_update_modules)}")

    _refresh_console_log("phase 3/3: checking ComfyUI status...")
    comfy_started = time.perf_counter()
    comfyui = _comfyui_git_status(force_refresh=True)
    comfy_elapsed = time.perf_counter() - comfy_started
    _refresh_console_log(
        "phase 3/3 done in {elapsed:.2f}s: ComfyUI status={status}, behind={behind}, ahead={ahead}, "
        "local={local}, remote={remote}".format(
            elapsed=comfy_elapsed,
            status=str(comfyui.get("update_status") or "unknown"),
            behind=str(comfyui.get("behind") if comfyui.get("behind") is not None else "-"),
            ahead=str(comfyui.get("ahead") if comfyui.get("ahead") is not None else "-"),
            local=_short_commit(str(comfyui.get("installed_commit") or "")),
            remote=_short_commit(str(comfyui.get("remote_commit") or "")),
        )
    )
    progress_cb(
        phase="done",
        current=0,
        total=0,
        remaining=0,
        modules_need_update=modules_need_update,
        modules_unknown_update=modules_unknown_update,
        message="done",
    )
    _LAZY_REFRESH_DONE = True
    total_elapsed = time.perf_counter() - refresh_started
    _refresh_console_log(
        "runtime refresh finished in {elapsed:.2f}s (modules_need_update={need}, modules_unknown_update={unknown})".format(
            elapsed=total_elapsed,
            need=modules_need_update,
            unknown=modules_unknown_update,
        )
    )
    # Mark that update status was explicitly refreshed and can be shown in UI cards.
    _set_custom_update_checked(True)
    return {
        "status": "ok",
        "refreshed_at": _now_iso(),
        "comfyui": comfyui,
        "sync_upstreams": sync_upstreams,
        "modules_need_update": modules_need_update,
        "modules_unknown_update": modules_unknown_update,
    }


def _ensure_runtime_state_ready() -> None:
    """Ensure runtime snapshot cache is initialized before serving API requests."""
    global _LAZY_REFRESH_DONE
    if _LAZY_REFRESH_DONE:
        return
    _load_module_state()
    # On process start, hide stale remote-update status until user runs explicit refresh.
    _set_custom_update_checked(False)
    _announce_tracked_module_updates(local_only=True)
    _track_comfyui_local_update()
    _LAZY_REFRESH_DONE = True


def _start_runtime_state_warmup() -> bool:
    """Start non-blocking runtime-state warmup once and return start status."""
    global _RUNTIME_WARMUP_THREAD
    if _LAZY_REFRESH_DONE:
        return False

    with _RUNTIME_WARMUP_LOCK:
        if _LAZY_REFRESH_DONE:
            return False
        existing = _RUNTIME_WARMUP_THREAD
        if existing is not None and existing.is_alive():
            return False
        # Reset stale remote-update visibility immediately on first access.
        _set_custom_update_checked(False)

        def _runner() -> None:
            """Warmup worker for runtime state cache used by first-open UI paths."""
            global _RUNTIME_WARMUP_THREAD
            try:
                _ensure_runtime_state_ready()
            except Exception as exc:  # pragma: no cover - diagnostic
                _LOGGER.warning("Runtime warmup failed: %s", exc, exc_info=True)
            finally:
                with _RUNTIME_WARMUP_LOCK:
                    _RUNTIME_WARMUP_THREAD = None

        _RUNTIME_WARMUP_THREAD = threading.Thread(
            target=_runner,
            name="ALEXZ_tools_RuntimeWarmup",
            daemon=True,
        )
        _RUNTIME_WARMUP_THREAD.start()
        return True


def _runtime_warmup_status() -> dict[str, Any]:
    """Return lightweight runtime warmup state for frontend polling hints."""
    with _RUNTIME_WARMUP_LOCK:
        thread = _RUNTIME_WARMUP_THREAD
        running = bool(thread is not None and thread.is_alive())
    done = bool(_LAZY_REFRESH_DONE)
    return {
        "running": running,
        "done": done,
    }


def _start_refresh_job(sync_upstreams: bool) -> dict[str, Any]:
    """Start background module refresh job if one is not already running."""
    global _REFRESH_THREAD
    global _REFRESH_CONSOLE_LOG_LAST
    with _REFRESH_LOCK:
        thread = _REFRESH_THREAD
        if thread is not None and thread.is_alive():
            return {"status": "running", "refresh": dict(_REFRESH_STATUS)}
        _REFRESH_CONSOLE_LOG_LAST = ""
        _REFRESH_STATUS.update(
            {
                "running": True,
                "phase": "starting",
                "current": 0,
                "total": 0,
                "remaining": 0,
                "modules_need_update": 0,
                "modules_unknown_update": 0,
                "module": "",
                "message": "starting",
                "error": "",
                "sync_upstreams": bool(sync_upstreams),
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "refreshed_at": "",
            }
        )

    def _runner() -> None:
        """Background job worker that runs long update/refresh operations."""
        global _REFRESH_THREAD
        try:
            _refresh_console_log(
                "job started (sync_upstreams={sync}, log_mode={mode})".format(
                    sync="on" if sync_upstreams else "off",
                    mode=_get_update_console_log_mode(),
                )
            )
            result = _refresh_module_runtime_state(sync_upstreams=sync_upstreams, progress_cb=_refresh_progress)
            _set_refresh_status(
                running=False,
                phase="done",
                message="done",
                module="",
                refreshed_at=result.get("refreshed_at", ""),
                modules_need_update=max(0, int(result.get("modules_need_update", 0))),
                modules_unknown_update=max(0, int(result.get("modules_unknown_update", 0))),
            )
            _refresh_console_log(
                "job finished: modules_need_update={need}, modules_unknown_update={unknown}".format(
                    need=max(0, int(result.get("modules_need_update", 0))),
                    unknown=max(0, int(result.get("modules_unknown_update", 0))),
                )
            )
        except Exception as exc:
            _refresh_console_log(f"job error: {exc}")
            _set_refresh_status(running=False, phase="error", message="error", error=str(exc), module="")
        finally:
            with _REFRESH_LOCK:
                _REFRESH_THREAD = None

    thread = threading.Thread(target=_runner, name="alexz-module-refresh", daemon=True)
    with _REFRESH_LOCK:
        _REFRESH_THREAD = thread
    thread.start()
    return {"status": "started", "refresh": _refresh_status_snapshot()}


def _set_update_status(**kwargs: Any) -> None:
    """Set shared module-update job status fields in a thread-safe way."""
    global _UPDATE_LOG_LAST
    set_update_status(
        lock=_UPDATE_LOCK,
        status=_UPDATE_STATUS,
        now_iso=_now_iso,
        **kwargs,
    )
    line = format_update_status_line(_UPDATE_STATUS)
    if line != _UPDATE_LOG_LAST:
        _UPDATE_LOG_LAST = line
        _LOGGER.info("Module update: %s", line)


def _update_status_snapshot() -> dict[str, Any]:
    """Return thread-safe snapshot of module-update job status."""
    return update_status_snapshot(lock=_UPDATE_LOCK, status=_UPDATE_STATUS)


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
    global _UPDATE_THREAD
    scope_norm = (scope or "").strip().lower()
    normalized_log_mode = _set_update_console_log_mode(log_mode)
    if scope_norm not in {"single", "all", "comfyui"}:
        return {"status": "error", "error": "scope must be 'single', 'all' or 'comfyui'"}

    if scope_norm == "single":
        canonical = _canonical_custom_module_name(module_name)
        if _module_dir(canonical) is None:
            return {"status": "error", "error": "module not found"}
    if scope_norm == "comfyui":
        if _comfyui_root() is None:
            return {"status": "error", "error": "ComfyUI root not found"}

    with _REFRESH_LOCK:
        if _REFRESH_THREAD is not None and _REFRESH_THREAD.is_alive():
            return {"status": "error", "error": "module refresh is running"}

    with _UPDATE_LOCK:
        thread = _UPDATE_THREAD
        if thread is not None and thread.is_alive():
            return {"status": "running", "update": dict(_UPDATE_STATUS)}
        _UPDATE_STATUS.update(
            {
                "running": True,
                "phase": "starting",
                "scope": scope_norm,
                "current": 0,
                "total": 0,
                "remaining": 0,
                "module": "",
                "message": "starting",
                "error": "",
                "updated": 0,
                "up_to_date": 0,
                "failed": 0,
                "requirements_changed": False,
                "requirements_modules": [],
                "results": [],
                "log_mode": normalized_log_mode,
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "finished_at": "",
            }
        )

    def _runner() -> None:
        """Background job worker that runs long update/refresh operations."""
        global _UPDATE_THREAD
        try:
            _update_console_log(
                f"job started (scope={scope_norm}, module={module_name or '-'}, log_mode={normalized_log_mode})"
            )
            if scope_norm == "comfyui":
                _update_console_log("ComfyUI update: pull start")
                _set_update_status(phase="update", current=0, total=1, remaining=1, module="ComfyUI", message="pull")
                item = _pull_comfyui()
                status = str(item.get("status") or "")
                _update_console_log(f"ComfyUI update: pull done (status={status})")
                updated_count = 1 if status == "updated" else 0
                uptodate_count = 1 if status == "up_to_date" else 0
                failed_count = 1 if status not in {"updated", "up_to_date"} else 0
                requirements_changed = bool(item.get("requirements_changed"))
                _set_update_status(
                    phase="update",
                    current=1,
                    total=1,
                    remaining=0,
                    module="ComfyUI",
                    message=status or "done",
                    updated=updated_count,
                    up_to_date=uptodate_count,
                    failed=failed_count,
                    requirements_changed=requirements_changed,
                    requirements_modules=[],
                    results=[item],
                )
                _refresh_module_runtime_state(sync_upstreams=False, progress_cb=lambda **kwargs: None)
                _set_update_status(
                    running=False,
                    phase="done",
                    message="done",
                    module="",
                    finished_at=_now_iso(),
                )
                _update_console_log("job finished (scope=comfyui)")
                return

            targets = _resolve_update_targets(scope_norm, module_name)
            total = len(targets)
            _update_console_log(
                "resolved targets: {total} ({mods})".format(
                    total=total,
                    mods=", ".join(targets) if targets else "-",
                )
            )
            _set_update_status(phase="update", total=total, remaining=total, message="running")
            if total == 0:
                _refresh_module_runtime_state(sync_upstreams=False, progress_cb=lambda **kwargs: None)
                _set_update_status(
                    running=False,
                    phase="done",
                    message="nothing_to_update",
                    results=[],
                    requirements_changed=False,
                    requirements_modules=[],
                    finished_at=_now_iso(),
                )
                _update_console_log("job finished: nothing to update")
                return

            updated_count = 0
            uptodate_count = 0
            failed_count = 0
            requirements_modules: list[str] = []
            results: list[dict[str, Any]] = []

            for idx, target in enumerate(targets, start=1):
                _set_update_status(
                    phase="update",
                    current=idx - 1,
                    total=total,
                    remaining=total - idx + 1,
                    module=target,
                    message="pull",
                )
                _update_console_log(f"[{idx}/{total}] {target}: pull start", level="verbose")
                pull_started = time.perf_counter()
                item = _pull_custom_module(target)
                pull_elapsed = time.perf_counter() - pull_started
                results.append(item)
                status = str(item.get("status") or "")
                _update_console_log(
                    "[{idx}/{total}] {target}: pull done (status={status}, elapsed={elapsed:.2f}s)".format(
                        idx=idx,
                        total=total,
                        target=target,
                        status=status or "unknown",
                        elapsed=pull_elapsed,
                    ),
                    level="verbose",
                )
                if status == "updated":
                    updated_count += 1
                elif status == "up_to_date":
                    uptodate_count += 1
                else:
                    failed_count += 1
                    _update_console_log(
                        "[{idx}/{total}] {target}: failed ({msg})".format(
                            idx=idx,
                            total=total,
                            target=target,
                            msg=str(item.get("message") or "unknown error"),
                        )
                    )
                if bool(item.get("requirements_changed")):
                    requirements_modules.append(target)
                _set_update_status(
                    phase="update",
                    current=idx,
                    total=total,
                    remaining=total - idx,
                    module=target,
                    message=status or "done",
                    updated=updated_count,
                    up_to_date=uptodate_count,
                    failed=failed_count,
                    requirements_changed=bool(requirements_modules),
                    requirements_modules=requirements_modules,
                    results=results,
                )

            _update_console_log("refreshing runtime state after update run...")
            _refresh_module_runtime_state(sync_upstreams=False, progress_cb=lambda **kwargs: None)
            _set_update_status(
                running=False,
                phase="done",
                message="done",
                module="",
                finished_at=_now_iso(),
            )
            _update_console_log(
                "job finished: updated={updated}, up_to_date={uptodate}, failed={failed}".format(
                    updated=updated_count,
                    uptodate=uptodate_count,
                    failed=failed_count,
                )
            )
        except Exception as exc:
            _update_console_log(f"job error: {exc}")
            _set_update_status(running=False, phase="error", message="error", error=str(exc), module="", finished_at=_now_iso())
        finally:
            with _UPDATE_LOCK:
                _UPDATE_THREAD = None

    thread = threading.Thread(target=_runner, name="alexz-module-update", daemon=True)
    with _UPDATE_LOCK:
        _UPDATE_THREAD = thread
    thread.start()
    return {"status": "started", "update": _update_status_snapshot()}


def _install_requirements_for_modules(modules: list[str]) -> dict[str, Any]:
    """Install requirements.txt for a list of modules after update confirmation."""
    if not isinstance(modules, list):
        return {"status": "error", "error": "modules must be a list"}
    canonical = [_canonical_custom_module_name(str(x)) for x in modules if str(x).strip()]
    canonical = [x for x in dict.fromkeys(canonical) if x and x != "unknown"]
    if not canonical:
        return {"status": "ok", "count": 0, "results": []}
    _LOGGER.info("Installing requirements for %d module(s): %s", len(canonical), ", ".join(canonical))

    results: list[dict[str, Any]] = []
    installed = 0
    failed = 0
    for module_name in canonical:
        item = _install_module_requirements(module_name)
        results.append(item)
        if str(item.get("status")) == "installed":
            installed += 1
        else:
            failed += 1
    _LOGGER.info(
        "Requirements install summary: total=%d installed=%d failed=%d",
        len(canonical),
        installed,
        failed,
    )
    return {"status": "ok", "count": len(canonical), "installed": installed, "failed": failed, "results": results}


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None):
    _LOGGER.info("✅ Module Nodes widget backend loaded")

    @PromptServer.instance.routes.post(ROUTE_MODULE_REFRESH)
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
            requested_log_mode = _normalize_log_mode(payload.get("log_mode") if isinstance(payload, dict) else None)
            _set_update_console_log_mode(requested_log_mode)
            return web.json_response(_start_refresh_job(sync_upstreams=do_sync))
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module refresh API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_MODULE_REFRESH_STATUS)
    async def alexz_tools_module_refresh_status(request):
        """API route that returns current module-refresh job status."""
        try:
            return web.json_response({"status": "ok", "refresh": _refresh_status_snapshot()})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module refresh status API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post(ROUTE_MODULE_ACKNOWLEDGE_ALL)
    async def alexz_tools_module_acknowledge_all(request):
        """API route that clears novelty markers for all modules."""
        try:
            result = _acknowledge_all_novelty()
            return web.json_response(result)
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module acknowledge-all API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post(ROUTE_MODULE_UPDATE)
    async def alexz_tools_module_update(request):
        """API route that starts asynchronous module update jobs."""
        try:
            if _INFO_ONLY_WIDGET_MODE:
                return web.json_response(
                    _info_only_rejection_payload("module_update"),
                    status=403,
                )
            payload = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            scope = str(payload.get("scope") or request.query.get("scope") or "single").strip().lower()
            module_name = str(payload.get("module") or request.query.get("module") or "").strip()
            requested_log_mode = _normalize_log_mode(payload.get("log_mode") or request.query.get("log_mode") or "summary")
            started = _start_module_update_job(scope=scope, module_name=module_name, log_mode=requested_log_mode)
            if started.get("status") == "error":
                return web.json_response(started, status=400)
            return web.json_response(started)
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module update API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_MODULE_UPDATE_STATUS)
    async def alexz_tools_module_update_status(request):
        """API route that returns current module-update job status."""
        try:
            return web.json_response({"status": "ok", "update": _update_status_snapshot()})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module update status API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post(ROUTE_MODULE_INSTALL_REQUIREMENTS)
    async def alexz_tools_module_install_requirements(request):
        """API route that installs Python requirements for selected modules."""
        try:
            if _INFO_ONLY_WIDGET_MODE:
                return web.json_response(
                    _info_only_rejection_payload("module_install_requirements"),
                    status=403,
                )
            payload = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            modules = payload.get("modules")
            result = _install_requirements_for_modules(modules if isinstance(modules, list) else [])
            status_code = 200 if result.get("status") == "ok" else 400
            return web.json_response(result, status=status_code)
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module requirements install API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post(ROUTE_COMFYUI_INSTALL_REQUIREMENTS)
    async def alexz_tools_comfyui_install_requirements(request):
        """API route that installs ComfyUI requirements in the active environment."""
        try:
            if _INFO_ONLY_WIDGET_MODE:
                return web.json_response(
                    _info_only_rejection_payload("comfyui_install_requirements"),
                    status=403,
                )
            result = _install_comfyui_requirements()
            status_code = 200 if result.get("status") == "installed" else 400
            return web.json_response(result, status=status_code)
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("ComfyUI requirements install API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_COMPONENT_REGISTRY)
    async def alexz_tools_component_registry(request):
        """API route that returns extensibility registry snapshot (nodes/widgets/api)."""
        try:
            refresh_raw = (request.query.get("refresh", "0") or "0").strip().lower()
            force_refresh = refresh_raw not in {"0", "false", "no", "off"}
            payload = _component_registry_payload(force_refresh=force_refresh)
            return web.json_response({"status": "ok", "registry": payload})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Component registry API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_NODE_CATALOG)
    async def alexz_tools_node_catalog(request):
        """API route that returns grouped module and node catalog data."""
        try:
            mode = _normalize_comfyui_mode(request.query.get("comfyui_mode", "") or request.query.get("mode", ""))
            _start_runtime_state_warmup()
            grouped = _build_group_catalog()
            modules_by_group = _build_group_modules(grouped)
            comfyui = _comfyui_git_status(mode=mode)
            show_custom_update_status = _custom_update_checked_flag()
            custom_modules_need_update = _count_custom_modules_need_update() if show_custom_update_status else 0
            custom_modules_unknown_update = _count_custom_modules_unknown_update() if show_custom_update_status else 0
            runtime_warmup = _runtime_warmup_status()
            groups = []
            for group_id, group_title in _GROUP_ORDER:
                nodes = grouped.get(group_id, [])
                modules = modules_by_group.get(group_id, [])
                groups.append(
                    {
                        "id": group_id,
                        "title": group_title,
                        "count": len(nodes),
                        "nodes": nodes,
                        "module_count": len(modules),
                        "modules": modules,
                    }
                )
            return web.json_response(
                {
                    "groups": groups,
                    "comfyui": comfyui,
                    "custom_modules_need_update": custom_modules_need_update,
                    "custom_modules_unknown_update": custom_modules_unknown_update,
                    "runtime_warmup": runtime_warmup,
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Node catalog API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_MODULE_INFO)
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
            _start_runtime_state_warmup()
            info = _resolve_module_info(
                group,
                module_name,
                force_refresh=force_refresh,
                sync_upstream=sync_upstream,
                cache_only=cache_only,
            )
            if force_refresh:
                _acknowledge_module_novelty(group, module_name)
                info = _resolve_module_info(
                    group,
                    module_name,
                    force_refresh=True,
                    sync_upstream=False,
                    cache_only=True,
                )
            return web.json_response({"group": group, "module": module_name, "info": info})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_COMFYUI_INFO)
    async def alexz_tools_comfyui_info(request):
        """API route that returns ComfyUI update and version status."""
        try:
            refresh_raw = (request.query.get("refresh", "1") or "1").strip().lower()
            force_refresh = refresh_raw not in {"0", "false", "no", "off"}
            ack_raw = (request.query.get("acknowledge", "1") or "1").strip().lower()
            acknowledge = ack_raw not in {"0", "false", "no", "off"}
            mode = _normalize_comfyui_mode(request.query.get("mode", ""))
            log_mode = _normalize_log_mode(request.query.get("log_mode", "summary"))
            _set_update_console_log_mode(log_mode)
            if force_refresh:
                _refresh_console_log(
                    "ComfyUI info refresh started (mode={mode}, acknowledge={ack}, log_mode={log})".format(
                        mode=mode,
                        ack="on" if acknowledge else "off",
                        log=log_mode,
                    )
                )
                _LOGGER.info(
                    "ComfyUI info refresh requested: mode=%s acknowledge=%s",
                    mode,
                    acknowledge,
                )
            if force_refresh and acknowledge:
                _acknowledge_comfyui_novelty()
            comfyui = _comfyui_git_status(force_refresh=force_refresh, mode=mode)
            if force_refresh:
                _refresh_console_log(
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
                _refresh_console_log(
                    "ComfyUI refs: path={path}, remote={remote_name}, branch={branch}, upstream={upstream}, remote_ref={remote_ref}".format(
                        path=str(comfyui.get("path") or "-"),
                        remote_name=str(comfyui.get("remote_name") or "-"),
                        branch=str(comfyui.get("branch") or "-"),
                        upstream=str(comfyui.get("upstream") or "-"),
                        remote_ref=str(comfyui.get("remote_ref") or "-"),
                    ),
                    level="verbose",
                )
                _LOGGER.info(
                    "ComfyUI info refresh finished: update_status=%s update_available=%s local=%s remote=%s mode=%s",
                    str(comfyui.get("update_status") or "unknown"),
                    bool(comfyui.get("update_available")),
                    str(comfyui.get("installed_commit_short") or "unknown"),
                    str(comfyui.get("remote_commit_short") or "unknown"),
                    str(comfyui.get("check_mode") or mode),
                )
                _refresh_console_log("ComfyUI info refresh finished")
            return web.json_response({"status": "ok", "comfyui": comfyui})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("ComfyUI info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_MODULE_LIST)
    async def alexz_tools_module_list(request):
        """API route that returns module list for the selected group."""
        query = (request.query.get("q", "") or "").strip().lower()
        try:
            catalog = _build_catalog()
            modules = []
            for module_name, nodes in catalog.items():
                if query and query not in module_name.lower():
                    continue
                modules.append({"module": module_name, "count": len(nodes)})
            return web.json_response({"query": query, "modules": modules})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module list API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get(ROUTE_MODULE_NODES)
    async def alexz_tools_module_nodes(request):
        """API route that returns node list for the selected module."""
        query = (request.query.get("module", "") or request.query.get("q", "")).strip()
        try:
            catalog = _build_catalog()
            modules = list(catalog.keys())
            selected_modules = _filter_modules(query, modules)

            results = []
            for module_name in selected_modules:
                nodes = catalog.get(module_name, [])
                results.append(
                    {
                        "module": module_name,
                        "count": len(nodes),
                        "nodes": nodes,
                    }
                )

            return web.json_response(
                {
                    "query": query,
                    "module_count": len(results),
                    "results": results,
                    "hint": "Введите имя python-модуля (например: ComfyUI_ALEXZ_tools).",
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module browser API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
