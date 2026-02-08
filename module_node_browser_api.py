from __future__ import annotations

import importlib
import json
import logging
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path
from typing import Any

try:
    import folder_paths
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - non-Comfy environment
    folder_paths = None
    PromptServer = None
    web = None


_LOGGER = logging.getLogger("ALEXZ_tools.ModuleBrowser")
_MODULE_INFO_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_MODULE_INFO_TTL_SEC = 30.0
_MANAGER_INDEX_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None
_MANAGER_GITHUB_STATS_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None
_MODULE_STATE_CACHE: dict[str, dict[str, Any]] | None = None
_GITHUB_RE = re.compile(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)", re.IGNORECASE)
_MODULE_STATE_PATH = Path(__file__).resolve().with_name("module_state_cache.json")
_GROUP_ORDER = (
    ("core", "Core_Nodes"),
    ("core_extras", "Core_Extras_Nodes"),
    ("api", "API_Nodes"),
    ("custom", "Custom_Nodes"),
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
}


def _short_commit(commit: str | None) -> str:
    value = (commit or "").strip()
    if not value:
        return "unknown"
    return value[:8]


def _module_root(node_cls: Any) -> str:
    module_name = getattr(node_cls, "__module__", "") or ""
    if not module_name:
        return "unknown"
    return module_name.split(".", 1)[0]


def _classify_by_relative_module(node_cls: Any) -> tuple[str, str]:
    rel = getattr(node_cls, "RELATIVE_PYTHON_MODULE", None)
    if not isinstance(rel, str) or not rel:
        return ("core", _module_root(node_cls))
    parts = [p for p in rel.split(".") if p]
    if len(parts) >= 2:
        root, module_name = parts[0], parts[1]
    elif len(parts) == 1:
        root, module_name = parts[0], parts[0]
    else:
        return ("core", _module_root(node_cls))

    if root == "custom_nodes":
        return ("custom", module_name)
    if root == "comfy_extras":
        return ("core_extras", module_name)
    if root == "comfy_api_nodes":
        return ("api", module_name)
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
    if folder_paths is not None and hasattr(folder_paths, "get_folder_paths"):
        try:
            roots = [Path(x) for x in folder_paths.get_folder_paths("custom_nodes") if x]
            if roots:
                return roots
        except Exception:
            pass
    return [Path(__file__).resolve().parents[1]]


def _normalize_repo_url(url: str | None) -> str | None:
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
    norm = _normalize_repo_url(url)
    if not norm:
        return None
    match = _GITHUB_RE.search(norm)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}".lower()


def _repo_name(url: str | None) -> str | None:
    gid = _github_id(url)
    if not gid:
        return None
    return gid.split("/", 1)[1]


def _pick_repo_url(entry: dict[str, Any]) -> str | None:
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
    for root in _custom_nodes_roots():
        db_path = root / "comfyui-manager" / "custom-node-list.json"
        if db_path.exists():
            return db_path
    return None


def _manager_github_stats_path() -> Path | None:
    for root in _custom_nodes_roots():
        db_path = root / "comfyui-manager" / "github-stats.json"
        if db_path.exists():
            return db_path
    return None


def _parse_datetime(value: str | None) -> datetime | None:
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
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manager_github_stats() -> dict[str, dict[str, dict[str, Any]]]:
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


def _run_git(args: list[str], timeout: float = 2.0) -> str | None:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


def _module_repo_url(module_name: str) -> str | None:
    module_name = (module_name or "").strip()
    if not module_name:
        return None
    for root in _custom_nodes_roots():
        module_dir = root / module_name
        if not module_dir.exists():
            continue
        out = _run_git(["git", "-C", str(module_dir), "config", "--get", "remote.origin.url"])
        if out:
            return _normalize_repo_url(out)
    return None


def _module_git_state(module_name: str) -> dict[str, Any]:
    module_name = (module_name or "").strip()
    if not module_name:
        return {}
    for root in _custom_nodes_roots():
        module_dir = root / module_name
        if not module_dir.exists():
            continue
        is_git = _run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"])
        if is_git != "true":
            continue

        state: dict[str, Any] = {
            "module_path": str(module_dir),
            "repository": _normalize_repo_url(
                _run_git(["git", "-C", str(module_dir), "config", "--get", "remote.origin.url"])
            ),
            "installed_commit": _run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"]),
            "installed_updated_at": _run_git(["git", "-C", str(module_dir), "log", "-1", "--format=%cI"]),
            "remote_updated_at": _run_git(["git", "-C", str(module_dir), "log", "-1", "--format=%cI", "@{u}"]),
            "branch": _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "HEAD"]),
            "upstream": _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]),
            "ahead": None,
            "behind": None,
            "remote_head": _run_git(["git", "-C", str(module_dir), "rev-parse", "@{u}"]),
        }

        counts = _run_git(["git", "-C", str(module_dir), "rev-list", "--left-right", "--count", "HEAD...@{u}"])
        if counts:
            parts = counts.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                state["ahead"] = int(parts[0])
                state["behind"] = int(parts[1])

        return state
    return {}


def _load_module_state() -> dict[str, dict[str, Any]]:
    global _MODULE_STATE_CACHE
    if _MODULE_STATE_CACHE is not None:
        return _MODULE_STATE_CACHE
    if not _MODULE_STATE_PATH.exists():
        _MODULE_STATE_CACHE = {}
        return _MODULE_STATE_CACHE
    try:
        with _MODULE_STATE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        _MODULE_STATE_CACHE = data if isinstance(data, dict) else {}
    except Exception:
        _MODULE_STATE_CACHE = {}
    return _MODULE_STATE_CACHE


def _save_module_state(state: dict[str, dict[str, Any]]) -> None:
    try:
        with _MODULE_STATE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
    except Exception as exc:
        _LOGGER.debug("Failed to save module state cache: %s", exc)


def _remember_module_state(module_name: str, result: dict[str, Any]) -> None:
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
    entry["module_path"] = result.get("module_path")
    entry["repository"] = result.get("repository")
    state[module_name] = entry
    result["last_checked_at"] = entry.get("last_checked_at")
    result["last_local_change_at"] = entry.get("last_local_change_at")
    startup_prev = (entry.get("startup_prev_commit") or "").strip()
    startup_new = (entry.get("startup_new_commit") or "").strip()
    result["updated_between_runs"] = bool(startup_prev and startup_new)
    result["startup_prev_commit_short"] = _short_commit(startup_prev) if startup_prev else ""
    result["startup_new_commit_short"] = _short_commit(startup_new) if startup_new else ""
    result["startup_update_at"] = entry.get("startup_update_at") or ""
    _save_module_state(state)


def _announce_tracked_module_updates() -> None:
    state = _load_module_state()
    if not isinstance(state, dict) or not state:
        return

    now = _now_iso()
    changed = False

    for module_name in sorted(state.keys()):
        entry = state.get(module_name, {})
        if not isinstance(entry, dict):
            continue

        prev_commit = (entry.get("installed_commit") or "").strip()
        if not prev_commit:
            continue

        git_state = _module_git_state(module_name)
        current_commit = (git_state.get("installed_commit") or "").strip()
        if not current_commit:
            continue

        entry["last_checked_at"] = now

        if current_commit != prev_commit:
            entry["installed_commit"] = current_commit
            entry["installed_updated_at"] = git_state.get("installed_updated_at") or entry.get("installed_updated_at")
            entry["remote_updated_at"] = git_state.get("remote_updated_at") or entry.get("remote_updated_at")
            entry["last_local_change_at"] = now
            entry["startup_prev_commit"] = prev_commit
            entry["startup_new_commit"] = current_commit
            entry["startup_update_at"] = now
            changed = True
        else:
            # Show "updated between runs" only for one startup cycle after actual change.
            entry.pop("startup_prev_commit", None)
            entry.pop("startup_new_commit", None)
            entry.pop("startup_update_at", None)

        state[module_name] = entry

    if changed:
        _save_module_state(state)


def _module_local_readme_summary(module_name: str) -> str | None:
    module_name = (module_name or "").strip()
    if not module_name:
        return None
    readme_names = ("README.md", "readme.md", "README.MD")
    for root in _custom_nodes_roots():
        module_dir = root / module_name
        if not module_dir.exists():
            continue
        for name in readme_names:
            path = module_dir / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                    continue
                if len(stripped) > 800:
                    stripped = stripped[:800] + "..."
                return stripped
    return None


def _resolve_module_info(group: str, module_name: str) -> dict[str, Any]:
    key = (group or "", module_name or "")
    now_ts = time.time()
    cached = _MODULE_INFO_CACHE.get(key)
    if cached is not None and (now_ts - cached[0]) < _MODULE_INFO_TTL_SEC:
        return dict(cached[1])

    result: dict[str, Any] = {
        "module": module_name,
        "group": group,
        "title": module_name,
        "author": "",
        "description": "",
        "repository": "",
        "owner_url": "",
        "module_path": "",
        "installed_commit": "",
        "installed_commit_short": "",
        "installed_updated_at": "",
        "remote_updated_at": "",
        "update_available": None,
        "update_status": "unknown",
        "last_checked_at": "",
        "last_local_change_at": "",
        "updated_between_runs": False,
        "startup_prev_commit_short": "",
        "startup_new_commit_short": "",
        "startup_update_at": "",
        "source": "none",
    }

    if group != "custom":
        if group in {"core", "core_extras", "api"}:
            result["author"] = "ComfyUI"
            result["repository"] = "https://github.com/comfyanonymous/ComfyUI"
            result["owner_url"] = result["repository"]
            result["description"] = {
                "core": "Built-in ComfyUI nodes.",
                "core_extras": "Built-in ComfyUI extras module.",
                "api": "Built-in ComfyUI API nodes module.",
            }.get(group, "")
            result["source"] = "builtin"
        _MODULE_INFO_CACHE[key] = (now_ts, dict(result))
        return result

    manager_data = _manager_index()
    module_l = (module_name or "").lower()
    git_state = _module_git_state(module_name)
    repo_url = _module_repo_url(module_name) or git_state.get("repository")
    repo_gid = _github_id(repo_url)
    meta = None
    if repo_gid:
        meta = manager_data["by_github"].get(repo_gid)
    if meta is None and module_l:
        meta = manager_data["by_id"].get(module_l)
    if meta is None and module_l:
        meta = manager_data["by_repo_name"].get(module_l)

    if meta is not None:
        result["title"] = meta.get("title") or module_name
        result["author"] = meta.get("author") or ""
        result["description"] = meta.get("description") or ""
        result["repository"] = meta.get("repository") or repo_url or ""
        result["source"] = "comfyui-manager"
    else:
        result["repository"] = repo_url or ""
        result["description"] = _module_local_readme_summary(module_name) or ""
        result["source"] = "local"

    if not result["author"] and repo_gid:
        result["author"] = repo_gid.split("/", 1)[0]
    if not result["description"]:
        result["description"] = "No description found."
    if result["repository"]:
        result["owner_url"] = result["repository"]
    if git_state:
        result["module_path"] = git_state.get("module_path") or ""
        result["installed_commit"] = git_state.get("installed_commit") or ""
        result["installed_commit_short"] = (result["installed_commit"] or "")[:8]
        result["installed_updated_at"] = git_state.get("installed_updated_at") or ""
        result["remote_updated_at"] = git_state.get("remote_updated_at") or ""
        behind = git_state.get("behind")
        remote_head = git_state.get("remote_head")
        if isinstance(behind, int):
            result["update_available"] = behind > 0
            result["update_status"] = "can_update" if behind > 0 else "up_to_date"
        elif remote_head and result["installed_commit"]:
            if remote_head == result["installed_commit"]:
                result["update_available"] = False
                result["update_status"] = "up_to_date"
            else:
                result["update_available"] = True
                result["update_status"] = "can_update"

    stats_meta = None
    stats = _manager_github_stats()
    norm_repo = _normalize_repo_url(result.get("repository"))
    if norm_repo:
        stats_meta = stats["by_url"].get(norm_repo)
    if stats_meta is None and repo_gid:
        stats_meta = stats["by_github"].get(repo_gid)
    if isinstance(stats_meta, dict):
        remote_raw = stats_meta.get("last_update")
        remote_dt = _parse_datetime(remote_raw)
        if remote_dt is not None:
            if not result.get("remote_updated_at"):
                result["remote_updated_at"] = _to_iso(remote_dt) or ""
            if result["update_available"] is None:
                local_dt = _parse_datetime(result.get("installed_updated_at"))
                if local_dt is not None:
                    # manager's last_update is coarse; treat >5 min as potential update
                    has_update = remote_dt > (local_dt + timedelta(minutes=5))
                    result["update_available"] = has_update
                    result["update_status"] = "can_update" if has_update else "up_to_date"
                else:
                    result["update_status"] = "unknown"

    _remember_module_state(module_name, result)
    _MODULE_INFO_CACHE[key] = (now_ts, dict(result))
    return result


def _collect_nodes() -> list[dict[str, Any]]:
    comfy_nodes = importlib.import_module("nodes")
    class_map = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
    display_map = getattr(comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

    items: list[dict[str, Any]] = []
    for node_name, node_cls in class_map.items():
        display_name = display_map.get(node_name, node_name)
        annotation = _ALEXZ_ANNOTATIONS.get(node_name) or _fallback_annotation(node_cls)
        group, module_bucket = _classify_by_relative_module(node_cls)
        items.append(
            {
                "node_name": node_name,
                "display_name": display_name,
                "module": module_bucket,
                "group": group,
                "category": getattr(node_cls, "CATEGORY", "") or "",
                "annotation": annotation,
            }
        )
    return items


def _build_catalog() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _collect_nodes():
        module_name = item["module"]
        grouped[module_name].append(item)

    for module_name in grouped:
        grouped[module_name].sort(key=lambda item: item["display_name"].lower())
    return dict(sorted(grouped.items(), key=lambda kv: kv[0].lower()))


def _build_group_catalog() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _collect_nodes():
        grouped[item["group"]].append(item)

    for group_name in grouped:
        grouped[group_name].sort(key=lambda item: item["display_name"].lower())
    return grouped


def _filter_modules(query: str, module_names: list[str]) -> list[str]:
    if not query:
        return module_names
    q = query.lower()
    exact = [name for name in module_names if name.lower() == q]
    if exact:
        return exact
    return [name for name in module_names if q in name.lower()]


if folder_paths is not None:
    try:
        _announce_tracked_module_updates()
    except Exception as exc:  # pragma: no cover - startup diagnostic
        _LOGGER.debug("Module update startup check failed: %s", exc)


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None):
    @PromptServer.instance.routes.get("/alexz_tools/node_catalog")
    async def alexz_tools_node_catalog(request):
        try:
            grouped = _build_group_catalog()
            groups = []
            for group_id, group_title in _GROUP_ORDER:
                nodes = grouped.get(group_id, [])
                groups.append(
                    {
                        "id": group_id,
                        "title": group_title,
                        "count": len(nodes),
                        "nodes": nodes,
                    }
                )
            return web.json_response({"groups": groups})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Node catalog API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_info")
    async def alexz_tools_module_info(request):
        group = (request.query.get("group", "") or "").strip().lower()
        module_name = (request.query.get("module", "") or "").strip()
        if not module_name:
            return web.json_response({"error": "module is required"}, status=400)
        try:
            info = _resolve_module_info(group, module_name)
            return web.json_response({"group": group, "module": module_name, "info": info})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_list")
    async def alexz_tools_module_list(request):
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

    @PromptServer.instance.routes.get("/alexz_tools/module_nodes")
    async def alexz_tools_module_nodes(request):
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
