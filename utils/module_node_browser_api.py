"""Utility/support module: `utils/module_node_browser_api.py`."""
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
from hashlib import sha1
from collections import defaultdict
from datetime import datetime, timezone
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
_CUSTOM_MODULE_ALIAS_CACHE: dict[str, str] | None = None
_COMFYUI_STATUS_CACHE: tuple[float, dict[str, Any]] | None = None
_COMFYUI_STATUS_TTL_SEC = 120.0
_LAZY_REFRESH_DONE = False
_REFRESH_LOCK = threading.Lock()
_REFRESH_THREAD: threading.Thread | None = None
_REFRESH_STATUS: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "current": 0,
    "total": 0,
    "remaining": 0,
    "modules_need_update": 0,
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


def _run_git(args: list[str], timeout: float = 2.0) -> str | None:
    """Run a git command in the target directory with safe non-interactive environment."""
    env = os.environ.copy()
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
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    return out or None


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
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


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
    before = (before_commit or "").strip()
    after = (after_commit or "").strip()
    if not before or not after or before == after:
        return False
    diff = _run_command(
        ["git", "-C", str(module_dir), "diff", "--name-only", f"{before}..{after}", "--", "requirements.txt"],
        timeout=20.0,
        disable_git_prompt=True,
    )
    if not diff.get("ok"):
        return False
    changed_files = [line.strip().lower() for line in str(diff.get("stdout") or "").splitlines() if line.strip()]
    return "requirements.txt" in changed_files


def _module_needs_update_now(module_name: str) -> bool:
    """Check whether local module commit differs from tracked remote commit."""
    git_state = _module_git_state(module_name)
    if not git_state:
        return False
    behind = git_state.get("behind")
    if isinstance(behind, int):
        return behind > 0
    remote_head = (git_state.get("remote_head") or "").strip()
    installed = (git_state.get("installed_commit") or "").strip()
    return bool(git_state.get("has_upstream") and remote_head and installed and remote_head != installed)


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


def _comfyui_requirements_path() -> Path | None:
    """Resolve requirements.txt path for the main ComfyUI repository."""
    root = _comfyui_root()
    if root is None:
        return None
    req = root / "requirements.txt"
    return req if req.exists() else None


def _comfyui_needs_update_now() -> bool:
    """Check whether local ComfyUI commit is behind remote tracking commit."""
    status = _comfyui_git_status(force_refresh=True)
    behind = status.get("behind")
    if isinstance(behind, int):
        return behind > 0
    return bool(status.get("update_status") == "can_update")


def _git_remote_names(repo_root: Path) -> list[str]:
    """Return list of configured git remote names for repository."""
    out = _run_git(["git", "-C", str(repo_root), "remote"])
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _git_pick_remote(repo_root: Path, upstream: str | None) -> str | None:
    """Choose preferred git remote (upstream, origin, or first available)."""
    upstream_text = (upstream or "").strip()
    if upstream_text and "/" in upstream_text:
        return upstream_text.split("/", 1)[0].strip() or None
    remotes = _git_remote_names(repo_root)
    if "origin" in remotes:
        return "origin"
    if "upstream" in remotes:
        return "upstream"
    return remotes[0] if remotes else None


def _git_ref_exists(repo_root: Path, ref_name: str) -> bool:
    """Check whether a local or remote git reference exists."""
    ref = (ref_name or "").strip()
    if not ref:
        return False
    return bool(_run_git(["git", "-C", str(repo_root), "rev-parse", "--verify", ref]))


def _git_resolve_remote_ref(
    repo_root: Path,
    remote_name: str,
    branch_name: str | None,
    upstream: str | None,
) -> tuple[str | None, str | None]:
    """Resolve remote tracking reference to compare local and upstream revisions."""
    upstream_text = (upstream or "").strip()
    if upstream_text and "/" in upstream_text:
        remote_branch = upstream_text.split("/", 1)[1].strip()
        return (upstream_text, remote_branch or None)

    branch = (branch_name or "").strip()
    if branch and branch != "HEAD":
        by_branch = f"{remote_name}/{branch}"
        if _git_ref_exists(repo_root, by_branch):
            return (by_branch, branch)

    head_ref = _run_git(
        ["git", "-C", str(repo_root), "symbolic-ref", "--quiet", f"refs/remotes/{remote_name}/HEAD"]
    )
    remote_branch = ""
    if head_ref:
        prefix = f"refs/remotes/{remote_name}/"
        if head_ref.startswith(prefix):
            remote_branch = head_ref[len(prefix) :].strip()

    if not remote_branch:
        remote_info = _run_git(["git", "-C", str(repo_root), "remote", "show", remote_name], timeout=8.0) or ""
        for line in remote_info.splitlines():
            text = line.strip()
            if text.lower().startswith("head branch:"):
                remote_branch = text.split(":", 1)[1].strip()
                break

    if not remote_branch:
        for candidate in ("main", "master"):
            ref = f"{remote_name}/{candidate}"
            if _git_ref_exists(repo_root, ref):
                remote_branch = candidate
                break

    if not remote_branch:
        return (None, None)
    return (f"{remote_name}/{remote_branch}", remote_branch)


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
    pull = _run_command(pull_cmd, timeout=timeout, disable_git_prompt=True)
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
        result["requirements_changed"] = _requirements_changed_between(root, before_commit, after_commit)
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
        "before_commit": "",
        "after_commit": "",
    }
    if module_dir is None:
        result["status"] = "not_found"
        result["message"] = "module directory not found"
        return result

    is_git = _run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        result["status"] = "no_git"
        result["message"] = "not a git repository"
        return result

    upstream = _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if not upstream:
        result["status"] = "no_upstream"
        result["message"] = "upstream is not configured"
        return result

    before_commit = _run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"]) or ""
    result["before_commit"] = before_commit
    pull = _run_command(
        ["git", "-C", str(module_dir), "pull", "--ff-only"],
        timeout=timeout,
        disable_git_prompt=True,
    )
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
        result["message"] = "module updated"
        result["requirements_changed"] = _requirements_changed_between(module_dir, before_commit, after_commit)
    else:
        result["status"] = "up_to_date"
        result["message"] = "already up to date"
    return result


def _install_module_requirements(module_name: str, timeout: float = 1200.0) -> dict[str, Any]:
    """Install Python dependencies from module requirements.txt in active runtime environment."""
    module = _canonical_custom_module_name(module_name)
    module_dir = _module_dir(module)
    result: dict[str, Any] = {
        "module": module,
        "status": "error",
        "message": "",
        "requirements_path": "",
    }
    if module_dir is None:
        result["status"] = "not_found"
        result["message"] = "module directory not found"
        return result

    requirements_path = module_dir / "requirements.txt"
    result["requirements_path"] = str(requirements_path)
    if not requirements_path.exists():
        result["status"] = "missing_requirements"
        result["message"] = "requirements.txt not found"
        return result

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    run = _run_command(cmd, timeout=timeout)
    if not run.get("ok"):
        result["status"] = "error"
        result["message"] = str(run.get("stderr") or run.get("stdout") or "pip install failed")
        return result
    result["status"] = "installed"
    result["message"] = "requirements installed"
    return result


def _install_comfyui_requirements(timeout: float = 1800.0) -> dict[str, Any]:
    """Install Python dependencies from ComfyUI requirements.txt in active runtime environment."""
    result: dict[str, Any] = {
        "module": "ComfyUI",
        "status": "error",
        "message": "",
        "requirements_path": "",
    }
    req = _comfyui_requirements_path()
    if req is None:
        result["status"] = "missing_requirements"
        result["message"] = "ComfyUI requirements.txt not found"
        return result
    result["requirements_path"] = str(req)
    run = _run_command([sys.executable, "-m", "pip", "install", "-r", str(req)], timeout=timeout)
    if not run.get("ok"):
        result["status"] = "error"
        result["message"] = str(run.get("stderr") or run.get("stdout") or "pip install failed")
        return result
    result["status"] = "installed"
    result["message"] = "ComfyUI requirements installed"
    return result

def _module_repo_url(module_name: str) -> str | None:
    """Resolve module repository URL using manager metadata and git remotes."""
    module_name = _canonical_custom_module_name((module_name or "").strip())
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
    """Collect local/remote git commit and timestamp state for one module."""
    module_name = _canonical_custom_module_name((module_name or "").strip())
    if not module_name:
        return {}
    for root in _custom_nodes_roots():
        module_dir = root / module_name
        if not module_dir.exists():
            continue
        is_git = _run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"])
        if is_git != "true":
            continue

        upstream = _run_git(
            ["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
        )
        state: dict[str, Any] = {
            "module_path": str(module_dir),
            "repository": _normalize_repo_url(
                _run_git(["git", "-C", str(module_dir), "config", "--get", "remote.origin.url"])
            ),
            "installed_commit": _run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"]),
            "installed_updated_at": _run_git(["git", "-C", str(module_dir), "log", "-1", "--format=%cI"]),
            "remote_updated_at": _run_git(["git", "-C", str(module_dir), "log", "-1", "--format=%cI", "@{u}"]),
            "branch": _run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "HEAD"]),
            "upstream": upstream,
            "has_upstream": bool(upstream),
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


def _sync_module_upstream(module_name: str, timeout: float = 15.0) -> bool:
    """Fetch module remotes and refresh local view of upstream references."""
    module_name = _canonical_custom_module_name((module_name or "").strip())
    if not module_name:
        return False
    for root in _custom_nodes_roots():
        module_dir = root / module_name
        if not module_dir.exists():
            continue
        is_git = _run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"])
        if is_git != "true":
            continue
        upstream = _run_git(
            ["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
        )
        if not upstream:
            return False
        _run_git(["git", "-C", str(module_dir), "fetch", "--quiet"], timeout=timeout)
        return True
    return False


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


def _comfyui_git_status(force_refresh: bool = False) -> dict[str, Any]:
    """Collect local/remote git status summary for ComfyUI repository."""
    global _COMFYUI_STATUS_CACHE
    now_ts = time.time()
    if (
        not force_refresh
        and _COMFYUI_STATUS_CACHE is not None
        and (now_ts - _COMFYUI_STATUS_CACHE[0]) < _COMFYUI_STATUS_TTL_SEC
    ):
        return dict(_COMFYUI_STATUS_CACHE[1])

    result: dict[str, Any] = {
        "path": "",
        "repository": "https://github.com/comfyanonymous/ComfyUI",
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
        "ahead": None,
        "behind": None,
        "update_available": None,
        "update_status": "unknown",
    }

    root = _comfyui_root()
    if root is None:
        _COMFYUI_STATUS_CACHE = (now_ts, dict(result))
        return result

    result["path"] = str(root)
    is_git = _run_git(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"])
    if is_git != "true":
        _COMFYUI_STATUS_CACHE = (now_ts, dict(result))
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
        _COMFYUI_STATUS_CACHE = (now_ts, dict(result))
        return result

    # Keep remote refs fresh to reflect actual GitHub state.
    _run_git(["git", "-C", str(root), "fetch", "--quiet", remote_name], timeout=20.0)
    remote_ref, _remote_branch = _git_resolve_remote_ref(root, remote_name, result["branch"], upstream)
    result["remote_ref"] = remote_ref or ""
    if not remote_ref:
        _COMFYUI_STATUS_CACHE = (now_ts, dict(result))
        return result

    result["remote_commit"] = _run_git(["git", "-C", str(root), "rev-parse", remote_ref]) or ""
    result["remote_commit_short"] = _short_commit(result["remote_commit"]) if result["remote_commit"] else ""
    result["remote_updated_at"] = _run_git(["git", "-C", str(root), "log", "-1", "--format=%cI", remote_ref]) or ""

    counts = _run_git(["git", "-C", str(root), "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"])
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
            result["update_available"] = True
            result["update_status"] = "can_update"

    _COMFYUI_STATUS_CACHE = (now_ts, dict(result))
    return result


def _load_module_state() -> dict[str, dict[str, Any]]:
    """Load persisted module snapshot state from extension cache file."""
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
    """Persist module snapshot state to extension cache file."""
    try:
        with _MODULE_STATE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
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


def _apply_node_change_info(result: dict[str, Any], group: str, module_name: str) -> None:
    """Attach node-level change markers to module info payload for UI rendering."""
    state = _load_module_state()
    tracker = state.get("__node_tracker__")
    if not isinstance(tracker, dict):
        return
    startup_changes = tracker.get("startup_changes")
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

    startup_new_modules = tracker.get("startup_new_modules")
    if isinstance(startup_new_modules, dict):
        group_new = startup_new_modules.get(group)
        if isinstance(group_new, list) and module_name in group_new:
            result["new_module_between_runs"] = True
            result["updated_between_runs"] = True


def _announce_tracked_module_updates() -> dict[str, int]:
    """Build per-module node-change info by comparing saved and current snapshots."""
    state = _load_module_state()
    if not isinstance(state, dict):
        return {"modules_need_update": 0}

    now = _now_iso()
    changed = False
    modules_need_update = 0

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
        entry = state.get(module_name, {})
        if not isinstance(entry, dict):
            entry = {}
        prev_commit = (entry.get("installed_commit") or "").strip()
        git_state = _module_git_state(module_name)
        current_commit = (git_state.get("installed_commit") or "").strip()
        before = dict(entry)

        entry["last_checked_at"] = now
        needs_update = False
        if git_state:
            entry["module_path"] = git_state.get("module_path") or entry.get("module_path")
            entry["repository"] = git_state.get("repository") or entry.get("repository")
            entry["installed_updated_at"] = git_state.get("installed_updated_at") or entry.get("installed_updated_at")
            entry["remote_updated_at"] = git_state.get("remote_updated_at") or entry.get("remote_updated_at")
            behind = git_state.get("behind")
            remote_head = (git_state.get("remote_head") or "").strip()
            if isinstance(behind, int):
                needs_update = behind > 0
            elif git_state.get("has_upstream") and remote_head and current_commit:
                needs_update = remote_head != current_commit
            entry["update_available"] = bool(needs_update)

        if current_commit:
            if prev_commit and current_commit != prev_commit:
                entry["installed_commit"] = current_commit
                entry["last_local_change_at"] = now
                entry["startup_prev_commit"] = prev_commit
                entry["startup_new_commit"] = current_commit
                entry["startup_update_at"] = now
            else:
                entry["installed_commit"] = current_commit
                # Show update marker only for one startup cycle after actual change.
                entry.pop("startup_prev_commit", None)
                entry.pop("startup_new_commit", None)
                entry.pop("startup_update_at", None)

        if needs_update:
            modules_need_update += 1

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

    for group_name, current_list in current_module_sets.items():
        prev_list_raw = prev_module_sets.get(group_name)
        if not isinstance(prev_list_raw, list):
            continue
        prev_set = {x for x in prev_list_raw if isinstance(x, str)}
        curr_set = {x for x in current_list if isinstance(x, str)}
        new_modules = sorted(curr_set - prev_set, key=str.lower)
        if new_modules:
            startup_new_modules[group_name] = new_modules

    if prev_snapshots != current_snapshots:
        changed = True
    if prev_module_sets != current_module_sets:
        changed = True
    tracker["snapshots"] = current_snapshots
    tracker["startup_changes"] = startup_changes
    tracker["module_sets"] = current_module_sets
    tracker["startup_new_modules"] = startup_new_modules
    tracker["updated_at"] = now
    state["__node_tracker__"] = tracker

    if changed:
        _save_module_state(state)
    return {"modules_need_update": modules_need_update}


def _module_local_readme_summary(module_name: str) -> str | None:
    """Read and extract short description snippet from module README file."""
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
                if (
                    not stripped
                    or stripped.startswith("#")
                    or stripped.startswith("!")
                    or stripped.startswith("<")
                ):
                    continue
                if len(stripped) > 800:
                    stripped = stripped[:800] + "..."
                return stripped
    return None


def _sanitize_module_description(text: str) -> str:
    """Normalize module description text for UI card rendering."""
    value = str(text or "")
    if not value:
        return ""
    out_lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Drop pure HTML marker lines such as `<div align="center">`.
        if stripped.startswith("<"):
            plain = _HTML_TAG_RE.sub("", stripped).strip()
            if not plain:
                continue
            stripped = plain
        else:
            stripped = _HTML_TAG_RE.sub("", stripped).strip()
            if not stripped:
                continue
        if stripped.startswith("!"):
            continue
        out_lines.append(stripped)
    if not out_lines:
        return ""
    summary = out_lines[0]
    if len(summary) > 800:
        summary = summary[:800] + "..."
    return summary


def _resolve_module_info(
    group: str,
    module_name: str,
    *,
    force_refresh: bool = False,
    sync_upstream: bool = False,
) -> dict[str, Any]:
    """Build complete module info payload with metadata, git state, and change markers."""
    group = (group or "").strip().lower()
    module_name = (module_name or "").strip()
    if group == "custom":
        module_name = _canonical_custom_module_name(module_name)

    key = (group or "", module_name or "")
    if force_refresh:
        _MODULE_INFO_CACHE.pop(key, None)
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
        "git_has_upstream": False,
        "git_ahead": None,
        "git_behind": None,
        "last_checked_at": "",
        "last_local_change_at": "",
        "updated_between_runs": False,
        "startup_prev_commit_short": "",
        "startup_new_commit_short": "",
        "startup_update_at": "",
        "new_nodes_between_runs": [],
        "updated_nodes_between_runs": [],
        "startup_node_update_at": "",
        "new_module_between_runs": False,
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
        _apply_node_change_info(result, group, module_name)
        _MODULE_INFO_CACHE[key] = (now_ts, dict(result))
        return result

    if sync_upstream:
        _sync_module_upstream(module_name)

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
        result["description"] = _sanitize_module_description(meta.get("description") or "")
        result["repository"] = meta.get("repository") or repo_url or ""
        result["source"] = "comfyui-manager"
    else:
        result["repository"] = repo_url or ""
        result["description"] = _sanitize_module_description(_module_local_readme_summary(module_name) or "")
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
        result["git_has_upstream"] = bool(git_state.get("has_upstream"))
        result["git_ahead"] = git_state.get("ahead")
        result["git_behind"] = git_state.get("behind")
        behind = git_state.get("behind")
        remote_head = git_state.get("remote_head")
        if isinstance(behind, int):
            result["update_available"] = behind > 0
            result["update_status"] = "can_update" if behind > 0 else "up_to_date"
        elif result["git_has_upstream"] and remote_head and result["installed_commit"]:
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

    _remember_module_state(module_name, result)
    _apply_node_change_info(result, group, module_name)
    _MODULE_INFO_CACHE[key] = (now_ts, dict(result))
    return result


def _collect_nodes() -> list[dict[str, Any]]:
    """Collect node definitions from registered ComfyUI mappings."""
    class_map, display_map = _node_mappings()

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
    """Build cached module-to-node catalog from discovered nodes."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _collect_nodes():
        module_name = item["module"]
        grouped[module_name].append(item)

    for module_name in grouped:
        grouped[module_name].sort(key=lambda item: item["display_name"].lower())
    return dict(sorted(grouped.items(), key=lambda kv: kv[0].lower()))


def _build_group_catalog() -> dict[str, list[dict[str, Any]]]:
    """Build grouped node catalog for one category."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in _collect_nodes():
        grouped[item["group"]].append(item)

    for group_name in grouped:
        grouped[group_name].sort(key=lambda item: item["display_name"].lower())
    return grouped


def _build_group_modules(grouped_nodes: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Build grouped module summaries for one category."""
    module_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for group_name, nodes in grouped_nodes.items():
        for node in nodes:
            module_name = str(node.get("module") or "unknown")
            module_counts[group_name][module_name] += 1

    for module_name in _discover_custom_modules():
        module_counts["custom"].setdefault(module_name, 0)

    out: dict[str, list[dict[str, Any]]] = {}
    for group_name, counts in module_counts.items():
        out[group_name] = [
            {"module": mod, "count": int(cnt)}
            for mod, cnt in sorted(counts.items(), key=lambda kv: kv[0].lower())
        ]
    return out


def _filter_modules(query: str, module_names: list[str]) -> list[str]:
    """Filter module list by case-insensitive text query over module names."""
    if not query:
        return module_names
    q = query.lower()
    exact = [name for name in module_names if name.lower() == q]
    if exact:
        return exact
    return [name for name in module_names if q in name.lower()]


def _set_refresh_status(**kwargs: Any) -> None:
    """Set shared refresh job status fields in a thread-safe way."""
    with _REFRESH_LOCK:
        _REFRESH_STATUS.update(kwargs)
        _REFRESH_STATUS["updated_at"] = _now_iso()


def _refresh_status_snapshot() -> dict[str, Any]:
    """Return thread-safe snapshot of refresh-job status."""
    with _REFRESH_LOCK:
        return dict(_REFRESH_STATUS)


def _refresh_progress(
    *,
    phase: str,
    current: int = 0,
    total: int = 0,
    remaining: int = 0,
    modules_need_update: int = 0,
    module: str = "",
    message: str = "",
) -> None:
    """Update refresh-job progress counters and status text."""
    _set_refresh_status(
        phase=phase,
        current=int(current),
        total=int(total),
        remaining=max(0, int(remaining)),
        modules_need_update=max(0, int(modules_need_update)),
        module=module,
        message=message,
    )


def _refresh_module_runtime_state(sync_upstreams: bool = False, progress_cb: Any | None = None) -> dict[str, Any]:
    """Recompute module snapshots and update persisted runtime tracking state."""
    global _LAZY_REFRESH_DONE
    global _CUSTOM_MODULE_ALIAS_CACHE
    global _COMFYUI_STATUS_CACHE
    _MODULE_INFO_CACHE.clear()
    _CUSTOM_MODULE_ALIAS_CACHE = None
    _COMFYUI_STATUS_CACHE = None
    if progress_cb is None:
        progress_cb = _refresh_progress
    if sync_upstreams:
        module_names = _discover_custom_modules()
        total = len(module_names)
        progress_cb(phase="sync", current=0, total=total, remaining=total, message="sync_upstreams")
        for idx, module_name in enumerate(module_names, start=1):
            synced = _sync_module_upstream(module_name)
            status = "synced" if synced else "skip"
            progress_cb(
                phase="sync",
                current=idx,
                total=total,
                remaining=total - idx,
                module=module_name,
                message=status,
            )
    else:
        progress_cb(phase="sync", current=0, total=0, remaining=0, message="fast_mode")

    progress_cb(phase="snapshots", current=0, total=0, remaining=0, message="recompute_snapshots")
    announce_summary = _announce_tracked_module_updates()
    modules_need_update = 0
    if isinstance(announce_summary, dict):
        modules_need_update = max(0, int(announce_summary.get("modules_need_update", 0)))
    comfyui = _comfyui_git_status(force_refresh=True)
    progress_cb(
        phase="done",
        current=0,
        total=0,
        remaining=0,
        modules_need_update=modules_need_update,
        message="done",
    )
    _LAZY_REFRESH_DONE = True
    return {
        "status": "ok",
        "refreshed_at": _now_iso(),
        "comfyui": comfyui,
        "sync_upstreams": sync_upstreams,
        "modules_need_update": modules_need_update,
    }


def _ensure_runtime_state_ready() -> None:
    """Ensure runtime snapshot cache is initialized before serving API requests."""
    global _LAZY_REFRESH_DONE
    if _LAZY_REFRESH_DONE:
        return
    _refresh_module_runtime_state(sync_upstreams=False, progress_cb=None)
    _LAZY_REFRESH_DONE = True


def _start_refresh_job(sync_upstreams: bool) -> dict[str, Any]:
    """Start background module refresh job if one is not already running."""
    global _REFRESH_THREAD
    with _REFRESH_LOCK:
        thread = _REFRESH_THREAD
        if thread is not None and thread.is_alive():
            return {"status": "running", "refresh": dict(_REFRESH_STATUS)}
        _REFRESH_STATUS.update(
            {
                "running": True,
                "phase": "starting",
                "current": 0,
                "total": 0,
                "remaining": 0,
                "modules_need_update": 0,
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
            result = _refresh_module_runtime_state(sync_upstreams=sync_upstreams, progress_cb=_refresh_progress)
            _set_refresh_status(
                running=False,
                phase="done",
                message="done",
                module="",
                refreshed_at=result.get("refreshed_at", ""),
                modules_need_update=max(0, int(result.get("modules_need_update", 0))),
            )
        except Exception as exc:
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
    with _UPDATE_LOCK:
        _UPDATE_STATUS.update(kwargs)
        _UPDATE_STATUS["updated_at"] = _now_iso()


def _update_status_snapshot() -> dict[str, Any]:
    """Return thread-safe snapshot of module-update job status."""
    with _UPDATE_LOCK:
        return dict(_UPDATE_STATUS)


def _resolve_update_targets(scope: str, module_name: str) -> list[str]:
    """Resolve concrete module names targeted by update request payload."""
    scope_norm = (scope or "").strip().lower()
    if scope_norm == "single":
        canonical = _canonical_custom_module_name(module_name)
        if not canonical or canonical == "unknown":
            return []
        return [canonical]

    if scope_norm != "all":
        return []

    targets: list[str] = []
    for mod in _discover_custom_modules():
        _sync_module_upstream(mod)
        if _module_needs_update_now(mod):
            targets.append(_canonical_custom_module_name(mod))
    # Keep order stable and deduplicate.
    return list(dict.fromkeys(targets))


def _start_module_update_job(scope: str, module_name: str) -> dict[str, Any]:
    """Start background module update job for selected custom modules."""
    global _UPDATE_THREAD
    scope_norm = (scope or "").strip().lower()
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
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "finished_at": "",
            }
        )

    def _runner() -> None:
        """Background job worker that runs long update/refresh operations."""
        global _UPDATE_THREAD
        try:
            if scope_norm == "comfyui":
                _set_update_status(phase="update", current=0, total=1, remaining=1, module="ComfyUI", message="pull")
                item = _pull_comfyui()
                status = str(item.get("status") or "")
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
                return

            targets = _resolve_update_targets(scope_norm, module_name)
            total = len(targets)
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
                item = _pull_custom_module(target)
                results.append(item)
                status = str(item.get("status") or "")
                if status == "updated":
                    updated_count += 1
                elif status == "up_to_date":
                    uptodate_count += 1
                else:
                    failed_count += 1
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

            _refresh_module_runtime_state(sync_upstreams=False, progress_cb=lambda **kwargs: None)
            _set_update_status(
                running=False,
                phase="done",
                message="done",
                module="",
                finished_at=_now_iso(),
            )
        except Exception as exc:
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
    return {"status": "ok", "count": len(canonical), "installed": installed, "failed": failed, "results": results}


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None):
    _LOGGER.info("✅ Module Nodes widget backend loaded")

    @PromptServer.instance.routes.post("/alexz_tools/module_refresh")
    async def alexz_tools_module_refresh(request):
        """API route that starts asynchronous module status refresh."""
        try:
            sync_raw = (request.query.get("sync_upstreams", "1") or "1").strip().lower()
            do_sync = sync_raw not in {"0", "false", "no", "off"}
            return web.json_response(_start_refresh_job(sync_upstreams=do_sync))
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module refresh API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_refresh_status")
    async def alexz_tools_module_refresh_status(request):
        """API route that returns current module-refresh job status."""
        try:
            return web.json_response({"status": "ok", "refresh": _refresh_status_snapshot()})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module refresh status API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/alexz_tools/module_update")
    async def alexz_tools_module_update(request):
        """API route that starts asynchronous module update jobs."""
        try:
            payload = {}
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            scope = str(payload.get("scope") or request.query.get("scope") or "single").strip().lower()
            module_name = str(payload.get("module") or request.query.get("module") or "").strip()
            started = _start_module_update_job(scope=scope, module_name=module_name)
            if started.get("status") == "error":
                return web.json_response(started, status=400)
            return web.json_response(started)
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module update API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_update_status")
    async def alexz_tools_module_update_status(request):
        """API route that returns current module-update job status."""
        try:
            return web.json_response({"status": "ok", "update": _update_status_snapshot()})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module update status API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.post("/alexz_tools/module_install_requirements")
    async def alexz_tools_module_install_requirements(request):
        """API route that installs Python requirements for selected modules."""
        try:
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

    @PromptServer.instance.routes.post("/alexz_tools/comfyui_install_requirements")
    async def alexz_tools_comfyui_install_requirements(request):
        """API route that installs ComfyUI requirements in the active environment."""
        try:
            result = _install_comfyui_requirements()
            status_code = 200 if result.get("status") == "installed" else 400
            return web.json_response(result, status=status_code)
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("ComfyUI requirements install API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/node_catalog")
    async def alexz_tools_node_catalog(request):
        """API route that returns grouped module and node catalog data."""
        try:
            _ensure_runtime_state_ready()
            grouped = _build_group_catalog()
            modules_by_group = _build_group_modules(grouped)
            comfyui = _comfyui_git_status()
            custom_modules_need_update = _count_custom_modules_need_update()
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
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Node catalog API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_info")
    async def alexz_tools_module_info(request):
        """API route that returns detailed information for one module."""
        group = (request.query.get("group", "") or "").strip().lower()
        module_name = (request.query.get("module", "") or "").strip()
        refresh_raw = (request.query.get("refresh", "0") or "0").strip().lower()
        sync_raw = (request.query.get("sync_upstream", "0") or "0").strip().lower()
        force_refresh = refresh_raw not in {"0", "false", "no", "off"}
        sync_upstream = sync_raw not in {"0", "false", "no", "off"}
        if not module_name:
            return web.json_response({"error": "module is required"}, status=400)
        try:
            _ensure_runtime_state_ready()
            info = _resolve_module_info(
                group,
                module_name,
                force_refresh=force_refresh,
                sync_upstream=sync_upstream,
            )
            return web.json_response({"group": group, "module": module_name, "info": info})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("Module info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/comfyui_info")
    async def alexz_tools_comfyui_info(request):
        """API route that returns ComfyUI update and version status."""
        try:
            refresh_raw = (request.query.get("refresh", "1") or "1").strip().lower()
            force_refresh = refresh_raw not in {"0", "false", "no", "off"}
            comfyui = _comfyui_git_status(force_refresh=force_refresh)
            return web.json_response({"status": "ok", "comfyui": comfyui})
        except Exception as exc:  # pragma: no cover - diagnostic
            _LOGGER.error("ComfyUI info API error: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)

    @PromptServer.instance.routes.get("/alexz_tools/module_list")
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

    @PromptServer.instance.routes.get("/alexz_tools/module_nodes")
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
