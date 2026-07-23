"""
Module: nodes/trove_search_ids.py
Author: AlexZ1967
Last updated: 2026-03-26

Description:
    Search Trove image results and extract NLA object ids.

Purpose:
    Provides a ComfyUI node that searches Trove via the official API-first flow
    and can optionally fall back to the legacy headless Chrome public UI flow.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any
from urllib.parse import quote_plus

import requests

from ..utils.network_diagnostics import build_network_diagnostic, summarize_network_diagnostic
from .trove_search_ids_ops import (
    TROVE_API_KEY_ENV,
    TROVE_API_RESULT_URL,
    TROVE_WEB_CATEGORY_URL,
    build_trove_api_params,
    extract_nla_obj_ids,
    extract_nla_obj_ids_from_api_payload,
    limit_ids,
    normalize_trove_api_category,
    normalize_trove_ui_category,
    resolve_trove_api_key,
    sanitize_trove_result,
)

_TROVE_API_TIMEOUT_SECONDS = 30.0


def _log(message: str) -> None:
    """Emit Trove search logs to ComfyUI console."""
    print(f"[TroveSearch] {message}")


def _find_chrome_binary() -> str:
    """Return first available Chrome/Chromium binary path."""
    for candidate in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    return ""


def _trove_category_search_url(query: str, category: str = "images") -> str:
    """Build Trove category search URL for browser rendering."""
    category_text = normalize_trove_ui_category(category)
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("`query` must not be empty.")
    return TROVE_WEB_CATEGORY_URL.format(category=category_text, query=quote_plus(query_text))


def _extract_nla_obj_ids(text: str) -> list[str]:
    """Compatibility wrapper for older tests/imports."""
    return extract_nla_obj_ids(text)


def _search_trove_ids_via_api(
    query: str,
    *,
    category: str = "images",
    api_key: str = "",
    max_results: int = 50,
    include_online_only: bool = True,
    timeout_seconds: float = _TROVE_API_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Search Trove through the official API and extract NLA object ids."""
    resolved_key, key_source = resolve_trove_api_key(api_key)
    params = build_trove_api_params(
        query,
        category=category,
        max_results=max_results,
        include_online_only=bool(include_online_only),
    )
    base: dict[str, Any] = {
        "mode": "api",
        "query": str(query or "").strip(),
        "category": normalize_trove_ui_category(category),
        "api_category": normalize_trove_api_category(category),
        "api_url": TROVE_API_RESULT_URL,
        "params": params,
        "api_key_source": key_source,
        "count": 0,
        "ids": [],
        "warning": "",
    }
    active_session = session or requests.Session()
    headers = {"X-API-KEY": resolved_key} if resolved_key else {}
    try:
        response = active_session.get(
            TROVE_API_RESULT_URL,
            params=params,
            headers=headers,
            timeout=float(timeout_seconds),
        )
    except requests.RequestException as exc:
        diagnostic = build_network_diagnostic(
            family="Trove",
            stage="API request",
            url=TROVE_API_RESULT_URL,
            reason=type(exc).__name__,
            hint="Check network connectivity, proxy, DNS, and Trove API availability.",
            detail=str(exc),
        )
        base["warning"] = summarize_network_diagnostic(diagnostic)
        base["diagnostic"] = diagnostic
        return base

    base["status_code"] = int(response.status_code)
    if int(response.status_code) != 200:
        status_code = int(response.status_code)
        if status_code == 429:
            hint = (
                "Anonymous Trove API request limit was reached; retry later or configure "
                f"{TROVE_API_KEY_ENV} for higher limits."
                if not resolved_key
                else "Trove API key request limit was reached; reduce request frequency and retry later."
            )
        elif status_code in {401, 403}:
            hint = (
                f"Anonymous access was rejected; configure {TROVE_API_KEY_ENV} or pass api_key in the node."
                if not resolved_key
                else "The Trove API key may be invalid, expired, or unauthorized."
            )
        else:
            hint = "Retry later and inspect the status and response excerpt in result_json."
        diagnostic = build_network_diagnostic(
            family="Trove",
            stage="API request",
            url=TROVE_API_RESULT_URL,
            status_code=status_code,
            reason=getattr(response, "reason", "") or "HTTP error",
            hint=hint,
            detail=str(response.text or "")[:2000],
        )
        base["warning"] = summarize_network_diagnostic(diagnostic)
        base["diagnostic"] = diagnostic
        return base

    try:
        payload = response.json()
    except ValueError as exc:
        diagnostic = build_network_diagnostic(
            family="Trove",
            stage="API JSON",
            url=TROVE_API_RESULT_URL,
            status_code=int(response.status_code),
            reason=type(exc).__name__,
            hint="Trove returned a non-JSON response; retry later or inspect result_json details.",
            detail=str(response.text or "")[:2000],
        )
        base["warning"] = summarize_network_diagnostic(diagnostic)
        base["diagnostic"] = diagnostic
        return base

    ids = limit_ids(extract_nla_obj_ids_from_api_payload(payload), int(max_results))
    base["count"] = len(ids)
    base["ids"] = ids
    total = None
    try:
        total = payload["category"][0]["records"].get("total")
    except (KeyError, IndexError, TypeError, AttributeError):
        total = None
    if total is not None:
        base["total"] = total
    if not ids:
        base["warning"] = "Trove API returned no `nla.obj-...` ids for this query."
    return base


def _search_trove_ids_via_chrome(
    query: str,
    *,
    category: str = "images",
    virtual_time_budget_ms: int = 20000,
    max_results: int = 100,
) -> dict[str, Any]:
    """Render Trove search page in headless Chrome and extract NLA object ids."""
    chrome_path = _find_chrome_binary()
    if not chrome_path:
        raise RuntimeError("Chrome/Chromium binary was not found in PATH.")

    search_url = _trove_category_search_url(query, category=category)
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--virtual-time-budget={max(1000, int(virtual_time_budget_ms))}",
        "--dump-dom",
        search_url,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    ids = limit_ids(extract_nla_obj_ids(stdout), int(max_results))

    warning = ""
    if not ids:
        if "Click the search button below to load results" in stdout:
            warning = (
                "Trove category page rendered, but results were not auto-expanded. "
                "Current public UI flow may require extra interaction."
            )
        elif "Making sure you're not a bot!" in stdout or "Anubis" in stdout:
            warning = "Trove anti-bot challenge intercepted the request."
        else:
            warning = "No `nla.obj-...` ids were found in rendered DOM."

    return {
        "mode": "browser",
        "query": str(query or "").strip(),
        "category": normalize_trove_ui_category(category),
        "search_url": search_url,
        "chrome_path": chrome_path,
        "returncode": int(proc.returncode),
        "count": len(ids),
        "ids": ids,
        "warning": warning,
        "stdout_excerpt": stdout[:4000],
        "stderr_excerpt": stderr[:2000],
    }


class SearchTroveImageIDs:
    """ComfyUI node that finds Trove image object ids for a text query."""

    @classmethod
    def INPUT_TYPES(cls):
        """Return ComfyUI INPUT_TYPES schema with defaults and UI options."""
        return {
            "required": {
                "query": (
                    "STRING",
                    {
                        "default": "Pavlova",
                        "multiline": False,
                        "tooltip": "Поисковый запрос для Trove Images, Maps & Artefacts.",
                    },
                ),
            },
            "optional": {
                "search_mode": (
                    ["api_first", "api_only", "browser_only"],
                    {
                        "default": "api_first",
                        "tooltip": "api_first = официальный Trove API v3, а при наличии ключа использует X-API-KEY. api_only не запускает Chrome. browser_only = legacy headless Chrome режим.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": f"Опциональный Trove API key. Пустое поле использует {TROVE_API_KEY_ENV}, а без переменной пробует запрос без авторизации; Trove может отклонить его с 401.",
                    },
                ),
                "category": (
                    ["images"],
                    {
                        "default": "images",
                        "tooltip": "Категория Trove. Пока поддерживается только images.",
                    },
                ),
                "max_results": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Максимум найденных `nla.obj-...` id на выходе.",
                    },
                ),
                "include_online_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Добавляет API facet l-availability=y/f, чтобы предпочитать онлайн-доступные записи.",
                    },
                ),
                "enable_browser_fallback": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Разрешить legacy headless Chrome fallback, если API-first не дал IDs. Требует установленный Chrome/Chromium и остается best-effort режимом.",
                    },
                ),
                "virtual_time_budget_ms": (
                    "INT",
                    {
                        "default": 20000,
                        "min": 1000,
                        "max": 120000,
                        "tooltip": "Время рендера headless Chrome перед `--dump-dom`.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("ids_text", "result_json", "count")
    FUNCTION = "search"
    CATEGORY = "image/io"

    def search(
        self,
        query: str,
        search_mode: str = "api_first",
        api_key: str = "",
        category: str = "images",
        max_results: int = 50,
        include_online_only: bool = True,
        enable_browser_fallback: bool = False,
        virtual_time_budget_ms: int = 20000,
    ):
        """Search Trove and return newline-separated ids plus diagnostic JSON."""
        mode = str(search_mode or "api_first").strip().lower()
        if mode == "browser_only":
            result = _search_trove_ids_via_chrome(
                query,
                category=category,
                virtual_time_budget_ms=int(virtual_time_budget_ms),
                max_results=int(max_results),
            )
        else:
            result = _search_trove_ids_via_api(
                query,
                category=category,
                api_key=api_key,
                max_results=int(max_results),
                include_online_only=bool(include_online_only),
            )
            ids = list(result.get("ids") or [])
            should_fallback = (
                mode == "api_first"
                and bool(enable_browser_fallback)
                and not ids
            )
            if should_fallback:
                api_result = dict(result)
                browser_result = _search_trove_ids_via_chrome(
                    query,
                    category=category,
                    virtual_time_budget_ms=int(virtual_time_budget_ms),
                    max_results=int(max_results),
                )
                result = {
                    **browser_result,
                    "mode": "api_first_with_browser_fallback",
                    "api_result": sanitize_trove_result(api_result),
                    "browser_result": browser_result,
                    "ids": list(browser_result.get("ids") or []),
                    "count": int(browser_result.get("count") or 0),
                    "warning": browser_result.get("warning") or api_result.get("warning") or "",
                }
        if result.get("warning"):
            _log(str(result["warning"]))
        result = sanitize_trove_result(result)
        ids = list(result.get("ids") or [])
        ids_text = "\n".join(ids)
        result_json = json.dumps(result, ensure_ascii=True, indent=2)
        return (ids_text, result_json, int(len(ids)))
