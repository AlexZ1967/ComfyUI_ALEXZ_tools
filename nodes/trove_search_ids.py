"""
Module: nodes/trove_search_ids.py
Author: AlexZ1967
Last updated: 2026-03-26

Description:
    Search Trove image results and extract NLA object ids.

Purpose:
    Provides a ComfyUI node that performs best-effort Trove image search without
    an API key by rendering the public search page in headless Chrome and
    extracting `nla.obj-...` ids from the rendered DOM.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import quote_plus


_TROVE_SEARCH_CATEGORY_URL = "https://trove.nla.gov.au/search/category/{category}?keyword={query}"


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
    category_text = str(category or "images").strip().lower() or "images"
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("`query` must not be empty.")
    return _TROVE_SEARCH_CATEGORY_URL.format(category=category_text, query=quote_plus(query_text))


def _extract_nla_obj_ids(text: str) -> list[str]:
    """Extract unique `nla.obj-...` identifiers while preserving order."""
    ids: list[str] = []
    seen: set[str] = set()
    for value in re.findall(r"nla\.obj-\d+", str(text or ""), flags=re.IGNORECASE):
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ids.append(normalized)
    return ids


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
    ids = _extract_nla_obj_ids(stdout)
    if int(max_results) > 0:
        ids = ids[: int(max_results)]

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
        "query": str(query or "").strip(),
        "category": str(category or "images").strip().lower(),
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
        category: str = "images",
        max_results: int = 50,
        virtual_time_budget_ms: int = 20000,
    ):
        """Search Trove and return newline-separated ids plus diagnostic JSON."""
        result = _search_trove_ids_via_chrome(
            query,
            category=category,
            virtual_time_budget_ms=int(virtual_time_budget_ms),
            max_results=int(max_results),
        )
        if result.get("warning"):
            _log(str(result["warning"]))
        ids = list(result.get("ids") or [])
        ids_text = "\n".join(ids)
        result_json = json.dumps(result, ensure_ascii=True, indent=2)
        return (ids_text, result_json, int(len(ids)))
