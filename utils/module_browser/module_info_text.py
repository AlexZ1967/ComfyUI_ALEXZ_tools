"""
Module: utils/module_browser/module_info_text.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Text helpers for custom-module metadata in Module Node Picker backend.

Purpose:
    Keeps README summary extraction and description sanitization logic isolated
    from API route handlers to simplify maintenance and testing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


def module_local_readme_summary(
    *,
    module_name: str,
    custom_nodes_roots: Callable[[], list[Path]],
) -> str | None:
    """Read short first meaningful line from module README if available."""
    name = (module_name or "").strip()
    if not name:
        return None
    readme_names = ("README.md", "readme.md", "README.MD")
    for root in custom_nodes_roots():
        module_dir = root / name
        if not module_dir.exists():
            continue
        for fname in readme_names:
            path = module_dir / fname
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


def sanitize_module_description(text: str, html_tag_re: re.Pattern[str]) -> str:
    """Normalize module description for stable card rendering in UI."""
    value = str(text or "")
    if not value:
        return ""
    out_lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<"):
            plain = html_tag_re.sub("", stripped).strip()
            if not plain:
                continue
            stripped = plain
        else:
            stripped = html_tag_re.sub("", stripped).strip()
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
