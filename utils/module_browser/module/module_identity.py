"""
Module: utils/module_browser/module_identity.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Custom-module discovery and canonical-name helper functions.

Purpose:
    Extracts module identity logic (discover/normalize/alias/canonicalize)
    from backend API facade to keep behavior testable and reusable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


def discover_custom_modules(*, custom_nodes_roots: Callable[[], list[Path]]) -> list[str]:
    """Discover installed custom module directories under custom_nodes roots."""
    names: set[str] = set()
    for root in custom_nodes_roots():
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


def normalize_module_token(name: str) -> str:
    """Normalize module token for case-insensitive matching and aliases."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def build_custom_module_aliases(
    *,
    discovered_modules: list[str],
    normalize_token: Callable[[str], str] = normalize_module_token,
) -> dict[str, str]:
    """Build alias map for custom module names and normalized tokens."""
    aliases: dict[str, str] = {}
    for module_name in discovered_modules:
        aliases[module_name] = module_name
        aliases[module_name.lower()] = module_name
        norm = normalize_token(module_name)
        if norm and norm not in aliases:
            aliases[norm] = module_name
    return aliases


def canonical_custom_module_name(
    module_name: str,
    *,
    aliases: dict[str, str],
    normalize_token: Callable[[str], str] = normalize_module_token,
) -> str:
    """Resolve user-provided module token to canonical custom module name."""
    name = (module_name or "").strip()
    if not name:
        return "unknown"

    direct = aliases.get(name) or aliases.get(name.lower())
    if direct:
        return direct

    norm = normalize_token(name)
    if norm:
        matched = aliases.get(norm)
        if matched:
            return matched
    return name
