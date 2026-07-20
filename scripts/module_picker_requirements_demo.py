#!/usr/bin/env python3
"""
Seed or clear demo `requirements pending` markers for Module Node Picker.

Usage:
  conda run -n p313 python scripts/module_picker_requirements_demo.py seed
  conda run -n p313 python scripts/module_picker_requirements_demo.py clear
  conda run -n p313 python scripts/module_picker_requirements_demo.py status
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "module_state_cache.json"
COMFY_ROOT = REPO_ROOT.parents[1]
MODULE_NAME = "ComfyUI_ALEXZ_tools"
COMFY_KEY = "__comfyui__"
BACKUP_KEY = "__demo_requirements_backup__"
PENDING_KEYS = (
    "pending_requirements_update",
    "pending_requirements_before_commit",
    "pending_requirements_after_commit",
    "pending_requirements_updated_at",
)
DEMO_BEFORE = "1111111111111111111111111111111111111111"
DEMO_AFTER = "2222222222222222222222222222222222222222"
DEMO_AT = "2026-07-19T12:00:00+00:00"


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_custom_entry(state: dict) -> dict:
    entry = dict(state.get(MODULE_NAME) or {})
    entry.setdefault("module_path", str(REPO_ROOT))
    entry.setdefault("repository", "https://github.com/AlexZ1967/ComfyUI_ALEXZ_tools")
    entry.setdefault("update_status", "up_to_date")
    entry.setdefault("update_available", False)
    state[MODULE_NAME] = entry
    return entry


def ensure_comfy_entry(state: dict) -> dict:
    entry = dict(state.get(COMFY_KEY) or {})
    entry.setdefault("installed_commit", "")
    entry.setdefault("installed_updated_at", "")
    status = dict(entry.get("status") or {})
    status.setdefault("path", str(COMFY_ROOT))
    entry["status"] = status
    state[COMFY_KEY] = entry
    return entry


def snapshot_pending(entry: dict) -> dict:
    return {key: entry[key] for key in PENDING_KEYS if key in entry}


def restore_pending(entry: dict, backup: dict) -> dict:
    for key in PENDING_KEYS:
        entry.pop(key, None)
    for key, value in backup.items():
        entry[key] = value
    return entry


def seed_entry(entry: dict) -> dict:
    if BACKUP_KEY not in entry:
        entry[BACKUP_KEY] = snapshot_pending(entry)
    entry["pending_requirements_update"] = True
    entry["pending_requirements_before_commit"] = DEMO_BEFORE
    entry["pending_requirements_after_commit"] = DEMO_AFTER
    entry["pending_requirements_updated_at"] = DEMO_AT
    return entry


def clear_entry(entry: dict) -> dict:
    backup = entry.pop(BACKUP_KEY, None)
    if isinstance(backup, dict):
        return restore_pending(entry, backup)
    for key in PENDING_KEYS:
        entry.pop(key, None)
    return entry


def entry_status(entry: dict) -> dict:
    return {
        "pending": bool(entry.get("pending_requirements_update")),
        "before": str(entry.get("pending_requirements_before_commit") or ""),
        "after": str(entry.get("pending_requirements_after_commit") or ""),
        "updated_at": str(entry.get("pending_requirements_updated_at") or ""),
    }


def main(argv: list[str]) -> int:
    mode = str(argv[1] if len(argv) > 1 else "status").strip().lower()
    if mode not in {"seed", "clear", "status"}:
        print(f"Unsupported mode: {mode}", file=sys.stderr)
        return 2

    state = load_state()
    custom_entry = ensure_custom_entry(state)
    comfy_entry = ensure_comfy_entry(state)

    if mode == "seed":
        state[MODULE_NAME] = seed_entry(custom_entry)
        state[COMFY_KEY] = seed_entry(comfy_entry)
        save_state(state)
    elif mode == "clear":
        state[MODULE_NAME] = clear_entry(custom_entry)
        state[COMFY_KEY] = clear_entry(comfy_entry)
        save_state(state)

    print(
        json.dumps(
            {
                "mode": mode,
                "state_path": str(STATE_PATH),
                "custom_module": {MODULE_NAME: entry_status(state[MODULE_NAME])},
                "comfyui": entry_status(state[COMFY_KEY]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
