"""
Module: utils/module_browser/component_registry_payload_ops.py
Author: AlexZ1967
Last updated: 2026-02-13

Description:
    Component registry payload orchestration helpers for Module Node Picker backend.

Purpose:
    Builds and caches Slice-0 component registry payload while preserving the
    existing API schema and persisted tracker behavior.
"""

from __future__ import annotations

from typing import Any, Callable


def collect_component_registry_payload(
    *,
    force_refresh: bool,
    now_ts: float,
    cache_payload: tuple[float, dict[str, Any]] | None,
    ttl_sec: float,
    build_default_component_registry: Callable[[], Any],
    load_module_state: Callable[[], dict[str, dict[str, Any]]],
    save_module_state: Callable[[dict[str, dict[str, Any]]], None],
    build_registry_snapshot: Callable[[Any], dict[str, list[str]]],
    compute_snapshot_signature: Callable[[dict[str, Any]], str],
    build_component_health_report: Callable[[], dict[str, Any]],
    schema_name: str,
    schema_version: int,
    now_iso: Callable[[], str],
) -> tuple[dict[str, Any], tuple[float, dict[str, Any]]]:
    """Return component registry payload and updated cache tuple."""
    if (
        not force_refresh
        and isinstance(cache_payload, tuple)
        and len(cache_payload) == 2
        and (now_ts - float(cache_payload[0])) < ttl_sec
    ):
        cached_payload = dict(cache_payload[1])
        return cached_payload, (float(cache_payload[0]), dict(cached_payload))

    registry = build_default_component_registry()
    state = load_module_state()
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
        "schema_name": schema_name,
        "schema_version": schema_version,
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
        "refreshed_at": now_iso(),
    }

    if (
        not isinstance(tracker_raw, dict)
        or tracker.get("snapshot") != current_snapshot
        or str(tracker.get("manifest_signature") or "") != current_signature
    ):
        state["__component_registry__"] = {
            "schema_name": schema_name,
            "schema_version": schema_version,
            "snapshot": current_snapshot,
            "manifest_signature": current_signature,
            "summary": dict(payload["summary"]),
            "updated_at": payload["refreshed_at"],
        }
        save_module_state(state)

    cache_value = (now_ts, dict(payload))
    return payload, cache_value

