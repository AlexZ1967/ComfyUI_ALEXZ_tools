"""Small shared helpers for user-facing network diagnostics."""

from __future__ import annotations

from typing import Any


def sanitize_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    """Return headers safe for diagnostic JSON."""
    safe: dict[str, str] = {}
    for key, value in dict(headers or {}).items():
        name = str(key)
        lowered = name.lower()
        if lowered in {"x-api-key", "authorization", "cookie", "set-cookie"}:
            safe[name] = "***"
        else:
            safe[name] = str(value)
    return safe


def build_network_diagnostic(
    *,
    family: str,
    stage: str,
    url: str,
    status_code: int | None = None,
    reason: str = "",
    hint: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """Build a compact, serializable network diagnostic payload."""
    payload: dict[str, Any] = {
        "family": str(family or "").strip(),
        "stage": str(stage or "").strip(),
        "url": str(url or "").strip(),
        "status_code": int(status_code) if status_code is not None else None,
        "reason": str(reason or "").strip(),
        "hint": str(hint or "").strip(),
    }
    if detail:
        payload["detail"] = str(detail)
    return payload


def summarize_network_diagnostic(diagnostic: dict[str, Any] | None) -> str:
    """Convert a diagnostic payload into one console-friendly sentence."""
    data = dict(diagnostic or {})
    family = str(data.get("family") or "Network").strip()
    stage = str(data.get("stage") or "request").strip()
    status = data.get("status_code")
    reason = str(data.get("reason") or "").strip()
    hint = str(data.get("hint") or "").strip()

    parts = [f"{family} {stage} failed"]
    if status is not None:
        parts.append(f"status={status}")
    if reason:
        parts.append(reason)
    message = ": ".join((parts[0], ", ".join(parts[1:]))) if len(parts) > 1 else parts[0]
    if hint:
        message = f"{message}. {hint}"
    return message
