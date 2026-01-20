import os
from pathlib import Path
from typing import Optional


try:  # pragma: no cover - optional dependency
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None


_GOOGLE_DRIVE_BASE = "https://drive.google.com/uc?export=download"


def _get_confirm_token(response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(32768):
            if chunk:
                handle.write(chunk)


def download_google_drive(file_id: str, destination: Path) -> str:
    if requests is None:
        raise RuntimeError("requests is required to download from Google Drive.")
    session = requests.Session()
    response = session.get(_GOOGLE_DRIVE_BASE, params={"id": file_id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        response = session.get(
            _GOOGLE_DRIVE_BASE, params={"id": file_id, "confirm": token}, stream=True
        )
    _save_response_content(response, destination)
    return str(destination)


def ensure_file(path: Path, file_id: Optional[str] = None) -> str:
    if path.exists():
        return str(path)
    if file_id is None:
        raise RuntimeError(f"Missing weight file: {path}")
    return download_google_drive(file_id, path)
