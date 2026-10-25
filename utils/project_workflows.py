"""Persist editor workflows atomically inside this custom-node project's workflows folder."""

import asyncio
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import tempfile
import threading
from urllib.parse import urlsplit

ROUTE = "/alexz/project-workflows"
MAX_BYTES = 16 * 1024 * 1024


class WorkflowConflict(ValueError):
    """Signal a file revision mismatch without modifying the destination."""


def _local_request(request):
    """Allow only local same-origin clients, requiring a non-simple write header."""
    try:
        host = urlsplit("//" + request.host).hostname
        if host not in {"localhost", "127.0.0.1", "::1"} or not ipaddress.ip_address(request.remote).is_loopback:
            return False
        origin = request.headers.get("Origin")
        if origin and origin != f"{request.scheme}://{request.host}":
            return False
        return request.method == "GET" or (request.content_type == "application/json" and request.headers.get("X-Alexz-Workflow") == "1")
    except (TypeError, ValueError):
        return False


class ProjectWorkflows:
    """Own a flat JSON-only directory and optimistic revisions for explicit editor saves."""

    def __init__(self, project):
        """Bind the store to the installed project, not to any client-provided path."""
        self.directory = Path(project).resolve() / "workflows"
        self.lock = threading.RLock()

    def _directory(self):
        """Reject a redirected workflow directory rather than writing outside the project."""
        if self.directory.is_symlink():
            raise ValueError("Папка workflows не должна быть символической ссылкой.")
        return self.directory

    @staticmethod
    def filename(value):
        """Normalize a single portable JSON filename without permitting path traversal."""
        if not isinstance(value, str):
            raise ValueError("Укажите имя JSON-файла.")
        name = value.strip()
        if not name or name.startswith(".") or any(c in name for c in "/\\:") or any(ord(c) < 32 for c in name):
            raise ValueError("Укажите только имя файла, без пути и служебных символов.")
        name = name[:-5] + ".json" if name.lower().endswith(".json") else name + ".json"
        if len(name.encode("utf-8")) > 240:
            raise ValueError("Слишком длинное имя файла.")
        return name

    @staticmethod
    def _revision(path):
        """Hash an existing regular file without following symbolic links."""
        if path.is_symlink():
            raise ValueError("Символические ссылки не поддерживаются.")
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError("Имя занято не обычным файлом.")
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def list(self):
        """List existing project JSON files with their current content revisions."""
        with self.lock:
            root = self._directory()
            entries = []
            if root.exists():
                for path in sorted(root.glob("*.json")):
                    if path.is_file() and not path.is_symlink() and not path.name.startswith("."):
                        entries.append({"filename": path.name, "revision": self._revision(path)})
            return {"directory": "workflows", "files": entries}

    def save(self, payload):
        """Create or replace a workflow only when its expected revision still matches."""
        if not isinstance(payload, dict):
            raise ValueError("Ожидается JSON object.")
        name = self.filename(payload.get("filename"))
        workflow = payload.get("workflow")
        if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list) or not isinstance(workflow.get("links"), list):
            raise ValueError("Ожидается обычный workflow с массивами nodes и links, а не API prompt.")
        workflow = dict(workflow)
        extra = workflow.get("extra", {})
        if not isinstance(extra, dict):
            raise ValueError("Поле extra должно быть JSON object.")
        workflow["extra"] = {**extra, "alexz_project_workflow": name}
        content = (json.dumps(workflow, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(content) > MAX_BYTES:
            raise ValueError("Workflow превышает допустимые 16 MiB.")
        expected = payload.get("expected_revision")
        if expected is not None and not isinstance(expected, str):
            raise ValueError("Некорректная версия файла.")
        with self.lock:
            root = self._directory()
            root.mkdir(parents=True, exist_ok=True)
            target = root / name
            current = self._revision(target)
            if current != expected:
                raise WorkflowConflict("Файл изменился на диске или уже существует. Закройте окно, откройте его снова и повторите сохранение.")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, prefix=".alexz-workflow-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(target.stat().st_mode & 0o777 if current is not None else 0o644)
                if current is None:
                    # Не затираем файл, созданный другим процессом после проверки.
                    try:
                        os.link(temporary, target)
                    except FileExistsError as exc:
                        raise WorkflowConflict("Файл уже создан. Перечитайте список файлов.") from exc
                else:
                    os.replace(temporary, target)
                return {"filename": name, "directory": "workflows", "revision": hashlib.sha256(content).hexdigest()}
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def register_routes(PromptServer, web, project):
    """Register local project listing/saving endpoints without exposing arbitrary paths."""
    store = ProjectWorkflows(project)
    routes = PromptServer.instance.routes

    @routes.get(ROUTE)
    async def project_workflows_list(request):
        """Return filenames available in the save dialog."""
        if not _local_request(request):
            return web.json_response({"error": "Сохранение workflow доступно только локальному same-origin клиенту."}, status=403)
        try:
            return web.json_response(await asyncio.to_thread(store.list))
        except (ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post(ROUTE)
    async def project_workflows_save(request):
        """Validate an explicit editor save and perform atomic disk IO off the event loop."""
        if not _local_request(request):
            return web.json_response({"error": "Сохранение workflow доступно только локальному same-origin клиенту."}, status=403)
        try:
            if request.content_length is not None and request.content_length > MAX_BYTES:
                return web.json_response({"error": "Workflow превышает допустимые 16 MiB."}, status=413)
            raw = await request.read()
            if len(raw) > MAX_BYTES:
                return web.json_response({"error": "Workflow превышает допустимые 16 MiB."}, status=413)
            payload = json.loads(raw)
            return web.json_response(await asyncio.to_thread(store.save, payload))
        except WorkflowConflict as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except (ValueError, OSError, UnicodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    return store
