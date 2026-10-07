"""Expose dependency-aware planning and explicit queued updates over local routes."""

from __future__ import annotations

import copy
import asyncio
import ipaddress
import json
import os
import subprocess
from subprocess import Popen
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .dependencies import Uncheckable, environment_snapshot
from .executor import validate_plan, write_status
from .planner import analyze_batch

ROUTE_STATUS = "/alexz_tools/update_plan_status"
ROUTE_CHECK = "/alexz_tools/update_plan_check"
ROUTE_APPLY = "/alexz_tools/update_plan_apply"
ROUTE_CANCEL = "/alexz_tools/update_plan_cancel"


def local_request(request) -> bool:
    """Require a local, same-origin JSON client with a non-simple request header."""
    try:
        host = urlsplit("//" + request.host).hostname
        peer = ipaddress.ip_address(request.remote)
        if not peer.is_loopback or host not in {"127.0.0.1", "localhost", "::1"}:
            return False
        origin = request.headers.get("Origin")
        if origin and origin != f"{request.scheme}://{request.host}":
            return False
        if request.method == "POST":
            return request.content_type == "application/json" and request.headers.get("X-Alexz-Update") == "1"
        return True
    except (ValueError, TypeError):
        return False


class UpdateService:
    """Own one plan and one explicit update queue for a local ComfyUI instance."""

    def __init__(self, roots, project: Path, port=lambda: 8188):
        """Inject allowed custom-node roots and the project's local artifact folder."""
        self.roots = roots
        self.project = project
        self.port = port
        self.artifacts = project / ".alexz-updates"
        self.lock = threading.Lock()
        self.state = {"phase": "idle", "message": "Проверка ещё не запускалась", "plan": None}
        self.plan = None
        self.queued = False

    def status(self) -> dict:
        """Return a detached public snapshot, including persistent worker progress."""
        with self.lock:
            result = copy.deepcopy(self.state)
        histories = sorted(self.artifacts.glob("*/status.json"), key=lambda path: path.stat().st_mtime)
        if histories:
            result["execution"] = json.loads(histories[-1].read_text(encoding="utf-8"))
            result["execution"]["directory"] = str(histories[-1].parent)
            with self.lock:
                if result["execution"]["phase"] in {"done", "error", "cancelled"}:
                    self.queued = False
        return {"status": "ok", **result}

    def modules(self, selection) -> dict[str, Path]:
        """Resolve names strictly inside configured roots without path traversal."""
        available = {}
        for raw in self.roots():
            root = Path(raw).resolve()
            for path in root.iterdir():
                if path.is_dir() and not path.is_symlink() and not path.name.startswith(".") and not path.name.endswith(".disabled"):
                    available[path.name] = path.resolve()
        if selection is None:
            return available
        if not isinstance(selection, list) or not selection or any(not isinstance(name, str) or name not in available for name in selection):
            raise Uncheckable("Укажите существующие имена custom modules")
        return {name: available[name] for name in dict.fromkeys(selection)}

    def check(self, selection=None) -> dict:
        """Start an explicit background check, never executing install scripts."""
        if Path(sys.prefix).name != "p313-torch214-cu132":
            raise Uncheckable("ComfyUI должен работать в p313-torch214-cu132")
        modules = self.modules(selection)
        execution = self.status().get("execution")
        if execution and execution["phase"] not in {"done", "error", "cancelled"}:
            raise Uncheckable("Есть незавершённое обновление; проверьте status.json")
        with self.lock:
            if self.state["phase"] == "checking" or self.queued:
                raise Uncheckable("Уже выполняется проверка или поставлено обновление")
            self.plan = None
            self.state = {"phase": "checking", "message": "Читается окружение", "plan": None}

        def worker():
            """Publish a complete immutable plan or an explicit failure."""
            try:
                snapshot = environment_snapshot()

                def progress(name, current, total):
                    """Publish bounded background progress."""
                    with self.lock:
                        self.state["message"] = f"{current + 1}/{total}: {name}"

                plan = analyze_batch(modules, snapshot, progress=progress)
                if environment_snapshot()["fingerprint"] != snapshot["fingerprint"]:
                    raise Uncheckable("Окружение изменилось во время проверки")
                plan.update(id=uuid.uuid4().hex, fingerprint=snapshot["fingerprint"], expires_at=time.time() + 3600)
                with self.lock:
                    self.plan = plan
                    self.state = {"phase": "ready", "message": "Проверка завершена", "plan": self.public_plan(plan)}
            except Exception as exc:  # Граница фонового job: не теряем результат проверки.
                with self.lock:
                    self.state = {"phase": "error", "message": str(exc), "plan": None}

        threading.Thread(target=worker, name="alexz-update-plan", daemon=True).start()
        return {"status": "started"}

    @staticmethod
    def public_plan(plan: dict) -> dict:
        """Exclude executable artifact URLs and internal filesystem paths from UI data."""
        result = copy.deepcopy(plan)
        result["results"] = [{key: value for key, value in item.items() if key not in {"path", "report"}}
                             for item in result["results"]]
        result["batch"].pop("report", None)
        return result

    def apply(self, payload: dict) -> dict:
        """Queue only an explicitly acknowledged, revalidated plan for shutdown."""
        if os.name != "posix":
            raise Uncheckable("Worker обновления в этой версии поддерживает POSIX; анализ доступен без worker")
        with self.lock:
            plan = self.plan
            if not plan or payload.get("plan_id") != plan["id"] or self.queued or self.state["phase"] != "ready":
                raise Uncheckable("План отсутствует, устарел или уже поставлен в очередь")
            if payload.get("confirmed") is not True:
                raise Uncheckable("Нужно подтверждение конкретного плана")
            mode = payload.get("mode", "checked")
            module = payload.get("module")
            selected = [item for item in plan["results"] if item["module"] == module] if module else [
                item for item in plan["results"] if item["status"] == "safe"]
            if not selected or mode not in {"checked", "code_only"}:
                raise Uncheckable("Недопустимый режим или выбор модулей")
            if mode == "code_only" and (len(selected) != 1 or payload.get("acknowledge_risk") is not True):
                raise Uncheckable("Обновление только кода требует отдельного подтверждения риска")
            if any(item["status"] in {"blocked", "up_to_date"} or not item.get("target") for item in selected):
                raise Uncheckable("Git-состояние модуля не допускает обновление")
            if mode == "checked" and (any(item["status"] != "safe" for item in selected) or (not module and plan["batch_error"])):
                raise Uncheckable("Выбранный набор не прошёл проверку")
            report = selected[0]["report"] if module and mode == "checked" else plan["batch"].get("report")
            execution = {"id": plan["id"], "fingerprint": plan["fingerprint"], "expires_at": plan["expires_at"], "port": self.port(),
                         "modules": selected, "report": report if mode == "checked" else {"version": "1", "install": []}}
            validate_plan(execution)
            directory = self.artifacts / plan["id"]
            directory.mkdir(parents=True, mode=0o700, exist_ok=False)
            path = directory / "plan.json"
            path.write_text(json.dumps(execution, ensure_ascii=False, indent=2), encoding="utf-8")
            path.chmod(0o600)
            write_status(directory / "status.json", "waiting_for_shutdown", "Остановите ComfyUI и дождитесь завершения worker")
            with (directory / "worker.log").open("w", encoding="utf-8") as log:
                try:
                    Popen([sys.executable, str(self.project / "scripts/module_update_worker.py"),
                                      str(path), str(os.getpid())], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
                except OSError as exc:
                    write_status(directory / "status.json", "error", str(exc))
                    raise
            self.queued = True
        return {"status": "queued", "directory": str(directory),
                "message": "Остановите ComfyUI. Дождитесь phase=done в status.json перед запуском. Зависимости не заменяются"}

    def cancel(self, payload: dict) -> dict:
        """Cancel a queued job only while it is waiting for shutdown."""
        with self.lock:
            if not self.plan or payload.get("plan_id") != self.plan["id"]:
                raise Uncheckable("План отсутствует")
            directory = self.artifacts / self.plan["id"]
            status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
            if status["phase"] != "waiting_for_shutdown":
                raise Uncheckable("Отмена доступна только до остановки ComfyUI")
            (directory / "cancel").touch()
        return {"status": "cancelling"}


def register_routes(PromptServer, web, roots, project: Path) -> UpdateService:
    """Register isolated local-only endpoints without enabling legacy update routes."""
    service = UpdateService(roots, project, port=lambda: getattr(PromptServer.instance, "port", 8188))
    routes = PromptServer.instance.routes

    async def handle(request, action):
        """Reject unauthorized calls and return structured errors."""
        if not local_request(request):
            return web.json_response({"error": "Обновления доступны только локальному same-origin клиенту"}, status=403)
        try:
            payload = await request.json() if request.method == "POST" else {}
            if not isinstance(payload, dict):
                raise Uncheckable("Ожидается JSON object")
            return web.json_response(await asyncio.to_thread(action, payload))
        except (Uncheckable, ValueError, OSError, subprocess.SubprocessError) as exc:
            return web.json_response({"error": str(exc)}, status=409)

    @routes.get(ROUTE_STATUS)
    async def update_plan_status(request):
        """Read current analysis and execution progress without initiating work."""
        return await handle(request, lambda _: service.status())

    @routes.post(ROUTE_CHECK)
    async def update_plan_check(request):
        """Start an explicit dependency-aware upstream check."""
        return await handle(request, lambda payload: service.check(payload.get("modules")))

    @routes.post(ROUTE_APPLY)
    async def update_plan_apply(request):
        """Queue an explicitly confirmed update for execution after shutdown."""
        return await handle(request, service.apply)

    @routes.post(ROUTE_CANCEL)
    async def update_plan_cancel(request):
        """Cancel an update that has not started changing code or dependencies."""
        return await handle(request, service.cancel)

    return service
