"""Expose dependency-aware planning and explicit queued updates over local routes."""

from __future__ import annotations

import copy
import asyncio
import ipaddress
import json
import logging
import os
import subprocess
from subprocess import Popen
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name

from .dependencies import Uncheckable, environment_snapshot, project_report
from .executor import validate_plan, write_status
from .planner import analyze_batch

ROUTE_STATUS = "/alexz_tools/update_plan_status"
ROUTE_CHECK = "/alexz_tools/update_plan_check"
ROUTE_APPLY = "/alexz_tools/update_plan_apply"
ROUTE_CANCEL = "/alexz_tools/update_plan_cancel"


def forward_worker_output(worker, path: Path) -> None:
    """Mirror worker output to its diagnostic log and the ComfyUI console."""
    with path.open("w", encoding="utf-8") as log:
        for line in worker.stdout:
            log.write(line)
            log.flush()
            emit = logging.error if "[error]" in line else logging.info
            emit(line.rstrip())
    worker.stdout.close()
    worker.wait()


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

    def __init__(self, roots, project: Path, port=lambda: 8188, *, capture_module=None, finalize_tracking=None, busy=lambda: False):
        """Inject allowed custom-node roots and the project's local artifact folder."""
        self.roots = roots
        self.project = project
        self.port = port
        self.busy = busy
        self.capture_module = capture_module
        self.finalize_tracking = finalize_tracking
        self.artifacts = project / ".alexz-updates"
        self.lock = threading.Lock()
        self.state = {"phase": "idle", "message": "Проверка ещё не запускалась", "plan": None}
        self.plan = None
        self.queued = False
        self.server_session = uuid.uuid4().hex
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.session_path = self.artifacts / "server-session"
        self.session_path.write_text(self.server_session, encoding="utf-8")
        self.saved_plan = self.artifacts / "checked-plan.json"
        if self.saved_plan.exists():
            try:
                saved = json.loads(self.saved_plan.read_text(encoding="utf-8"))
                if saved["expires_at"] > time.time():
                    self.plan = saved
                    self.state = {"phase": "ready", "message": "Сохранённый отчёт проверки", "plan": self.public_plan(saved)}
            except (OSError, ValueError, KeyError, TypeError):
                pass

    def status(self) -> dict:
        """Return a detached public snapshot, including persistent worker progress."""
        with self.lock:
            result = copy.deepcopy(self.state)
        execution = self.status_without_lock()
        if execution:
            result["execution"] = execution
            with self.lock:
                if execution["phase"] in {"done", "error", "cancelled"}:
                    self.queued = False
                    if execution["phase"] in {"done", "error"} and self.plan and self.plan["id"] == Path(execution["directory"]).name:
                        if execution["phase"] == "done":
                            try:
                                self.advance_plan(Path(execution["directory"]))
                            except (OSError, ValueError, KeyError, TypeError, Uncheckable):
                                logging.exception("[ALEXZ update] Не удалось сохранить оставшихся кандидатов")
                                self.plan = None
                        else:
                            self.plan = None
                        if self.plan is None:
                            self.state = {"phase": "idle", "message": "Для следующего апдейта повторите проверку", "plan": None}
                            self.saved_plan.unlink(missing_ok=True)
                        result.update(copy.deepcopy(self.state))
                    if self.state["phase"] == "checking" or (self.plan and self.plan["id"] != Path(execution["directory"]).name):
                        result.pop("execution", None)
            if execution["phase"] == "done":
                try:
                    plan_path = Path(execution["directory"]) / "plan.json"
                    session = json.loads(plan_path.read_text(encoding="utf-8")).get("server_session")
                except (OSError, ValueError, TypeError):
                    session = None
                # После нового запуска код уже загружен: прежнее предложение restart не нужно.
                if session and session != self.server_session:
                    result.pop("execution", None)
        return {"status": "ok", **result, "updated_modules": self.successful_update_modules(),
                "restart_required_modules": self.successful_update_modules(current_session=True)}

    def successful_update_modules(self, *, current_session=False) -> list[str]:
        """Recover successful module updates from local worker history without Git."""
        modules = set()
        for path in self.artifacts.glob("*/status.json"):
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                if status["phase"] != "done":
                    continue
                plan = json.loads((path.parent / "plan.json").read_text(encoding="utf-8"))
                if current_session and plan.get("server_session") != self.server_session:
                    continue
                for item in plan["modules"]:
                    if item.get("before") and item.get("target") and item["before"] != item["target"]:
                        modules.add(item["module"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(modules)

    def advance_plan(self, directory: Path) -> None:
        """Retain pending candidates and rebase pinned reports after a successful job."""
        execution = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        snapshot = environment_snapshot()
        if snapshot["fingerprint"] != self.plan["fingerprint"]:
            previous = json.loads((self.artifacts / f"{self.plan['id']}-environment.json").read_text(encoding="utf-8"))
            expected = {name: item["version"] for name, item in previous["packages"].items()}
            for item in execution["report"]["install"]:
                expected[canonicalize_name(item["metadata"]["name"])] = item["metadata"]["version"]
            actual = {name: item["version"] for name, item in snapshot["packages"].items()}
            if actual != expected:
                raise Uncheckable("Окружение изменилось вне выполненного плана; повторите проверку")
        plan = copy.deepcopy(self.plan)
        completed = {item["module"] for item in execution["modules"]}

        def rebase(report, roots):
            """Reuse checked wheels, excluding packages installed by the completed job."""
            report = copy.deepcopy(report)
            report["install"] = [item for item in report["install"]
                                 if snapshot["packages"].get(canonicalize_name(item["metadata"]["name"]), {}).get("version")
                                 != item["metadata"]["version"]]
            projected = project_report(snapshot, report, roots)
            if projected["conflicts"]:
                raise Uncheckable("; ".join(projected["conflicts"]))
            return {**projected, "report": report}

        for item in plan["results"]:
            if item["module"] in completed:
                item.update(status="up_to_date", before=item["target"], additions=[], reasons=[],
                            report={"version": "1", "install": []})
            elif item["status"] == "safe":
                roots = item.get("requirements", []) + item.get("constraints", []) if item.get("requirements_changed") else []
                item.update(rebase(item["report"], roots))
        remaining = [item for item in plan["results"] if item["status"] == "safe"]
        if not plan["batch_error"]:
            roots = [req for item in remaining if item.get("requirements_changed")
                     for req in item.get("requirements", []) + item.get("constraints", [])]
            plan["batch"] = rebase(plan["batch"]["report"], roots)
        plan["batch_count"] = 0 if plan["batch_error"] else len(remaining)
        plan["counts"] = {status: sum(item["status"] == status for item in plan["results"])
                          for status in ("safe", "risk", "unknown", "blocked", "up_to_date")}
        plan.update(id=uuid.uuid4().hex, fingerprint=snapshot["fingerprint"])
        plan["environment"] = {key: snapshot[key] for key in ("python", "prefix", "fingerprint")}
        snapshot_path = self.artifacts / f"{plan['id']}-environment.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
        snapshot_path.chmod(0o600)
        temporary = self.saved_plan.with_suffix(".tmp")
        temporary.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(self.saved_plan)
        self.plan = plan
        self.state = {"phase": "ready", "message": "Оставшиеся кандидаты сохранены", "plan": self.public_plan(plan)}

    def modules(self, selection) -> dict[str, Path]:
        """Resolve names strictly inside configured roots without path traversal."""
        available = {}
        for raw in self.roots():
            root = Path(raw).resolve()
            for path in root.iterdir():
                if (path.is_dir() and not path.is_symlink() and path.name != "__pycache__"
                        and not path.name.startswith(".") and not path.name.endswith(".disabled")):
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
                captured = {}

                def capture(item):
                    """Save tracking facts collected by the module's analysis task."""
                    if self.capture_module:
                        captured[item["module"]] = self.capture_module(item)

                def progress(name, current, total):
                    """Publish bounded background progress."""
                    with self.lock:
                        self.state["message"] = f"Проверено {current}/{total}" + (f": {name}" if name else "")

                plan = analyze_batch(modules, snapshot, progress=progress, module_callback=capture)
                if environment_snapshot()["fingerprint"] != snapshot["fingerprint"]:
                    raise Uncheckable("Окружение изменилось во время проверки")
                plan.update(id=uuid.uuid4().hex, fingerprint=snapshot["fingerprint"], expires_at=time.time() + 3600)
                logging.info("[ALEXZ update] Проверка завершена: готово к обновлению %s модулей", plan["batch_count"])
                for item in plan["results"]:
                    if item["status"] in {"risk", "unknown", "blocked"}:
                        logging.warning("[ALEXZ update] %s: %s", item["module"], "; ".join(item["reasons"]))
                if plan["batch_error"]:
                    logging.warning("[ALEXZ update] Общий набор: %s", plan["batch_error"])
                # Сохраняем исходный снимок для диагностики расхождений с отдельным worker.
                snapshot_path = self.artifacts / f"{plan['id']}-environment.json"
                snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
                snapshot_path.chmod(0o600)
                if self.finalize_tracking:
                    with self.lock:
                        self.state["message"] = "Обновляется каталог из собранных данных"
                    self.finalize_tracking(captured)
                temporary = self.saved_plan.with_suffix(".tmp")
                temporary.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
                temporary.chmod(0o600)
                temporary.replace(self.saved_plan)
                with self.lock:
                    self.plan = plan
                    self.state = {"phase": "ready", "message": "Проверка завершена", "plan": self.public_plan(plan)}
            except Exception as exc:  # Граница фонового job: не теряем результат проверки.
                logging.exception("[ALEXZ update] Ошибка проверки модулей")
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
        """Start an explicitly acknowledged, revalidated plan in a detached worker."""
        if os.name != "posix":
            raise Uncheckable("Worker обновления в этой версии поддерживает POSIX; анализ доступен без worker")
        with self.lock:
            plan = self.plan
            if not plan or payload.get("plan_id") != plan["id"] or self.queued or self.state["phase"] != "ready":
                raise Uncheckable("План отсутствует, устарел или уже поставлен в очередь")
            if payload.get("confirmed") is not True:
                raise Uncheckable("Нужно подтверждение конкретного плана")
            execution_status = self.status_without_lock()
            if execution_status and execution_status["phase"] not in {"done", "error", "cancelled"}:
                raise Uncheckable("Есть незавершённое обновление. Сначала отмените ожидающую очередь")
            if self.busy():
                raise Uncheckable("Остановите генерацию и очистите очередь перед обновлением")
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
                         "modules": selected, "report": report if mode == "checked" else {"version": "1", "install": []},
                         "execution_mode": "online", "session_path": str(self.session_path), "server_session": self.server_session}
            validate_plan(execution)
            directory = self.artifacts / plan["id"]
            directory.mkdir(parents=True, mode=0o700, exist_ok=False)
            path = directory / "plan.json"
            path.write_text(json.dumps(execution, ensure_ascii=False, indent=2), encoding="utf-8")
            path.chmod(0o600)
            write_status(directory / "status.json", "queued", "Обновление запускается. Не запускайте генерацию и не перезапускайте ComfyUI")
            try:
                worker_process = Popen([sys.executable, "-u", str(self.project / "scripts/module_update_worker.py"),
                                  str(path), str(os.getpid()), "--immediate"], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                if isinstance(worker_process.pid, int):
                    (directory / "worker.json").write_text(json.dumps({"pid": worker_process.pid}), encoding="utf-8")
                    threading.Thread(target=forward_worker_output, args=(worker_process, directory / "worker.log"),
                                     name="alexz-update-output", daemon=True).start()
            except OSError as exc:
                write_status(directory / "status.json", "error", str(exc))
                raise
            self.queued = True
        return {"status": "queued", "directory": str(directory),
                "message": "Выполняется обновление. Дождитесь завершения, затем перезапустите ComfyUI"}

    def status_without_lock(self) -> dict | None:
        """Read the latest execution without recursively acquiring the state lock."""
        histories = sorted(self.artifacts.glob("*/status.json"), key=lambda path: path.stat().st_mtime)
        if not histories:
            return None
        path = histories[-1]
        status = json.loads(path.read_text(encoding="utf-8"))
        metadata = path.parent / "worker.json"
        if status["phase"] not in {"done", "error", "cancelled"} and metadata.exists():
            pid = json.loads(metadata.read_text(encoding="utf-8"))["pid"]
            try:
                os.kill(pid, 0)
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    raise ProcessLookupError
            except ProcessLookupError:
                # Сверяем статус ещё раз: worker мог записать результат между чтениями.
                status = json.loads(path.read_text(encoding="utf-8"))
                if status["phase"] not in {"done", "error", "cancelled"}:
                    write_status(path, "error", "Worker завершился без результата. Проверьте worker.log и повторите проверку")
                    status = json.loads(path.read_text(encoding="utf-8"))
        return {**status, "directory": str(path.parent)}

    def cancel(self, payload: dict) -> dict:
        """Cancel a legacy waiting job even when its in-memory plan was lost."""
        with self.lock:
            status = self.status_without_lock()
            if not status or payload.get("plan_id") != Path(status["directory"]).name:
                raise Uncheckable("Ожидающее обновление отсутствует")
            if status["phase"] != "waiting_for_shutdown":
                raise Uncheckable("Начавшееся обновление нельзя отменить; дождитесь результата")
            (Path(status["directory"]) / "cancel").touch()
        return {"status": "cancelling"}


def register_routes(PromptServer, web, roots, project: Path, *, capture_module=None, finalize_tracking=None) -> UpdateService:
    """Register isolated local-only endpoints without enabling legacy update routes."""
    service = UpdateService(roots, project, port=lambda: getattr(PromptServer.instance, "port", 8188),
                            capture_module=capture_module, finalize_tracking=finalize_tracking,
                            busy=lambda: PromptServer.instance.prompt_queue.get_tasks_remaining() > 0)
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
            logging.error("[ALEXZ update] %s", exc)
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
