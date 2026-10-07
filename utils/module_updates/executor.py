"""Apply pinned update plans after revalidating repositories and the environment."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.utils import canonicalize_name

from .dependencies import Uncheckable, dependency_conflicts, environment_snapshot
from .planner import git, run


def validate_plan(plan: dict) -> None:
    """Reject a stale plan without changing any repository or package."""
    if time.time() > plan["expires_at"]:
        raise Uncheckable("План устарел; повторите проверку")
    if environment_snapshot()["fingerprint"] != plan["fingerprint"]:
        raise Uncheckable("Окружение изменилось; повторите проверку")
    for item in plan["modules"]:
        validate_repository(item)


def validate_repository(item: dict) -> None:
    """Check one repository immediately before applying its commit."""
    repo = Path(item["path"])
    if repo.is_symlink() or repo.resolve() != repo:
        raise Uncheckable("Путь репозитория заменён symlink")
    if git(repo, "rev-parse", "--show-toplevel") != str(repo.resolve()):
        raise Uncheckable("Изменился путь репозитория")
    if git(repo, "rev-parse", "HEAD") != item["before"] or git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise Uncheckable(f"{item['module']}: изменилось локальное состояние")
    git(repo, "merge-base", "--is-ancestor", item["before"], item["target"])


def ensure_server_stopped(plan: dict) -> None:
    """Refuse execution if ComfyUI has already restarted on the same local port."""
    for host in ("127.0.0.1", "::1"):
        try:
            connection = socket.create_connection((host, plan.get("port", 8188)), timeout=1)
        except ConnectionRefusedError:
            continue
        except OSError as exc:
            raise Uncheckable("Не удалось подтвердить остановку ComfyUI") from exc
        connection.close()
        raise Uncheckable("ComfyUI уже запущен. Дождитесь завершения worker до запуска сервера")


def download_wheels(report: dict, directory: Path) -> list[Path]:
    """Download only the exact SHA256-verified wheel artifacts in the checked plan."""
    files = []
    for item in report["install"]:
        info = item["download_info"]
        url = info["url"]
        parsed = urllib.parse.urlsplit(url)
        digest = info.get("archive_info", {}).get("hashes", {}).get("sha256")
        name = Path(urllib.parse.unquote(parsed.path)).name
        if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org" or not name.endswith(".whl") or not digest:
            raise Uncheckable("План содержит wheel без доверенного URL/SHA256")
        target = directory / name
        with urllib.request.urlopen(url, timeout=60) as response:
            if urllib.parse.urlsplit(response.url).hostname != "files.pythonhosted.org":
                raise Uncheckable("Wheel перенаправлен на другой источник")
            with target.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        with target.open("rb") as stream:
            actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_digest != digest:
            raise Uncheckable("SHA256 wheel не совпадает с планом")
        files.append(target)
    return files


def execute(plan: dict, directory: Path, progress) -> None:
    """Apply a pinned plan, backing up the environment before package additions."""
    validate_plan(plan)
    ensure_server_stopped(plan)
    before = environment_snapshot()
    wheels = download_wheels(plan["report"], directory)
    if wheels:
        progress("backup", "Создаётся резервная копия Conda-окружения")
        conda = Path(sys.prefix).parents[1] / "bin" / "conda"
        if not conda.is_file():
            raise Uncheckable("Не найден Conda для резервного копирования")
        run([str(conda), "create", "--yes", "--prefix", str(directory / "environment-backup"),
             "--clone", sys.prefix], timeout=1800)
    validate_plan(plan)
    ensure_server_stopped(plan)
    applied = []
    try:
        for item in plan["modules"]:
            ensure_server_stopped(plan)
            validate_repository(item)
            progress("updating", f"Обновляется {item['module']}")
            repo = Path(item["path"])
            git(repo, "update-ref", f"refs/alexz-backups/{plan['id']}", item["before"])
            git(repo, "merge", "--ff-only", item["target"])
            applied.append(item)
        if wheels:
            ensure_server_stopped(plan)
            progress("dependencies", "Добавляются проверенные wheel-зависимости")
            run([sys.executable, "-m", "pip", "--isolated", "install", "--no-index", "--no-deps",
                 *map(str, wheels)], timeout=1200)
            after = environment_snapshot()
            if any(after["packages"].get(name, {}).get("version") != pkg["version"] for name, pkg in before["packages"].items()):
                raise Uncheckable("Изменились защищённые пакеты после установки")
            for item in plan["report"]["install"]:
                name = item["metadata"]["name"]
                if after["packages"].get(canonicalize_name(name), {}).get("version") != item["metadata"]["version"]:
                    raise Uncheckable(f"Установка {name} не совпала с планом")
            roots = [req for item in plan["modules"] for req in item.get("requirements", [])]
            baseline = dependency_conflicts(before["packages"], before["markers"])
            if set(dependency_conflicts(after["packages"], after["markers"], roots)) - set(baseline):
                raise Uncheckable("После установки обнаружены новые конфликты")
        if any(git(Path(item["path"]), "rev-parse", "HEAD") != item["target"] for item in applied):
            raise Uncheckable("Commit после обновления не совпал с планом")
        progress("done", "Обновление выполнено. Запустите/перезапустите ComfyUI и проверьте загрузку модулей")
    except (Uncheckable, OSError, subprocess.TimeoutExpired) as exc:
        # pip не транзакционен. Код можно вернуть только при отсутствии новых правок.
        rolled_back = []
        for item in reversed(applied):
            repo = Path(item["path"])
            if git(repo, "rev-parse", "HEAD") == item["target"] and not git(repo, "status", "--porcelain", "--untracked-files=all"):
                git(repo, "reset", "--keep", item["before"])
                rolled_back.append(item["module"])
        raise Uncheckable(f"{exc}; откат кода: {', '.join(rolled_back)}. "
                          f"При частичной установке восстановите окружение из {directory / 'environment-backup'}") from exc


def write_status(path: Path, phase: str, message: str) -> None:
    """Atomically persist worker progress for the next ComfyUI startup."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"phase": phase, "message": message}, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def wait_and_execute(plan_path: Path, parent_pid: int) -> None:
    """Wait for the current ComfyUI process to exit before modifying dependencies."""
    directory = plan_path.parent
    status = directory / "status.json"
    progress = lambda phase, message: write_status(status, phase, message)
    try:
        import fcntl

        with (directory.parent / "worker.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Uncheckable("Другой worker уже выполняет обновление") from exc
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            progress("waiting_for_shutdown", "План поставлен в очередь. Остановите ComfyUI; worker выполнит обновление и завершится")
            while True:
                if (directory / "cancel").exists():
                    progress("cancelled", "Обновление отменено; код и зависимости не изменялись")
                    return
                try:
                    os.kill(parent_pid, 0)
                except ProcessLookupError:
                    break
                if time.time() > plan["expires_at"]:
                    raise Uncheckable("Время ожидания истекло; обновление отменено")
                time.sleep(1)
            if (directory / "cancel").exists():
                progress("cancelled", "Обновление отменено")
                return
            progress("preparing", "Проверяется план перед выполнением")
            execute(plan, directory, progress)
    except Exception as exc:  # Граница отдельного worker: ошибка должна попасть в журнал.
        progress("error", str(exc))
