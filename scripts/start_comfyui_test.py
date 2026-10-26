"""Run an isolated CPU ComfyUI instance with disposable server state."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDA_ENV = "p313-torch214-cu132"


def build_command(comfy_root, state, port):
    """Keep every writable ComfyUI directory inside one disposable test session."""
    command = [sys.executable, "-u", str(comfy_root / "main.py"), "--cpu", "--listen", "127.0.0.1",
               "--port", str(port), "--disable-auto-launch", "--disable-all-custom-nodes",
               "--disable-partner-nodes", "--whitelist-custom-nodes", PROJECT_ROOT.name]
    for name in ("user", "input", "output", "temp"):
        command += [f"--{name}-directory", str(state / name)]
    return command


def stop_process(process):
    """Stop only the owned process group before deleting its working directories."""
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("Не удалось дождаться остановки тестового процесса.")


def main(argv=None):
    """Create fresh state per launch and clean it after normal exit or interruption."""
    parser = argparse.ArgumentParser(description="Изолированный тестовый ComfyUI; состояние удаляется после остановки.")
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--comfy-root", type=Path, default=PROJECT_ROOT.parents[1])
    parser.add_argument("--dry-run", action="store_true", help="Показать команду и проверить создание/очистку каталогов без запуска сервера.")
    args = parser.parse_args(argv)
    if os.environ.get("CONDA_DEFAULT_ENV") != CONDA_ENV:
        parser.error(f"Запускайте через conda run --no-capture-output -n {CONDA_ENV} python scripts/start_comfyui_test.py")
    comfy_root = args.comfy_root.resolve()
    if not (comfy_root / "main.py").is_file() or not 1 <= args.port <= 65535:
        parser.error("Проверьте ComfyUI main.py и номер порта.")
    # Занятый порт не означает разрешение останавливать чужой сервер.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(f"Порт {args.port} занят; существующий сервер не изменён.")
    artifacts = PROJECT_ROOT / ".playwright-mcp" / "comfyui-test"
    artifacts.mkdir(parents=True, exist_ok=True)
    session_file = artifacts / f"session-{args.port}.json"
    process = None
    previous_handlers = {}

    def interrupted(_signum, _frame):
        """Unwind the owned child and temporary state on terminal signals."""
        raise KeyboardInterrupt

    with tempfile.TemporaryDirectory(prefix="run-", dir=artifacts) as directory:
        state = Path(directory)
        for name in ("user", "input", "output", "temp"):
            (state / name).mkdir()
        for name in ("workflows", "subgraphs"):
            (state / "user" / "default" / name).mkdir(parents=True, exist_ok=True)
        command = build_command(comfy_root, state, args.port)
        print(f"Тестовый ComfyUI: http://127.0.0.1:{args.port}; CPU, только {PROJECT_ROOT.name}", flush=True)
        print(f"Временное состояние: {state}", flush=True)
        if args.dry_run:
            print(json.dumps({"command": command, "state": str(state)}, ensure_ascii=False), flush=True)
            return 0
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.signal(sig, interrupted)
            process = subprocess.Popen(command, cwd=comfy_root, start_new_session=True)
            session_file.write_text(json.dumps({"state": str(state), "launcher_pid": os.getpid(),
                                                "server_pid": process.pid, "port": args.port}), encoding="utf-8")
            return process.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            if process is not None:
                stop_process(process)
            if session_file.exists() and json.loads(session_file.read_text()).get("state") == str(state):
                session_file.unlink()
            print("Тестовый сервер остановлен; временные workflow и каталоги удаляются.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
