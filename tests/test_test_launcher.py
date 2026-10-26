"""Check disposable ComfyUI server state without starting a real ComfyUI process."""

import importlib.util
import json
from pathlib import Path
import socket

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/start_comfyui_test.py"


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    """Load the launcher against a fake installation and a disposable project root."""
    spec = importlib.util.spec_from_file_location("alexz_test_launcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comfy = tmp_path / "ComfyUI с пробелами"
    project = comfy / "custom_nodes" / "ComfyUI_ALEXZ_tools"
    project.mkdir(parents=True)
    (comfy / "user/default/workflows").mkdir(parents=True)
    (comfy / "user/default/workflows/keep.json").write_text("user workflow")
    (comfy / "main.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nuser=Path(args[args.index('--user-directory')+1])\n"
        "(user/'default/workflows/test.json').write_text('{}')\n"
        "record=Path(__file__).parent/'runs.json'\n"
        "runs=json.loads(record.read_text()) if record.exists() else []\n"
        "runs.append({'state':str(user.parent),'args':args})\n"
        "record.write_text(json.dumps(runs))\n")
    monkeypatch.setattr(module, "PROJECT_ROOT", project)
    return module, comfy, project


def free_port():
    """Choose an unused loopback port without a hardcoded dependency on the real server."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_each_test_launch_is_fresh_and_cleans_owned_state(launcher):
    """Keep real user files and remove only directories created for each test run."""
    module, comfy, project = launcher
    for _ in range(2):
        assert module.main(["--port", str(free_port())]) == 0
    runs = json.loads((comfy / "runs.json").read_text())
    assert len(runs) == 2 and runs[0]["state"] != runs[1]["state"]
    for run in runs:
        state = Path(run["state"])
        assert not state.exists()
        assert state.parent == project / ".playwright-mcp/comfyui-test"
        for directory in ("user", "input", "output", "temp"):
            assert run["args"][run["args"].index(f"--{directory}-directory") + 1] == str(state / directory)
        assert "--cpu" in run["args"] and "--disable-all-custom-nodes" in run["args"]
    assert not list((project / ".playwright-mcp/comfyui-test").glob("session-*.json"))
    assert (comfy / "user/default/workflows/keep.json").read_text() == "user workflow"


def test_busy_test_port_is_not_reused_or_stopped(launcher):
    """Refuse an occupied port before creating a session or starting another process."""
    module, comfy, project = launcher
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        with pytest.raises(SystemExit) as result:
            module.main(["--port", str(existing.getsockname()[1])])
        assert result.value.code == 2
        assert existing.getsockname()[1] > 0
    assert not (comfy / "runs.json").exists()
    assert not (project / ".playwright-mcp").exists()


def test_test_launcher_dry_run_and_required_environment(launcher, monkeypatch):
    """Clean dry-run state and reject accidental execution from another environment."""
    module, comfy, project = launcher
    assert module.main(["--port", str(free_port()), "--dry-run"]) == 0
    assert not (comfy / "runs.json").exists()
    assert not list((project / ".playwright-mcp/comfyui-test").iterdir())
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    with pytest.raises(SystemExit) as result:
        module.main(["--port", str(free_port())])
    assert result.value.code == 2
