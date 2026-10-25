"""Validate the opt-in ComfyUI development launcher without starting a server."""

import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/start_comfyui_dev.sh"


@pytest.fixture
def launch_environment(tmp_path):
    """Provide an isolated installation and a Conda recorder for command-contract checks."""
    comfy = tmp_path / "ComfyUI с пробелами"
    (comfy / "custom_nodes" / ROOT.name).mkdir(parents=True)
    (comfy / "custom_nodes" / "ComfyUI-VideoHelperSuite").mkdir()
    (comfy / "custom_nodes" / "comfyui-manager").mkdir()
    (comfy / "custom_nodes" / "ComfyUI-Crystools").mkdir()
    (comfy / "main.py").write_text("# Тестовая установка\n")
    recorder = tmp_path / "fake conda"
    recorder.write_text("#!/usr/bin/env python\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ['ALEXZ_TEST_RECORD']).write_text(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd()}))\n")
    recorder.chmod(0o755)
    record = tmp_path / "record.json"
    env = {**os.environ, "ALEXZ_COMFY_ROOT": str(comfy), "CONDA_EXE": str(recorder), "ALEXZ_TEST_RECORD": str(record)}
    return comfy, env, record


def launch(env, *args):
    """Run the shell entry point with captured diagnostics."""
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, text=True, capture_output=True)


def test_dev_command_limits_modules_and_preserves_arguments(launch_environment):
    """Forward paths and ports as distinct arguments and use the required environment."""
    comfy, env, record = launch_environment
    result = launch(env, "--", "--port", "8189", "--output-directory", "/tmp/video results")
    assert result.returncode == 0, result.stderr
    command = json.loads(record.read_text())
    assert command["cwd"] == str(comfy)
    assert command["args"] == ["run", "--no-capture-output", "-n", "p313-torch214-cu132", "python", "-u", str(comfy / "main.py"),
                               "--disable-all-custom-nodes", "--disable-partner-nodes", "--whitelist-custom-nodes", ROOT.name, "ComfyUI-VideoHelperSuite", "comfyui-manager", "ComfyUI-Crystools",
                               "--port", "8189", "--output-directory", "/tmp/video results"]


def test_dev_dry_run_does_not_execute_conda(launch_environment):
    """Make the launch plan inspectable without starting Python or a second ComfyUI."""
    _, env, record = launch_environment
    result = launch(env, "--dry-run")
    assert result.returncode == 0
    assert "--whitelist-custom-nodes" in result.stdout
    assert not record.exists()


def test_dev_extra_node_is_explicit_and_checked(launch_environment):
    """Allow an additional installed package while retaining the default whitelist."""
    comfy, env, record = launch_environment
    result = launch(env, "--extra-node", "missing")
    assert result.returncode == 2 and "missing" in result.stderr
    assert not record.exists()
    (comfy / "custom_nodes" / "extra custom pack").mkdir()
    result = launch(env, "--extra-node", "extra custom pack", "--", "--cpu")
    assert result.returncode == 0
    assert json.loads(record.read_text())["args"][-2:] == ["extra custom pack", "--cpu"]


def test_dev_missing_dependency_and_help(launch_environment):
    """Fail before launch when required modules are missing, and keep help independent."""
    comfy, env, record = launch_environment
    (comfy / "custom_nodes" / "ComfyUI-VideoHelperSuite").rmdir()
    result = launch(env)
    assert result.returncode == 2 and "ComfyUI-VideoHelperSuite" in result.stderr
    assert not record.exists()
    env["ALEXZ_COMFY_ROOT"] = str(comfy / "missing")
    assert launch(env, "--help").returncode == 0
    assert launch(env, "--extra-node").returncode == 2
