"""Exercise update plans and execution against disposable Git repositories."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from module_updates import dependencies, executor, planner, service


def snapshot(packages=None):
    """Create a deterministic metadata-only environment fixture."""
    return {"packages": packages or {}, "markers": dependencies.default_environment(),
            "python": sys.executable, "prefix": sys.prefix, "fingerprint": "fixture"}


def package(version, requires=()):
    """Create installed distribution metadata."""
    return {"version": version, "requires_dist": list(requires)}


def test_environment_snapshot_ignores_vendored_metadata(tmp_path, monkeypatch):
    """Ignore sys.path-only copies while detecting actual installed changes."""
    installed = tmp_path / "site-packages"
    vendor = tmp_path / "vendor"
    for root, version in ((installed, "2.0"), (vendor, "1.0")):
        metadata = root / f"demo-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Name: demo\nVersion: {version}\n")
    monkeypatch.setattr(dependencies.sysconfig, "get_paths", lambda: {
        "purelib": str(installed), "platlib": str(installed)})
    before = dependencies.environment_snapshot()
    monkeypatch.syspath_prepend(str(vendor))
    assert dependencies.environment_snapshot() == before
    assert before["packages"]["demo"]["version"] == "2.0"
    (installed / "demo-2.0.dist-info" / "METADATA").write_text("Name: demo\nVersion: 3.0\n")
    assert dependencies.environment_snapshot()["fingerprint"] != before["fingerprint"]


@pytest.fixture
def pip_dry_run(monkeypatch):
    """Replace only pip resolution while retaining real disposable Git operations."""
    state = SimpleNamespace(calls=[], error=None)
    original = planner.run

    def run(command, **kwargs):
        """Publish an empty wheel report or a controlled resolver failure."""
        if "pip" not in command:
            return original(command, **kwargs)
        state.calls.append(command)
        assert "--dry-run" in command
        if state.error:
            raise dependencies.Uncheckable(state.error)
        Path(command[command.index("--report") + 1]).write_text(json.dumps({"version": "1", "install": []}))
        return ""

    monkeypatch.setattr(planner, "run", run)
    return state


@pytest.fixture
def repository(tmp_path):
    """Prepare a local upstream and installed clone without touching custom nodes."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    planner.run(["git", "init", "-b", "main", str(upstream)])
    planner.git(upstream, "config", "user.name", "Fixture")
    planner.git(upstream, "config", "user.email", "fixture@example.invalid")
    (upstream / "requirements.txt").write_text("demo>=1\n")
    planner.git(upstream, "add", ".")
    planner.git(upstream, "commit", "-m", "initial")
    installed = tmp_path / "custom_nodes" / "demo"
    planner.run(["git", "clone", str(upstream), str(installed)])

    def commit(files):
        """Create an upstream revision for a test case."""
        for name, content in files.items():
            path = upstream / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        planner.git(upstream, "add", ".")
        planner.git(upstream, "commit", "-m", "update")
        return planner.git(upstream, "rev-parse", "HEAD")

    return upstream, installed, commit


def test_unchanged_requirements_checked_before_update(repository):
    """Analyze without changing HEAD, tracked files or FETCH_HEAD."""
    _, installed, commit = repository
    before = planner.git(installed, "rev-parse", "HEAD")
    target = commit({"node.py": "# new code\n"})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "safe"
    assert result["requirements_changed"] is False
    assert result["target"] == target
    assert planner.git(installed, "rev-parse", "@{u}") == target
    assert planner.git(installed, "rev-parse", "HEAD") == before
    assert not (installed / "node.py").exists()
    assert not (installed / ".git/FETCH_HEAD").exists()


def test_changed_requirements_can_already_be_satisfied(repository, pip_dry_run):
    """Run pip dry-run even when changed declarations appear already satisfied."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=1.0\n"})
    with patch.object(planner, "run", wraps=planner.run) as command:
        result = planner.analyze_module("demo", installed, snapshot({"demo": package("1.2")}))
    assert result["status"] == "safe"
    assert result["requirements_changed"]
    assert len(pip_dry_run.calls) == 1


def test_existing_package_replacement_is_risk(repository, pip_dry_run):
    """Do not approve a request to replace a protected distribution."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=2\n"})
    pip_dry_run.error = "ResolutionImpossible: demo>=2 conflicts with protected demo==1"
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "risk"
    assert "ResolutionImpossible" in result["reasons"][0]
    assert len(pip_dry_run.calls) == 1


def test_command_failure_retains_stdout_conflict_explanation():
    """Keep resolver explanations emitted to stdout alongside stderr errors."""
    with pytest.raises(dependencies.Uncheckable) as error:
        planner.run([sys.executable, "-c", "import sys; print('constraint demo==1'); "
                     "print('ResolutionImpossible', file=sys.stderr); sys.exit(1)"])
    assert "constraint demo==1" in str(error.value)
    assert "ResolutionImpossible" in str(error.value)


@pytest.mark.parametrize("name", ["install.py", "setup.py", "setup.cfg", "pyproject.toml"])
def test_packaging_changes_with_unchanged_requirements_allow_code_update(repository, name):
    """Allow code updates without running changed install scripts or pip."""
    _, installed, commit = repository
    commit({name: "changed\n"})
    with patch.object(planner, "resolve_dependencies") as resolve:
        result = planner.analyze_module("demo", installed, snapshot())
    assert result["status"] == "safe"
    assert result["report"]["install"] == []
    resolve.assert_not_called()


def test_changed_requirements_with_install_script_still_require_review(repository, pip_dry_run):
    """Keep executable installation changes outside the checked dependency path."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=1.0\n", "install.py": "raise RuntimeError()\n"})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "risk"
    assert len(pip_dry_run.calls) == 1
    assert "Dry-run зависимостей пройден" in result["reasons"][0]


def test_pyproject_metadata_change_does_not_block_requirements_dry_run(repository, pip_dry_run):
    """Check changed requirements despite added version and description metadata."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=1.0\n", "pyproject.toml": '[project]\nversion = "0.2"\ndescription = "New"\n'})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "safe"
    assert len(pip_dry_run.calls) == 1


def test_build_backend_change_is_reviewed_after_dry_run(repository, pip_dry_run):
    """Distinguish build behavior from harmless project metadata."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=1.0\n", "pyproject.toml": '[build-system]\nbuild-backend = "custom.build"\n'})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "risk"
    assert len(pip_dry_run.calls) == 1
    assert "build-system" in result["reasons"][0]


def test_unchanged_requirements_batch_never_installs_missing_packages(repository):
    """Do not turn a code-only batch into an environment repair operation."""
    _, installed, commit = repository
    commit({"node.py": "new\n"})
    with patch.object(planner, "resolve_dependencies") as resolve:
        result = planner.analyze_batch({"demo": installed}, snapshot())
    assert result["batch_count"] == 1
    assert result["batch"]["report"]["install"] == []
    resolve.assert_not_called()


def test_unrelated_untracked_file_does_not_block_update(repository):
    """Preserve local files while permitting an unrelated fast-forward."""
    _, installed, commit = repository
    commit({"node.py": "new"})
    (installed / "local.txt").write_text("keep")
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "safe"
    assert (installed / "local.txt").read_text() == "keep"
    assert not planner.git(installed, "stash", "list")


def test_git_failure_is_unknown_not_unchanged(tmp_path):
    """Fail closed on invalid repositories."""
    result = planner.analyze_module("demo", tmp_path, snapshot())
    assert result["status"] == "unknown"
    assert result["requirements_changed"] is None


def test_nested_includes_and_constraints():
    """Expand relative includes with platform markers intact."""
    files = {"requirements.txt": "-r nested/deps.txt\n-c constraints.txt\n",
             "nested/deps.txt": "-r ../shared.txt\n", "shared.txt": "demo>=1; python_version >= '3.10'\n",
             "constraints.txt": "demo<3\n"}
    result = dependencies.read_requirements(files.get)
    assert len(result["files"]) == 4
    assert result["constraints"] == ["demo<3"]
    assert "python_version" in result["requirements"][0]


@pytest.mark.parametrize("line", ["-r ../outside.txt", "-r requirements.txt", "-e .",
                                 "demo @ git+https://example.invalid/repo", "--extra-index-url https://example.invalid",
                                 "-r missing.txt"])
def test_uncheckable_requirement_inputs(line):
    """Reject executable, external, cyclic and missing requirements inputs."""
    with pytest.raises(dependencies.Uncheckable):
        dependencies.read_requirements({"requirements.txt": line}.get)


def test_environment_graph_and_requested_extras():
    """Detect new missing extras while retaining pre-existing conflicts separately."""
    packages = {"old": package("1", ["absent>=1"]), "demo": package("1", ["addon>=2; extra == 'fast'"])}
    current = snapshot(packages)
    result = dependencies.project_report(current, {"version": "1", "install": []}, ["demo[fast]>=1"])
    assert result["baseline"] == ["old: отсутствует absent>=1"]
    assert result["conflicts"] == ["demo: отсутствует addon>=2"]


def test_report_replacement_never_approved():
    """Defend against a pip report that changes an existing distribution."""
    with pytest.raises(dependencies.Uncheckable, match="защищённый"):
        dependencies.project_report(snapshot({"torch": package("2.14.0")}),
            {"version": "1", "install": [{"metadata": {"name": "torch", "version": "2.15.0"}}]}, [])


def test_recursive_dry_run_rejects_transitive_urls(tmp_path):
    """Never give pip executable transitive requirements from wheel metadata."""
    seen = []

    def dry_run(command, timeout=30):
        """Emulate an unsafe transitive requirement in a wheel report."""
        seen.append(command)
        report = Path(command[command.index("--report") + 1])
        report.write_text(json.dumps({"version": "1", "install": [{"metadata": {
            "name": "new", "version": "1", "requires_dist": ["child @ git+https://example.invalid/repo"]}}]}))
        return ""

    with patch.object(planner, "run", side_effect=dry_run):
        with pytest.raises(dependencies.Uncheckable, match="Transitive URL"):
            planner.resolve_dependencies(snapshot(), ["new>=1"], [])
    assert len(seen) == 1
    assert "--dry-run" in seen[0] and "--no-deps" in seen[0] and "--only-binary=:all:" in seen[0]


def test_combined_plan_is_checked_separately():
    """Do not offer a batch button for individually compatible conflicting modules."""
    items = [{"module": name, "status": "safe", "requirements": [req], "constraints": [], "additions": []}
             for name, req in (("a", "new==1"), ("b", "new==2"))]
    with patch.object(planner, "analyze_module", side_effect=items), patch.object(
            planner, "resolve_dependencies", side_effect=dependencies.DependencyRisk("joint conflict")):
        result = planner.analyze_batch({"a": Path("a"), "b": Path("b")}, snapshot())
    assert result["counts"]["safe"] == 2
    assert result["batch_count"] == 0
    assert result["batch_error"] == "joint conflict"


def execution_plan(installed, target):
    """Create a short-lived exact-commit execution fixture."""
    return {"id": "fixture", "expires_at": time.time() + 60, "fingerprint": "fixture",
            "report": {"version": "1", "install": []}, "modules": [{"module": "demo", "path": str(installed),
            "before": planner.git(installed, "rev-parse", "HEAD"), "target": target}]}


def test_executor_updates_exact_commit_in_disposable_repo(repository, tmp_path):
    """Exercise real fast-forward execution without pip or installed custom nodes."""
    _, installed, commit = repository
    target = commit({"node.py": "new"})
    planner.git(installed, "fetch", "origin")
    plan = execution_plan(installed, target)
    progress = []
    with patch.object(executor, "environment_snapshot", return_value=snapshot()), patch.object(executor, "ensure_server_stopped"):
        executor.execute(plan, tmp_path, lambda phase, message: progress.append(phase))
    assert planner.git(installed, "rev-parse", "HEAD") == target
    assert planner.git(installed, "rev-parse", "refs/alexz-backups/fixture") == plan["modules"][0]["before"]
    assert progress[-1] == "done"


@pytest.mark.parametrize("change", ["expired", "environment", "dirty", "head"])
def test_executor_rejects_stale_plans_before_mutation(repository, change):
    """Check expiration, environment, HEAD and working-tree binding."""
    _, installed, commit = repository
    target = commit({"node.py": "new"})
    planner.git(installed, "fetch", "origin")
    plan = execution_plan(installed, target)
    if change == "expired":
        plan["expires_at"] = 0
    if change == "dirty":
        (installed / "node.py").write_text("keep")
    if change == "head":
        planner.git(installed, "merge", "--ff-only", target)
    with patch.object(executor, "environment_snapshot", return_value=snapshot() if change != "environment" else {"fingerprint": "new"}):
        with pytest.raises(dependencies.Uncheckable):
            executor.validate_plan(plan)


@pytest.mark.parametrize("peer,host,origin,header,allowed", [
    ("127.0.0.1", "127.0.0.1:8188", "http://127.0.0.1:8188", "1", True),
    ("192.168.1.2", "127.0.0.1:8188", None, "1", False),
    ("127.0.0.1", "evil.invalid:8188", None, "1", False),
    ("127.0.0.1", "127.0.0.1:8188", "https://evil.invalid", "1", False),
    ("127.0.0.1", "127.0.0.1:8188", None, None, False),
])
def test_local_route_authorization(peer, host, origin, header, allowed):
    """Require loopback, trusted Host, same origin and explicit JSON mutation header."""
    request = SimpleNamespace(remote=peer, host=host, scheme="http", method="POST", content_type="application/json",
                              headers={"Origin": origin, "X-Alexz-Update": header})
    assert service.local_request(request) is allowed


def test_service_rejects_path_traversal_and_symlinks(repository, tmp_path):
    """Never accept arbitrary client paths or symlink escapes."""
    _, installed, _ = repository
    (installed.parent / "escape").symlink_to(tmp_path, target_is_directory=True)
    updater = service.UpdateService(lambda: [installed.parent], tmp_path / "project")
    for selection in (["../upstream"], ["escape"], "demo", []):
        with pytest.raises(dependencies.Uncheckable):
            updater.modules(selection)


def test_service_excludes_python_cache_from_modules(repository, tmp_path):
    """Exclude Python caches without excluding legitimate module directories."""
    _, installed, _ = repository
    (installed.parent / "__pycache__").mkdir()
    updater = service.UpdateService(lambda: [installed.parent], tmp_path / "project")
    assert set(updater.modules(None)) == {"demo"}
    with pytest.raises(dependencies.Uncheckable):
        updater.modules(["__pycache__"])


def test_apply_requires_acknowledgment_and_does_not_enable_legacy_routes(repository, tmp_path):
    """Reject an unconfirmed execution before any subprocess is spawned."""
    _, installed, _ = repository
    updater = service.UpdateService(lambda: [installed.parent], tmp_path / "project")
    updater.plan = {"id": "fixture"}
    updater.state["phase"] = "ready"
    with patch.object(service, "Popen") as spawn:
        with pytest.raises(dependencies.Uncheckable, match="подтверждение"):
            updater.apply({"plan_id": "fixture"})
        spawn.assert_not_called()


def test_real_http_routes_remain_read_only_until_explicit_request(tmp_path):
    """Exercise status and rejected POST routes through a real local aiohttp server."""
    from aiohttp import ClientSession, web

    async def scenario():
        """Use an ephemeral port and avoid all Git or package mutations."""
        routes = web.RouteTableDef()
        server = SimpleNamespace(instance=SimpleNamespace(routes=routes))
        service.register_routes(server, web, lambda: [tmp_path], tmp_path)
        app = web.Application()
        app.add_routes(routes)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with ClientSession() as client:
                async with client.get(f"http://127.0.0.1:{port}" + service.ROUTE_STATUS) as response:
                    assert response.status == 200
                    assert (await response.json())["phase"] == "idle"
                async with client.post(f"http://127.0.0.1:{port}" + service.ROUTE_APPLY, json={}) as response:
                    assert response.status == 403
                async with client.post(f"http://127.0.0.1:{port}" + service.ROUTE_APPLY, json={},
                                       headers={"X-Alexz-Update": "1"}) as response:
                    assert response.status == 409
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_real_pip_dry_run_resolves_local_wheels_without_installing(tmp_path):
    """Resolve real wheel metadata offline without mutating the Conda environment."""
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name, requires in (("alexz_fixture_a", "Requires-Dist: alexz-fixture-b>=1\n"), ("alexz_fixture_b", "")):
        with zipfile.ZipFile(wheels / f"{name}-1.0-py3-none-any.whl", "w") as wheel:
            prefix = f"{name}-1.0.dist-info/"
            wheel.writestr(prefix + "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n{requires}")
            wheel.writestr(prefix + "WHEEL", "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            wheel.writestr(prefix + "RECORD", "")
    original = dependencies.environment_snapshot()
    real_run = planner.run

    def offline_run(command, timeout=30):
        """Redirect only the test resolver to disposable wheel fixtures."""
        command = list(command)
        index = command.index("--index-url")
        command[index:index + 2] = ["--no-index", "--find-links", str(wheels)]
        return real_run(command, timeout)

    with patch.object(planner, "run", side_effect=offline_run):
        result = planner.resolve_dependencies(original, ["alexz-fixture-a>=1"], [])
    assert {item["name"] for item in result["additions"]} == {"alexz-fixture-a", "alexz-fixture-b"}
    assert not result["conflicts"]
    assert dependencies.environment_snapshot()["fingerprint"] == original["fingerprint"]


def test_cancelled_worker_never_executes_plan(tmp_path):
    """Cancel a queued worker while its parent is still running."""
    directory = tmp_path / "job"
    directory.mkdir()
    plan = directory / "plan.json"
    plan.write_text(json.dumps({"expires_at": time.time() + 60}))
    (directory / "cancel").touch()
    with patch.object(executor, "execute") as execute:
        executor.wait_and_execute(plan, os.getpid())
    execute.assert_not_called()
    assert json.loads((directory / "status.json").read_text())["phase"] == "cancelled"


def test_constraints_do_not_request_uninstalled_packages():
    """Do not treat an unused constraint as a missing required dependency."""
    with patch.object(planner, "run") as command:
        result = planner.resolve_dependencies(snapshot({"demo": package("1")}), ["demo>=1"], ["unused<2"])
    assert not result["conflicts"]
    assert not result["additions"]
    command.assert_not_called()


def test_server_probe_fails_closed_on_permission_error():
    """An inaccessible socket is not evidence that ComfyUI has stopped."""
    with patch.object(executor.socket, "create_connection", side_effect=PermissionError):
        with pytest.raises(dependencies.Uncheckable, match="подтвердить"):
            executor.ensure_server_stopped({"port": 8188})


def test_queue_starts_immediate_worker_without_mutating_in_request(repository, tmp_path):
    """Exercise a confirmed queue without actually starting an update worker."""
    _, installed, commit = repository
    target = commit({"node.py": "new"})
    planner.git(installed, "fetch", "origin")
    updater = service.UpdateService(lambda: [installed.parent], tmp_path / "project")
    plan = execution_plan(installed, target)
    item = {**plan["modules"][0], "status": "safe", "report": plan["report"]}
    updater.plan = {**plan, "results": [item], "batch": {"report": plan["report"]}, "batch_error": ""}
    updater.state["phase"] = "ready"
    old_head = planner.git(installed, "rev-parse", "HEAD")
    with patch.object(executor, "environment_snapshot", return_value=snapshot()), patch.object(service, "Popen") as spawn:
        result = updater.apply({"plan_id": "fixture", "confirmed": True, "mode": "checked"})
        assert result["status"] == "queued"
        spawn.assert_called_once()
        assert planner.git(installed, "rev-parse", "HEAD") == old_head
        assert "--immediate" in spawn.call_args.args[0]
        with pytest.raises(dependencies.Uncheckable, match="Начавшееся"):
            updater.cancel({"plan_id": "fixture"})
    assert json.loads((Path(result["directory"]) / "plan.json").read_text())["execution_mode"] == "online"


def test_worker_output_reaches_main_console_and_local_log(tmp_path, caplog):
    """Forward real subprocess stdout and stderr without changing installed nodes."""
    import logging
    worker = subprocess.Popen([sys.executable, "-u", "-c",
        "import sys; print('Updating aaa..bbb\\nFast-forward', flush=True); "
        "print('[ALEXZ update][error] fixture failure', file=sys.stderr, flush=True)"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    path = tmp_path / "worker.log"
    with caplog.at_level(logging.INFO):
        service.forward_worker_output(worker, path)
    assert worker.returncode == 0
    assert "Fast-forward" in path.read_text()
    assert "Fast-forward" in caplog.text
    assert any(record.levelno == logging.ERROR and "fixture failure" in record.message for record in caplog.records)


def test_executor_rolls_back_code_on_package_install_failure(repository, tmp_path):
    """Preserve a rollback commit and revert fixture code after a simulated pip failure."""
    _, installed, commit = repository
    target = commit({"node.py": "new"})
    planner.git(installed, "fetch", "origin")
    plan = execution_plan(installed, target)
    # Только fixture wheel: downloader и Conda/pip заменены, установка не выполняется.
    (installed / "runtime.json").write_text("keep")
    wheel = tmp_path / "fixture.whl"
    wheel.touch()
    with patch.object(executor, "environment_snapshot", return_value=snapshot()), patch.object(executor, "ensure_server_stopped"), \
         patch.object(executor, "download_wheels", return_value=[wheel]), patch.object(executor, "run", side_effect=["", dependencies.Uncheckable("fixture pip failure")]):
        with pytest.raises(dependencies.Uncheckable, match="откат кода: demo"):
            executor.execute(plan, tmp_path, lambda *_: None)
    assert planner.git(installed, "rev-parse", "HEAD") == plan["modules"][0]["before"]
    assert (installed / "runtime.json").read_text() == "keep"


def test_untracked_python_caches_do_not_block_analysis(repository):
    """Import-created caches must not mark an otherwise compatible update blocked."""
    _, installed, commit = repository
    commit({"node.py": "new"})
    cache = installed / "__pycache__/node.cpython-313.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"generated cache")
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "safe"
    assert cache.read_bytes() == b"generated cache"


@pytest.mark.parametrize("ignored", [False, True])
def test_untracked_collision_is_reported_by_path(repository, ignored):
    """Preserve even ignored local files when upstream adds a colliding path."""
    _, installed, commit = repository
    commit({"node.py": "upstream"})
    (installed / "node.py").write_text("local")
    if ignored:
        (installed / ".git/info/exclude").write_text("node.py\n")
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "blocked"
    assert "node.py" in result["reasons"][0]
    assert (installed / "node.py").read_text() == "local"


def test_automatic_files_are_backed_up_and_real_merge_succeeds(repository, tmp_path):
    """Exercise tracked caches and malformed upstream EOL attributes together."""
    from module_updates.worktree import inspect_worktree, prepare_automatic_files
    upstream, installed, commit = repository
    (upstream / "node.py").write_bytes(b"print(1)\r\n")
    (upstream / "__pycache__").mkdir()
    (upstream / "__pycache__/node.cpython-313.pyc").write_bytes(b"old cache")
    planner.git(upstream, "add", ".")
    planner.git(upstream, "commit", "-m", "CRLF committed before attributes")
    (upstream / ".gitattributes").write_text("*.py text eol=crlf\n")
    planner.git(upstream, "add", ".gitattributes")
    planner.git(upstream, "commit", "-m", "attributes without renormalization")
    planner.git(installed, "pull", "--ff-only")
    before = planner.git(installed, "rev-parse", "HEAD")
    cache = installed / "__pycache__/node.cpython-313.pyc"
    cache.write_bytes(b"runtime cache")
    node_before = (installed / "node.py").read_bytes()
    attributes = installed / ".git/info/attributes"
    attributes.write_bytes(b"*.txt -text\n")
    commit({"node.py": "print(2)\n", "__pycache__/node.cpython-313.pyc": "new cache"})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "safe"
    working = inspect_worktree(installed, before, result["target"])
    assert set(working["automatic"]) == {"node.py", "__pycache__/node.cpython-313.pyc"}
    assert attributes.read_bytes() == b"*.txt -text\n"
    with prepare_automatic_files(installed, before, working["automatic"], tmp_path / "backup"):
        planner.git(installed, "merge", "--ff-only", "--no-overwrite-ignore", result["target"])
    assert (installed / "node.py").read_bytes().replace(b"\r\n", b"\n") == b"print(2)\n"
    assert cache.read_bytes() == b"new cache"
    assert (tmp_path / "backup/node.py").read_bytes() == node_before
    assert (tmp_path / "backup/__pycache__/node.cpython-313.pyc").read_bytes() == b"runtime cache"
    assert attributes.read_bytes() == b"*.txt -text\n"


def test_automatic_file_preparation_restores_files_when_merge_fails(repository, tmp_path):
    """Restore cached bytes and temporary attributes when HEAD remains unchanged."""
    from module_updates.worktree import prepare_automatic_files
    upstream, installed, _ = repository
    cache_name = "__pycache__/node.cpython-313.pyc"
    (upstream / "__pycache__").mkdir()
    (upstream / cache_name).write_bytes(b"committed")
    planner.git(upstream, "add", ".")
    planner.git(upstream, "commit", "-m", "cache fixture")
    planner.git(installed, "pull", "--ff-only")
    (installed / cache_name).write_bytes(b"generated")
    before = planner.git(installed, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError):
        with prepare_automatic_files(installed, before, [cache_name], tmp_path / "backup"):
            raise RuntimeError("fixture failure")
    assert (installed / cache_name).read_bytes() == b"generated"
    assert not (installed / ".git/info/attributes").exists()


def test_real_tracked_edit_still_blocks_overlapping_update(repository):
    """Do not mistake a code change for EOL normalization."""
    _, installed, commit = repository
    commit({"node.py": "original\n"})
    planner.git(installed, "pull", "--ff-only")
    commit({"node.py": "upstream\n"})
    (installed / "node.py").write_text("local code\n")
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "blocked"
    assert "node.py" in result["reasons"][0]


@pytest.mark.parametrize("cache_updated", [False, True])
def test_worker_updates_with_local_runtime_files(repository, tmp_path, cache_updated):
    """Integrate runtime backup, fast-forward and preservation of untouched cache."""
    _, installed, commit = repository
    cache_name = "__pycache__/node.cpython-313.pyc"
    commit({cache_name: "committed cache", "node.py": "first\n"})
    planner.git(installed, "pull", "--ff-only")
    changes = {"node.py": "second\n"}
    if cache_updated:
        changes[cache_name] = "upstream cache"
    target = commit(changes)
    planner.git(installed, "fetch", "origin")
    (installed / cache_name).write_bytes(b"runtime cache")
    (installed / "custom_models.json").write_text('{"runtime":true}')
    plan = execution_plan(installed, target)
    with patch.object(executor, "environment_snapshot", return_value=snapshot()), patch.object(executor, "ensure_server_stopped"):
        executor.execute(plan, tmp_path, lambda *_: None)
    assert planner.git(installed, "rev-parse", "HEAD") == target
    assert (installed / cache_name).read_bytes() == (b"upstream cache" if cache_updated else b"runtime cache")
    assert (installed / "custom_models.json").read_text() == '{"runtime":true}'
    assert (tmp_path / "local-files/demo" / cache_name).read_bytes() == b"runtime cache"
    assert not (installed / ".git/info/attributes").exists()


def test_batch_parallelism_is_bounded_and_progress_counts_completions():
    """Require overlapping tasks, bounded concurrency and deterministic output."""
    import threading
    gate = threading.Barrier(4, timeout=5)
    lock = threading.Lock()
    active = peak = 0
    captures = []
    progress = []

    def analyze(name, repo, environment, **kwargs):
        """Synchronize four fixture tasks without contacting remote repositories."""
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        gate.wait()
        with lock:
            active -= 1
        return {"module": name, "status": "up_to_date"}

    with patch.object(planner, "analyze_module", side_effect=analyze):
        result = planner.analyze_batch({str(n): Path(str(n)) for n in range(8)}, snapshot(), workers=20,
            progress=lambda name, done, total: progress.append((done, total)),
            module_callback=lambda item: captures.append(item["module"]))
    assert peak == 4
    assert progress == [(n, 8) for n in range(9)]
    assert [item["module"] for item in result["results"]] == list(map(str, range(8)))
    assert sorted(captures) == list(map(str, range(8)))


def test_identical_dependency_checks_are_shared_with_joint_analysis(repository, pip_dry_run):
    """Resolve identical module requirements and joint requirements exactly once."""
    _, installed, commit = repository
    commit({"node.py": "new", "requirements.txt": "demo>=1.0\n"})
    # Два разных имени fixture с одним readonly repo, без параллельного Git fetch.
    planner.git(installed, "fetch", "origin")
    with patch.object(planner, "resolve_dependencies", wraps=planner.resolve_dependencies) as resolve:
        result = planner.analyze_batch({"a": installed, "b": installed}, snapshot({"demo": package("1")}), fetch=False)
    assert result["batch_count"] == 2
    assert resolve.call_count == 1


def test_service_finalizes_tracking_once_before_publishing_ready(repository, tmp_path):
    """Make cached tracking available before frontend sees a completed plan."""
    import threading
    _, installed, _ = repository
    finished = threading.Event()
    updater = service.UpdateService(lambda: [installed.parent], tmp_path,
        capture_module=lambda item: {"git": item["module"]},
        finalize_tracking=lambda captured: (assert_capture(captured), finished.set()))

    def assert_capture(captured):
        """Verify cached capture and ordering while the job is still checking."""
        assert captured == {"demo": {"git": "demo"}}
        assert updater.state["phase"] == "checking"

    with patch.object(service, "environment_snapshot", return_value=snapshot()), \
         patch.object(planner, "analyze_module", return_value={"module": "demo", "status": "up_to_date"}):
        updater.check(["demo"])
        assert finished.wait(5)
        # Join the fixture job to avoid racing patch teardown.
        for thread in threading.enumerate():
            if thread.name == "alexz-update-plan":
                thread.join(5)
    assert updater.state["phase"] == "ready"


def test_cached_tracking_avoids_second_git_scan_and_manager_probe(monkeypatch):
    """Preserve marker publication while forbidding all live tracking probes."""
    import importlib
    import types
    root = "ComfyUI_ALEXZ_tools"
    if root not in sys.modules:
        stub = types.ModuleType(root)
        stub.__path__ = [str(Path(__file__).resolve().parents[1])]
        monkeypatch.setitem(sys.modules, root, stub)
    api = importlib.import_module(root + ".utils.module_node_browser_api")
    saved = []
    with patch.object(api, "_module_git_state", side_effect=AssertionError("second Git scan")), \
         patch.object(api, "_module_worktree_signature", side_effect=AssertionError("second worktree scan")), \
         patch.object(api, "_manager_installed_update_overrides", side_effect=AssertionError("Manager probe")) as manager_probe, \
         patch.object(api, "_load_module_state", return_value={"unchecked": {"installed_commit": "old", "update_available": True, "worktree_signature": "keep"}}), \
         patch.object(api, "_save_module_state", side_effect=saved.append), \
         patch.object(api, "_discover_custom_modules", return_value=["demo", "unchecked"]), \
         patch.object(api, "_canonical_custom_module_name", side_effect=lambda name: name), \
         patch.object(api, "_manager_meta_for_module", return_value=None), \
         patch.object(api, "_infer_update_from_manager_stats", return_value=(None, "")), \
         patch.object(api, "_build_node_snapshots", return_value={}):
        report = api._announce_tracked_module_updates(precomputed={"demo": {
            "git": {"installed_commit": "aaa", "remote_head": "bbb", "behind": 1}, "worktree": "fixture"}})
    assert report["modules_need_update"] == 2
    assert saved[-1]["demo"]["installed_commit"] == "aaa"
    assert saved[-1]["demo"]["update_available"] is True
    assert saved[-1]["unchecked"]["update_available"] is True
    assert saved[-1]["unchecked"]["worktree_signature"] == "keep"
    manager_probe.assert_not_called()


def test_pip_report_reuses_batch_baseline():
    """Compute projected conflicts without recomputing the original environment."""
    environment = {**snapshot(), "baseline": ["existing conflict"]}
    with patch.object(dependencies, "dependency_conflicts", return_value=[]) as conflicts:
        report = dependencies.project_report(environment, {"version": "1", "install": []}, [])
    assert conflicts.call_count == 1
    assert report["baseline"] == ["existing conflict"]
    assert not report["conflicts"]


def test_immediate_worker_does_not_wait_for_live_parent(tmp_path):
    """Run immediately despite a living parent, without touching real modules."""
    directory = tmp_path / "job"
    directory.mkdir()
    plan = directory / "plan.json"
    plan.write_text(json.dumps({"expires_at": time.time() + 60}))
    with patch.object(executor, "execute") as execute, patch.object(executor.os, "kill") as kill:
        executor.wait_and_execute(plan, os.getpid(), immediate=True)
    execute.assert_called_once()
    kill.assert_not_called()


def test_cancel_legacy_job_after_server_restart(tmp_path):
    """Recover cancellation without the former server's in-memory report."""
    updater = service.UpdateService(lambda: [], tmp_path)
    directory = updater.artifacts / "old-job"
    directory.mkdir()
    executor.write_status(directory / "status.json", "waiting_for_shutdown", "waiting")
    assert updater.plan is None
    assert updater.cancel({"plan_id": "old-job"})["status"] == "cancelling"
    assert (directory / "cancel").exists()


def test_saved_report_survives_restart_and_expired_report_does_not(tmp_path):
    """Restore only unexpired reports; application still revalidates each plan."""
    updater = service.UpdateService(lambda: [], tmp_path)
    plan = {"id": "saved", "expires_at": time.time() + 60, "results": [], "batch": {}}
    updater.saved_plan.write_text(json.dumps(plan))
    restored = service.UpdateService(lambda: [], tmp_path)
    assert restored.status()["plan"]["id"] == "saved"
    plan["expires_at"] = time.time() - 1
    updater.saved_plan.write_text(json.dumps(plan))
    assert service.UpdateService(lambda: [], tmp_path).status()["plan"] is None


def test_online_worker_refuses_changed_server_session(tmp_path):
    """Detect an exec-based restart even when the process PID stays unchanged."""
    session = tmp_path / "server-session"
    session.write_text("new")
    plan = {"execution_mode": "online", "session_path": str(session), "server_session": "old"}
    with pytest.raises(dependencies.Uncheckable, match="перезапущен"):
        executor.ensure_update_allowed(plan)


@pytest.mark.parametrize("queue", [{"queue_running": [1], "queue_pending": []},
                                   {"queue_running": [], "queue_pending": [1]}, {}])
def test_online_worker_refuses_busy_or_invalid_queue(tmp_path, queue):
    """Never start mutation while queued workflows exist or status is malformed."""
    from io import BytesIO
    session = tmp_path / "server-session"
    session.write_text("same")
    plan = {"execution_mode": "online", "session_path": str(session), "server_session": "same", "port": 8188}
    with patch.object(executor.urllib.request, "urlopen", return_value=BytesIO(json.dumps(queue).encode())):
        with pytest.raises(dependencies.Uncheckable):
            executor.ensure_update_allowed(plan)


def test_online_worker_accepts_idle_queue(tmp_path):
    """Allow immediate execution with an unchanged server and an empty queue."""
    from io import BytesIO
    session = tmp_path / "server-session"
    session.write_text("same")
    plan = {"execution_mode": "online", "session_path": str(session), "server_session": "same", "port": 8188}
    with patch.object(executor.urllib.request, "urlopen", return_value=BytesIO(b'{"queue_running": [], "queue_pending": []}')):
        executor.ensure_update_allowed(plan)


def test_online_executor_fast_forwards_before_server_shutdown(repository, tmp_path):
    """Update disposable code immediately while the calling process remains alive."""
    from io import BytesIO
    _, installed, commit = repository
    target = commit({"node.py": "online new"})
    planner.git(installed, "fetch", "origin")
    session = tmp_path / "server-session"
    session.write_text("same")
    plan = {**execution_plan(installed, target), "execution_mode": "online", "port": 8188,
            "session_path": str(session), "server_session": "same"}
    phases = []
    with patch.object(executor, "environment_snapshot", return_value=snapshot()), \
         patch.object(executor.urllib.request, "urlopen", side_effect=lambda *a, **kw: BytesIO(b'{"queue_running": [], "queue_pending": []}')):
        executor.execute(plan, tmp_path, lambda phase, message: phases.append(phase))
    assert planner.git(installed, "rev-parse", "HEAD") == target
    assert phases[-1] == "done"


def test_service_rejects_update_during_generation(repository, tmp_path):
    """Refuse a confirmed plan before spawning when the server has queued work."""
    _, installed, commit = repository
    target = commit({"node.py": "new"})
    updater = service.UpdateService(lambda: [], tmp_path, busy=lambda: True)
    plan = execution_plan(installed, target)
    updater.plan = {**plan, "results": [{**plan["modules"][0], "status": "safe"}], "batch": {}, "batch_error": ""}
    updater.state["phase"] = "ready"
    with patch.object(service, "Popen") as spawn:
        with pytest.raises(dependencies.Uncheckable, match="генерацию"):
            updater.apply({"plan_id": "fixture", "confirmed": True})
    spawn.assert_not_called()


def test_finished_job_does_not_hide_new_report(tmp_path):
    """Clear an applied plan, retaining a subsequent check and its new buttons."""
    updater = service.UpdateService(lambda: [], tmp_path)
    directory = updater.artifacts / "old-job"
    directory.mkdir()
    executor.write_status(directory / "status.json", "done", "done")
    updater.plan = {"id": "new-job"}
    updater.state = {"phase": "ready", "message": "ready", "plan": {"id": "new-job"}}
    status = updater.status()
    assert "execution" not in status
    assert status["plan"]["id"] == "new-job"


@pytest.mark.parametrize("phase", ["done", "error", "waiting_for_shutdown"])
def test_restart_hides_only_successful_previous_execution(tmp_path, phase):
    """Keep restart guidance until restart, and retain failures and pending jobs."""
    updater = service.UpdateService(lambda: [], tmp_path)
    directory = updater.artifacts / "job"
    directory.mkdir()
    (directory / "plan.json").write_text(json.dumps({"server_session": updater.server_session}))
    executor.write_status(directory / "status.json", phase, "fixture")
    assert updater.status()["execution"]["phase"] == phase
    restored = service.UpdateService(lambda: [], tmp_path)
    status = restored.status()
    if phase == "done":
        assert "execution" not in status
        assert json.loads((directory / "status.json").read_text())["phase"] == "done"
    else:
        assert status["execution"]["phase"] == phase


def test_successful_update_markers_survive_restart_and_ignore_failed_jobs(tmp_path):
    """Restore only confirmed updates, including older successful jobs."""
    updater = service.UpdateService(lambda: [], tmp_path)
    for name, phase in (("updated", "done"), ("failed", "error"), ("pending", "updating")):
        directory = updater.artifacts / name
        directory.mkdir()
        executor.write_status(directory / "status.json", phase, phase)
        (directory / "plan.json").write_text(json.dumps({"modules": [
            {"module": name, "before": "old", "target": "new"}]}))
    assert updater.status()["updated_modules"] == ["updated"]
    assert service.UpdateService(lambda: [], tmp_path).status()["updated_modules"] == ["updated"]


def test_dead_worker_is_reported_instead_of_blocking_forever(tmp_path):
    """Expose interrupted operations without attempting to resume or reapply them."""
    updater = service.UpdateService(lambda: [], tmp_path)
    directory = updater.artifacts / "job"
    directory.mkdir()
    executor.write_status(directory / "status.json", "updating", "updating")
    (directory / "worker.json").write_text(json.dumps({"pid": 99999999}))
    with patch.object(service.os, "kill", side_effect=ProcessLookupError):
        assert updater.status()["execution"]["phase"] == "error"


@pytest.mark.parametrize("with_additions", [False, True])
def test_individual_updates_keep_remaining_plan_and_restart_status(tmp_path, with_additions):
    """Apply consecutive fixture jobs, retaining candidates and session restart flags."""
    updater = service.UpdateService(lambda: [], tmp_path)
    before = snapshot()
    before["baseline"] = []
    after = snapshot({"demo": package("1")}) if with_additions else before
    if with_additions:
        after["fingerprint"] = "after"
    report = {"version": "1", "install": [{"metadata": {"name": "demo", "version": "1"}}] if with_additions else []}
    items = [{"module": name, "status": "safe", "before": "old", "target": "new",
              "report": report, "requirements_changed": with_additions, "requirements": ["demo>=1"] if with_additions else [],
              "constraints": [], "additions": []} for name in ("first", "second")]
    updater.plan = {"id": "checked", "fingerprint": "fixture", "expires_at": time.time() + 3600,
                    "results": items, "batch": {"report": report}, "batch_error": "", "batch_count": 2}
    (updater.artifacts / "checked-environment.json").write_text(json.dumps(before))
    updater.state = {"phase": "ready", "plan": updater.public_plan(updater.plan)}
    with patch.object(service, "environment_snapshot", return_value=after):
        for index, name in enumerate(("first", "second")):
            plan_id = updater.plan["id"]
            with patch.object(service, "validate_plan"), patch.object(service, "Popen"):
                result = updater.apply({"plan_id": plan_id, "module": name, "confirmed": True})
            directory = Path(result["directory"])
            execution = json.loads((directory / "plan.json").read_text())
            assert [item["module"] for item in execution["modules"]] == [name]
            if index == 1:
                assert execution["report"]["install"] == []
            executor.write_status(directory / "status.json", "done", "done")
            status = updater.status()
            assert status["plan"]["batch_count"] == 1 - index
            assert status["plan"]["id"] != plan_id
            assert status["restart_required_modules"] == sorted(("first", "second")[:index + 1])
            assert updater.status()["plan"]["id"] == status["plan"]["id"]
        with patch.object(service, "Popen") as spawn:
            with pytest.raises(dependencies.Uncheckable):
                updater.apply({"plan_id": updater.plan["id"], "module": "first", "confirmed": True})
        spawn.assert_not_called()
    restored = service.UpdateService(lambda: [], tmp_path)
    assert restored.status()["restart_required_modules"] == []
    assert restored.status()["updated_modules"] == ["first", "second"]
    assert restored.status()["plan"]["batch_count"] == 0


def test_individual_update_does_not_accept_unrelated_environment_changes(tmp_path):
    """Do not rebase checked candidates over an external package replacement."""
    updater = service.UpdateService(lambda: [], tmp_path)
    updater.plan = {"id": "job", "fingerprint": "old"}
    directory = updater.artifacts / "job"
    directory.mkdir()
    (directory / "plan.json").write_text(json.dumps({"report": {"install": []}, "modules": []}))
    (updater.artifacts / "job-environment.json").write_text(json.dumps(snapshot({"demo": package("1")})))
    with patch.object(service, "environment_snapshot", return_value=snapshot({"demo": package("2")})):
        with pytest.raises(dependencies.Uncheckable, match="вне выполненного плана"):
            updater.advance_plan(directory)
