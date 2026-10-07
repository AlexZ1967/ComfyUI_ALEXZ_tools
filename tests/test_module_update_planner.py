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


def test_changed_requirements_can_already_be_satisfied(repository):
    """Avoid pip when changed declarations are satisfied by installed versions."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=1.0\n"})
    with patch.object(planner, "run", wraps=planner.run) as command:
        result = planner.analyze_module("demo", installed, snapshot({"demo": package("1.2")}))
    assert result["status"] == "safe"
    assert result["requirements_changed"]
    assert not any("pip" in call.args[0] for call in command.call_args_list)


def test_existing_package_replacement_is_risk(repository):
    """Do not approve a request to replace a protected distribution."""
    _, installed, commit = repository
    commit({"requirements.txt": "demo>=2\n"})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "risk"
    assert "защищена" in result["reasons"][0]


@pytest.mark.parametrize("name", ["install.py", "setup.py", "setup.cfg", "pyproject.toml"])
def test_packaging_changes_require_separate_review(repository, name):
    """Do not classify executable install behavior as a checked dependency update."""
    _, installed, commit = repository
    commit({name: "changed\n"})
    result = planner.analyze_module("demo", installed, snapshot({"demo": package("1")}))
    assert result["status"] == "risk"


def test_dirty_repository_is_blocked_without_stash(repository):
    """Preserve both tracked edits and untracked files."""
    _, installed, commit = repository
    commit({"node.py": "new"})
    (installed / "local.txt").write_text("keep")
    result = planner.analyze_module("demo", installed, snapshot())
    assert result["status"] == "blocked"
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
        (installed / "local").write_text("keep")
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


def test_queue_stages_only_and_cancel_is_available(repository, tmp_path):
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
        assert updater.cancel({"plan_id": "fixture"})["status"] == "cancelling"
    assert (Path(result["directory"]) / "cancel").exists()


def test_executor_rolls_back_code_on_package_install_failure(repository, tmp_path):
    """Preserve a rollback commit and revert fixture code after a simulated pip failure."""
    _, installed, commit = repository
    target = commit({"node.py": "new"})
    planner.git(installed, "fetch", "origin")
    plan = execution_plan(installed, target)
    # Только fixture wheel: downloader и Conda/pip заменены, установка не выполняется.
    wheel = tmp_path / "fixture.whl"
    wheel.touch()
    with patch.object(executor, "environment_snapshot", return_value=snapshot()), patch.object(executor, "ensure_server_stopped"), \
         patch.object(executor, "download_wheels", return_value=[wheel]), patch.object(executor, "run", side_effect=["", dependencies.Uncheckable("fixture pip failure")]):
        with pytest.raises(dependencies.Uncheckable, match="откат кода: demo"):
            executor.execute(plan, tmp_path, lambda *_: None)
    assert planner.git(installed, "rev-parse", "HEAD") == plan["modules"][0]["before"]
