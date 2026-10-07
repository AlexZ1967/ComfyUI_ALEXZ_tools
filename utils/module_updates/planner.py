"""Build immutable, conservative Git and dependency update plans."""

from __future__ import annotations

import difflib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .dependencies import DependencyRisk, Uncheckable, dependency_conflicts, project_report, read_requirements


def run(command: list[str], timeout: float = 30) -> str:
    """Run an argument-vector command with bounded time and noninteractive Git."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "PIP_CONFIG_FILE": os.devnull}
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode:
        raise Uncheckable((result.stderr or result.stdout or "Команда завершилась с ошибкой")[-5000:])
    return result.stdout.strip()


def git(repo: Path, *args: str) -> str:
    """Run Git without local hook execution."""
    return run(["git", "-c", "core.hooksPath=/dev/null", "-c", "merge.autoStash=false", "-C", str(repo), *args])


def read_commit_file(repo: Path, commit: str, path: str) -> str | None:
    """Read a regular UTF-8 blob, distinguishing missing files from Git failures."""
    entry = git(repo, "ls-tree", commit, "--", path)
    if not entry:
        return None
    if not entry.startswith(("100644 blob ", "100755 blob ")):
        raise Uncheckable(f"{path}: ожидается обычный файл, не symlink")
    return git(repo, "show", f"{commit}:{path}")


def resolve_dependencies(snapshot: dict, requirements: list[str], constraints: list[str]) -> dict:
    """Resolve wheels while pinning all existing distributions, without installing."""
    baseline = dependency_conflicts(snapshot["packages"], snapshot["markers"])
    installed_constraints = [raw for raw in constraints if canonicalize_name(Requirement(raw).name) in snapshot["packages"]]
    current = dependency_conflicts(snapshot["packages"], snapshot["markers"], requirements + installed_constraints)
    for raw in requirements + constraints:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({**snapshot["markers"], "extra": ""}):
            continue
        installed = snapshot["packages"].get(canonicalize_name(req.name))
        if installed and req.specifier and not req.specifier.contains(installed["version"], prereleases=True):
            raise DependencyRisk(f"Нужен {req.name}{req.specifier}; защищена установленная версия {installed['version']}")
    # Даже изменённые requirements могут уже удовлетворяться текущим окружением.
    if not set(current) - set(baseline):
        return {"additions": [], "baseline": baseline, "conflicts": [], "report": {"version": "1", "install": []}}
    with tempfile.TemporaryDirectory(prefix="alexz-update-check-") as directory:
        root = Path(directory)
        protected = root / "protected.txt"
        protected.write_text("\n".join(f"{name}=={pkg['version']}" for name, pkg in sorted(snapshot["packages"].items()))
                             + "\n" + "\n".join(constraints), encoding="utf-8")
        requirements_path = root / "requirements.txt"
        report_path = root / "report.json"
        command = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
                   "--dry-run", "--no-deps", "--no-input", "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                   "--retries", "0", "--timeout", "20", "--report", str(report_path),
                   "-c", str(protected), "-r", str(requirements_path)]
        # --no-deps предотвращает выполнение VCS/build backend из transitive URL.
        # Metadata wheels читаем до передачи дочерних requirements в pip.
        expanded = set(requirements)
        deadline = time.monotonic() + 180
        for _round in range(16):
            requirements_path.write_text("\n".join(sorted(expanded)), encoding="utf-8")
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Uncheckable("Истекло время разрешения зависимостей")
                run(command, timeout=remaining)
            except Uncheckable as exc:
                if "ResolutionImpossible" in str(exc):
                    raise DependencyRisk(str(exc)) from exc
                raise
            report = json.loads(report_path.read_text(encoding="utf-8"))
            previous = set(expanded)
            extras = {}
            for raw in expanded:
                req = Requirement(raw)
                extras.setdefault(canonicalize_name(req.name), set()).update(req.extras)
            for item in report["install"]:
                metadata = item["metadata"]
                owner = canonicalize_name(metadata["name"])
                for raw in metadata.get("requires_dist", []):
                    req = Requirement(raw)
                    if req.marker and not any(req.marker.evaluate({**snapshot["markers"], "extra": extra})
                                              for extra in {"", *extras.get(owner, set())}):
                        continue
                    if req.url:
                        raise Uncheckable(f"Transitive URL-зависимость требует отдельной проверки: {req.name}")
                    expanded.add(str(req).split(";", 1)[0])
            if expanded == previous:
                break
        else:
            raise Uncheckable("Превышена глубина графа зависимостей")
    names = set(snapshot["packages"]) | {canonicalize_name(item["metadata"]["name"]) for item in report["install"]}
    applicable_constraints = [raw for raw in constraints if canonicalize_name(Requirement(raw).name) in names]
    projected = project_report(snapshot, report, requirements + applicable_constraints)
    projected["report"] = report
    return projected


def analyze_module(name: str, repo: Path, snapshot: dict, *, fetch: bool = True) -> dict:
    """Inspect an upstream commit without updating a repository's working tree."""
    result = {"module": name, "status": "unknown", "reasons": [], "additions": [],
              "requirements_changed": None, "diff": "", "path": str(repo)}
    try:
        if git(repo, "rev-parse", "--show-toplevel") != str(repo.resolve()):
            raise Uncheckable("Каталог не является отдельным Git-репозиторием")
        result["before"] = git(repo, "rev-parse", "HEAD")
        branch = git(repo, "symbolic-ref", "--short", "HEAD")
        remote = git(repo, "config", "--get", f"branch.{branch}.remote")
        remote_ref = git(repo, "config", "--get", f"branch.{branch}.merge")
        if remote.startswith("-") or remote == "." or not remote_ref.startswith("refs/heads/"):
            raise Uncheckable("Нужен настроенный upstream удалённой ветки")
        if git(repo, "status", "--porcelain", "--untracked-files=all"):
            result["status"] = "blocked"
            result["reasons"] = ["Есть локальные изменения; автоматический stash запрещён"]
            return result
        if fetch:
            tracking_ref = git(repo, "rev-parse", "--symbolic-full-name", "@{u}")
            if not tracking_ref.startswith("refs/remotes/"):
                raise Uncheckable("Нужна remote-tracking ветка")
            git(repo, "fetch", "--no-tags", "--no-write-fetch-head", remote, f"{remote_ref}:{tracking_ref}")
            result["target"] = git(repo, "rev-parse", tracking_ref)
        else:
            result["target"] = git(repo, "rev-parse", "@{u}")
        if result["before"] == result["target"]:
            result["status"] = "up_to_date"
            return result
        git(repo, "merge-base", "--is-ancestor", result["before"], result["target"])
        before = read_requirements(lambda path: read_commit_file(repo, result["before"], path))
        after = read_requirements(lambda path: read_commit_file(repo, result["target"], path))
        result["requirements_changed"] = before["files"] != after["files"]
        result["requirements"] = after["requirements"]
        result["constraints"] = after["constraints"]
        result["diff"] = "\n".join(difflib.unified_diff(
            json.dumps(before["files"], indent=2, ensure_ascii=False).splitlines(),
            json.dumps(after["files"], indent=2, ensure_ascii=False).splitlines(),
            fromfile="installed", tofile="upstream", lineterm=""))
        changed = git(repo, "diff", "--name-only", result["before"], result["target"]).splitlines()
        metadata = [path for path in changed if Path(path).name in {"install.py", "setup.py", "setup.cfg", "pyproject.toml"}]
        # Новый install script может менять окружение вне pip requirements.
        if metadata or (result["requirements_changed"] and read_commit_file(repo, result["target"], "install.py") is not None):
            result["status"] = "risk"
            result["reasons"] = ["Изменён packaging/install script: нужна отдельная проверка"]
            return result
        resolved = resolve_dependencies(snapshot, after["requirements"], after["constraints"])
        result.update(resolved)
        if resolved["conflicts"]:
            result["status"] = "risk"
            result["reasons"] = resolved["conflicts"]
        else:
            result["status"] = "safe"
            result["reasons"] = ["Конфликтов зависимостей не обнаружено; существующие пакеты сохраняются"]
    except DependencyRisk as exc:
        result["status"] = "risk"
        result["reasons"] = [str(exc)]
    except (Uncheckable, subprocess.TimeoutExpired, OSError, ValueError, KeyError) as exc:
        # Ошибки Git/сети/формата никогда не трактуются как неизменные requirements.
        result["status"] = "unknown"
        result["reasons"] = [str(exc)]
    return result


def analyze_batch(modules: dict[str, Path], snapshot: dict, *, fetch: bool = True, progress=None) -> dict:
    """Validate individual updates and the union selected for the batch action."""
    results = []
    for name, repo in sorted(modules.items()):
        if progress:
            progress(name, len(results), len(modules))
        results.append(analyze_module(name, repo, snapshot, fetch=fetch))
    candidates = [item for item in results if item["status"] == "safe"]
    requirements = sorted({req for item in candidates for req in item["requirements"]})
    constraints = sorted({req for item in candidates for req in item["constraints"]})
    batch = {"additions": [], "conflicts": [], "baseline": [], "report": {"version": "1", "install": []}}
    batch_error = ""
    if candidates:
        try:
            batch = resolve_dependencies(snapshot, requirements, constraints)
            if batch["conflicts"]:
                batch_error = "\n".join(batch["conflicts"])
        except (Uncheckable, subprocess.TimeoutExpired, OSError, ValueError, KeyError) as exc:
            batch_error = str(exc)
    counts = {status: sum(item["status"] == status for item in results)
              for status in ("safe", "risk", "unknown", "blocked", "up_to_date")}
    return {"results": results, "counts": counts, "batch": batch,
            "batch_error": batch_error, "batch_count": 0 if batch_error else len(candidates),
            "environment": {key: snapshot[key] for key in ("python", "prefix", "fingerprint")},
            "baseline": dependency_conflicts(snapshot["packages"], snapshot["markers"])}
