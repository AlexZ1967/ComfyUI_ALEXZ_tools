"""Read declarative requirements and validate a projected Python environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import posixpath
import sys
import sysconfig
from pathlib import PurePosixPath
from typing import Callable

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


class Uncheckable(ValueError):
    """Indicate that declarative dependency analysis cannot establish a result."""


class DependencyRisk(Uncheckable):
    """Indicate a known conflict with the protected environment."""


def environment_snapshot() -> dict:
    """Read installed distribution metadata without importing third-party packages."""
    packages = {}
    # ComfyUI/extensions могут добавлять setuptools/_vendor в sys.path.
    # Сравниваем установленные пакеты окружения, а не эти встроенные копии.
    paths = sysconfig.get_paths()
    installation_paths = list(dict.fromkeys([paths["purelib"], paths["platlib"]]))
    for dist in importlib.metadata.distributions(path=installation_paths):
        name = dist.metadata.get("Name")
        if name:
            packages[canonicalize_name(name)] = {
                "version": dist.version,
                "requires_dist": sorted(dist.requires or []),
            }
    data = {"python": sys.executable, "prefix": sys.prefix, "packages": packages,
            "markers": default_environment()}
    data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def read_requirements(read_file: Callable[[str], str | None]) -> dict:
    """Expand repository-local includes, rejecting executable or external inputs."""
    files: dict[str, str] = {}
    requirements: list[str] = []
    constraints: list[str] = []
    active: set[str] = set()

    def visit(path: str, constraint: bool = False) -> None:
        """Read one plain requirements file recursively."""
        path = posixpath.normpath(path)
        if PurePosixPath(path).is_absolute() or path == ".." or path.startswith("../"):
            raise Uncheckable("Include выходит за пределы репозитория")
        if path in active:
            raise Uncheckable("Циклический include в requirements")
        if len(files) >= 64 and path not in files:
            raise Uncheckable("Слишком много вложенных requirements")
        text = read_file(path)
        if text is None:
            raise Uncheckable(f"Не найден {path}")
        if len(text) > 256000:
            raise Uncheckable("Слишком большой requirements")
        files[path] = text
        active.add(path)
        try:
            for raw in text.splitlines():
                line = raw.split(" #", 1)[0].strip()
                if not line or line.startswith("#"):
                    continue
                include = None
                for flag, kind in (("--requirement", False), ("--constraint", True), ("-r", False), ("-c", True)):
                    if line.startswith(flag + " ") or line.startswith(flag + "="):
                        include = (line[len(flag):].lstrip(" ="), kind)
                        break
                    if flag in {"-r", "-c"} and line.startswith(flag) and len(line) > 2:
                        include = (line[2:].strip(), kind)
                        break
                if include:
                    visit(posixpath.join(posixpath.dirname(path), include[0]), constraint or include[1])
                    continue
                try:
                    req = Requirement(line)
                except InvalidRequirement as exc:
                    raise Uncheckable(f"Неподдерживаемая строка в {path}: {line}") from exc
                if req.url:
                    raise Uncheckable(f"URL/VCS-зависимость требует отдельной проверки: {req.name}")
                (constraints if constraint else requirements).append(str(req))
        finally:
            active.remove(path)

    if read_file("requirements.txt") is not None:
        visit("requirements.txt")
    return {"files": files, "requirements": sorted(set(requirements)),
            "constraints": sorted(set(constraints))}


def dependency_conflicts(packages: dict, markers: dict, roots: list[str] = ()) -> list[str]:
    """Check all installed distributions and requested extras against a package set."""
    problems: set[str] = set()
    extras: dict[str, set[str]] = {}
    pending = list(roots)
    visited: set[tuple[str, str]] = set()
    while pending:
        raw = pending.pop()
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            problems.add(f"Некорректная metadata: {raw}")
            continue
        name = canonicalize_name(req.name)
        if req.marker and not req.marker.evaluate({**markers, "extra": ""}):
            continue
        extras.setdefault(name, set()).update(req.extras)
        for extra in req.extras:
            if (name, extra) not in visited and name in packages:
                visited.add((name, extra))
                for child in packages[name]["requires_dist"]:
                    child_req = Requirement(child)
                    if not child_req.marker or child_req.marker.evaluate({**markers, "extra": extra}):
                        pending.append(str(child_req).split(";", 1)[0])

    def check(raw: str, owner: str, owner_extras: set[str]) -> None:
        """Evaluate one dependency, preserving platform and extras markers."""
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            problems.add(f"{owner}: некорректная metadata {raw}")
            return
        if req.marker and not any(req.marker.evaluate({**markers, "extra": extra}) for extra in {"", *owner_extras}):
            return
        name = canonicalize_name(req.name)
        installed = packages.get(name)
        if not installed:
            problems.add(f"{owner}: отсутствует {req.name}{req.specifier}")
        elif req.specifier and not req.specifier.contains(installed["version"], prereleases=True):
            problems.add(f"{owner}: нужен {req.name}{req.specifier}, установлен {installed['version']}")
        elif req.url:
            problems.add(f"{owner}: происхождение URL-зависимости {req.name} не проверено")

    for name, pkg in packages.items():
        for raw in pkg["requires_dist"]:
            check(raw, name, extras.get(name, set()))
    for raw in roots:
        check(raw, "requirements", set())
    return sorted(problems)


def project_report(snapshot: dict, report: dict, roots: list[str]) -> dict:
    """Reject replacements and new conflicts introduced by a pip installation plan."""
    if report.get("version") != "1" or not isinstance(report.get("install"), list):
        raise Uncheckable("Неподдерживаемый формат pip report")
    packages = dict(snapshot["packages"])
    additions = []
    for item in report["install"]:
        metadata = item["metadata"]
        name = canonicalize_name(metadata["name"])
        version = metadata["version"]
        if name in packages and packages[name]["version"] != version:
            raise Uncheckable(f"План заменяет защищённый пакет {name}")
        if item.get("is_yanked") or item.get("is_direct"):
            raise Uncheckable(f"Непроверенный источник пакета {name}")
        packages[name] = {"version": version, "requires_dist": metadata.get("requires_dist", [])}
        additions.append({"name": name, "version": version})
    baseline = snapshot["baseline"] if "baseline" in snapshot else dependency_conflicts(snapshot["packages"], snapshot["markers"])
    projected = dependency_conflicts(packages, snapshot["markers"], roots)
    return {"additions": additions, "baseline": baseline,
            "conflicts": sorted(set(projected) - set(baseline))}
