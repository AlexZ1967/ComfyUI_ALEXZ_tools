"""Distinguish merge blockers from generated files and Git EOL normalization."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from .dependencies import Uncheckable


def raw_git(repo: Path, *args: str) -> bytes:
    """Read Git data without text decoding or path quoting."""
    result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
                            capture_output=True, timeout=30, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    if result.returncode:
        raise Uncheckable(result.stderr.decode("utf-8", errors="replace")[-3000:])
    return result.stdout


def paths(repo: Path, *args: str) -> list[str]:
    """Decode NUL-delimited Git paths, including spaces and newlines."""
    return [os.fsdecode(item) for item in raw_git(repo, *args).split(b"\0") if item]


def ordinary_file(repo: Path, name: str) -> bool:
    """Reject symlinks and nonregular working files before reading or copying."""
    path = repo / name
    return path.is_file() and not any(part.is_symlink() for part in [path, *path.parents])


def inspect_worktree(repo: Path, before: str, target: str) -> dict:
    """Report actual fast-forward blockers without changing any repository files."""
    if raw_git(repo, "ls-files", "-u"):
        return {"blockers": ["Незавершённый Git merge"], "automatic": []}
    staged = paths(repo, "diff", "--cached", "--name-only", "-z")
    if staged:
        return {"blockers": [f"Изменения в Git index: {name}" for name in staged], "automatic": []}
    modified = paths(repo, "diff", "--name-only", "-z", before)
    automatic, blockers = [], []
    for name in modified:
        tree = raw_git(repo, "ls-tree", before, "--", name)
        mode = tree.split(b" ", 1)[0]
        safe = ordinary_file(repo, name) and mode in {b"100644", b"100755"}
        if safe:
            path = repo / name
            safe = bool(path.stat().st_mode & 0o111) == (mode == b"100755")
        if safe:
            old = raw_git(repo, "show", f"{before}:{name}")
            current = (repo / name).read_bytes()
            cache = "__pycache__" in Path(name).parts and name.endswith((".pyc", ".pyo"))
            eol_only = b"\0" not in old + current and old.replace(b"\r\n", b"\n") == current.replace(b"\r\n", b"\n")
            if cache or eol_only:
                automatic.append(name)
                continue
        # Реальный изменённый код/includes может отличаться от проверяемого commit.
        blockers.append(f"Изменено содержимое отслеживаемого файла: {name}")
    # Включаем ignored-файлы: Git по умолчанию может перезаписать их при merge.
    untracked = paths(repo, "ls-files", "--others", "-z")
    added = set(paths(repo, "diff", "--no-renames", "--name-only", "--diff-filter=A", "-z", before, target))
    for name in untracked:
        name = name.rstrip("/")
        if any(name == new or name.startswith(new + "/") or new.startswith(name + "/") for new in added):
            blockers.append(f"Локальный файл занимает путь обновления: {name}")
    return {"blockers": blockers, "automatic": automatic}


@contextmanager
def prepare_automatic_files(repo: Path, before: str, names: list[str], backup: Path):
    """Back up verified generated/EOL files and temporarily suppress EOL conversion."""
    if not names:
        yield
        return
    # info/attributes временно исправляет ошибочную нормализацию upstream без правки .gitattributes.
    attributes = Path(os.fsdecode(raw_git(repo, "rev-parse", "--git-path", "info/attributes").strip()))
    if not attributes.is_absolute():
        attributes = repo / attributes
    if attributes.is_symlink():
        raise Uncheckable("Git info/attributes является symlink")
    original = attributes.read_bytes() if attributes.exists() else None
    override = (original or b"") + b"\n" + "".join(json.dumps(name, ensure_ascii=False) + " -text\n" for name in names).encode()
    originals = {}
    head_files = {}
    for name in names:
        if not ordinary_file(repo, name):
            raise Uncheckable(f"Файл заменён во время проверки: {name}")
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, destination)
        originals[name] = (repo / name).read_bytes()
        head_files[name] = raw_git(repo, "show", f"{before}:{name}")
        cache = "__pycache__" in Path(name).parts and name.endswith((".pyc", ".pyo"))
        if not cache and originals[name].replace(b"\r\n", b"\n") != head_files[name].replace(b"\r\n", b"\n"):
            raise Uncheckable(f"Содержимое файла изменилось перед обновлением: {name}")
    attributes.parent.mkdir(parents=True, exist_ok=True)
    if original is not None:
        (backup / "git-info-attributes.backup").write_bytes(original)
    try:
        attributes.write_bytes(override)
        for name in names:
            (repo / name).write_bytes(head_files[name])
        yield
    finally:
        head = raw_git(repo, "rev-parse", "HEAD").decode().strip()
        changed = set(paths(repo, "diff", "--name-only", "-z", before, head))
        for name in names:
            if name not in changed and ordinary_file(repo, name) and (repo / name).read_bytes() == head_files[name]:
                (repo / name).write_bytes(originals[name])
        if attributes.read_bytes() != override:
            raise Uncheckable("Git info/attributes изменён извне; проверьте резервную копию")
        if original is None:
            attributes.unlink()
        else:
            attributes.write_bytes(original)
