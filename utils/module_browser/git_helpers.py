"""
Module: utils/module_browser/git_helpers.py
Author: AlexZ1967
Last updated: 2026-02-12

Description:
    Git helper functions extracted from Module Node Picker backend API.

Purpose:
    Keeps git state/sync logic in a focused reusable module while route-level
    API behavior remains unchanged through facade wrappers.
"""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any, Callable


def git_remote_names(repo_root: Path, *, run_git: Callable[[list[str], float], str | None]) -> list[str]:
    """Return configured git remote names for a repository."""
    out = run_git(["git", "-C", str(repo_root), "remote"], 2.0)
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def git_pick_remote(
    repo_root: Path,
    upstream: str | None,
    *,
    git_remote_names_fn: Callable[[Path], list[str]],
) -> str | None:
    """Pick preferred remote name: upstream remote, then origin, then upstream, then first."""
    upstream_text = (upstream or "").strip()
    if upstream_text and "/" in upstream_text:
        return upstream_text.split("/", 1)[0].strip() or None
    remotes = git_remote_names_fn(repo_root)
    if "origin" in remotes:
        return "origin"
    if "upstream" in remotes:
        return "upstream"
    return remotes[0] if remotes else None


def git_ref_exists(
    repo_root: Path,
    ref_name: str,
    *,
    run_git: Callable[[list[str], float], str | None],
) -> bool:
    """Check whether a git reference exists in repository."""
    ref = (ref_name or "").strip()
    if not ref:
        return False
    return bool(run_git(["git", "-C", str(repo_root), "rev-parse", "--verify", ref], 2.0))


def git_resolve_remote_ref(
    repo_root: Path,
    remote_name: str,
    branch_name: str | None,
    upstream: str | None,
    *,
    run_git: Callable[[list[str], float], str | None],
    git_ref_exists_fn: Callable[[Path, str], bool],
) -> tuple[str | None, str | None]:
    """Resolve tracking remote ref used for local-vs-remote comparison."""
    upstream_text = (upstream or "").strip()
    if upstream_text and "/" in upstream_text:
        remote_branch = upstream_text.split("/", 1)[1].strip()
        return (upstream_text, remote_branch or None)

    branch = (branch_name or "").strip()
    if branch and branch != "HEAD":
        by_branch = f"{remote_name}/{branch}"
        if git_ref_exists_fn(repo_root, by_branch):
            return (by_branch, branch)

    head_ref = run_git(
        ["git", "-C", str(repo_root), "symbolic-ref", "--quiet", f"refs/remotes/{remote_name}/HEAD"],
        2.0,
    )
    remote_branch = ""
    if head_ref:
        prefix = f"refs/remotes/{remote_name}/"
        if head_ref.startswith(prefix):
            remote_branch = head_ref[len(prefix) :].strip()

    if not remote_branch:
        remote_info = run_git(["git", "-C", str(repo_root), "remote", "show", remote_name], 8.0) or ""
        for line in remote_info.splitlines():
            text = line.strip()
            if text.lower().startswith("head branch:"):
                remote_branch = text.split(":", 1)[1].strip()
                break

    if not remote_branch:
        for candidate in ("main", "master"):
            ref = f"{remote_name}/{candidate}"
            if git_ref_exists_fn(repo_root, ref):
                remote_branch = candidate
                break

    if not remote_branch:
        return (None, None)
    return (f"{remote_name}/{remote_branch}", remote_branch)


def resolve_release_ref(
    repo_root: Path,
    remote_name: str,
    tag_name: str,
    *,
    run_git: Callable[[list[str], float], str | None],
    git_ref_exists_fn: Callable[[Path, str], bool],
) -> tuple[str | None, str]:
    """Resolve local git ref for release tag, fetching tag when needed."""
    tag_text = (tag_name or "").strip()
    if not tag_text:
        return (None, "")
    tag_ref = f"refs/tags/{tag_text}"
    if git_ref_exists_fn(repo_root, tag_ref):
        commit = run_git(["git", "-C", str(repo_root), "rev-list", "-n", "1", tag_ref], 2.0) or ""
        return (tag_ref, commit)

    run_git(["git", "-C", str(repo_root), "fetch", "--quiet", remote_name, "tag", tag_text], 20.0)
    if not git_ref_exists_fn(repo_root, tag_ref):
        run_git(["git", "-C", str(repo_root), "fetch", "--quiet", remote_name, "--tags"], 25.0)
    if not git_ref_exists_fn(repo_root, tag_ref):
        return (None, "")
    commit = run_git(["git", "-C", str(repo_root), "rev-list", "-n", "1", tag_ref], 2.0) or ""
    return (tag_ref, commit)


def module_repo_url(
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    custom_nodes_roots: Callable[[], list[Path]],
    run_git: Callable[[list[str], float], str | None],
    normalize_repo_url: Callable[[str | None], str | None],
) -> str | None:
    """Resolve custom-module repository URL from local git remotes."""
    module = canonical_custom_module_name((module_name or "").strip())
    if not module:
        return None
    for root in custom_nodes_roots():
        module_dir = root / module
        if not module_dir.exists():
            continue
        out = run_git(["git", "-C", str(module_dir), "config", "--get", "remote.origin.url"], 2.0)
        if out:
            return normalize_repo_url(out)
    return None


def module_worktree_signature(
    module_name: str,
    *,
    module_dir_resolver: Callable[[str], Path | None],
    run_git: Callable[[list[str], float], str | None],
) -> str:
    """Return short digest for local git worktree changes of custom module."""
    module_dir = module_dir_resolver(module_name)
    if module_dir is None:
        return ""
    is_git = run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"], 2.0)
    if is_git != "true":
        return ""
    status = run_git(["git", "-C", str(module_dir), "status", "--porcelain"], 2.0)
    if not status:
        return ""
    lines = sorted(line.strip() for line in status.splitlines() if line.strip())
    if not lines:
        return ""
    return sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def module_git_state(
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    custom_nodes_roots: Callable[[], list[Path]],
    run_git: Callable[[list[str], float], str | None],
    normalize_repo_url: Callable[[str | None], str | None],
    git_pick_remote_fn: Callable[[Path, str | None], str | None],
    git_resolve_remote_ref_fn: Callable[[Path, str, str | None, str | None], tuple[str | None, str | None]],
) -> dict[str, Any]:
    """Collect git state for one custom module from local and tracking refs."""
    module = canonical_custom_module_name((module_name or "").strip())
    if not module:
        return {}
    for root in custom_nodes_roots():
        module_dir = root / module
        if not module_dir.exists():
            continue
        is_git = run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"], 2.0)
        if is_git != "true":
            continue

        branch = run_git(["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "HEAD"], 2.0) or ""
        upstream = run_git(
            ["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            2.0,
        )
        remote_name = git_pick_remote_fn(module_dir, upstream)
        remote_ref = ""
        remote_branch = ""
        if remote_name:
            resolved_ref, resolved_branch = git_resolve_remote_ref_fn(module_dir, remote_name, branch, upstream)
            remote_ref = resolved_ref or ""
            remote_branch = resolved_branch or ""

        remote_repo_url = ""
        if remote_name:
            remote_repo_url = (
                run_git(["git", "-C", str(module_dir), "config", "--get", f"remote.{remote_name}.url"], 2.0) or ""
            )
        if not remote_repo_url:
            remote_repo_url = run_git(["git", "-C", str(module_dir), "config", "--get", "remote.origin.url"], 2.0) or ""

        remote_target = upstream or remote_ref
        remote_head = run_git(["git", "-C", str(module_dir), "rev-parse", remote_target], 2.0) if remote_target else None
        remote_updated_at = (
            run_git(["git", "-C", str(module_dir), "log", "-1", "--format=%cI", remote_target], 2.0)
            if remote_target
            else None
        )
        state: dict[str, Any] = {
            "module_path": str(module_dir),
            "repository": normalize_repo_url(remote_repo_url),
            "installed_commit": run_git(["git", "-C", str(module_dir), "rev-parse", "HEAD"], 2.0),
            "installed_updated_at": run_git(["git", "-C", str(module_dir), "log", "-1", "--format=%cI"], 2.0),
            "remote_updated_at": remote_updated_at,
            "branch": branch,
            "remote_name": remote_name or "",
            "remote_ref": remote_target,
            "remote_branch": remote_branch,
            "upstream": upstream,
            "has_upstream": bool(remote_target),
            "ahead": None,
            "behind": None,
            "remote_head": remote_head,
        }

        counts_target = f"HEAD...{remote_target}" if remote_target else ""
        counts = (
            run_git(["git", "-C", str(module_dir), "rev-list", "--left-right", "--count", counts_target], 2.0)
            if counts_target
            else None
        )
        if counts:
            parts = counts.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                state["ahead"] = int(parts[0])
                state["behind"] = int(parts[1])

        return state
    return {}


def sync_module_upstream(
    module_name: str,
    *,
    canonical_custom_module_name: Callable[[str], str],
    custom_nodes_roots: Callable[[], list[Path]],
    run_git: Callable[[list[str], float], str | None],
    git_pick_remote_fn: Callable[[Path, str | None], str | None],
    bootstrap_module_remote_fn: Callable[[str, Path], bool],
    timeout: float = 15.0,
) -> bool:
    """Fetch module upstream refs to refresh local tracking state."""
    module = canonical_custom_module_name((module_name or "").strip())
    if not module:
        return False
    for root in custom_nodes_roots():
        module_dir = root / module
        if not module_dir.exists():
            continue
        is_git = run_git(["git", "-C", str(module_dir), "rev-parse", "--is-inside-work-tree"], 2.0)
        if is_git != "true":
            continue
        upstream = run_git(
            ["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            2.0,
        )
        remote_name = git_pick_remote_fn(module_dir, upstream)
        if not remote_name and bootstrap_module_remote_fn(module, module_dir):
            upstream = run_git(
                ["git", "-C", str(module_dir), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
                2.0,
            )
            remote_name = git_pick_remote_fn(module_dir, upstream)
        if not remote_name:
            return False
        run_git(["git", "-C", str(module_dir), "fetch", "--quiet", remote_name], timeout)
        return True
    return False

