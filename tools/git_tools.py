"""
tools/git_tools.py
===================

Git integration tools: git_status, git_diff, git_log, git_commit,
git_branch.

All functions shell out to the system `git` binary (no GitPython
dependency, keeping the install footprint small for Termux) and return
structured ToolResult objects, consistent with the rest of the tools
package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from tools.filesystem import ToolResult


def _run_git(args: List[str], cwd: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run a git subcommand and return the completed process (no shell=True
    needed since args are passed as a list, avoiding shell injection risk)."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _is_git_repo(cwd: str) -> bool:
    return (Path(cwd) / ".git").exists()


def git_status(project_root: str) -> ToolResult:
    """Return `git status --porcelain` plus a human-readable summary."""
    if not _is_git_repo(project_root):
        return ToolResult(success=False, tool="git_status", error="Not a git repository (no .git directory found).")

    try:
        proc = _run_git(["status", "--porcelain=v1", "--branch"], cwd=project_root)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return ToolResult(success=False, tool="git_status", error=f"git status failed: {exc}")

    if proc.returncode != 0:
        return ToolResult(success=False, tool="git_status", error=proc.stderr.strip() or "git status failed")

    lines = proc.stdout.splitlines()
    branch_line = lines[0] if lines and lines[0].startswith("##") else ""
    file_lines = [l for l in lines if not l.startswith("##")]

    changes = []
    for line in file_lines:
        if len(line) < 3:
            continue
        status_code = line[:2].strip()
        filepath = line[3:]
        changes.append({"status": status_code, "path": filepath})

    return ToolResult(
        success=True,
        tool="git_status",
        data={"branch_info": branch_line.lstrip("# ").strip(), "changes": changes, "clean": len(changes) == 0},
        message=f"{len(changes)} changed file(s)" if changes else "Working tree clean",
    )


def git_diff(project_root: str, path: Optional[str] = None, staged: bool = False) -> ToolResult:
    """Return the unified diff of working tree (or staged) changes.

    Args:
        path: Optional path to limit the diff to a single file/directory.
        staged: If True, show staged changes (`git diff --cached`)
            instead of unstaged working-tree changes.
    """
    if not _is_git_repo(project_root):
        return ToolResult(success=False, tool="git_diff", error="Not a git repository (no .git directory found).")

    args = ["diff", "--cached"] if staged else ["diff"]
    if path:
        args.append("--")
        args.append(path)

    try:
        proc = _run_git(args, cwd=project_root)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return ToolResult(success=False, tool="git_diff", error=f"git diff failed: {exc}")

    if proc.returncode != 0:
        return ToolResult(success=False, tool="git_diff", error=proc.stderr.strip() or "git diff failed")

    return ToolResult(
        success=True,
        tool="git_diff",
        data={"diff": proc.stdout, "has_changes": bool(proc.stdout.strip())},
        message="Diff retrieved" if proc.stdout.strip() else "No differences found",
    )


def git_log(project_root: str, max_count: int = 20, path: Optional[str] = None) -> ToolResult:
    """Return recent commit history as structured entries."""
    if not _is_git_repo(project_root):
        return ToolResult(success=False, tool="git_log", error="Not a git repository (no .git directory found).")

    # Use a unit-separator-delimited format string so we can split commit
    # fields reliably even if the commit message itself contains pipes.
    sep = "\x1f"
    fmt = f"%H{sep}%an{sep}%ad{sep}%s"
    args = ["log", f"--max-count={max_count}", f"--pretty=format:{fmt}", "--date=short"]
    if path:
        args.append("--")
        args.append(path)

    try:
        proc = _run_git(args, cwd=project_root)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return ToolResult(success=False, tool="git_log", error=f"git log failed: {exc}")

    if proc.returncode != 0:
        return ToolResult(success=False, tool="git_log", error=proc.stderr.strip() or "git log failed")

    commits = []
    for line in proc.stdout.splitlines():
        parts = line.split(sep)
        if len(parts) == 4:
            commits.append(
                {
                    "hash": parts[0][:10],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                }
            )

    return ToolResult(
        success=True,
        tool="git_log",
        data={"commits": commits, "count": len(commits)},
        message=f"Retrieved {len(commits)} commit(s)",
    )


def git_commit(project_root: str, message: str, add_all: bool = False) -> ToolResult:
    """Create a git commit. Optionally stages all changes first (`git add -A`).

    This does not require the safety-confirmation flow used by destructive
    filesystem operations, since git commits are non-destructive and easily
    reversible (git revert / reset), but the executor may still choose to
    confirm before calling this if desired.
    """
    if not _is_git_repo(project_root):
        return ToolResult(success=False, tool="git_commit", error="Not a git repository (no .git directory found).")

    if not message or not message.strip():
        return ToolResult(success=False, tool="git_commit", error="Commit message cannot be empty.")

    if add_all:
        try:
            add_proc = _run_git(["add", "-A"], cwd=project_root)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            return ToolResult(success=False, tool="git_commit", error=f"git add failed: {exc}")
        if add_proc.returncode != 0:
            return ToolResult(success=False, tool="git_commit", error=add_proc.stderr.strip() or "git add failed")

    try:
        proc = _run_git(["commit", "-m", message], cwd=project_root)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return ToolResult(success=False, tool="git_commit", error=f"git commit failed: {exc}")

    if proc.returncode != 0:
        # Common case: nothing to commit. Surface stdout too, since git
        # often reports "nothing to commit" on stdout, not stderr.
        combined = (proc.stderr.strip() or proc.stdout.strip() or "git commit failed")
        return ToolResult(success=False, tool="git_commit", error=combined)

    return ToolResult(
        success=True,
        tool="git_commit",
        data={"output": proc.stdout.strip()},
        message=f"Committed: {message}",
    )


def git_branch(project_root: str, create: Optional[str] = None) -> ToolResult:
    """List branches, or create+switch to a new branch if `create` is given."""
    if not _is_git_repo(project_root):
        return ToolResult(success=False, tool="git_branch", error="Not a git repository (no .git directory found).")

    if create:
        try:
            proc = _run_git(["checkout", "-b", create], cwd=project_root)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            return ToolResult(success=False, tool="git_branch", error=f"git checkout -b failed: {exc}")
        if proc.returncode != 0:
            return ToolResult(success=False, tool="git_branch", error=proc.stderr.strip() or "Branch creation failed")
        return ToolResult(
            success=True,
            tool="git_branch",
            data={"created": create, "current": create},
            message=f"Created and switched to branch '{create}'",
        )

    try:
        proc = _run_git(["branch", "--list"], cwd=project_root)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return ToolResult(success=False, tool="git_branch", error=f"git branch failed: {exc}")

    if proc.returncode != 0:
        return ToolResult(success=False, tool="git_branch", error=proc.stderr.strip() or "git branch failed")

    branches = []
    current = None
    for line in proc.stdout.splitlines():
        name = line.strip().lstrip("* ").strip()
        is_current = line.strip().startswith("*")
        if is_current:
            current = name
        if name:
            branches.append(name)

    return ToolResult(
        success=True,
        tool="git_branch",
        data={"branches": branches, "current": current},
        message=f"{len(branches)} branch(es); current: {current}",
    )
