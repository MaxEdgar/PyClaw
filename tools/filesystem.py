"""
tools/filesystem.py
====================

Filesystem tools exposed to the agent: read_file, write_file, append_file,
delete_file, move_file, copy_file, list_directory, create_directory, and
file_info.

Every function in this module:
    * Validates and resolves paths (rejecting paths that escape the
      configured project root, to keep the agent sandboxed).
    * Catches and translates exceptions into structured ToolResult objects
      instead of raising, so the agent loop can always inspect `.success`.
    * Returns a structured result rather than printing directly -- the UI
      layer decides how to render results.

All paths accepted by these functions may be relative (resolved against the
project root) or absolute; absolute paths are still required to live inside
the project root unless the path is explicitly outside the sandbox and the
caller has disabled sandboxing (not exposed to the LLM by default).
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.safety import warn_delete, warn_overwrite


@dataclass
class ToolResult:
    """Uniform result wrapper returned by every filesystem tool function."""

    success: bool
    tool: str
    data: Any = None
    error: Optional[str] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
            "message": self.message,
        }


class PathSandboxError(Exception):
    """Raised when a requested path escapes the allowed project root."""


def _resolve(project_root: str, path: str) -> Path:
    """Resolve `path` against `project_root`, refusing to escape the root.

    A relative path is joined to the project root. An absolute path must
    still be located inside the project root. This prevents the agent
    (or a malicious prompt-injected tool call) from reading or writing
    arbitrary system files like /etc/passwd or ~/.ssh/id_rsa.
    """
    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError:
        raise PathSandboxError(
            f"Path '{path}' resolves outside the project root '{root}'. "
            "Refusing for safety."
        )
    return resolved


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '12.3 KB')."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


# ----------------------------------------------------------------------
# read_file
# ----------------------------------------------------------------------
def read_file(
    project_root: str,
    path: str,
    max_bytes: int = 200_000,
    encoding: str = "utf-8",
) -> ToolResult:
    """Read a text file's contents, truncating if it exceeds max_bytes."""
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="read_file", error=str(exc))

    if not resolved.exists():
        return ToolResult(success=False, tool="read_file", error=f"File not found: {path}")
    if resolved.is_dir():
        return ToolResult(success=False, tool="read_file", error=f"Path is a directory, not a file: {path}")

    try:
        raw = resolved.read_bytes()
    except PermissionError:
        return ToolResult(success=False, tool="read_file", error=f"Permission denied: {path}")
    except OSError as exc:
        return ToolResult(success=False, tool="read_file", error=f"OS error reading {path}: {exc}")

    truncated = len(raw) > max_bytes
    content_bytes = raw[:max_bytes] if truncated else raw

    try:
        content = content_bytes.decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError) as exc:
        return ToolResult(success=False, tool="read_file", error=f"Could not decode file: {exc}")

    return ToolResult(
        success=True,
        tool="read_file",
        data={
            "path": str(resolved),
            "content": content,
            "truncated": truncated,
            "total_bytes": len(raw),
        },
        message=f"Read {len(content_bytes)} bytes from {path}" + (" (truncated)" if truncated else ""),
    )


# ----------------------------------------------------------------------
# write_file
# ----------------------------------------------------------------------
def write_file(
    project_root: str,
    path: str,
    content: str,
    require_confirmation: bool = True,
    confirmed: bool = False,
    encoding: str = "utf-8",
) -> ToolResult:
    """Write (overwrite) a file's contents.

    If the file already exists and `require_confirmation` is True, the
    caller must pass `confirmed=True` (i.e. the executor must have already
    obtained approval via the patch/diff approval flow) or this function
    will refuse and report that confirmation is required. New file
    creation does not require confirmation since nothing is destroyed.
    """
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="write_file", error=str(exc))

    file_exists = resolved.exists()
    if file_exists and require_confirmation and not confirmed:
        return ToolResult(
            success=False,
            tool="write_file",
            error="confirmation_required",
            message=f"Overwriting '{path}' requires user approval before proceeding.",
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
    except PermissionError:
        return ToolResult(success=False, tool="write_file", error=f"Permission denied: {path}")
    except OSError as exc:
        return ToolResult(success=False, tool="write_file", error=f"OS error writing {path}: {exc}")

    return ToolResult(
        success=True,
        tool="write_file",
        data={"path": str(resolved), "bytes_written": len(content.encode(encoding))},
        message=f"{'Overwrote' if file_exists else 'Created'} {path}",
    )


# ----------------------------------------------------------------------
# append_file
# ----------------------------------------------------------------------
def append_file(project_root: str, path: str, content: str, encoding: str = "utf-8") -> ToolResult:
    """Append text to a file, creating it if it does not exist."""
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="append_file", error=str(exc))

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "a", encoding=encoding) as fh:
            fh.write(content)
    except PermissionError:
        return ToolResult(success=False, tool="append_file", error=f"Permission denied: {path}")
    except OSError as exc:
        return ToolResult(success=False, tool="append_file", error=f"OS error appending to {path}: {exc}")

    return ToolResult(
        success=True,
        tool="append_file",
        data={"path": str(resolved), "bytes_appended": len(content.encode(encoding))},
        message=f"Appended {len(content)} characters to {path}",
    )


# ----------------------------------------------------------------------
# delete_file
# ----------------------------------------------------------------------
def delete_file(
    project_root: str,
    path: str,
    require_confirmation: bool = True,
    confirmed: bool = False,
) -> ToolResult:
    """Delete a file or directory (recursively for directories).

    Requires explicit confirmation by default. The executor is expected to
    call `tools.safety.warn_delete` (directly, via the UI) and only pass
    `confirmed=True` once the user has approved.
    """
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="delete_file", error=str(exc))

    if not resolved.exists():
        return ToolResult(success=False, tool="delete_file", error=f"Path not found: {path}")

    if require_confirmation and not confirmed:
        return ToolResult(
            success=False,
            tool="delete_file",
            error="confirmation_required",
            message=f"Deleting '{path}' requires user approval before proceeding.",
        )

    try:
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
    except PermissionError:
        return ToolResult(success=False, tool="delete_file", error=f"Permission denied: {path}")
    except OSError as exc:
        return ToolResult(success=False, tool="delete_file", error=f"OS error deleting {path}: {exc}")

    return ToolResult(success=True, tool="delete_file", data={"path": str(resolved)}, message=f"Deleted {path}")


# ----------------------------------------------------------------------
# move_file
# ----------------------------------------------------------------------
def move_file(
    project_root: str,
    src: str,
    dst: str,
    require_confirmation: bool = True,
    confirmed: bool = False,
) -> ToolResult:
    """Move/rename a file or directory from src to dst."""
    try:
        resolved_src = _resolve(project_root, src)
        resolved_dst = _resolve(project_root, dst)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="move_file", error=str(exc))

    if not resolved_src.exists():
        return ToolResult(success=False, tool="move_file", error=f"Source not found: {src}")

    if resolved_dst.exists() and require_confirmation and not confirmed:
        return ToolResult(
            success=False,
            tool="move_file",
            error="confirmation_required",
            message=f"Destination '{dst}' already exists and would be overwritten.",
        )

    try:
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved_src), str(resolved_dst))
    except PermissionError:
        return ToolResult(success=False, tool="move_file", error=f"Permission denied moving {src} -> {dst}")
    except OSError as exc:
        return ToolResult(success=False, tool="move_file", error=f"OS error moving {src} -> {dst}: {exc}")

    return ToolResult(
        success=True,
        tool="move_file",
        data={"src": str(resolved_src), "dst": str(resolved_dst)},
        message=f"Moved {src} -> {dst}",
    )


# ----------------------------------------------------------------------
# copy_file
# ----------------------------------------------------------------------
def copy_file(
    project_root: str,
    src: str,
    dst: str,
    require_confirmation: bool = True,
    confirmed: bool = False,
) -> ToolResult:
    """Copy a file or directory tree from src to dst."""
    try:
        resolved_src = _resolve(project_root, src)
        resolved_dst = _resolve(project_root, dst)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="copy_file", error=str(exc))

    if not resolved_src.exists():
        return ToolResult(success=False, tool="copy_file", error=f"Source not found: {src}")

    if resolved_dst.exists() and require_confirmation and not confirmed:
        return ToolResult(
            success=False,
            tool="copy_file",
            error="confirmation_required",
            message=f"Destination '{dst}' already exists and would be overwritten.",
        )

    try:
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        if resolved_src.is_dir():
            shutil.copytree(resolved_src, resolved_dst, dirs_exist_ok=True)
        else:
            shutil.copy2(resolved_src, resolved_dst)
    except PermissionError:
        return ToolResult(success=False, tool="copy_file", error=f"Permission denied copying {src} -> {dst}")
    except OSError as exc:
        return ToolResult(success=False, tool="copy_file", error=f"OS error copying {src} -> {dst}: {exc}")

    return ToolResult(
        success=True,
        tool="copy_file",
        data={"src": str(resolved_src), "dst": str(resolved_dst)},
        message=f"Copied {src} -> {dst}",
    )


# ----------------------------------------------------------------------
# list_directory
# ----------------------------------------------------------------------
def list_directory(
    project_root: str,
    path: str = ".",
    show_hidden: bool = False,
) -> ToolResult:
    """List immediate contents of a directory with basic metadata."""
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="list_directory", error=str(exc))

    if not resolved.exists():
        return ToolResult(success=False, tool="list_directory", error=f"Path not found: {path}")
    if not resolved.is_dir():
        return ToolResult(success=False, tool="list_directory", error=f"Path is not a directory: {path}")

    entries: List[Dict[str, Any]] = []
    try:
        for entry in sorted(resolved.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if not show_hidden and entry.name.startswith("."):
                continue
            try:
                st = entry.stat()
                entries.append(
                    {
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size": st.st_size,
                        "size_human": _human_size(st.st_size),
                        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    }
                )
            except OSError:
                # Broken symlink or permission issue on a single entry --
                # skip it rather than failing the whole listing.
                continue
    except PermissionError:
        return ToolResult(success=False, tool="list_directory", error=f"Permission denied: {path}")

    return ToolResult(
        success=True,
        tool="list_directory",
        data={"path": str(resolved), "entries": entries, "count": len(entries)},
        message=f"Listed {len(entries)} entries in {path}",
    )


# ----------------------------------------------------------------------
# create_directory
# ----------------------------------------------------------------------
def create_directory(project_root: str, path: str) -> ToolResult:
    """Create a directory (and any missing parents)."""
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="create_directory", error=str(exc))

    if resolved.exists():
        if resolved.is_dir():
            return ToolResult(
                success=True,
                tool="create_directory",
                data={"path": str(resolved)},
                message=f"Directory already exists: {path}",
            )
        return ToolResult(success=False, tool="create_directory", error=f"Path exists and is a file: {path}")

    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return ToolResult(success=False, tool="create_directory", error=f"Permission denied: {path}")
    except OSError as exc:
        return ToolResult(success=False, tool="create_directory", error=f"OS error creating {path}: {exc}")

    return ToolResult(success=True, tool="create_directory", data={"path": str(resolved)}, message=f"Created directory {path}")


# ----------------------------------------------------------------------
# file_info
# ----------------------------------------------------------------------
def file_info(project_root: str, path: str) -> ToolResult:
    """Return metadata about a file or directory (size, permissions, mtime)."""
    try:
        resolved = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="file_info", error=str(exc))

    if not resolved.exists():
        return ToolResult(success=False, tool="file_info", error=f"Path not found: {path}")

    try:
        st = resolved.stat()
    except OSError as exc:
        return ToolResult(success=False, tool="file_info", error=f"OS error stating {path}: {exc}")

    info = {
        "path": str(resolved),
        "is_dir": resolved.is_dir(),
        "is_file": resolved.is_file(),
        "is_symlink": resolved.is_symlink(),
        "size": st.st_size,
        "size_human": _human_size(st.st_size),
        "permissions": stat.filemode(st.st_mode),
        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_ctime)),
    }
    return ToolResult(success=True, tool="file_info", data=info, message=f"Retrieved info for {path}")
