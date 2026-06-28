"""
tools/search.py
================

Search and project-indexing tools: search_files, grep_text, find_extensions,
find_large_files, and project_summary.

All functions operate recursively from the project root (or a given
sub-path), skip the configured excluded directories (e.g. .git,
node_modules, __pycache__), and return structured results rather than
printing -- consistent with tools/filesystem.py.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.filesystem import PathSandboxError, ToolResult, _resolve, _human_size

# Reasonable default cap on matches returned, so a runaway grep over a huge
# repo doesn't blow the model's context window.
DEFAULT_MAX_RESULTS = 200

# Files we never attempt to treat as text (binary-ish extensions), used by
# grep_text to skip files that would just produce garbage matches.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a",
    ".pyc", ".pyo", ".class", ".jar",
    ".sqlite", ".db", ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
}


def _iter_files(root: Path, excluded_dirs: Tuple[str, ...]) -> List[Path]:
    """Walk `root` recursively, yielding file paths and pruning excluded dirs.

    Pruning excluded directories *during* the walk (rather than filtering
    afterward) is important for performance on large repos with huge
    node_modules / .git directories.
    """
    results: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs and not d.startswith(".git")]
        for fname in filenames:
            results.append(Path(dirpath) / fname)
    return results


# ----------------------------------------------------------------------
# search_files: find files by glob-style name pattern
# ----------------------------------------------------------------------
def search_files(
    project_root: str,
    pattern: str,
    path: str = ".",
    excluded_dirs: Tuple[str, ...] = (),
    max_results: int = DEFAULT_MAX_RESULTS,
) -> ToolResult:
    """Find files whose name matches a glob pattern (e.g. '*.py', 'test_*').

    Matching is case-insensitive and applied to the filename only (not the
    full path), which matches the intuitive behavior of tools like `find
    -iname`.
    """
    try:
        resolved_root = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="search_files", error=str(exc))

    if not resolved_root.exists():
        return ToolResult(success=False, tool="search_files", error=f"Path not found: {path}")

    matches: List[str] = []
    project_path = Path(project_root).expanduser().resolve()
    for file_path in _iter_files(resolved_root, excluded_dirs):
        if fnmatch.fnmatch(file_path.name.lower(), pattern.lower()):
            try:
                rel = file_path.relative_to(project_path)
            except ValueError:
                rel = file_path
            matches.append(str(rel))
            if len(matches) >= max_results:
                break

    return ToolResult(
        success=True,
        tool="search_files",
        data={
            "pattern": pattern,
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= max_results,
        },
        message=f"Found {len(matches)} file(s) matching '{pattern}'",
    )


# ----------------------------------------------------------------------
# grep_text: search file contents for a regex or plain substring
# ----------------------------------------------------------------------
def grep_text(
    project_root: str,
    query: str,
    path: str = ".",
    excluded_dirs: Tuple[str, ...] = (),
    regex: bool = False,
    case_sensitive: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
    context_lines: int = 0,
) -> ToolResult:
    """Search file contents recursively for `query`, returning matching lines.

    Args:
        query: Plain substring or regular expression to search for.
        regex: If True, treat `query` as a regular expression; otherwise
            it is matched as a literal substring (escaped internally).
        case_sensitive: Whether matching is case sensitive.
        context_lines: Number of lines of context to include before/after
            each match.
    """
    try:
        resolved_root = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="grep_text", error=str(exc))

    if not resolved_root.exists():
        return ToolResult(success=False, tool="grep_text", error=f"Path not found: {path}")

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(query if regex else re.escape(query), flags)
    except re.error as exc:
        return ToolResult(success=False, tool="grep_text", error=f"Invalid regular expression: {exc}")

    project_path = Path(project_root).expanduser().resolve()
    matches: List[Dict[str, Any]] = []

    for file_path in _iter_files(resolved_root, excluded_dirs):
        if file_path.suffix.lower() in BINARY_EXTENSIONS:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue

        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if compiled.search(line):
                try:
                    rel = file_path.relative_to(project_path)
                except ValueError:
                    rel = file_path

                start = max(0, idx - context_lines)
                end = min(len(lines), idx + context_lines + 1)
                snippet = "\n".join(lines[start:end])

                matches.append(
                    {
                        "file": str(rel),
                        "line_number": idx + 1,
                        "line": line.strip(),
                        "context": snippet if context_lines else None,
                    }
                )
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    return ToolResult(
        success=True,
        tool="grep_text",
        data={
            "query": query,
            "matches": matches,
            "count": len(matches),
            "truncated": len(matches) >= max_results,
        },
        message=f"Found {len(matches)} match(es) for '{query}'",
    )


# ----------------------------------------------------------------------
# find_extensions: count files grouped by extension
# ----------------------------------------------------------------------
def find_extensions(
    project_root: str,
    path: str = ".",
    excluded_dirs: Tuple[str, ...] = (),
) -> ToolResult:
    """Count files by extension, useful for quickly understanding a repo's
    language composition (e.g. {'.py': 42, '.md': 5, '.json': 3})."""
    try:
        resolved_root = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="find_extensions", error=str(exc))

    if not resolved_root.exists():
        return ToolResult(success=False, tool="find_extensions", error=f"Path not found: {path}")

    counter: Counter = Counter()
    for file_path in _iter_files(resolved_root, excluded_dirs):
        ext = file_path.suffix.lower() or "(no extension)"
        counter[ext] += 1

    breakdown = [{"extension": ext, "count": count} for ext, count in counter.most_common()]

    return ToolResult(
        success=True,
        tool="find_extensions",
        data={"breakdown": breakdown, "total_files": sum(counter.values())},
        message=f"Found {len(breakdown)} distinct extension(s) across {sum(counter.values())} files",
    )


# ----------------------------------------------------------------------
# find_large_files: list files above a size threshold
# ----------------------------------------------------------------------
def find_large_files(
    project_root: str,
    path: str = ".",
    excluded_dirs: Tuple[str, ...] = (),
    min_size_bytes: int = 1_000_000,
    max_results: int = 50,
) -> ToolResult:
    """Find files larger than `min_size_bytes`, sorted largest first."""
    try:
        resolved_root = _resolve(project_root, path)
    except PathSandboxError as exc:
        return ToolResult(success=False, tool="find_large_files", error=str(exc))

    if not resolved_root.exists():
        return ToolResult(success=False, tool="find_large_files", error=f"Path not found: {path}")

    project_path = Path(project_root).expanduser().resolve()
    sized: List[Tuple[Path, int]] = []
    for file_path in _iter_files(resolved_root, excluded_dirs):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size >= min_size_bytes:
            sized.append((file_path, size))

    sized.sort(key=lambda item: item[1], reverse=True)
    sized = sized[:max_results]

    results = []
    for file_path, size in sized:
        try:
            rel = file_path.relative_to(project_path)
        except ValueError:
            rel = file_path
        results.append({"path": str(rel), "size": size, "size_human": _human_size(size)})

    return ToolResult(
        success=True,
        tool="find_large_files",
        data={"files": results, "count": len(results)},
        message=f"Found {len(results)} file(s) >= {_human_size(min_size_bytes)}",
    )


# ----------------------------------------------------------------------
# project_summary: high-level overview of the codebase
# ----------------------------------------------------------------------
def project_summary(
    project_root: str,
    excluded_dirs: Tuple[str, ...] = (),
    max_top_files: int = 10,
) -> ToolResult:
    """Produce a high-level summary of the project: file/dir counts,
    extension breakdown, total size, and notable top-level entries.

    This is the tool the planner reaches for first on a fresh session to
    orient itself before diving into specific files.
    """
    root = Path(project_root).expanduser().resolve()
    if not root.exists():
        return ToolResult(success=False, tool="project_summary", error=f"Project root not found: {project_root}")

    all_files = _iter_files(root, excluded_dirs)
    total_size = 0
    ext_counter: Counter = Counter()
    for f in all_files:
        try:
            total_size += f.stat().st_size
        except OSError:
            continue
        ext_counter[f.suffix.lower() or "(no extension)"] += 1

    top_level_entries = []
    try:
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if entry.name in excluded_dirs or entry.name.startswith("."):
                continue
            top_level_entries.append(entry.name + ("/" if entry.is_dir() else ""))
    except PermissionError:
        pass

    # Heuristic detection of common project markers, to help the model
    # quickly recognize the tech stack without reading every file.
    markers = {
        "Python": ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"],
        "Node.js": ["package.json"],
        "Rust": ["Cargo.toml"],
        "Go": ["go.mod"],
        "Java/Gradle": ["build.gradle", "pom.xml"],
        "Docker": ["Dockerfile", "docker-compose.yml"],
        "Git": [".git"],
    }
    detected_stack = [
        name for name, files in markers.items() if any((root / f).exists() for f in files)
    ]

    summary = {
        "project_root": str(root),
        "total_files": len(all_files),
        "total_size": total_size,
        "total_size_human": _human_size(total_size),
        "extension_breakdown": [{"extension": e, "count": c} for e, c in ext_counter.most_common(max_top_files)],
        "top_level_entries": top_level_entries,
        "detected_stack": detected_stack,
    }

    return ToolResult(
        success=True,
        tool="project_summary",
        data=summary,
        message=f"Project has {len(all_files)} files ({_human_size(total_size)}) across {len(ext_counter)} extension types",
    )
