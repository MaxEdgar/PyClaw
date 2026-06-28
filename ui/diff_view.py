"""
ui/diff_view.py
=================

Unified diff generation and rendering for the patch approval system.

Before PyClaw ever modifies an existing file, it generates a unified
diff (old content vs proposed new content) and shows it to the user in a
syntax-highlighted, color-coded view, requiring explicit Y/N approval.
Files are never modified automatically -- this module is the single choke
point through which all edits to existing files must pass.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from tools.safety import InputFunc
from ui.glyphs import safe_box

console = Console()


@dataclass
class Patch:
    """A proposed change to a single file."""

    path: str
    old_content: str
    new_content: str
    is_new_file: bool = False

    @property
    def unified_diff(self) -> str:
        """Compute the unified diff text between old and new content."""
        old_lines = self.old_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)
        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{self.path}",
            tofile=f"b/{self.path}",
            lineterm="",
        )
        return "\n".join(diff_lines)

    @property
    def has_changes(self) -> bool:
        return self.old_content != self.new_content

    @property
    def stats(self) -> tuple:
        """Return (lines_added, lines_removed) counts."""
        added = 0
        removed = 0
        for line in self.unified_diff.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        return added, removed


def render_diff(patch: Patch) -> None:
    """Print a color-coded unified diff panel for the given patch."""
    if patch.is_new_file:
        console.print(
            Panel(
                Syntax(patch.new_content, _guess_lexer(patch.path), theme="ansi_dark", line_numbers=True),
                title=f"[bold green]New file: {patch.path}[/bold green]",
                border_style="green",
                box=safe_box(),
            )
        )
        return

    if not patch.has_changes:
        console.print(Panel(f"No changes to {patch.path}.", border_style="yellow", box=safe_box()))
        return

    diff_text = patch.unified_diff
    added, removed = patch.stats

    body = Text()
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            body.append(line + "\n", style="bold cyan")
        elif line.startswith("@@"):
            body.append(line + "\n", style="bold magenta")
        elif line.startswith("+"):
            body.append(line + "\n", style="green")
        elif line.startswith("-"):
            body.append(line + "\n", style="red")
        else:
            body.append(line + "\n", style="dim")

    console.print(
        Panel(
            body,
            title=f"[bold]Patch: {patch.path}[/bold]  "
            f"([green]+{added}[/green] / [red]-{removed}[/red])",
            border_style="blue",
            box=safe_box(),
        )
    )


def _guess_lexer(path: str) -> str:
    """Best-effort lexer name guess from a file extension, for syntax
    highlighting new-file previews. Falls back to plain text."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".bash": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".html": "html",
        ".css": "css",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".sql": "sql",
    }
    for ext, lexer in ext_map.items():
        if path.lower().endswith(ext):
            return lexer
    return "text"


def approve_patch(patch: Patch, input_func: Optional[InputFunc] = None) -> bool:
    """Display the patch and block for Y/N approval.

    This is the ONLY function in PyClaw that grants permission to
    write changes to an *existing* file. New files created via write_file
    do not need this (nothing is destroyed), but any modification to
    existing content must flow through here first.
    """
    if patch.is_new_file:
        prompt_label = "Create this file?"
    elif not patch.has_changes:
        return False  # nothing to approve -- no-op patch
    else:
        prompt_label = "Approve patch?"

    if input_func is None:
        # Default console path: render the diff to the terminal, then
        # block on stdin for the Y/N answer.
        render_diff(patch)
        console.print(f"\n[bold cyan]{prompt_label}[/bold cyan] [Y] Yes   [N] No")
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # Custom input_func path (e.g. the Textual TUI's modal bridge): build a
    # plain-text rendering of the diff (no Rich markup) so it can be shown
    # verbatim in a non-console widget, and pass it through input_func.
    plain_diff = patch.new_content if patch.is_new_file else patch.unified_diff
    added, removed = (0, 0) if patch.is_new_file else patch.stats
    header = f"{prompt_label}\n\nFile: {patch.path}"
    if not patch.is_new_file:
        header += f"  (+{added} / -{removed})"
    full_prompt = f"{header}\n\n{plain_diff}\n\n[Y]es / [N]o"
    try:
        answer = input_func(full_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")
