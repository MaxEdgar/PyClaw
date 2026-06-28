"""
tools/safety.py
================

Centralized safety logic for PyClaw.

This module is the single source of truth for:
    * Detecting "dangerous" shell commands (rm, mv, chmod, sudo, dd, etc.)
    * Detecting destructive filesystem operations (delete, overwrite)
    * Producing the actual Y/N confirmation prompts shown to the user

Keeping this logic in one place means every tool (shell, filesystem) that
needs to ask "are you sure?" behaves consistently, and the dangerous-command
patterns only need to be maintained in one location.

Nothing in this module ever bypasses confirmation automatically -- the
default posture is "ask unless explicitly told not to".
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Callable, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ui.glyphs import safe_box

console = Console()

# ----------------------------------------------------------------------
# Dangerous shell command detection
# ----------------------------------------------------------------------

# Command names that are considered dangerous regardless of arguments.
# These are matched against the *first token* of each pipeline segment so
# that `foo | rm -rf bar` is also caught, not just literal leading `rm`.
DANGEROUS_COMMANDS = {
    "rm",
    "mv",
    "chmod",
    "chown",
    "sudo",
    "dd",
    "mkfs",
    "fdisk",
    "shred",
    "kill",
    "killall",
    "reboot",
    "shutdown",
    "poweroff",
    "format",
    "diskutil",
    "parted",
}

# Regex patterns that flag dangerous *intent* even when the command name
# itself looks innocuous (e.g. piping into bash, redirecting over a device).
DANGEROUS_PATTERNS = [
    re.compile(r">\s*/dev/"),          # writing directly to a device node
    re.compile(r":\(\)\s*\{.*\};\s*:"),  # fork-bomb pattern
    re.compile(r"rm\s+-[a-z]*r[a-z]*f"),  # rm -rf / -fr in any order
    re.compile(r"curl[^|]*\|\s*(sh|bash)"),  # curl | sh remote execution
    re.compile(r"wget[^|]*\|\s*(sh|bash)"),  # wget | sh remote execution
]


@dataclass
class SafetyVerdict:
    """Result of a safety check on a command or filesystem action."""

    dangerous: bool
    reason: str = ""


def _split_pipeline_segments(command: str) -> List[str]:
    """Split a shell command on pipe/semicolon/&& boundaries.

    This is a best-effort, non-executing split used purely for safety
    scanning -- it does not need to be a fully correct shell parser, just
    good enough to find the leading command word of each clause.
    """
    parts = re.split(r"\|\||&&|[|;]", command)
    return [p.strip() for p in parts if p.strip()]


def check_shell_command(command: str) -> SafetyVerdict:
    """Inspect a shell command string and decide if it needs confirmation.

    Returns a SafetyVerdict; callers (tools/shell.py) are responsible for
    actually prompting the user via `confirm_action`.
    """
    if not command or not command.strip():
        return SafetyVerdict(dangerous=False)

    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return SafetyVerdict(
                dangerous=True,
                reason=f"Command matches a known dangerous pattern: '{pattern.pattern}'",
            )

    for segment in _split_pipeline_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            # Unbalanced quotes etc. -- treat conservatively as dangerous
            # since we can't reliably analyze it.
            return SafetyVerdict(
                dangerous=True,
                reason="Command could not be safely parsed; treating as dangerous.",
            )
        if not tokens:
            continue
        head = tokens[0]
        # Strip a leading path, e.g. /bin/rm -> rm
        head_name = head.rsplit("/", 1)[-1]
        if head_name in DANGEROUS_COMMANDS:
            return SafetyVerdict(
                dangerous=True,
                reason=f"Command '{head_name}' can modify or destroy data/system state.",
            )

    return SafetyVerdict(dangerous=False)


# ----------------------------------------------------------------------
# Confirmation prompts
# ----------------------------------------------------------------------

# Injectable input function so the TUI (or tests) can override how
# confirmation is collected, instead of always blocking on builtin input().
InputFunc = Callable[[str], str]


def confirm_action(
    message: str,
    detail: Optional[str] = None,
    danger: bool = True,
    input_func: Optional[InputFunc] = None,
) -> bool:
    """Show a confirmation panel and block for a Y/N answer.

    Args:
        message: Short summary of the action requiring confirmation.
        detail: Optional longer explanation / context (e.g. the diff or
            the exact command line).
        danger: If True, render with a warning style; if False, render
            with a neutral/informational style (still requires Y/N).
        input_func: Optional override for collecting the response, used
            by the Textual UI to route through its own input widget
            instead of blocking stdin.

    Returns:
        True if the user approved the action, False otherwise.
    """
    title = "[bold red]WARNING[/bold red]" if danger else "[bold yellow]Confirm[/bold yellow]"
    lines = [message]
    if detail:
        lines.append("")
        lines.append(detail)
    lines.append("")
    lines.append("Proceed? [Y/n]")
    plain_prompt = "\n".join(lines)

    if input_func is None:
        # Default console path: render a Rich panel, then block on stdin.
        console.print(
            Panel(
                plain_prompt,
                title=title,
                border_style="red" if danger else "yellow",
                expand=False,
                box=safe_box(),
            )
        )
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    # Custom input_func path (e.g. the Textual TUI's modal bridge): pass
    # the full human-readable prompt text through so the alternate UI can
    # display it verbatim instead of a bare "> ".
    try:
        answer = input_func(plain_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


def warn_delete(path: str, input_func: Optional[InputFunc] = None) -> bool:
    """Specialized confirmation for file/directory deletion."""
    return confirm_action(
        message="This action will permanently delete a file or directory.",
        detail=f"Target: {path}",
        danger=True,
        input_func=input_func,
    )


def warn_overwrite(path: str, input_func: Optional[InputFunc] = None) -> bool:
    """Specialized confirmation for overwriting an existing file."""
    return confirm_action(
        message="This action will overwrite an existing file's contents.",
        detail=f"Target: {path}",
        danger=False,
        input_func=input_func,
    )


def warn_shell_command(command: str, reason: str, input_func: Optional[InputFunc] = None) -> bool:
    """Specialized confirmation for a potentially dangerous shell command."""
    return confirm_action(
        message="This shell command may modify or destroy data.",
        detail=f"Command: {command}\nReason flagged: {reason}",
        danger=True,
        input_func=input_func,
    )
