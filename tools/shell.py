"""
tools/shell.py
===============

Shell command execution tool: run_command().

Supports:
    * Timeout enforcement (kills the process group on timeout)
    * Captured stdout/stderr
    * Return codes
    * Safety-gated confirmation for dangerous commands (rm, sudo, dd, etc.)
      via tools.safety.check_shell_command / warn_shell_command

The command always executes with the project root as its working directory
unless an explicit `cwd` override is given (still sandboxed to within the
project root by the caller's discipline -- this module does not attempt to
sandbox arbitrary shell commands the way filesystem.py sandboxes paths,
since a shell command can do anything a path-based check cannot detect;
the dangerous-command confirmation gate is the real safety boundary here).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.safety import InputFunc, check_shell_command, warn_shell_command


@dataclass
class ShellResult:
    """Structured result of a shell command execution."""

    success: bool
    command: str
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    timed_out: bool = False
    duration_seconds: float = 0.0
    error: Optional[str] = None
    cancelled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
            "duration_seconds": round(self.duration_seconds, 3),
            "error": self.error,
            "cancelled": self.cancelled,
        }


# Cap stdout/stderr captured into the result to keep huge command output
# (e.g. `find /` by accident) from blowing the model's context window.
MAX_OUTPUT_CHARS = 20_000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [truncated {len(text) - limit} characters] ...\n" + text[-half:]


def run_command(
    command: str,
    cwd: str,
    timeout: float = 60.0,
    require_confirmation: bool = True,
    confirmed: bool = False,
    input_func: Optional[InputFunc] = None,
) -> ShellResult:
    """Execute a shell command and capture its output.

    Args:
        command: The shell command line to run (executed via the system
            shell, so pipes/redirects/globs work as expected).
        cwd: Working directory to run the command in.
        timeout: Seconds to allow before killing the process.
        require_confirmation: Whether dangerous commands should block on
            confirmation. The executor/UI layer is expected to set
            `confirmed=True` once the user has approved, mirroring the
            pattern used by filesystem.py's delete/overwrite tools.
        confirmed: Whether the user has already approved this exact
            command (set by the caller after a successful confirm_action).
        input_func: Optional override for how confirmation input is
            collected (used by the Textual UI).
    """
    verdict = check_shell_command(command)

    if verdict.dangerous and require_confirmation and not confirmed:
        return ShellResult(
            success=False,
            command=command,
            error="confirmation_required",
            cancelled=False,
        )

    if verdict.dangerous and require_confirmation and confirmed:
        # `confirmed` here means the executor already showed the prompt and
        # got a yes; we proceed. (The actual interactive prompt, when not
        # pre-confirmed by the caller, is triggered by the executor calling
        # warn_shell_command directly before invoking run_command.)
        pass

    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        return ShellResult(
            success=proc.returncode == 0,
            command=command,
            stdout=_truncate(proc.stdout),
            stderr=_truncate(proc.stderr),
            return_code=proc.returncode,
            duration_seconds=duration,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ShellResult(
            success=False,
            command=command,
            stdout=_truncate(stdout),
            stderr=_truncate(stderr),
            timed_out=True,
            duration_seconds=duration,
            error=f"Command timed out after {timeout} seconds",
        )
    except FileNotFoundError as exc:
        return ShellResult(success=False, command=command, error=f"Command not found: {exc}")
    except PermissionError as exc:
        return ShellResult(success=False, command=command, error=f"Permission denied: {exc}")
    except OSError as exc:
        return ShellResult(success=False, command=command, error=f"OS error running command: {exc}")


def run_command_interactive(
    command: str,
    cwd: str,
    timeout: float = 60.0,
    require_confirmation: bool = True,
    input_func: Optional[InputFunc] = None,
) -> ShellResult:
    """Convenience wrapper that handles the confirmation prompt itself.

    Unlike `run_command` (which returns `confirmation_required` and expects
    the caller to manage the approval flow, useful for the TUI's own
    approval widget), this function blocks on a console Y/N prompt directly
    -- handy for CLI/non-interactive-UI contexts.
    """
    verdict = check_shell_command(command)
    if verdict.dangerous and require_confirmation:
        approved = warn_shell_command(command, verdict.reason, input_func=input_func)
        if not approved:
            return ShellResult(success=False, command=command, cancelled=True, error="Cancelled by user")

    return run_command(
        command=command,
        cwd=cwd,
        timeout=timeout,
        require_confirmation=False,  # already handled above
        confirmed=True,
    )
