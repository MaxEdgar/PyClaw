"""
agent/project_instructions.py
================================

Optional project-scoped instructions file: PYCLAW.md.

Inspired by the AGENTS.md/instructions-file pattern used by several coding
agents (OpenClaw's workspace bootstrap files among them): if a project
contains a PYCLAW.md at its root, PyClaw reads it automatically on every
request for that project and includes it in the system prompt -- no
command needed, no keyword matching, no per-conversation setup.

This is deliberately distinct from memory/skills.py:
    * A skill is scoped to the USER, applies across every project, and only
      activates when its trigger keywords match the current request.
    * PYCLAW.md is scoped to the PROJECT, applies to everyone working in
      it (it's meant to be committed to the repo, like a README), and
      always applies whenever that project is open -- no matching needed,
      since project-level conventions ("this repo uses 4-space indents,
      never touch vendor/") are relevant to every request, not just some.

Like skills, PYCLAW.md is plain text the model reads as additional
context. It cannot execute code and does not bypass the tool-call
approval/safety flow in agent/tool_router.py and tools/safety.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# The filename PyClaw looks for at the project root. Uppercase + .md to
# match the convention of README.md, LICENSE, CONTRIBUTING.md, etc. --
# something that stands out in a directory listing as a "read me first"
# file, and that's easy to commit alongside the rest of the project.
PROJECT_INSTRUCTIONS_FILENAME = "PYCLAW.md"

# Cap how much of the file gets read, mirroring the same reasoning as
# AgentConfig.max_file_read_bytes -- a runaway or accidentally-huge
# instructions file should never silently consume the model's whole
# context budget every single turn.
MAX_INSTRUCTIONS_BYTES = 20_000


def read_project_instructions(project_root: str) -> Optional[str]:
    """Return the contents of <project_root>/PYCLAW.md if it exists and is
    readable, truncated to MAX_INSTRUCTIONS_BYTES. Returns None if the
    file doesn't exist or can't be read -- this is an optional, best-effort
    feature, never a required one, so any failure here is silent rather
    than surfaced as an error to the user.
    """
    path = Path(project_root).expanduser() / PROJECT_INSTRUCTIONS_FILENAME
    if not path.is_file():
        return None

    try:
        raw = path.read_bytes()
    except OSError:
        return None

    truncated = len(raw) > MAX_INSTRUCTIONS_BYTES
    content_bytes = raw[:MAX_INSTRUCTIONS_BYTES] if truncated else raw

    try:
        text = content_bytes.decode("utf-8", errors="replace").strip()
    except LookupError:
        return None

    if not text:
        return None

    if truncated:
        text += f"\n\n[... {PROJECT_INSTRUCTIONS_FILENAME} truncated at {MAX_INSTRUCTIONS_BYTES} bytes ...]"

    return text


def render_for_prompt(instructions: str) -> str:
    """Wrap project instructions in a clear delimiter for system-prompt
    injection, matching the style used for skill blocks (see
    memory/skills.py:Skill.render_for_prompt) so the model can tell apart
    project-level conventions from user-level skills and its base
    instructions."""
    return f"[PROJECT INSTRUCTIONS: {PROJECT_INSTRUCTIONS_FILENAME}]\n{instructions}\n[END PROJECT INSTRUCTIONS]"
