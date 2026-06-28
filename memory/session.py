"""
memory/session.py
===================

Session-level memory: tracks the current task, recently touched files, and
the most recent plan, persisted to JSON so a session can be resumed after
restarting PyClaw.

This is intentionally separate from memory/history.py (full chat
transcript) -- session memory is a small, curated summary of "where was I"
state, optimized for being cheaply re-injected into the model's context on
resume, rather than a complete log.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SESSION_PATH = Path.home() / ".pyclaw" / "session.json"

# Cap how many recent files we remember, to keep the persisted state small
# and the re-injected context compact.
MAX_RECENT_FILES = 20


@dataclass
class TodoItem:
    """A single persistent todo item, distinct from SessionState.last_plan:
    last_plan is the agent's short-lived plan for ONE request and is
    overwritten every turn; todos are user-managed and meant to survive
    across many separate requests and conversations -- e.g. tracking a
    multi-day migration where each day is its own session. Inspired by
    OpenClaw's "Workboard" concept, scoped down to PyClaw's single-user,
    single-project context (no multi-agent assignment, just a list)."""

    text: str
    done: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionState:
    """In-memory representation of the current session's working state."""

    current_task: Optional[str] = None
    recent_files: List[str] = field(default_factory=list)
    last_plan: List[str] = field(default_factory=list)
    project_root: Optional[str] = None
    todos: List[TodoItem] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        todos_raw = data.get("todos", [])
        todos = [
            TodoItem(
                text=t.get("text", ""),
                done=t.get("done", False),
                created_at=t.get("created_at", time.time()),
            )
            for t in todos_raw
            if isinstance(t, dict)
        ]
        return cls(
            current_task=data.get("current_task"),
            recent_files=list(data.get("recent_files", [])),
            last_plan=list(data.get("last_plan", [])),
            project_root=data.get("project_root"),
            todos=todos,
            updated_at=data.get("updated_at", time.time()),
        )


class SessionMemory:
    """Manages loading, mutating, and persisting SessionState to disk."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or DEFAULT_SESSION_PATH)
        self.state = self._load()

    # ------------------------------------------------------------------
    def _load(self) -> SessionState:
        if not self.path.exists():
            return SessionState()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return SessionState.from_dict(data)
        except (json.JSONDecodeError, OSError):
            # Corrupt session file -- start fresh rather than crashing.
            return SessionState()

    def save(self) -> None:
        """Atomically persist the current session state to disk."""
        self.state.updated_at = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.state.to_dict(), fh, indent=2)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def set_task(self, task: str) -> None:
        self.state.current_task = task
        self.save()

    def clear_task(self) -> None:
        self.state.current_task = None
        self.save()

    def touch_file(self, path: str) -> None:
        """Record that `path` was recently read/edited, moving it to the
        front of the recent-files list and capping the list length."""
        files = [f for f in self.state.recent_files if f != path]
        files.insert(0, path)
        self.state.recent_files = files[:MAX_RECENT_FILES]
        self.save()

    def set_plan(self, steps: List[str]) -> None:
        self.state.last_plan = list(steps)
        self.save()

    def set_project_root(self, path: str) -> None:
        self.state.project_root = path
        self.save()

    # ------------------------------------------------------------------
    # Todo list (persistent across /clear and across sessions -- see
    # TodoItem's docstring for why this is distinct from last_plan)
    # ------------------------------------------------------------------
    def add_todo(self, text: str) -> int:
        """Add a todo item and return its 1-based index for reference in
        /todo done <n>."""
        self.state.todos.append(TodoItem(text=text.strip()))
        self.save()
        return len(self.state.todos)

    def mark_todo_done(self, index: int) -> bool:
        """Mark the todo at 1-based `index` as done. Returns False if the
        index is out of range, so the caller can report a clear error
        instead of silently doing nothing."""
        pos = index - 1
        if pos < 0 or pos >= len(self.state.todos):
            return False
        self.state.todos[pos].done = True
        self.save()
        return True

    def clear_todos(self) -> None:
        """Remove every todo item (used by /todo clear -- a deliberate,
        explicit action distinct from /clear, which does NOT touch todos)."""
        self.state.todos = []
        self.save()

    def remove_completed_todos(self) -> int:
        """Drop every todo already marked done, keeping the still-open
        ones. Returns how many were removed."""
        before = len(self.state.todos)
        self.state.todos = [t for t in self.state.todos if not t.done]
        self.save()
        return before - len(self.state.todos)

    def reset(self) -> None:
        """Clear conversation-scoped session state (used by the /clear
        slash command). Deliberately preserves `todos`: /clear resets the
        current conversation, it does not mean "abandon my task list" --
        a todo list is meant to survive across many separate conversations
        (see TodoItem's docstring). Use /todo clear to explicitly clear
        todos instead."""
        self.state = SessionState(project_root=self.state.project_root, todos=self.state.todos)
        self.save()

    # ------------------------------------------------------------------
    # Rendering for re-injection into the model's context
    # ------------------------------------------------------------------
    def summary_text(self) -> str:
        """Produce a short human-readable summary of session state, suitable
        for showing in the sidebar or re-injecting as context on resume."""
        lines = []
        if self.state.current_task:
            lines.append(f"Current task: {self.state.current_task}")
        if self.state.last_plan:
            lines.append("Last plan:")
            for i, step in enumerate(self.state.last_plan, 1):
                lines.append(f"  {i}. {step}")
        if self.state.recent_files:
            lines.append("Recently touched files: " + ", ".join(self.state.recent_files[:5]))
        if self.state.todos:
            open_count = sum(1 for t in self.state.todos if not t.done)
            lines.append(f"Todos: {open_count} open / {len(self.state.todos)} total (see /todo list)")
        if not lines:
            return "No session history yet."
        return "\n".join(lines)

    def render_todos_text(self, done_glyph: str = "x", open_glyph: str = " ") -> str:
        """Render the full todo list as a readable checklist, used by
        /todo list.

        Args:
            done_glyph, open_glyph: characters used inside "[ ]" / "[x]"
                style brackets for done/open items. Callers in ui/ may
                pass terminal-safe Unicode glyphs (see ui/glyphs.py); this
                module stays UI-agnostic and does not import from ui/
                itself, consistent with the rest of memory/'s design.
        """
        if not self.state.todos:
            return "No todos yet. Add one with /todo add <text>."

        lines = []
        for i, item in enumerate(self.state.todos, 1):
            mark = done_glyph if item.done else open_glyph
            lines.append(f"{i}. [{mark}] {item.text}")
        return "\n".join(lines)
