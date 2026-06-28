"""
memory/history.py
===================

Persistent chat history storage.

Stores the full conversation transcript (every user message, assistant
response, and tool call/result) as JSON, so `/history` can show past
turns and a session can be resumed with full context after a restart.

History is stored as a flat JSON Lines (.jsonl) file, one record per line,
which is append-friendly (no need to rewrite the whole file on every
message) and resilient to a trailing partial write (a malformed last line
is simply skipped on load rather than corrupting the whole file).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_HISTORY_PATH = Path.home() / ".pyclaw" / "history.jsonl"


@dataclass
class HistoryEntry:
    """A single recorded event in the conversation history."""

    role: str  # "user" | "assistant" | "tool" | "system"
    content: str
    timestamp: float
    tool_name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            tool_name=data.get("tool_name"),
            meta=data.get("meta"),
        )


class HistoryStore:
    """Append-only JSONL-backed chat history store."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or DEFAULT_HISTORY_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a single entry to the history file."""
        entry = HistoryEntry(role=role, content=content, timestamp=time.time(), tool_name=tool_name, meta=meta)
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except OSError:
            # History is a convenience feature; failing to persist it
            # should never crash the agent loop.
            pass

    def load_all(self) -> List[HistoryEntry]:
        """Load the full history, skipping any malformed trailing lines."""
        if not self.path.exists():
            return []
        entries: List[HistoryEntry] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entries.append(HistoryEntry.from_dict(data))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return entries

    def load_recent(self, limit: int = 20) -> List[HistoryEntry]:
        """Return the most recent `limit` entries."""
        entries = self.load_all()
        return entries[-limit:]

    def clear(self) -> None:
        """Erase all history (used by the /clear slash command)."""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass

    def render_recent_text(self, limit: int = 10) -> str:
        """Render the last `limit` entries as a readable transcript, used
        by the /history slash command."""
        entries = self.load_recent(limit)
        if not entries:
            return "No history yet."
        lines = []
        for entry in entries:
            ts = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
            label = entry.role.upper() if not entry.tool_name else f"TOOL:{entry.tool_name}"
            snippet = entry.content if len(entry.content) <= 200 else entry.content[:200] + "..."
            lines.append(f"[{ts}] {label}: {snippet}")
        return "\n".join(lines)
