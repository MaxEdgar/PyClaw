"""
memory/skills.py
==================

Persistent, user-defined "skills" for PyClaw's agent.

A skill is a structured, reusable piece of guidance the user teaches
PyClaw once and which is then available in every future session --
distinct from a one-off instruction typed into the chat, which is
forgotten once the conversation ends.

Each skill is stored as a single JSON file under `~/.pyclaw/skills/` and
has this shape:

    {
      "name": "release-checklist",
      "description": "Steps to follow before tagging a release",
      "trigger_keywords": ["release", "version bump", "changelog"],
      "instructions": "1. Run the test suite...\n2. Update CHANGELOG.md...",
      "created_at": 1719200000.0,
      "updated_at": 1719200000.0
    }

Skills relevant to the user's current request (matched by simple keyword
overlap against `trigger_keywords` and the skill name) are injected into
the system prompt by agent/executor.py, so the model sees them as
additional ground-truth instructions rather than something it has to be
told fresh every conversation.

This module intentionally does NOT execute anything on its own -- a skill
is plain text guidance for the model, not code. There is no eval(), no
arbitrary script execution, and no network access here; skills are exactly
as safe as any other text the model reads, and every action a skill leads
to still passes through the normal tool-call approval/safety flow in
agent/tool_router.py and tools/safety.py.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SKILLS_DIR = Path.home() / ".pyclaw" / "skills"

# A skill name becomes its filename, so it's restricted to a safe,
# predictable character set -- this also doubles as a defense against
# path-traversal via a crafted skill name (e.g. "../../etc/passwd").
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class InvalidSkillNameError(Exception):
    """Raised when a skill name doesn't meet the safe-filename pattern."""


def validate_skill_name(name: str) -> str:
    """Validate and normalize a skill name, raising InvalidSkillNameError
    with a clear message if it doesn't fit the safe pattern."""
    candidate = name.strip().lower().replace(" ", "-")
    if not _NAME_PATTERN.match(candidate):
        raise InvalidSkillNameError(
            f"Invalid skill name '{name}'. Use 1-64 characters: letters, numbers, "
            "hyphens, or underscores, starting with a letter or number."
        )
    return candidate


@dataclass
class Skill:
    """A single structured, persistent skill."""

    name: str
    description: str
    instructions: str
    trigger_keywords: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            instructions=data.get("instructions", ""),
            trigger_keywords=list(data.get("trigger_keywords", [])),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    def matches(self, request_text: str) -> bool:
        """Return True if this skill looks relevant to a given user request,
        via simple case-insensitive keyword overlap. This is intentionally
        simple (no embeddings, no fuzzy matching) so behavior is
        predictable and explainable -- a skill activates because a
        specific word matched, not because of an opaque similarity score.
        """
        haystack = request_text.lower()
        if self.name.replace("-", " ") in haystack:
            return True
        return any(kw.lower() in haystack for kw in self.trigger_keywords if kw.strip())

    def render_for_prompt(self) -> str:
        """Render this skill as a block suitable for injection into the
        system prompt context, clearly delimited so the model can tell
        it apart from the base instructions and from other skills."""
        return (
            f"[SKILL: {self.name}]\n"
            f"When to use: {self.description}\n"
            f"{self.instructions}\n"
            f"[END SKILL: {self.name}]"
        )


class SkillStore:
    """Loads, saves, lists, and deletes Skill objects from disk."""

    def __init__(self, directory: Optional[str] = None):
        self.directory = Path(directory or DEFAULT_SKILLS_DIR)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, name: str) -> Path:
        safe_name = validate_skill_name(name)
        return self.directory / f"{safe_name}.json"

    def create(
        self,
        name: str,
        description: str,
        instructions: str,
        trigger_keywords: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Skill:
        """Create (or update, if overwrite=True) a skill and persist it.

        Raises FileExistsError if a skill with this name already exists
        and overwrite is False, so callers (the /skill create command) can
        prompt for confirmation before clobbering an existing skill.
        """
        path = self._path_for(name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"A skill named '{name}' already exists. Use overwrite=True to replace it.")

        safe_name = validate_skill_name(name)
        now = time.time()
        existing_created_at = now
        if path.exists():
            try:
                existing_created_at = self.load(safe_name).created_at
            except (OSError, json.JSONDecodeError):
                pass

        skill = Skill(
            name=safe_name,
            description=description.strip(),
            instructions=instructions.strip(),
            trigger_keywords=[kw.strip() for kw in (trigger_keywords or []) if kw.strip()],
            created_at=existing_created_at,
            updated_at=now,
        )
        path.write_text(json.dumps(skill.to_dict(), indent=2), encoding="utf-8")
        return skill

    def load(self, name: str) -> Skill:
        """Load a single skill by name. Raises FileNotFoundError if it
        doesn't exist."""
        path = self._path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"No skill named '{name}'.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Skill.from_dict(data)

    def list_all(self) -> List[Skill]:
        """Load every stored skill, skipping any file that fails to parse
        (corrupt/partially-written) rather than crashing the whole list."""
        skills = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                skills.append(Skill.from_dict(data))
            except (OSError, json.JSONDecodeError):
                continue
        return skills

    def delete(self, name: str) -> bool:
        """Delete a skill by name. Returns True if it existed and was
        deleted, False if it didn't exist."""
        path = self._path_for(name)
        if not path.exists():
            return False
        path.unlink()
        return True

    def find_relevant(self, request_text: str, limit: int = 5) -> List[Skill]:
        """Return up to `limit` stored skills relevant to a given request,
        for injection into the system prompt by agent/executor.py."""
        matches = [s for s in self.list_all() if s.matches(request_text)]
        return matches[:limit]
