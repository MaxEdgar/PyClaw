"""
agent/planner.py
==================

Planning module: given a user request, asks the LLM to produce a short
numbered plan (3-6 steps) before any tool calls happen. The plan is shown
to the user in the UI's "Agent Plan" panel and steps are checked off as
the executor makes progress.

If the model fails to return valid plan JSON (small local models can be
inconsistent), this module falls back to a single generic plan step so
the agent loop can still proceed rather than blocking entirely on a
planning failure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from llm.client import LLMClient, LLMConnectionError, LLMResponseError
from llm.prompts import PLANNER_PROMPT
from ui.glyphs import glyph


@dataclass
class PlanStep:
    description: str
    done: bool = False


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)

    def mark_done(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            self.steps[index].done = True

    def as_text_list(self) -> List[str]:
        return [s.description for s in self.steps]

    def render_checklist(self) -> str:
        """Render the plan as a checkbox list, e.g. 'Scan project' marked
        done/pending. Uses terminal-safe glyphs (see ui/glyphs.py) so this
        degrades to plain ASCII ([x]/[ ]) on terminals that can't render
        Unicode checkmarks, instead of printing garbled '?' characters."""
        lines = []
        for step in self.steps:
            mark = glyph("check") if step.done else glyph("box_empty")
            lines.append(f"{mark} {step.description}")
        return "\n".join(lines)


# Fallback plan used when the model's planning response can't be parsed,
# or when the LLM server is briefly unreachable -- ensures the agent loop
# always has *something* to show and proceed with.
FALLBACK_PLAN_STEPS = [
    "Investigate the project structure relevant to the request",
    "Locate and read the relevant files",
    "Analyze the problem or requirement",
    "Make the necessary changes or produce an answer",
    "Verify the result",
]


def _extract_json_object(text: str) -> Optional[dict]:
    """Try to pull a JSON object out of the model's planning response,
    tolerating markdown fences or minor surrounding text."""
    text = text.strip()
    candidates = [text]

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class Planner:
    """Produces a Plan for a given user request using the LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def create_plan(self, user_request: str, project_context: str = "") -> Plan:
        """Ask the LLM for a short plan; fall back to a generic plan on
        any failure (parse error, connection error, malformed response)."""
        context_block = f"\nProject context:\n{project_context}\n" if project_context else ""
        user_content = f"User request: {user_request}{context_block}"

        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self.llm_client.chat(messages, temperature=0.1, max_tokens=400)
        except (LLMConnectionError, LLMResponseError):
            return Plan(steps=[PlanStep(description=s) for s in FALLBACK_PLAN_STEPS])

        obj = _extract_json_object(response.content)
        if not obj or "steps" not in obj or not isinstance(obj["steps"], list):
            return Plan(steps=[PlanStep(description=s) for s in FALLBACK_PLAN_STEPS])

        steps = [PlanStep(description=str(s)) for s in obj["steps"] if str(s).strip()]
        if not steps:
            return Plan(steps=[PlanStep(description=s) for s in FALLBACK_PLAN_STEPS])

        return Plan(steps=steps)
