"""
agent/intent.py
=================

Intent classification layer.

Every user message is classified into exactly one intent BEFORE any
planning, tool execution, or multi-step reasoning happens. This exists to
fix a real behavioral bug: short conversational inputs like "hi" or "yo"
were being treated as actionable tasks, triggering a planning round-trip,
filesystem scanning, and sometimes tool calls for messages that needed
none of that.

Intent types:
    CHAT_INTENT          -- conversational/social input. No planning, no
                             tools, no agent loop. Answered directly.
    TASK_INTENT          -- an explicit request to do something (create,
                             fix, build, refactor, ...). Full agent
                             behavior (planning + tools) is allowed.
    SYSTEM_INTENT        -- a slash command (/model, /help, ...). These
                             are already handled before reaching the
                             executor at all (see ui/tui.py and main.py),
                             so this classifier never actually receives
                             one in practice, but the type exists so the
                             classification is complete and callers that
                             do see a leading "/" can route accordingly
                             rather than mis-classifying it as chat/task.
    TOOL_REQUEST_INTENT  -- an explicit, unambiguous request to run a
                             specific tool-shaped action (e.g. "run
                             pytest", "show me the git diff"). Treated
                             the same as TASK_INTENT for safety-gating
                             purposes (both permit planning + tools); kept
                             as a distinct label because the calling code
                             and prompts may want to skip planning even
                             when allowing tool use, since a single named
                             action rarely needs a multi-step plan.

Classification is intentionally a fast, local, deterministic heuristic --
not a model call. Spending an LLM round-trip just to decide whether to
make ANOTHER LLM round-trip would defeat the entire point of fixing the
"hi triggers a planning loop" bug: classification has to be instant and
free, every single message, including on low-end hardware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Intent(str, Enum):
    CHAT = "CHAT_INTENT"
    TASK = "TASK_INTENT"
    SYSTEM = "SYSTEM_INTENT"
    TOOL_REQUEST = "TOOL_REQUEST_INTENT"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float  # 0.0-1.0, informational -- see classify()'s docstring
    reason: str        # short human-readable explanation, useful for /doctor-style debugging


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

# Casual greetings/acknowledgements that are CHAT_INTENT essentially by
# definition, regardless of word count. Matched as whole-message or
# whole-word, not substring (so "hire" doesn't match "hi").
_GREETING_WORDS = {
    "hi", "hello", "hey", "yo", "sup", "heya", "hiya", "howdy",
    "morning", "evening", "afternoon",
    "thanks", "thank", "thx", "ty", "cool", "nice", "ok", "okay", "k",
    "bye", "goodbye", "cya", "later",
    "lol", "lmao", "haha", "nice one",
}

# Short conversational phrases (multi-word but still pure small talk).
# Checked as exact-match or startswith, after stripping punctuation.
_GREETING_PHRASES = (
    "how are you", "how's it going", "hows it going", "whats up", "what's up",
    "good morning", "good evening", "good afternoon", "good night",
    "what's good", "whats good", "you there", "are you there", "you good",
    "how's your day", "hows your day",
    "thanks for", "thank you for", "thanks a lot", "appreciate it", "appreciate you",
)

# Verbs/keywords that signal an actionable request. Presence of any of
# these is treated as strong evidence of TASK_INTENT even in a short
# message (e.g. "fix bug" is 2 words but clearly a task).
_ACTION_VERBS = (
    "create", "make", "build", "write", "generate", "add", "implement",
    "fix", "debug", "solve", "repair", "patch",
    "delete", "remove", "rm",
    "move", "rename", "copy",
    "refactor", "rewrite", "optimize", "optimise", "improve", "clean up", "cleanup",
    "update", "edit", "modify", "change",
    "run", "execute", "test", "install", "deploy", "commit", "push", "pull",
    "search", "find", "grep", "look for", "locate",
    "read", "open", "show me", "display", "list files", "explain this",
    "summarize", "summarise", "analyze", "analyse", "review",
    "convert", "migrate", "upgrade", "downgrade",
)

# Explicit tool-shaped requests -- naming a concrete, single action
# closely matching one of PyClaw's real tools (see llm/prompts.py's
# TOOL_SPECS). Distinct from a general TASK_INTENT verb because these
# usually don't need a multi-step plan -- just run the one thing asked.
_TOOL_REQUEST_PATTERNS = (
    re.compile(r"\bgit (status|diff|log|commit|branch)\b"),
    re.compile(r"\brun (pytest|the tests?|tests?)\b"),
    re.compile(r"\bshow( me)? the (diff|status|log)\b"),
    re.compile(r"\blist (files|directory|dir)\b"),
    re.compile(r"\brun\s+\S+"),  # "run <command>"
)

# Question words that often precede a genuine request for information
# about the codebase (still TASK_INTENT-adjacent in that they may need a
# tool to answer truthfully) as opposed to small talk. Kept separate from
# _ACTION_VERBS since these are interrogative, not imperative.
_INFO_QUESTION_PREFIXES = (
    "what does", "what is", "what are", "explain", "why does", "why is",
    "how does", "how do i", "where is", "where does", "what's in", "whats in",
)

_PUNCTUATION_RE = re.compile(r"[!?.,;:]+$")
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _normalize(text: str) -> str:
    return _PUNCTUATION_RE.sub("", text.strip().lower())


def classify(user_input: str) -> IntentResult:
    """Classify a single user message into an Intent, with no LLM call.

    Confidence is a coarse, informational signal (not used for any
    threshold math beyond what's described in each branch below) meant
    for /doctor-style debugging of why a message was classified a given
    way -- it is not a probability in any calibrated sense.
    """
    raw = user_input.strip()
    normalized = _normalize(raw)

    if not raw:
        return IntentResult(Intent.CHAT, 1.0, "empty input")

    if raw.startswith("/"):
        return IntentResult(Intent.SYSTEM, 1.0, "leading '/' -- slash command")

    words = _WORD_RE.findall(normalized)
    word_count = len(words)

    # 1. Exact/near-exact greeting match -- highest-confidence CHAT_INTENT,
    # checked before anything else so "hi" can never be reclassified by a
    # coincidental keyword collision.
    if normalized in _GREETING_WORDS or normalized in _GREETING_PHRASES:
        return IntentResult(Intent.CHAT, 1.0, f"exact greeting match: '{normalized}'")

    if any(normalized.startswith(phrase) for phrase in _GREETING_PHRASES):
        return IntentResult(Intent.CHAT, 0.95, f"greeting phrase prefix: '{normalized}'")

    # 2. Explicit tool-shaped request -- checked before the general task
    # check since these are a more specific, higher-confidence match.
    for pattern in _TOOL_REQUEST_PATTERNS:
        if pattern.search(normalized):
            return IntentResult(Intent.TOOL_REQUEST, 0.9, f"matched tool-request pattern: '{pattern.pattern}'")

    # 3. Action verb present -- strong TASK_INTENT signal regardless of
    # length, since "fix bug" (2 words) is clearly a task despite being
    # short. This deliberately runs BEFORE the short-input fallback so a
    # short-but-actionable request is never misclassified as chat just
    # for being short.
    for verb in _ACTION_VERBS:
        if normalized.startswith(verb + " ") or normalized == verb or f" {verb} " in f" {normalized} ":
            return IntentResult(Intent.TASK, 0.85, f"action verb detected: '{verb}'")

    # 4. Simplification heuristic: three words or fewer with no action
    # verb and no recognized greeting/question form -- default to
    # CHAT_INTENT. This is the specific rule requested to stop short,
    # ambiguous inputs from accidentally triggering the agent loop, and
    # runs BEFORE the informational-question check below: a short
    # fragment like "what is this" reads as much closer to small talk
    # than a real request to inspect the codebase, and the spec's intent
    # is clearly "when in doubt and short, treat as chat."
    if word_count <= 3:
        return IntentResult(Intent.CHAT, 0.8, f"<= 3 words ({word_count}) with no action verb -- defaulting to chat")

    # 5. Informational question about the project -- treated as TASK so
    # the agent is allowed to actually read files to answer truthfully,
    # rather than guessing from the conversation alone. Only reached for
    # longer questions (> 3 words), since short ones were already routed
    # to chat above.
    if any(normalized.startswith(prefix) for prefix in _INFO_QUESTION_PREFIXES):
        return IntentResult(Intent.TASK, 0.75, "informational question about project contents")

    # 6. Longer input with no other signal: still default to TASK rather
    # than CHAT. A long message with no recognized verb is far more
    # likely to be a request phrased unusually (e.g. "the login page
    # crashes when I click submit") than genuine small talk -- erring
    # toward TASK here means PyClaw can still investigate and answer
    # using tools, while erring toward CHAT would silently refuse to look
    # at the codebase for a real question that just didn't use one of the
    # exact verbs in _ACTION_VERBS.
    return IntentResult(Intent.TASK, 0.55, f"{word_count} words, no specific verb matched -- defaulting to task")


def allows_agent_behavior(intent: Intent) -> bool:
    """Safety gate: returns True only for intents permitted to trigger
    planning, tool execution, or any multi-step agent loop.

    This is the single choke point enforcing the rule "no planning, tool
    execution, or agent loop may be activated unless intent is TASK or
    TOOL_REQUEST" -- callers (agent/executor.py) check this before doing
    anything else, rather than each call site re-implementing the same
    condition slightly differently.
    """
    return intent in (Intent.TASK, Intent.TOOL_REQUEST)
