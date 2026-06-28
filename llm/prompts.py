"""
llm/prompts.py
================

System prompts and tool-call schema documentation fed to the LLM.

Small local models (3B-7B class) are not reliably trained on native
"function calling" formats the way large hosted models are, so PyClaw
uses a simpler, more robust convention: the model is instructed to emit a
single fenced JSON object describing the tool it wants to call, and the
tool router (agent/tool_router.py) parses that JSON out of the model's
free-text response. This module defines:

    * TOOL_SPECS: a human-readable (and model-readable) description of
      every available tool, its arguments, and when to use it.
    * SYSTEM_PROMPT: the main system prompt establishing the agent's
      persona, behavior rules, and the tool-calling contract.
    * PLANNER_PROMPT: a focused prompt used by agent/planner.py to produce
      a short numbered plan before any tool calls happen.
    * build_tool_message(): formats a tool result for re-injection into
      the conversation so the model can use it to continue reasoning.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# ----------------------------------------------------------------------
# Tool specifications
# ----------------------------------------------------------------------
# Each entry documents the tool name, a short description, and its
# arguments. This is rendered into the system prompt verbatim so even a
# small model has a concrete contract to follow.
TOOL_SPECS: List[Dict[str, Any]] = [
    {"name": "read_file", "description": "Read a text file's contents.", "args": {"path": "string"}},
    {"name": "write_file", "description": "Create or overwrite a file with new content.", "args": {"path": "string", "content": "string"}},
    {"name": "append_file", "description": "Append text to the end of a file.", "args": {"path": "string", "content": "string"}},
    {"name": "delete_file", "description": "Delete a file or directory. Requires user approval.", "args": {"path": "string"}},
    {"name": "move_file", "description": "Move or rename a file or directory.", "args": {"src": "string", "dst": "string"}},
    {"name": "copy_file", "description": "Copy a file or directory.", "args": {"src": "string", "dst": "string"}},
    {"name": "list_directory", "description": "List the contents of a directory.", "args": {"path": "string"}},
    {"name": "create_directory", "description": "Create a new directory.", "args": {"path": "string"}},
    {"name": "file_info", "description": "Get metadata (size, permissions, mtime) about a file or directory.", "args": {"path": "string"}},
    {"name": "search_files", "description": "Find files by glob filename pattern, e.g. '*.py'.", "args": {"pattern": "string", "path": "string (optional)"}},
    {"name": "grep_text", "description": "Search file contents recursively for a string or regex.", "args": {"query": "string", "path": "string (optional)", "regex": "boolean (optional)"}},
    {"name": "find_extensions", "description": "Count files grouped by file extension.", "args": {"path": "string (optional)"}},
    {"name": "find_large_files", "description": "Find files above a size threshold.", "args": {"min_size_bytes": "integer (optional)"}},
    {"name": "project_summary", "description": "Get a high-level overview of the project (file counts, stack, structure).", "args": {}},
    {"name": "run_command", "description": "Run a shell command. Dangerous commands require user approval.", "args": {"command": "string", "timeout": "number (optional)"}},
    {"name": "git_status", "description": "Show git working tree status.", "args": {}},
    {"name": "git_diff", "description": "Show git diff of unstaged (or staged) changes.", "args": {"path": "string (optional)", "staged": "boolean (optional)"}},
    {"name": "git_log", "description": "Show recent commit history.", "args": {"max_count": "integer (optional)"}},
    {"name": "git_commit", "description": "Create a git commit.", "args": {"message": "string", "add_all": "boolean (optional)"}},
    {"name": "git_branch", "description": "List branches, or create a new one.", "args": {"create": "string (optional)"}},
    {"name": "propose_patch", "description": "Propose a file edit as a unified diff for user approval before writing.", "args": {"path": "string", "new_content": "string"}},
    {"name": "final_answer", "description": "Provide the final answer to the user with no further tool calls.", "args": {"text": "string"}},
]


def _render_tool_specs() -> str:
    """Render TOOL_SPECS as a readable block for inclusion in the system prompt."""
    lines = []
    for spec in TOOL_SPECS:
        args_desc = ", ".join(f"{k}: {v}" for k, v in spec["args"].items()) or "(no arguments)"
        lines.append(f"- {spec['name']}({args_desc}) -- {spec['description']}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are PyClaw, a local AI coding assistant running entirely offline
on the user's own machine via a local language model server. You behave like a
careful, competent pair-programmer: thorough, concise, and honest about uncertainty.

You operate on a real project on disk. You have access to a fixed set of tools for
reading, searching, editing, and running commands within the project. You do not have
any abilities beyond these tools -- you cannot browse the internet, and you cannot act
outside the current project directory.

AVAILABLE TOOLS:
{_render_tool_specs()}

TOOL-CALLING FORMAT:
When you need to use a tool, respond with ONLY a single JSON object (no other text,
no markdown fences) in exactly this shape:

{{"tool": "<tool_name>", "arguments": {{...}}}}

Rules:
1. Emit at most ONE tool call per response. Wait for the result before calling another tool.
2. When you have enough information to answer the user, call the "final_answer" tool
   with your complete answer in the "text" argument, OR simply respond in plain natural
   language with no JSON if no further tool use is needed.
3. Never invent file contents or search results. Only state facts you have actually
   observed via a tool result shown to you in this conversation.
4. Before modifying an existing file, prefer calling "propose_patch" so the user can
   review a diff, rather than calling "write_file" directly on a pre-existing file.
5. For any destructive operation (delete_file, or a shell command involving rm/mv/
   chmod/chown/sudo/dd), expect that the system will ask the user for approval; if it
   is rejected, acknowledge that and propose an alternative or stop.
6. Keep your reasoning concise. Do not narrate excessively between tool calls.
7. If a tool result indicates an error, adapt your plan rather than repeating the same
   failing call.

Always ground your answers in the actual project contents you have read via tools."""


# A deliberately separate, minimal system prompt used ONLY for messages
# classified as CHAT_INTENT (see agent/intent.py). This prompt contains
# NO tool specifications and NO tool-calling format instructions at all
# -- not because the model is told not to use tools, but because the
# concept of tools is simply absent from what it's given. A model cannot
# emit a "{"tool": ...}" JSON block referencing a tool it was never told
# about in this turn's context, which is a stronger guarantee than asking
# it not to. This is what makes greetings/small talk answer instantly
# with a short natural reply instead of triggering planning or a tool
# call: there is nothing tool-shaped anywhere in the prompt to trigger on.
CHAT_SYSTEM_PROMPT = """You are PyClaw, a local AI coding assistant. The user just sent a short,
casual, conversational message (a greeting, acknowledgement, or small talk) rather than
a request to do something with their project.

Reply naturally and briefly, like a helpful colleague would. Do not mention tools,
plans, or your capabilities unless asked. Do not produce JSON, code blocks, or
structured output of any kind -- just a short, friendly, plain-text reply."""


PLANNER_PROMPT = """You are the planning module of PyClaw, a local coding assistant.
Given the user's request and a brief project summary, produce a short, numbered plan
(3-6 steps) describing how you will approach the task using the available tools
(reading files, searching, editing, running commands, git operations).

Respond with ONLY a JSON object of this shape, no other text:

{"steps": ["step one description", "step two description", ...]}

Keep each step short (one line). The plan should reflect a realistic order of
operations: orient/investigate first, then analyze, then act, then verify."""


def build_tool_message(tool_name: str, result: Dict[str, Any]) -> Dict[str, str]:
    """Format a tool execution result as a 'tool' role message for re-injection
    into the conversation history, so the model can read it on its next turn.

    Since llama.cpp's OpenAI-compatible endpoint does not necessarily support
    the native `role: tool` message type used by hosted APIs, we encode the
    result as a clearly-delimited user-role message instead, which is the
    most broadly compatible approach across local model templates.
    """
    payload = json.dumps(result, indent=2, default=str)
    content = f"[TOOL RESULT: {tool_name}]\n{payload}\n[END TOOL RESULT]"
    return {"role": "user", "content": content}


def build_user_message(text: str) -> Dict[str, str]:
    """Wrap raw user input as a chat message."""
    return {"role": "user", "content": text}


def build_chat_system_message() -> Dict[str, str]:
    """Return the minimal CHAT_INTENT-only system prompt as a chat message.
    See CHAT_SYSTEM_PROMPT's docstring for why this is a wholly separate,
    tool-free prompt rather than a variant of the main one."""
    return {"role": "system", "content": CHAT_SYSTEM_PROMPT}


def build_system_message(context_block: str = "") -> Dict[str, str]:
    """Return the main system prompt as a chat message.

    Args:
        context_block: Optional pre-rendered text block of additional
            context relevant to the current request -- the project's
            PYCLAW.md instructions and/or any matched user-defined skills,
            already concatenated and delimited by the caller (see
            agent/executor.py:_refresh_system_message). Kept as a plain
            string parameter here, rather than importing memory.skills or
            agent.project_instructions directly, so this module stays a
            pure prompt-formatting layer with no dependency on how that
            context is stored or matched.
    """
    content = SYSTEM_PROMPT
    if context_block.strip():
        content += (
            "\n\nThe following additional context applies to this request -- project-specific "
            "conventions from PYCLAW.md and/or user-defined skills relevant to what was just "
            "asked. Follow it where it applies:\n\n" + context_block
        )
    return {"role": "system", "content": content}
