"""
agent/tool_router.py
======================

Parses structured tool-call JSON emitted by the LLM and dispatches it to
the corresponding tool implementation in the `tools/` package.

This is the bridge between "the model said it wants to call X" and
"X actually executes". It owns:

    * Extracting a JSON tool call from a (possibly noisy) model response.
    * Validating the tool name and arguments.
    * Routing destructive actions (delete_file, dangerous shell commands,
      overwriting existing files) through the confirmation/patch-approval
      flow before they execute.
    * Normalizing every tool's result into a plain dict suitable for
      `llm.prompts.build_tool_message`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from config import Config
from tools import filesystem, git_tools, search, shell
from tools.safety import check_shell_command, warn_shell_command
from ui.diff_view import Patch, approve_patch


class ToolParseError(Exception):
    """Raised when the model's response cannot be parsed as a tool call."""


@dataclass
class ParsedToolCall:
    tool: str
    arguments: Dict[str, Any]


# Tools that are pure "say something to the user" actions rather than real
# side-effecting tools -- handled specially by the executor.
TERMINAL_TOOLS = {"final_answer"}


def extract_tool_call(model_text: str) -> Optional[ParsedToolCall]:
    """Attempt to extract a {"tool": ..., "arguments": ...} JSON object from
    the model's raw text output.

    Small local models sometimes wrap JSON in markdown fences or add a
    sentence before/after it despite instructions not to, so this function
    tries a few extraction strategies in order of strictness:
        1. The entire trimmed text is valid JSON.
        2. A fenced ```json ... ``` block is present.
        3. The first balanced {...} substring found via brace matching.
    Returns None if no valid tool call could be extracted (meaning the
    model intended a plain natural-language answer instead).
    """
    text = model_text.strip()
    if not text:
        return None

    candidates = []

    # Strategy 1: whole response is JSON
    candidates.append(text)

    # Strategy 2: fenced code block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    # Strategy 3: first balanced brace substring
    brace_substr = _first_balanced_braces(text)
    if brace_substr:
        candidates.append(brace_substr)

    for candidate in candidates:
        parsed = _try_parse_tool_json(candidate)
        if parsed is not None:
            return parsed

    return None


def _first_balanced_braces(text: str) -> Optional[str]:
    """Find the first top-level balanced {...} substring in `text`."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _try_parse_tool_json(candidate: str) -> Optional[ParsedToolCall]:
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    tool_name = obj.get("tool")
    if not tool_name or not isinstance(tool_name, str):
        return None
    arguments = obj.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return ParsedToolCall(tool=tool_name, arguments=arguments)


class ToolRouter:
    """Dispatches parsed tool calls to their implementations.

    Holds references to the active Config (for project_root, excluded_dirs,
    safety settings) so individual dispatch methods don't need to thread
    those through manually.
    """

    def __init__(self, config: Config, input_func: Optional[Callable[[str], str]] = None):
        self.config = config
        self.input_func = input_func

    # ------------------------------------------------------------------
    def dispatch(self, call: ParsedToolCall) -> Dict[str, Any]:
        """Execute a parsed tool call and return a plain-dict result.

        Unknown tool names produce a structured error result rather than
        raising, so the agent loop can feed the error back to the model
        and let it self-correct.
        """
        handler = self._handlers().get(call.tool)
        if handler is None:
            return {
                "success": False,
                "tool": call.tool,
                "error": f"Unknown tool '{call.tool}'. Available tools: {', '.join(sorted(self._handlers().keys()))}",
            }
        try:
            return handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 - tool execution must never crash the loop
            return {"success": False, "tool": call.tool, "error": f"Unhandled exception in tool '{call.tool}': {exc}"}

    # ------------------------------------------------------------------
    def _handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        return {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "append_file": self._append_file,
            "delete_file": self._delete_file,
            "move_file": self._move_file,
            "copy_file": self._copy_file,
            "list_directory": self._list_directory,
            "create_directory": self._create_directory,
            "file_info": self._file_info,
            "search_files": self._search_files,
            "grep_text": self._grep_text,
            "find_extensions": self._find_extensions,
            "find_large_files": self._find_large_files,
            "project_summary": self._project_summary,
            "run_command": self._run_command,
            "git_status": self._git_status,
            "git_diff": self._git_diff,
            "git_log": self._git_log,
            "git_commit": self._git_commit,
            "git_branch": self._git_branch,
            "propose_patch": self._propose_patch,
        }

    @property
    def root(self) -> str:
        return self.config.project_root

    @property
    def excluded_dirs(self) -> Tuple[str, ...]:
        return tuple(self.config.agent.excluded_dirs)

    # ------------------------------------------------------------------
    # Filesystem tool handlers
    # ------------------------------------------------------------------
    def _read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.read_file(
            self.root,
            args.get("path", ""),
            max_bytes=self.config.agent.max_file_read_bytes,
        )
        return result.to_dict()

    def _write_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path", "")
        content = args.get("content", "")
        require_conf = self.config.agent.require_confirmation

        # If the file already exists, route through the diff/approval flow
        # rather than writing blind, mirroring propose_patch's behavior.
        existing = filesystem.read_file(self.root, path, max_bytes=10_000_000)
        if existing.success and require_conf:
            patch = Patch(path=path, old_content=existing.data["content"], new_content=content, is_new_file=False)
            approved = approve_patch(patch, input_func=self.input_func)
            if not approved:
                return {"success": False, "tool": "write_file", "error": "User did not approve the patch."}
            result = filesystem.write_file(self.root, path, content, require_confirmation=True, confirmed=True)
            return result.to_dict()

        result = filesystem.write_file(self.root, path, content, require_confirmation=False, confirmed=True)
        return result.to_dict()

    def _append_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.append_file(self.root, args.get("path", ""), args.get("content", ""))
        return result.to_dict()

    def _delete_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path", "")
        require_conf = self.config.agent.require_confirmation
        if require_conf:
            from tools.safety import warn_delete

            approved = warn_delete(path, input_func=self.input_func)
            if not approved:
                return {"success": False, "tool": "delete_file", "error": "User did not approve deletion."}
        result = filesystem.delete_file(self.root, path, require_confirmation=False, confirmed=True)
        return result.to_dict()

    def _move_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.move_file(
            self.root,
            args.get("src", ""),
            args.get("dst", ""),
            require_confirmation=self.config.agent.require_confirmation,
            confirmed=False,
        )
        if result.error == "confirmation_required":
            from tools.safety import confirm_action

            approved = confirm_action(
                f"Move '{args.get('src')}' to '{args.get('dst')}' (destination exists and will be overwritten)?",
                danger=True,
                input_func=self.input_func,
            )
            if approved:
                result = filesystem.move_file(self.root, args.get("src", ""), args.get("dst", ""), require_confirmation=False, confirmed=True)
        return result.to_dict()

    def _copy_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.copy_file(
            self.root,
            args.get("src", ""),
            args.get("dst", ""),
            require_confirmation=self.config.agent.require_confirmation,
            confirmed=False,
        )
        if result.error == "confirmation_required":
            from tools.safety import confirm_action

            approved = confirm_action(
                f"Copy '{args.get('src')}' to '{args.get('dst')}' (destination exists and will be overwritten)?",
                danger=False,
                input_func=self.input_func,
            )
            if approved:
                result = filesystem.copy_file(self.root, args.get("src", ""), args.get("dst", ""), require_confirmation=False, confirmed=True)
        return result.to_dict()

    def _list_directory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.list_directory(self.root, args.get("path", "."))
        return result.to_dict()

    def _create_directory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.create_directory(self.root, args.get("path", ""))
        return result.to_dict()

    def _file_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = filesystem.file_info(self.root, args.get("path", ""))
        return result.to_dict()

    # ------------------------------------------------------------------
    # Search tool handlers
    # ------------------------------------------------------------------
    def _search_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = search.search_files(
            self.root,
            args.get("pattern", "*"),
            path=args.get("path", "."),
            excluded_dirs=self.excluded_dirs,
        )
        return result.to_dict()

    def _grep_text(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = search.grep_text(
            self.root,
            args.get("query", ""),
            path=args.get("path", "."),
            excluded_dirs=self.excluded_dirs,
            regex=bool(args.get("regex", False)),
            case_sensitive=bool(args.get("case_sensitive", False)),
        )
        return result.to_dict()

    def _find_extensions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = search.find_extensions(self.root, path=args.get("path", "."), excluded_dirs=self.excluded_dirs)
        return result.to_dict()

    def _find_large_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = search.find_large_files(
            self.root,
            path=args.get("path", "."),
            excluded_dirs=self.excluded_dirs,
            min_size_bytes=int(args.get("min_size_bytes", 1_000_000)),
        )
        return result.to_dict()

    def _project_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        result = search.project_summary(self.root, excluded_dirs=self.excluded_dirs)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Shell tool handler
    # ------------------------------------------------------------------
    def _run_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        command = args.get("command", "")
        timeout = float(args.get("timeout", 60.0))
        require_conf = self.config.agent.require_confirmation

        verdict = check_shell_command(command)
        if verdict.dangerous and require_conf:
            approved = warn_shell_command(command, verdict.reason, input_func=self.input_func)
            if not approved:
                return {"success": False, "tool": "run_command", "error": "User did not approve the command.", "cancelled": True}

        result = shell.run_command(
            command,
            cwd=self.root,
            timeout=timeout,
            require_confirmation=False,
            confirmed=True,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Git tool handlers
    # ------------------------------------------------------------------
    def _git_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return git_tools.git_status(self.root).to_dict()

    def _git_diff(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return git_tools.git_diff(self.root, path=args.get("path"), staged=bool(args.get("staged", False))).to_dict()

    def _git_log(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return git_tools.git_log(self.root, max_count=int(args.get("max_count", 20))).to_dict()

    def _git_commit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return git_tools.git_commit(self.root, args.get("message", ""), add_all=bool(args.get("add_all", False))).to_dict()

    def _git_branch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return git_tools.git_branch(self.root, create=args.get("create")).to_dict()

    # ------------------------------------------------------------------
    # Patch proposal handler
    # ------------------------------------------------------------------
    def _propose_patch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path", "")
        new_content = args.get("new_content", "")

        existing = filesystem.read_file(self.root, path, max_bytes=10_000_000)
        is_new_file = not existing.success
        old_content = existing.data["content"] if existing.success else ""

        patch = Patch(path=path, old_content=old_content, new_content=new_content, is_new_file=is_new_file)

        if not is_new_file and not patch.has_changes:
            return {"success": True, "tool": "propose_patch", "message": f"No changes needed for {path}.", "data": {"applied": False}}

        approved = approve_patch(patch, input_func=self.input_func)
        if not approved:
            return {"success": False, "tool": "propose_patch", "error": "User did not approve the patch.", "data": {"applied": False}}

        result = filesystem.write_file(self.root, path, new_content, require_confirmation=False, confirmed=True)
        out = result.to_dict()
        out["data"] = out.get("data") or {}
        out["data"]["applied"] = True
        out["data"]["diff"] = patch.unified_diff
        return out
