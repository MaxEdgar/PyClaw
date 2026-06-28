"""
agent/executor.py
====================

The executor runs the core agent loop:

    1. Receive a user request.
    2. (Optionally) generate a Plan via agent/planner.py.
    3. Repeatedly: ask the LLM for the next step; if it's a tool call,
       execute it via agent/tool_router.py and feed the result back; if
       it's a plain-text final answer, stop and return it.
    4. Enforce a maximum iteration count so a confused model can't loop
       forever.

The executor is UI-agnostic: it communicates progress via callbacks
(on_token, on_tool_call, on_tool_result, on_plan) rather than printing
directly, so both the Rich/Textual TUI and a hypothetical headless mode
can drive it identically.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from memory.skills import SkillStore

from agent.intent import Intent, allows_agent_behavior, classify
from agent.planner import Plan, Planner
from agent.project_instructions import read_project_instructions, render_for_prompt
from agent.tool_router import ToolRouter, extract_tool_call
from config import Config
from llm.client import LLMClient, LLMConnectionError, LLMResponseError
from llm.prompts import build_chat_system_message, build_system_message, build_tool_message, build_user_message
from memory.history import HistoryStore
from memory.session import SessionMemory

# Callback type aliases for clarity.
OnTokenCallback = Callable[[str], None]
OnToolCallCallback = Callable[[str, Dict[str, Any]], None]
OnToolResultCallback = Callable[[str, Dict[str, Any]], None]
OnPlanCallback = Callable[[Plan], None]
OnPlanStepCallback = Callable[[int], None]
OnStatusCallback = Callable[[str], None]


@dataclass
class ExecutionResult:
    """Final outcome of a single run() call."""

    final_text: str
    plan: Optional[Plan] = None
    iterations: int = 0
    tool_calls_made: int = 0
    stopped_reason: str = "final_answer"  # final_answer | max_iterations | error
    error: Optional[str] = None


@dataclass
class ExecutorCallbacks:
    """Bundle of optional UI callbacks the executor will invoke as it runs."""

    on_token: Optional[OnTokenCallback] = None
    on_tool_call: Optional[OnToolCallCallback] = None
    on_tool_result: Optional[OnToolResultCallback] = None
    on_plan: Optional[OnPlanCallback] = None
    on_plan_step_done: Optional[OnPlanStepCallback] = None
    on_status: Optional[OnStatusCallback] = None


class Executor:
    """Coordinates planner, LLM client, and tool router for one user turn."""

    # How long _project_context_summary() trusts its cached result before
    # re-walking the project directory. Short enough that a structural
    # change (new files created mid-session) is picked up again quickly;
    # long enough to skip the repeat disk I/O for consecutive requests
    # typed within the same minute of conversation. See
    # _project_context_summary's docstring for the full reasoning.
    PROJECT_SUMMARY_CACHE_SECONDS = 30.0

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        history: Optional[HistoryStore] = None,
        session: Optional[SessionMemory] = None,
        input_func: Optional[Callable[[str], str]] = None,
        skill_store: Optional["SkillStore"] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.history = history or HistoryStore()
        self.session = session or SessionMemory()
        self.tool_router = ToolRouter(config, input_func=input_func)
        self.planner = Planner(llm_client)

        from memory.skills import SkillStore

        self.skill_store = skill_store or SkillStore()

        # Running chat message list for the current conversation. Seeded
        # with the system prompt; user/assistant/tool turns are appended
        # as the loop progresses. messages[0] is re-rendered at the start
        # of each run() call to include any skills relevant to that turn's
        # request (see _refresh_system_message), so it always reflects the
        # current request rather than whatever was relevant on turn one.
        self.messages: List[Dict[str, str]] = [build_system_message()]

    # ------------------------------------------------------------------
    def reset_conversation(self) -> None:
        """Clear the in-memory message list back to just the system prompt
        (used by the /clear slash command)."""
        self.messages = [build_system_message()]

    # ------------------------------------------------------------------
    def _refresh_system_message(self, user_request: str) -> None:
        """Rebuild messages[0] to include skills relevant to this specific
        request, plus the project's PYCLAW.md instructions file if one
        exists. Run once per turn, before the user message is appended,
        so relevance is judged against what the user is asking *right
        now* rather than whatever was true on the first turn of the
        conversation."""
        try:
            relevant = self.skill_store.find_relevant(user_request)
        except Exception:  # noqa: BLE001 - a broken skill file must never break the agent loop
            relevant = []
        skills_block = "\n\n".join(s.render_for_prompt() for s in relevant)

        try:
            project_instructions = read_project_instructions(self.config.project_root)
        except Exception:  # noqa: BLE001 - an unreadable PYCLAW.md must never break the agent loop
            project_instructions = None
        project_block = render_for_prompt(project_instructions) if project_instructions else ""

        combined_block = "\n\n".join(b for b in (project_block, skills_block) if b)
        self.messages[0] = build_system_message(context_block=combined_block)

    # Phrases that suggest a request is a simple question rather than a
    # multi-step task -- used by _should_make_plan() to skip the planning
    # round-trip (a full extra LLM call) for requests unlikely to need
    # one. Deliberately conservative: only short requests starting with
    # one of these are skipped, so anything ambiguous still gets a plan
    # rather than risk under-planning a real multi-step task.
    _SIMPLE_REQUEST_PREFIXES = (
        "what does", "what is", "what are", "explain", "why does", "why is",
        "how does", "how do i", "show me", "where is", "where does",
        "what's", "whats", "summarize", "summarise", "list",
    )
    _SIMPLE_REQUEST_MAX_WORDS = 20

    @classmethod
    def _should_make_plan(cls, user_request: str) -> bool:
        """Decide whether a request looks simple enough to skip the
        planning step. This is a heuristic, not a guarantee -- it only
        ever returns False (skip planning) on a fairly strong, narrow
        signal (short length AND a recognized simple-question opener);
        anything else still gets a plan. Skipping planning does not skip
        tool use: the executor's main tool-call loop runs exactly the
        same either way, so an under-confident skip costs at most one
        extra back-and-forth, not a missing capability.
        """
        text = user_request.strip().lower()
        if not text or len(text.split()) > cls._SIMPLE_REQUEST_MAX_WORDS:
            return True
        return not any(text.startswith(prefix) for prefix in cls._SIMPLE_REQUEST_PREFIXES)

    # ------------------------------------------------------------------
    def run(
        self,
        user_request: str,
        callbacks: Optional[ExecutorCallbacks] = None,
        make_plan: Optional[bool] = None,
        cancel_event: Optional["threading.Event"] = None,
    ) -> ExecutionResult:
        """Run the full agent loop for a single user request and return the
        final answer text.

        Args:
            make_plan: Whether to run the planning step first. Defaults to
                None, which applies _should_make_plan()'s heuristic (skip
                planning for short, simple-looking questions, saving one
                LLM round-trip per message on the common case) -- pass an
                explicit True/False to override the heuristic.
            cancel_event: Optional threading.Event. If set (by the UI, e.g.
                a Ctrl+X keybinding), the loop stops at the next safe checkpoint
                (between tool-call iterations, or mid-stream on the next token)
                and returns whatever partial answer/plan exists so far rather
                than blocking until the model naturally finishes.
        """
        cb = callbacks or ExecutorCallbacks()

        # --------------------------------------------------------------
        # Intent classification gate -- runs before ANYTHING else, mirrors
        # exactly the contract described in agent/intent.py: no planning,
        # no tool execution, and no agent loop may begin unless the intent
        # is allowed to (TASK_INTENT or TOOL_REQUEST_INTENT). A CHAT_INTENT
        # message (greetings, small talk, short acknowledgements) is
        # answered directly below and returns immediately -- it never
        # reaches the planner, the tool router, or the main loop.
        # --------------------------------------------------------------
        intent_result = classify(user_request)
        self.history.append("user", user_request)

        agent_enabled = getattr(self.config.agent, "planning_enabled", True)

        if not agent_enabled or not allows_agent_behavior(intent_result.intent):
            return self._run_direct_response(user_request, cb, intent_result, agent_disabled=not agent_enabled)

        if make_plan is None:
            # TOOL_REQUEST_INTENT skips planning even though tools are
            # allowed: a single named action ("run pytest", "git status")
            # rarely benefits from a multi-step plan, and skipping it saves
            # an LLM round-trip on the common case -- the same reasoning
            # _should_make_plan already applied heuristically, now backed
            # by the classifier's more specific signal when available.
            if intent_result.intent == Intent.TOOL_REQUEST:
                make_plan = False
            else:
                make_plan = self._should_make_plan(user_request)

        self._refresh_system_message(user_request)
        self.session.set_task(user_request)
        self.messages.append(build_user_message(user_request))

        plan: Optional[Plan] = None
        if make_plan:
            if cb.on_status:
                cb.on_status("Planning...")
            try:
                plan = self.planner.create_plan(user_request, project_context=self._project_context_summary())
            except Exception:  # noqa: BLE001 - planning must never crash the loop
                plan = None
            if plan is not None:
                self.session.set_plan(plan.as_text_list())
                if cb.on_plan:
                    cb.on_plan(plan)

        if cancel_event is not None and cancel_event.is_set():
            return self._cancelled_result(plan, iterations=0, tool_calls_made=0)

        max_iterations = self.config.agent.max_tool_iterations
        tool_calls_made = 0

        for iteration in range(1, max_iterations + 1):
            if cancel_event is not None and cancel_event.is_set():
                return self._cancelled_result(plan, iterations=iteration - 1, tool_calls_made=tool_calls_made)

            if cb.on_status:
                cb.on_status("Thinking...")

            try:
                model_text = self._get_model_response(cb, cancel_event=cancel_event)
            except (LLMConnectionError, LLMResponseError) as exc:
                error_msg = (
                    f"Could not get a response from the LLM server: {exc}\n\n"
                    "Check that your local llama.cpp server is running and reachable "
                    "(see /model to view or change the server URL)."
                )
                self.history.append("assistant", error_msg)
                return ExecutionResult(
                    final_text=error_msg,
                    plan=plan,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    stopped_reason="error",
                    error=str(exc),
                )

            parsed_call = extract_tool_call(model_text)

            if cancel_event is not None and cancel_event.is_set():
                # Cancelled while/after generating this response -- stop here
                # rather than dispatching a (possibly destructive) tool call
                # the user no longer wants executed.
                return self._cancelled_result(plan, iterations=iteration, tool_calls_made=tool_calls_made)

            if parsed_call is None:
                # Plain natural-language answer -- the model is done.
                self.messages.append({"role": "assistant", "content": model_text})
                self.history.append("assistant", model_text)
                if plan is not None:
                    for i in range(len(plan.steps)):
                        plan.mark_done(i)
                        if cb.on_plan_step_done:
                            cb.on_plan_step_done(i)
                return ExecutionResult(
                    final_text=model_text,
                    plan=plan,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    stopped_reason="final_answer",
                )

            if parsed_call.tool == "final_answer":
                final_text = str(parsed_call.arguments.get("text", "")).strip() or model_text
                self.messages.append({"role": "assistant", "content": model_text})
                self.history.append("assistant", final_text)
                return ExecutionResult(
                    final_text=final_text,
                    plan=plan,
                    iterations=iteration,
                    tool_calls_made=tool_calls_made,
                    stopped_reason="final_answer",
                )

            # Record the assistant's tool-call message in conversation history
            # so the model retains awareness of what it already requested.
            self.messages.append({"role": "assistant", "content": model_text})
            self.history.append("assistant", model_text)

            if cb.on_tool_call:
                cb.on_tool_call(parsed_call.tool, parsed_call.arguments)
            if cb.on_status:
                cb.on_status(f"Running tool: {parsed_call.tool}...")

            result = self.tool_router.dispatch(parsed_call)
            tool_calls_made += 1

            # Track touched files in session memory for sidebar display.
            path_arg = parsed_call.arguments.get("path") or parsed_call.arguments.get("src")
            if path_arg and parsed_call.tool in ("read_file", "write_file", "propose_patch", "append_file"):
                self.session.touch_file(str(path_arg))

            if cb.on_tool_result:
                cb.on_tool_result(parsed_call.tool, result)
            self.history.append("tool", str(result), tool_name=parsed_call.tool)

            tool_message = build_tool_message(parsed_call.tool, result)
            self.messages.append(tool_message)

        # Exhausted max_iterations without a final answer.
        timeout_msg = (
            f"Stopped after {max_iterations} tool-call iterations without a final answer. "
            "The task may be too complex for the current model/iteration limit, or the "
            "model may be stuck in a loop. You can raise max_tool_iterations in config, "
            "or try breaking the request into smaller steps."
        )
        self.history.append("assistant", timeout_msg)
        return ExecutionResult(
            final_text=timeout_msg,
            plan=plan,
            iterations=max_iterations,
            tool_calls_made=tool_calls_made,
            stopped_reason="max_iterations",
        )

    # ------------------------------------------------------------------
    def _get_model_response(self, cb: ExecutorCallbacks, cancel_event: Optional["threading.Event"] = None) -> str:
        """Get the model's next response, streaming tokens through on_token
        if streaming is enabled, or making a single blocking call otherwise.

        If `cancel_event` becomes set while a stream is in progress, this
        stops consuming further chunks immediately (rather than waiting for
        the model to finish) and returns whatever text was collected so far.
        Note: the underlying HTTP connection to the LLM server is closed by
        the generator going out of scope -- the server may keep generating
        briefly in the background, but PyClaw stops listening for it.
        """
        if self.config.model.stream and cb.on_token:
            collected = []
            for chunk in self.llm_client.stream_chat(self.messages):
                if cancel_event is not None and cancel_event.is_set():
                    break
                if chunk.delta:
                    collected.append(chunk.delta)
                    cb.on_token(chunk.delta)
                if chunk.finished:
                    break
            return "".join(collected)

        response = self.llm_client.chat(self.messages)
        return response.content

    # ------------------------------------------------------------------
    def _cancelled_result(self, plan: Optional[Plan], iterations: int, tool_calls_made: int) -> ExecutionResult:
        """Build the ExecutionResult returned when a turn is cancelled
        mid-flight by the user (e.g. via a Ctrl+key cancel binding)."""
        message = "Cancelled by user."
        self.history.append("assistant", message)
        return ExecutionResult(
            final_text=message,
            plan=plan,
            iterations=iterations,
            tool_calls_made=tool_calls_made,
            stopped_reason="cancelled",
        )

    # ------------------------------------------------------------------
    def _run_direct_response(
        self,
        user_request: str,
        cb: ExecutorCallbacks,
        intent_result,
        agent_disabled: bool,
    ) -> ExecutionResult:
        """Direct-response path: a single LLM call with NO tool
        information in its context, no planning, and no agent loop.

        Reached in exactly two situations, both of which must bypass all
        agent machinery entirely per the safety rule in agent/intent.py:
            1. The message was classified as CHAT_INTENT (or SYSTEM_INTENT,
               which should never actually reach the executor in practice
               since slash commands are intercepted earlier, but is
               included here defensively rather than falling through to
               the agent loop if it ever did).
            2. The user has disabled agent behavior entirely via the
               planning_enabled config toggle (see config.py /
               AgentConfig.planning_enabled and the /agent slash command),
               in which case EVERY message takes this path regardless of
               what the classifier says -- direct-response mode means
               direct-response for everything, not just greetings.

        This does not touch self.messages (the main agent conversation
        history used by the tool-call loop) at all, so a chat exchange
        never pollutes the context the next real task-turn will see, and
        a later TASK_INTENT message starts with exactly the same
        system-prompt-plus-skills context it always would have.
        """
        if cb.on_status:
            cb.on_status("Thinking..." if agent_disabled else "Responding...")

        chat_messages = [build_chat_system_message(), build_user_message(user_request)]

        try:
            if self.config.model.stream and cb.on_token:
                collected = []
                for chunk in self.llm_client.stream_chat(chat_messages):
                    if chunk.delta:
                        collected.append(chunk.delta)
                        cb.on_token(chunk.delta)
                    if chunk.finished:
                        break
                final_text = "".join(collected)
            else:
                response = self.llm_client.chat(chat_messages)
                final_text = response.content
        except (LLMConnectionError, LLMResponseError) as exc:
            final_text = (
                f"Could not get a response from the LLM server: {exc}\n\n"
                "Check that your local llama.cpp server is running and reachable."
            )
            self.history.append("assistant", final_text)
            return ExecutionResult(final_text=final_text, stopped_reason="error", error=str(exc))

        final_text = final_text.strip() or "..."
        self.history.append("assistant", final_text)
        return ExecutionResult(
            final_text=final_text,
            plan=None,
            iterations=0,
            tool_calls_made=0,
            stopped_reason="chat" if not agent_disabled else "agent_disabled",
        )

    # ------------------------------------------------------------------
    def _project_context_summary(self) -> str:
        """Build a brief project context string to seed the planner with,
        without consuming much context budget.

        Cached for PROJECT_SUMMARY_CACHE_SECONDS: this does a full
        recursive directory walk (see tools/search.py:project_summary),
        which is real disk I/O that doesn't need to repeat on every single
        request -- a project's file structure essentially never changes
        between two messages typed a few seconds apart, so re-walking the
        whole tree each time is wasted work, more noticeably so on a
        low-end machine or a large project. The cache is keyed on
        project_root so switching projects with /project still gets a
        fresh walk immediately rather than serving a stale cached summary
        from a different directory.
        """
        import time

        from tools import search

        now = time.monotonic()
        cached = getattr(self, "_project_summary_cache", None)
        if cached is not None:
            cached_root, cached_at, cached_text = cached
            if cached_root == self.config.project_root and (now - cached_at) < self.PROJECT_SUMMARY_CACHE_SECONDS:
                return cached_text

        result = search.project_summary(self.config.project_root, excluded_dirs=tuple(self.config.agent.excluded_dirs))
        if not result.success:
            text = ""
        else:
            data = result.data
            stack = ", ".join(data.get("detected_stack", [])) or "unknown"
            text = (
                f"Project root: {data.get('project_root')}\n"
                f"Total files: {data.get('total_files')}\n"
                f"Detected stack: {stack}\n"
                f"Top-level entries: {', '.join(data.get('top_level_entries', [])[:15])}"
            )

        self._project_summary_cache = (self.config.project_root, now, text)
        return text
