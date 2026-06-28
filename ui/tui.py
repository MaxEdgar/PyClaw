"""
ui/tui.py
==========

Textual-based terminal UI for PyClaw.

Layout:

    +------------------------------------------------------------+
    | Header: model / project / connection status                |
    +---------------+----------------------------------------------+
    | Sidebar       | Chat history (scrolling)                     |
    | Plan          |                                               |
    | Tool Activity |                                               |
    +---------------+----------------------------------------------+
    | Status bar                                                   |
    | Input box                                                    |
    +------------------------------------------------------------+

The agent loop (agent/executor.py) is synchronous and blocking (it makes
HTTP calls and may prompt for confirmation), so it is run inside a Textual
`work` thread worker. Streaming tokens, tool-call events, and plan updates
are marshalled back to the UI thread via `call_from_thread` so widget
updates always happen on the Textual event loop.

Confirmation prompts (delete/overwrite/dangerous shell commands) normally
use blocking `input()` calls (tools/safety.py, ui/diff_view.py). Inside the
Textual app we override that behavior by injecting a thread-safe
`input_func` that blocks the worker thread on a `threading.Event` while
posting a modal confirmation screen to the UI thread -- giving the same
Y/N approval flow without breaking Textual's async model.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, OptionList, Static
from textual.widgets.option_list import Option

from agent.executor import Executor, ExecutorCallbacks
from agent.planner import Plan
from config import Config
from llm.client import LLMClient
from memory.history import HistoryStore
from memory.session import SessionMemory
from memory.skills import InvalidSkillNameError, SkillStore, validate_skill_name
from ui import panels
from ui.glyphs import glyph


class SplashScreen(Screen):
    """Brief startup animation shown when PyClaw first launches.

    Design constraints, deliberately:
        * Never blocks usability for long -- auto-dismisses itself after
          SPLASH_DURATION_SECONDS regardless of animation state, and any
          keypress dismisses it immediately. A loading screen that can't
          be skipped is a liability, not polish, especially for someone
          opening PyClaw repeatedly through a workday.
        * No network calls, no disk I/O beyond what App.__init__ already
          did before this screen even mounts -- the animation itself does
          nothing that could fail, hang, or vary in duration based on
          system state. It's purely decorative and timed off a fixed
          local interval, so it behaves identically on a fast or slow
          machine (the rest of the app's actual startup work, like
          connecting to the LLM server, already happened in
          PyClawApp.__init__ before this screen is even pushed).
        * Built from plain Static widgets and Rich Text, consistent with
          the rest of the UI -- no animation library dependency.
    """

    SPLASH_DURATION_SECONDS = 1.2
    FRAME_INTERVAL_SECONDS = 0.08

    CSS = """
    SplashScreen {
        align: center middle;
        background: $background;
    }
    #splash-content {
        align: center middle;
        width: auto;
        height: auto;
    }
    """

    # A small set of frames revealing the wordmark progressively, plus a
    # simple "loading dots" cycle underneath -- smooth and simple rather
    # than a busy effect, since this is seen every single launch and
    # should never feel like it's in the way.
    _WORDMARK = "PyClaw"

    def __init__(self) -> None:
        super().__init__()
        self._frame = 0
        self._total_frames = max(1, int(self.SPLASH_DURATION_SECONDS / self.FRAME_INTERVAL_SECONDS))

    def compose(self) -> ComposeResult:
        with Vertical(id="splash-content"):
            yield Static("", id="splash-wordmark")
            yield Static("", id="splash-subtitle")
            yield Static("", id="splash-progress")

    def on_mount(self) -> None:
        self._render_frame()
        self.set_interval(self.FRAME_INTERVAL_SECONDS, self._advance)
        self.set_timer(self.SPLASH_DURATION_SECONDS, self._finish)

    def _advance(self) -> None:
        self._frame += 1
        self._render_frame()
        if self._frame >= self._total_frames:
            self._finish()

    def _render_frame(self) -> None:
        progress = min(1.0, self._frame / self._total_frames)
        revealed = int(len(self._WORDMARK) * progress)

        wordmark_text = Text()
        wordmark_text.append(self._WORDMARK[:revealed], style="bold cyan")
        wordmark_text.append(self._WORDMARK[revealed:], style="dim")
        self.query_one("#splash-wordmark", Static).update(wordmark_text)

        self.query_one("#splash-subtitle", Static).update(
            Text("local AI coding agent", style="dim italic")
        )

        # A simple cycling dot indicator, three dots filling in and
        # resetting, giving a sense of motion without anything jarring.
        dot_count = (self._frame % 4)
        dots = "." * dot_count + " " * (3 - dot_count)
        self.query_one("#splash-progress", Static).update(Text(f"starting{dots}", style="dim"))

    def _finish(self) -> None:
        if not self.is_attached:
            return  # already dismissed (e.g. via keypress) -- avoid a double-pop
        self.dismiss()

    def on_key(self, event) -> None:
        # Any key skips the splash immediately -- never force the user to
        # sit through an animation they've already seen a hundred times.
        self.dismiss()


class ConfirmModal(ModalScreen[bool]):
    """A modal Y/N confirmation dialog used for destructive-action approval.

    Dismissing with True/False resolves the underlying threading.Event-based
    wait in `_threadsafe_input`, unblocking the executor's worker thread.
    """

    CSS = """
    ConfirmModal {
        align: center middle;
    }
    #confirm-box {
        width: 70%;
        max-width: 90;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    #confirm-buttons {
        margin-top: 1;
        align: center middle;
    }
    """

    def __init__(self, message: str):
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        from rich.markup import escape

        with Vertical(id="confirm-box"):
            yield Static(escape(self.message), id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Static("(Y)es", id="confirm-yes")
                yield Static("   ")
                yield Static("(N)o", id="confirm-no")

    def on_key(self, event) -> None:
        key = event.key.lower()
        if key == "y":
            self.dismiss(True)
        elif key in ("n", "escape"):
            self.dismiss(False)


class ChatLog(VerticalScroll):
    """Scrolling chat history widget. Each entry is appended as a Static."""

    def add_user_message(self, text: str) -> None:
        self.mount(Static(Text(f"You: {text}", style="bold white")))
        self.scroll_end(animate=False)

    def add_assistant_start(self) -> Static:
        widget = Static(Text("Agent: ", style="bold green"))
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def add_system_message(self, text: str, style: str = "dim italic") -> None:
        self.mount(Static(Text(text, style=style)))
        self.scroll_end(animate=False)

    def add_tool_message(self, tool: str, summary: str) -> None:
        self.mount(Static(Text(f"  {glyph('arrow')} {tool}: {summary}", style="yellow")))
        self.scroll_end(animate=False)


class PyClawApp(App):
    """Main Textual application for PyClaw."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #sidebar-column {
        width: 34;
        border: round $primary;
    }
    #chat-column {
        width: 1fr;
        border: round $secondary;
    }
    #status-bar {
        height: 3;
        border: round $success;
    }
    #command-autocomplete {
        height: auto;
        max-height: 8;
        border: round $accent;
        display: none;
    }
    #command-autocomplete.visible {
        display: block;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+x", "cancel_turn", "Cancel"),
    ]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.llm_client = LLMClient(config.model)
        self.history = HistoryStore()
        self.session = SessionMemory()
        self.session.set_project_root(config.project_root)

        # Thread-safe confirmation plumbing: the worker thread blocks on
        # this event after requesting a modal; on_confirm_result() (run on
        # the UI thread) sets the stored answer and the event.
        self._confirm_event = threading.Event()
        self._confirm_answer = False
        self._active_confirm_screen: Optional[ConfirmModal] = None

        # Cancellation plumbing: set by the Ctrl+X binding, checked by the
        # executor loop between iterations and mid-stream (see
        # agent/executor.py's `cancel_event` parameter). Cleared at the
        # start of every new turn.
        self._cancel_event = threading.Event()

        # Guided /skill create flow state: when not None, the next plain
        # (non-slash) message submitted is treated as an answer to the
        # current step of skill creation rather than a request to the
        # agent. See _handle_skill_command / _continue_skill_creation.
        self._pending_skill_creation: Optional[Dict[str, Any]] = None

        # Shell-style Up/Down command history (separate from
        # memory.history.HistoryStore, which persists the full
        # conversation transcript to disk -- this is just an in-memory
        # list of raw input strings for the current process, reset on
        # restart, exactly like a shell's in-session history).
        self._input_history: List[str] = []
        self._history_index: Optional[int] = None

        self.skill_store = SkillStore()

        self.executor = Executor(
            config=config,
            llm_client=self.llm_client,
            history=self.history,
            session=self.session,
            input_func=self._threadsafe_input,
            skill_store=self.skill_store,
        )

        self._recent_tool_calls: List[Tuple[str, str]] = []
        self._current_plan: Optional[Plan] = None
        self._is_running = False
        self._turn_start_time = 0.0
        # Tracked separately from _is_running: whether the LLM server was
        # reachable at the last check. Drives the Offline vs Idle status
        # distinction (see ui/panels.py:build_status_panel).
        self._connected = True

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar-column"):
                yield Static(id="sidebar-panel")
                yield Static(id="plan-panel")
                yield Static(id="tool-panel")
            with Vertical(id="chat-column"):
                yield ChatLog(id="chat-log")
        yield Static(id="status-bar")
        yield OptionList(id="command-autocomplete")
        yield Input(placeholder="Ask PyClaw to do something... (/help for commands)", id="input-box")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "PyClaw"
        # Show the splash screen immediately, before any real startup work
        # (theme registration, the LLM health check, etc.) runs -- the
        # animation should appear instantly rather than waiting on
        # anything that could be slow. Real startup work happens in the
        # background while the splash plays, and the main UI's content is
        # already correct underneath by the time the splash dismisses
        # (whether that's via its own timer or the user skipping it).
        self.push_screen(SplashScreen())
        self.run_worker(self._startup_worker, thread=True)

    def _startup_worker(self) -> None:
        """Runs in a background thread, in parallel with the splash
        animation. Does the one genuinely slow part of startup
        (health_check(), a real blocking HTTP call) off the main thread,
        then marshals the rest of on_mount's original work back via
        call_from_thread -- consistent with every other network call in
        this file (see _recheck_connection_worker for the same pattern).
        """
        connected = self.llm_client.health_check()
        self.call_from_thread(self._finish_startup, connected)

    def _finish_startup(self, connected: bool) -> None:
        from ui.themes import THEMES, get_theme

        # Register every PyClaw theme upfront (not just the active one) so
        # they all show up in Textual's built-in command palette
        # (Ctrl+P -> "Change theme"), alongside Textual's own built-ins
        # like Solarized Light, Nord, Gruvbox, and Tokyo Night.
        for theme_def in THEMES.values():
            self._register_textual_theme(theme_def)

        try:
            self._apply_theme(get_theme(self.config.theme_name))
        except KeyError:
            # Not one of PyClaw's own themes -- it may be a Textual
            # built-in (Solarized Light, Nord, Gruvbox, ...) picked via
            # Ctrl+P's native palette and persisted by watch_theme(). Try
            # setting it directly; if that name isn't valid either (e.g.
            # stale config from a Textual version with different built-ins),
            # fall back to Textual's own default rather than crashing startup.
            try:
                self.theme = self.config.theme_name
            except Exception:  # noqa: BLE001 - theming is cosmetic, never fatal
                pass
        self._refresh_sidebar()
        self._refresh_plan_panel()
        self._refresh_tool_panel()
        self.query_one("#input-box", Input).focus()
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_system_message(
            "Welcome to PyClaw. Type a request, or /help for commands. "
            "Press Ctrl+X to cancel a request in progress."
        )

        self._connected = connected
        if self._connected:
            from main import try_auto_detect_model

            try_auto_detect_model(
                self.config, self.llm_client,
                announce=lambda msg: chat_log.add_system_message(msg, style="dim"),
            )
            self.llm_client = LLMClient(self.config.model)  # carries any detected model_name
            self.executor.llm_client = self.llm_client
            self.executor.planner.llm_client = self.llm_client
            self._refresh_sidebar()
        else:
            chat_log.add_system_message(
                f"Could not reach an LLM server at {self.config.model.base_url}. "
                "Start your model server (llama.cpp, Ollama, LM Studio, ...), then try /model "
                "to reconnect, or restart PyClaw.",
                style="bold red",
            )

        self._refresh_status("Idle")
        self.set_interval(15.0, self._recheck_connection)

    def _recheck_connection(self) -> None:
        """Periodic re-check of LLM server reachability, triggered by a
        15s timer (started in on_mount).

        IMPORTANT: set_interval callbacks run on the main UI thread/event
        loop. health_check() makes a real blocking HTTP request (up to two
        sequential 5s-timeout attempts -- see llm/client.py), and calling
        that directly here would freeze the ENTIRE app for up to 10
        seconds every 15 seconds while the server is unreachable --
        including totally unrelated features like Ctrl+P's native command
        palette, since the whole event loop is blocked, not just this
        feature. The actual network call is therefore dispatched to a
        background worker thread; only the lightweight "did the status
        change" UI update happens back on the main thread.
        """
        if self._is_running:
            return
        self.run_worker(self._recheck_connection_worker, thread=True)

    def _recheck_connection_worker(self) -> None:
        """Runs in a background thread (see _recheck_connection). Performs
        the actual blocking health_check() call off the main thread, then
        marshals the (cheap) UI update back via call_from_thread."""
        was_connected = self._connected
        is_connected = self.llm_client.health_check()
        if is_connected != was_connected:
            self.call_from_thread(self._apply_connection_change, was_connected, is_connected)

    def _apply_connection_change(self, was_connected: bool, is_connected: bool) -> None:
        """Runs on the main thread (via call_from_thread from
        _recheck_connection_worker). Updates status and, if the server
        just came back online, re-runs model auto-detection -- the same
        UI-touching work _recheck_connection used to do directly, now
        safely marshalled back instead of running inline on a background
        thread (Textual widgets are not thread-safe to touch directly)."""
        self._connected = is_connected
        self._refresh_status("Idle" if is_connected else "Offline")
        if is_connected and not was_connected:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.add_system_message("LLM server is now reachable.", style="dim")
            from main import try_auto_detect_model

            try_auto_detect_model(
                self.config, self.llm_client,
                announce=lambda msg: chat_log.add_system_message(msg, style="dim"),
            )
            self.llm_client = LLMClient(self.config.model)
            self.executor.llm_client = self.llm_client
            self.executor.planner.llm_client = self.llm_client
            self._refresh_sidebar()

    # ------------------------------------------------------------------
    # Panel refresh helpers
    # ------------------------------------------------------------------
    def _refresh_sidebar(self) -> None:
        panel = panels.build_sidebar(self.config, self.session.summary_text())
        self.query_one("#sidebar-panel", Static).update(panel)

    def _refresh_plan_panel(self) -> None:
        panel = panels.build_plan_panel(self._current_plan)
        self.query_one("#plan-panel", Static).update(panel)

    def _refresh_tool_panel(self) -> None:
        panel = panels.build_tool_activity_panel(self._recent_tool_calls)
        self.query_one("#tool-panel", Static).update(panel)

    def _refresh_status(self, status: str) -> None:
        elapsed = (time.monotonic() - self._turn_start_time) if self._is_running else 0.0
        rendered = panels.build_status_panel(status, elapsed=elapsed, connected=self._connected)
        self.query_one("#status-bar", Static).update(rendered)

        input_box = self.query_one("#input-box", Input)
        if not self._connected:
            input_box.placeholder = "No LLM server reachable -- start one, then try /model to reconnect"
        elif self._is_running:
            input_box.placeholder = f"PyClaw is busy: {status}  (Ctrl+X to cancel)"
        else:
            input_box.placeholder = "Ask PyClaw to do something... (/help for commands)"

    # ------------------------------------------------------------------
    # Slash-command autocomplete
    # ------------------------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        """Show a live, filtered list of matching slash commands as soon as
        the input starts with '/'. Hidden again once the text no longer
        looks like a command-in-progress (empty, or doesn't start with '/',
        or already has a space -- meaning the user moved on to typing
        arguments and the command itself is already chosen)."""
        text = event.value
        autocomplete = self.query_one("#command-autocomplete", OptionList)

        if not text.startswith("/") or " " in text:
            self._hide_autocomplete(autocomplete)
            return

        matches = [(cmd, desc) for cmd, desc, _usage in panels.SLASH_COMMANDS if cmd.startswith(text)]
        if not matches:
            self._hide_autocomplete(autocomplete)
            return

        autocomplete.clear_options()
        for cmd, desc in matches:
            autocomplete.add_option(Option(f"{cmd}  -  {desc}", id=cmd))
        autocomplete.highlighted = 0
        autocomplete.add_class("visible")

    def _hide_autocomplete(self, autocomplete: Optional["OptionList"] = None) -> None:
        widget = autocomplete or self.query_one("#command-autocomplete", OptionList)
        widget.remove_class("visible")
        widget.clear_options()

    def on_option_list_option_selected(self, event: "OptionList.OptionSelected") -> None:
        """When a suggestion is picked (click, or Enter while the list has
        focus), complete it into the input box with a trailing space ready
        for arguments, and hide the dropdown."""
        if event.option_list.id != "command-autocomplete":
            return
        chosen = event.option.id
        if not chosen:
            return
        input_box = self.query_one("#input-box", Input)
        input_box.value = f"{chosen} "
        input_box.focus()
        self._hide_autocomplete()

    def on_key(self, event) -> None:
        """Let Tab complete the currently highlighted autocomplete suggestion,
        and Up/Down recall previously submitted input (shell-style history),
        since both are familiar muscle memory from a real terminal."""
        autocomplete = self.query_one("#command-autocomplete", OptionList)
        if "visible" in autocomplete.classes and event.key == "tab":
            if autocomplete.highlighted is None:
                return
            option = autocomplete.get_option_at_index(autocomplete.highlighted)
            if option is not None and option.id:
                event.prevent_default()
                input_box = self.query_one("#input-box", Input)
                input_box.value = f"{option.id} "
                input_box.focus()
                self._hide_autocomplete(autocomplete)
            return

        if event.key in ("up", "down"):
            self._navigate_input_history(event.key)

    def _navigate_input_history(self, direction: str) -> None:
        """Recall a previously submitted message into the input box, the
        same way pressing Up/Down recalls previous commands in a shell.

        `_input_history` is appended to on every non-empty submission (see
        on_input_submitted); `_history_index` is None while not browsing
        history (the input box reflects whatever the user is actively
        typing), and becomes a real index into `_input_history` once Up is
        pressed, walking further back on repeated presses and forward
        again on Down, until it falls off the front of the list back to
        an empty box.
        """
        if not self._input_history:
            return
        input_box = self.query_one("#input-box", Input)

        if direction == "up":
            if self._history_index is None:
                self._history_index = len(self._input_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            input_box.value = self._input_history[self._history_index]
        else:  # "down"
            if self._history_index is None:
                return
            if self._history_index < len(self._input_history) - 1:
                self._history_index += 1
                input_box.value = self._input_history[self._history_index]
            else:
                self._history_index = None
                input_box.value = ""

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._hide_autocomplete()
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        # Record into Up/Down history before any branching, so both
        # slash commands and plain requests are recallable -- skip an
        # exact repeat of the immediately previous entry, the same way
        # most shells avoid cluttering history with "ls" "ls" "ls".
        if not self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
        self._history_index = None

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        if self._pending_skill_creation is not None:
            self._continue_skill_creation(text)
            return

        if self._is_running:
            self.query_one("#chat-log", ChatLog).add_system_message(
                "Still processing the previous request -- please wait.", style="dim italic"
            )
            return

        # NOTE: this used to call self.llm_client.health_check() directly
        # here -- a blocking HTTP request (up to ~4s worst case) on every
        # single message submission, freezing the whole UI thread each
        # time. on_input_submitted is a main-thread message handler, so it
        # cannot block on network I/O the way a background worker can.
        # Instead, rely on self._connected, which the background
        # _recheck_connection_worker (see on_mount) keeps fresh every 15s
        # without blocking anything. If that cached value is stale (e.g.
        # the server just died in the last few seconds), the agent
        # worker's first LLM call will hit a real LLMConnectionError and
        # report it clearly -- see Executor.run's error handling -- rather
        # than this method needing to verify connectivity itself.
        if not self._connected:
            self.query_one("#chat-log", ChatLog).add_system_message(
                f"No LLM server reachable at {self.config.model.base_url} (as of the last check). "
                "Start your model server, then try again -- "
                "or use /model preset <name> / /model alias use <name> to point at a different one.",
                style="bold red",
            )
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_user_message(text)
        self._is_running = True
        self._turn_start_time = time.monotonic()
        self._recent_tool_calls = []
        self._cancel_event.clear()
        self._refresh_tool_panel()
        self._refresh_status("Thinking...")
        self.run_agent_turn(text)

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------
    def _handle_slash_command(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", ChatLog)
        parts = text.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd in ("/help", "/?"):
            self._print_panel(panels.build_help_panel())
        elif cmd == "/clear":
            self.executor.reset_conversation()
            self.session.reset()
            self.history.clear()
            self._current_plan = None
            self._refresh_sidebar()
            self._refresh_plan_panel()
            chat_log.add_system_message("Conversation and session memory cleared.")
        elif cmd == "/history":
            chat_log.add_system_message(self.history.render_recent_text(20))
        elif cmd == "/project":
            if len(parts) > 1:
                new_path = " ".join(parts[1:])
                try:
                    self.config.set_project_root(new_path)
                    self.session.set_project_root(self.config.project_root)
                    self._refresh_sidebar()
                    chat_log.add_system_message(f"Project root set to {self.config.project_root}")
                except OSError as exc:
                    chat_log.add_system_message(f"Failed to set project root: {exc}", style="bold red")
            else:
                chat_log.add_system_message(f"Current project root: {self.config.project_root}")
        elif cmd == "/model":
            self._handle_model_command(parts)
        elif cmd == "/tools":
            self._print_panel(panels.build_tools_panel())
        elif cmd == "/memory":
            chat_log.add_system_message(self.session.summary_text())
        elif cmd == "/skill":
            self._handle_skill_command(text, parts)
        elif cmd == "/theme":
            self._handle_theme_command(parts)
        elif cmd == "/doctor":
            self._handle_doctor_command()
        elif cmd == "/todo":
            self._handle_todo_command(parts)
        elif cmd == "/agent":
            self._handle_agent_command(parts)
        elif cmd in ("/quit", "/exit"):
            self.exit()
        else:
            chat_log.add_system_message(f"Unknown command: {cmd}. Try /help.", style="bold red")

    def _numbered_model_choices(self) -> List[Dict[str, Any]]:
        """Build the combined, numbered list of switchable model choices --
        every built-in preset, then every saved alias -- in a fixed order
        so "/model 3" means the same thing each time it's shown, matching
        OpenClaw's "/model list" -> "/model 3" pattern (see Models CLI in
        OpenClaw's docs) rather than a separate picker UI."""
        from config import MODEL_PRESETS

        choices = []
        for name, preset in MODEL_PRESETS.items():
            choices.append({"kind": "preset", "name": name, **preset})
        for name, snap in self.config.model_aliases.items():
            choices.append({"kind": "alias", "name": name, **snap})
        return choices

    def _render_numbered_model_list(self) -> str:
        choices = self._numbered_model_choices()
        m = self.config.model
        lines = [
            f"Current: {m.model_name} @ {m.base_url}",
            "",
            "Choose with /model <number> or /model <name>:",
        ]
        for i, choice in enumerate(choices, 1):
            tag = "alias" if choice["kind"] == "alias" else "preset"
            lines.append(f"  {i}. {choice['name']} [{tag}] -- {choice['model_name']} @ {choice['base_url']}")
        lines.append("")
        lines.append(
            "/model alias save <name> to save the current backend as a new alias  *  "
            "/model <field> <value> to set base_url/model_name/temperature/etc. directly"
        )
        return "\n".join(lines)

    def _switch_to_choice(self, choice: Dict[str, Any]) -> None:
        chat_log = self.query_one("#chat-log", ChatLog)
        self.config.update_model(
            base_url=choice["base_url"],
            model_name=choice["model_name"],
            api_key=choice.get("api_key"),
        )
        self.llm_client = LLMClient(self.config.model)
        self.executor.llm_client = self.llm_client
        self.executor.planner.llm_client = self.llm_client
        self._refresh_sidebar()
        chat_log.add_system_message(
            f"Switched to {choice['name']} ({self.config.model.model_name} @ {self.config.model.base_url}) -- checking connection..."
        )
        # Verify reachability in the background rather than blocking the
        # whole UI thread on this HTTP call -- see _recheck_connection's
        # docstring for why a synchronous health_check() here would freeze
        # the entire app, not just this one feature, on a slow connection
        # or a slow/low-end machine.
        self.run_worker(lambda: self._verify_switch_worker(choice["name"]), thread=True)

    def _verify_switch_worker(self, choice_name: str) -> None:
        is_connected = self.llm_client.health_check()
        self.call_from_thread(self._report_switch_connectivity, choice_name, is_connected)

    def _report_switch_connectivity(self, choice_name: str, is_connected: bool) -> None:
        self._connected = is_connected
        self._refresh_status("Idle" if is_connected else "Offline")
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_system_message(
            f"{choice_name}: {'connected' if is_connected else 'could not reach it'}.",
            style="dim" if is_connected else "bold red",
        )

    def _handle_model_command(self, parts: List[str]) -> None:
        chat_log = self.query_one("#chat-log", ChatLog)
        if len(parts) == 1:
            chat_log.add_system_message(self._render_numbered_model_list())
            return

        sub_raw = parts[1].strip()
        sub = sub_raw.lower()

        # "/model <number>" -- the fast path, matching OpenClaw's
        # "/model list" then "/model 3" pattern.
        if sub_raw.isdigit():
            choices = self._numbered_model_choices()
            index = int(sub_raw)
            if 1 <= index <= len(choices):
                self._switch_to_choice(choices[index - 1])
            else:
                chat_log.add_system_message(
                    f"No choice #{index}. Run /model to see the numbered list (1-{len(choices)}).",
                    style="bold red",
                )
            return

        # "/model <name>" -- switch directly by preset or alias name, e.g.
        # "/model ollama" or "/model primary", without needing the
        # "preset"/"alias use" subcommand prefix.
        named_match = next((c for c in self._numbered_model_choices() if c["name"].lower() == sub), None)
        if named_match is not None:
            self._switch_to_choice(named_match)
            return

        if sub == "list":
            chat_log.add_system_message(self._render_numbered_model_list())
            return

        if sub == "alias":
            self._handle_model_alias_command(parts)
            return

        if sub == "preset":
            if len(parts) < 3:
                chat_log.add_system_message("Usage: /model preset <name>  (try /model to see options)")
                return
            preset_name = parts[2].strip()
            if self.config.apply_preset(preset_name):
                self.llm_client = LLMClient(self.config.model)
                self.executor.llm_client = self.llm_client
                self.executor.planner.llm_client = self.llm_client
                self._refresh_sidebar()
                chat_log.add_system_message(
                    f"Switched to preset '{preset_name}' "
                    f"(base_url={self.config.model.base_url}, model={self.config.model.model_name}) -- checking connection..."
                )
                self.run_worker(lambda: self._verify_switch_worker(f"preset '{preset_name}'"), thread=True)
            else:
                from config import MODEL_PRESETS

                chat_log.add_system_message(
                    f"Unknown preset '{preset_name}'. Available: {', '.join(MODEL_PRESETS.keys())}",
                    style="bold red",
                )
            return

        if len(parts) < 3:
            chat_log.add_system_message(
                f"Unknown model/alias/preset '{sub_raw}'. Run /model to see the numbered list, "
                "or /model <field> <value> to set base_url/model_name/temperature/etc. directly.",
                style="bold red",
            )
            return
        field, value = parts[1], parts[2]
        numeric_fields = {"temperature": float, "context_size": int, "max_tokens": int, "top_p": float}
        try:
            if field in numeric_fields:
                value = numeric_fields[field](value)
            self.config.update_model(**{field: value})
            self.llm_client = LLMClient(self.config.model)
            self.executor.llm_client = self.llm_client
            self.executor.planner.llm_client = self.llm_client
            self._refresh_sidebar()
            chat_log.add_system_message(f"Updated {field} = {value}")
        except (ValueError, TypeError) as exc:
            chat_log.add_system_message(f"Invalid value for {field}: {exc}", style="bold red")

    def _handle_model_alias_command(self, parts: List[str]) -> None:
        """Handle /model alias save|use|list|delete <name>.

        Named aliases let you snapshot the currently active backend+model
        under a short name (e.g. "primary", "fast") and switch back to it
        later with one word, instead of re-typing base_url/model_name/
        api_key every time -- the local equivalent of OpenClaw's named
        model slots, adapted to PyClaw's single-active-backend design.
        """
        chat_log = self.query_one("#chat-log", ChatLog)
        # parts is ["/model", "alias", "<rest>"] thanks to the maxsplit=2
        # split in _handle_slash_command -- re-split the remainder here so
        # "/model alias save primary" still works with just two args.
        rest = parts[2].split(maxsplit=1) if len(parts) > 2 else []

        if not rest:
            if self.config.model_aliases:
                lines = [f"{a}: {s['model_name']} @ {s['base_url']}" for a, s in self.config.model_aliases.items()]
                chat_log.add_system_message("Saved aliases:\n" + "\n".join(lines))
            else:
                chat_log.add_system_message(
                    "No aliases saved yet. Usage: /model alias save <name> | use <name> | delete <name>"
                )
            return

        verb = rest[0].lower()
        name = rest[1].strip() if len(rest) > 1 else None

        if verb == "save":
            if not name:
                chat_log.add_system_message("Usage: /model alias save <name>")
                return
            self.config.save_model_alias(name)
            chat_log.add_system_message(
                f"Saved current model ({self.config.model.model_name} @ {self.config.model.base_url}) as alias '{name}'."
            )
            return

        if verb == "use":
            if not name:
                chat_log.add_system_message("Usage: /model alias use <name>")
                return
            if self.config.use_model_alias(name):
                self.llm_client = LLMClient(self.config.model)
                self.executor.llm_client = self.llm_client
                self.executor.planner.llm_client = self.llm_client
                self._refresh_sidebar()
                chat_log.add_system_message(
                    f"Switched to alias '{name}' ({self.config.model.model_name} @ {self.config.model.base_url}) -- checking connection..."
                )
                self.run_worker(lambda: self._verify_switch_worker(f"alias '{name}'"), thread=True)
            else:
                chat_log.add_system_message(
                    f"No alias named '{name}'. Saved aliases: {', '.join(self.config.model_aliases.keys()) or '(none)'}",
                    style="bold red",
                )
            return

        if verb == "delete":
            if not name:
                chat_log.add_system_message("Usage: /model alias delete <name>")
                return
            if self.config.delete_model_alias(name):
                chat_log.add_system_message(f"Deleted alias '{name}'.")
            else:
                chat_log.add_system_message(f"No alias named '{name}'.", style="bold red")
            return

        chat_log.add_system_message("Usage: /model alias save <name> | use <name> | delete <name> | list")

    def _print_panel(self, panel) -> None:
        # Static widgets can render any Rich renderable, including Panels,
        # so slash-command output reuses the same panel builders as the
        # sidebar for visual consistency.
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.mount(Static(panel))
        chat_log.scroll_end(animate=False)

    # ------------------------------------------------------------------
    # /skill: user-defined, persistent agent skills
    # ------------------------------------------------------------------
    def _handle_skill_command(self, full_text: str, parts: List[str]) -> None:
        """Dispatch /skill subcommands: list, show <name>, delete <name>,
        create (starts a guided multi-step flow, see
        _continue_skill_creation). See memory/skills.py for the storage
        format and docs/SKILLS.md for the user-facing guide."""
        chat_log = self.query_one("#chat-log", ChatLog)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "" or sub == "list":
            skills = self.skill_store.list_all()
            if not skills:
                chat_log.add_system_message(
                    "No skills defined yet. Use /skill create to teach PyClaw something reusable."
                )
            else:
                lines = [f"{s.name} -- {s.description}" for s in skills]
                chat_log.add_system_message("\n".join(lines))
            return

        if sub == "show":
            if len(parts) < 3:
                chat_log.add_system_message("Usage: /skill show <name>")
                return
            try:
                skill = self.skill_store.load(parts[2])
                detail = (
                    f"{skill.name}\n"
                    f"Description: {skill.description}\n"
                    f"Trigger keywords: {', '.join(skill.trigger_keywords) or '(none)'}\n\n"
                    f"Instructions:\n{skill.instructions}"
                )
                chat_log.add_system_message(detail)
            except FileNotFoundError as exc:
                chat_log.add_system_message(str(exc), style="bold red")
            return

        if sub == "delete":
            if len(parts) < 3:
                chat_log.add_system_message("Usage: /skill delete <name>")
                return
            name = parts[2]
            if self.skill_store.delete(name):
                chat_log.add_system_message(f"Deleted skill '{name}'.")
            else:
                chat_log.add_system_message(f"No skill named '{name}'.", style="bold red")
            return

        if sub == "create":
            name = parts[2] if len(parts) > 2 else None
            self._pending_skill_creation = {"step": "name" if not name else "description", "name": name}
            if name:
                chat_log.add_system_message(
                    f"Creating skill '{name}'. Describe when this skill should be used "
                    "(one line, e.g. 'Steps to follow before tagging a release'):"
                )
            else:
                chat_log.add_system_message("What should this skill be called? (letters, numbers, hyphens)")
            return

        chat_log.add_system_message("Usage: /skill [list|create|show <name>|delete <name>]")

    def _continue_skill_creation(self, text: str) -> None:
        """Advance the guided /skill create flow by one step using the
        plain-text message the user just submitted as the answer to the
        current step. Called from on_input_submitted instead of routing
        the message to the agent, while a creation flow is pending."""
        chat_log = self.query_one("#chat-log", ChatLog)
        state = self._pending_skill_creation
        step = state["step"]

        if step == "name":
            try:
                state["name"] = validate_skill_name(text)
            except InvalidSkillNameError as exc:
                chat_log.add_system_message(f"{exc} Try again, or type /skill create to cancel and restart.", style="bold red")
                return
            state["step"] = "description"
            chat_log.add_system_message("Describe when this skill should be used (one line):")
            return

        if step == "description":
            state["description"] = text
            state["step"] = "keywords"
            chat_log.add_system_message(
                "Trigger keywords, comma-separated (words in a request that should activate "
                "this skill, e.g. 'release, version bump, changelog') -- or leave blank:"
            )
            return

        if step == "keywords":
            state["trigger_keywords"] = [k.strip() for k in text.split(",") if k.strip()]
            state["step"] = "instructions"
            chat_log.add_system_message(
                "Now the actual instructions -- what should PyClaw do? "
                "(this is the part injected into its context when the skill is relevant):"
            )
            return

        if step == "instructions":
            state["instructions"] = text
            try:
                skill = self.skill_store.create(
                    name=state["name"],
                    description=state.get("description", ""),
                    instructions=state["instructions"],
                    trigger_keywords=state.get("trigger_keywords", []),
                    overwrite=state.get("confirmed_overwrite", False),
                )
                chat_log.add_system_message(f"Skill '{skill.name}' saved.")
                self._pending_skill_creation = None
            except FileExistsError:
                state["step"] = "confirm_overwrite"
                chat_log.add_system_message(
                    f"A skill named '{state['name']}' already exists. Overwrite it? (y/n)"
                )
            return

        if step == "confirm_overwrite":
            if text.strip().lower() in ("y", "yes"):
                state["confirmed_overwrite"] = True
                state["step"] = "instructions"
                self._continue_skill_creation(state["instructions"])
            else:
                chat_log.add_system_message("Cancelled -- the existing skill was not changed.")
                self._pending_skill_creation = None
            return

    # ------------------------------------------------------------------
    # /theme
    # ------------------------------------------------------------------
    def _handle_theme_command(self, parts: List[str]) -> None:
        chat_log = self.query_one("#chat-log", ChatLog)
        from ui.themes import THEMES, get_theme, list_themes_text

        if len(parts) == 1 or parts[1].lower() == "list":
            chat_log.add_system_message(list_themes_text())
            return

        if parts[1].lower() == "set":
            if len(parts) < 3:
                chat_log.add_system_message("Usage: /theme set <name>  (try /theme list to see options)")
                return
            try:
                theme_def = get_theme(parts[2])
            except KeyError as exc:
                chat_log.add_system_message(str(exc), style="bold red")
                return
            self.config.theme_name = theme_def.name
            self.config.save()
            self._apply_theme(theme_def)
            chat_log.add_system_message(f"Theme set to '{theme_def.name}' ({theme_def.label}).")
            return

        chat_log.add_system_message("Usage: /theme [list|set <name>]")

    def _register_textual_theme(self, theme_def) -> bool:
        """Register one PyClaw ThemeDef as a Textual Theme, WITHOUT
        switching to it. Called once per theme at startup (see on_mount)
        so every PyClaw theme appears in Textual's built-in command
        palette (Ctrl+P -> "Change theme") even before the user ever
        runs /theme. Returns True on success, False if Textual's theme
        API isn't available -- callers should treat that as "skip
        theming, keep going" rather than a fatal error."""
        try:
            from textual.theme import Theme as TextualTheme

            textual_theme = TextualTheme(
                name=theme_def.name,
                dark=theme_def.dark,
                primary=theme_def.colors["primary"],
                secondary=theme_def.colors["secondary"],
                accent=theme_def.colors["accent"],
                success=theme_def.colors["success"],
                warning=theme_def.colors["warning"],
                error=theme_def.colors["error"],
                background=theme_def.colors["background"],
                surface=theme_def.colors["surface"],
                foreground=theme_def.colors["foreground"],
            )
            self.register_theme(textual_theme)
            return True
        except Exception:  # noqa: BLE001 - theming is cosmetic, never fatal
            return False

    def _apply_theme(self, theme_def) -> None:
        """Activate an already-registered theme by name. Registers it
        first if it somehow wasn't (defensive -- on_mount registers all
        of them upfront, but this keeps _apply_theme safe to call standalone)."""
        if self._register_textual_theme(theme_def):
            try:
                self.theme = theme_def.name
            except Exception:  # noqa: BLE001 - theming is cosmetic, never fatal
                pass

    def watch_theme(self, theme_name: str) -> None:
        """Textual calls this automatically whenever self.theme changes --
        including when the user picks a theme through Textual's own
        native command palette (Ctrl+P), not just through /theme set.
        Persisting here means a PyClaw theme picked via Ctrl+P survives a
        restart exactly like one picked via /theme set does; a Textual
        built-in theme (Solarized Light, Nord, etc.) picked the same way
        is also saved, so it comes back next launch too."""
        if getattr(self, "config", None) is None:
            return  # called during App.__init__ before self.config exists
        if theme_name != self.config.theme_name:
            self.config.theme_name = theme_name
            self.config.save()

    # ------------------------------------------------------------------
    # /doctor: configuration and safety audit
    # ------------------------------------------------------------------
    def _handle_doctor_command(self) -> None:
        """Run agent/doctor.py's checks and render the findings with
        color-coded severity, so a danger-level finding is impossible to
        miss versus an informational ok."""
        from agent.doctor import run_doctor

        chat_log = self.query_one("#chat-log", ChatLog)
        findings = run_doctor(self.config, self.llm_client)

        style_by_severity = {"ok": "green", "warning": "yellow", "danger": "bold red"}
        label_by_severity = {"ok": "OK", "warning": "WARN", "danger": "DANGER"}

        chat_log.add_system_message("Running configuration and safety audit...", style="dim italic")
        for finding in findings:
            style = style_by_severity.get(finding.severity, "white")
            label = label_by_severity.get(finding.severity, "?")
            chat_log.add_system_message(f"[{label}] {finding.title}\n    {finding.detail}", style=style)

        danger_count = sum(1 for f in findings if f.severity == "danger")
        warning_count = sum(1 for f in findings if f.severity == "warning")
        summary_style = "bold red" if danger_count else ("yellow" if warning_count else "green")
        chat_log.add_system_message(
            f"{danger_count} danger, {warning_count} warning, "
            f"{len(findings) - danger_count - warning_count} ok",
            style=summary_style,
        )

    # ------------------------------------------------------------------
    # /todo: persistent task list across requests and sessions
    # ------------------------------------------------------------------
    def _handle_todo_command(self, parts: List[str]) -> None:
        """Handle /todo add|done|list|clear.

        Distinct from the per-request Agent Plan panel: a todo list is
        user-managed and persists across /clear and across separate
        PyClaw launches, for tracking work that spans many conversations
        (see memory/session.py:TodoItem for the full reasoning).
        """
        from ui.glyphs import glyph

        chat_log = self.query_one("#chat-log", ChatLog)
        sub = parts[1].lower() if len(parts) > 1 else "list"

        if sub == "add":
            text = parts[2].strip() if len(parts) > 2 else ""
            if not text:
                chat_log.add_system_message("Usage: /todo add <text>")
                return
            index = self.session.add_todo(text)
            chat_log.add_system_message(f"Added todo #{index}: {text}")
            return

        if sub == "done":
            if len(parts) < 3 or not parts[2].strip().isdigit():
                chat_log.add_system_message("Usage: /todo done <number>  (see /todo list for numbers)")
                return
            index = int(parts[2].strip())
            if self.session.mark_todo_done(index):
                chat_log.add_system_message(f"Marked todo #{index} done.")
            else:
                chat_log.add_system_message(f"No todo #{index}. See /todo list.", style="bold red")
            return

        if sub == "clear":
            self.session.clear_todos()
            chat_log.add_system_message("Cleared the todo list.")
            return

        if sub == "list" or sub == "":
            rendered = self.session.render_todos_text(done_glyph=glyph("check"), open_glyph=" ")
            chat_log.add_system_message(rendered)
            return

        chat_log.add_system_message("Usage: /todo [add <text>|done <n>|list|clear]")

    # ------------------------------------------------------------------
    # /agent: enable/disable the agent planning + tool-execution system
    # ------------------------------------------------------------------
    def _handle_agent_command(self, parts: List[str]) -> None:
        """Toggle agent.planning_enabled (see config.py and
        agent/executor.py:_run_direct_response). When off, PyClaw answers
        every message directly with no planning, no tool execution, and
        no multi-step reasoning -- a stronger, user-controlled version of
        what the intent classifier already does automatically for
        greetings, applied to everything regardless of intent.
        """
        chat_log = self.query_one("#chat-log", ChatLog)
        sub = parts[1].lower() if len(parts) > 1 else ""

        if sub == "":
            state = "enabled" if self.config.agent.planning_enabled else "disabled"
            chat_log.add_system_message(
                f"Agent system is currently {state}. Usage: /agent [on|off|status]"
            )
            return

        if sub in ("on", "enable", "enabled"):
            self.config.agent.planning_enabled = True
            self.config.save()
            chat_log.add_system_message(
                "Agent system enabled. Planning and tool execution are active for tasks."
            )
            return

        if sub in ("off", "disable", "disabled"):
            self.config.agent.planning_enabled = False
            self.config.save()
            chat_log.add_system_message(
                "Agent system disabled. PyClaw will respond directly to every message -- "
                "no planning, no tool execution, no file/codebase access. Use /agent on to re-enable."
            )
            return

        if sub == "status":
            state = "enabled" if self.config.agent.planning_enabled else "disabled"
            chat_log.add_system_message(f"Agent system is currently {state}.")
            return

        chat_log.add_system_message("Usage: /agent [on|off|status]")

    # ------------------------------------------------------------------
    # Agent execution (background worker)
    # ------------------------------------------------------------------
    def run_agent_turn(self, user_text: str) -> None:
        self.run_worker(lambda: self._agent_worker(user_text), thread=True, exclusive=True)

    def _agent_worker(self, user_text: str) -> None:
        # NOTE: this function executes in a real OS thread (thread=True
        # above is a plain sync callable, not a coroutine -- Textual thread
        # workers must not be `async def`), so it's safe to call the
        # blocking Executor.run() directly. All UI mutation from here must
        # go through call_from_thread.
        assistant_widget_holder: Dict[str, Any] = {}

        def on_status(status: str) -> None:
            self.call_from_thread(self._refresh_status, status)

        def on_plan(plan: Plan) -> None:
            self._current_plan = plan
            self.call_from_thread(self._refresh_plan_panel)

        def on_plan_step_done(index: int) -> None:
            self.call_from_thread(self._refresh_plan_panel)

        def on_tool_call(tool: str, args: Dict[str, Any]) -> None:
            summary = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
            self._recent_tool_calls.append((tool, summary))
            self.call_from_thread(self._refresh_tool_panel)
            self.call_from_thread(
                self.query_one("#chat-log", ChatLog).add_tool_message, tool, summary or "(no args)"
            )

        def on_tool_result(tool: str, result: Dict[str, Any]) -> None:
            message = result.get("message") or result.get("error") or ""
            self.call_from_thread(
                self.query_one("#chat-log", ChatLog).add_system_message,
                f"    {message}",
                "dim",
            )

        def on_token(token: str) -> None:
            # Lazily create the assistant message widget on first token.
            if "widget" not in assistant_widget_holder:
                widget = self.call_from_thread(self.query_one("#chat-log", ChatLog).add_assistant_start)
                assistant_widget_holder["widget"] = widget
                assistant_widget_holder["text"] = "Agent: "
            assistant_widget_holder["text"] += token
            self.call_from_thread(
                assistant_widget_holder["widget"].update, Text(assistant_widget_holder["text"], style="green")
            )

        callbacks = ExecutorCallbacks(
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_plan=on_plan,
            on_plan_step_done=on_plan_step_done,
            on_status=on_status,
        )

        try:
            result = self.executor.run(user_text, callbacks=callbacks, cancel_event=self._cancel_event)
        except Exception as exc:  # noqa: BLE001 - never let the worker crash the app
            self.call_from_thread(
                self.query_one("#chat-log", ChatLog).add_system_message,
                f"Unexpected error: {exc}",
                "bold red",
            )
            result = None

        # If nothing was streamed (e.g. non-streaming mode, or the whole
        # turn was tool calls followed by a non-streamed final answer),
        # make sure the final text still appears in the chat log.
        if result is not None and "widget" not in assistant_widget_holder:
            style = "bold yellow" if result.stopped_reason == "cancelled" else "green"
            prefix = "" if result.stopped_reason == "cancelled" else "Agent: "
            self.call_from_thread(
                self.query_one("#chat-log", ChatLog).add_system_message,
                f"{prefix}{result.final_text}",
                style,
            )

        self._is_running = False
        self.call_from_thread(self._refresh_status, "Idle")
        self.call_from_thread(self._refresh_sidebar)

    # ------------------------------------------------------------------
    # Thread-safe confirmation bridge
    # ------------------------------------------------------------------
    def _threadsafe_input(self, prompt: str) -> str:
        """Replacement for builtin input() used by tools/safety.py and
        ui/diff_view.py when running inside the Textual app.

        This blocks the calling (worker) thread until the UI thread shows
        a ConfirmModal and the user answers, then returns "y" or "n" so
        the existing `answer in ("y", "yes")` checks in safety.py / diff
        approval keep working unchanged.
        """
        self._confirm_event.clear()

        def show_modal() -> None:
            def handle_result(confirmed: Optional[bool]) -> None:
                self._active_confirm_screen = None
                self._confirm_answer = bool(confirmed)
                self._confirm_event.set()

            screen = ConfirmModal(prompt or "Confirm action?")
            self._active_confirm_screen = screen
            self.push_screen(screen, handle_result)

        self.call_from_thread(show_modal)
        self._confirm_event.wait()
        return "y" if self._confirm_answer else "n"

    # ------------------------------------------------------------------
    def action_clear_chat(self) -> None:
        self._handle_slash_command("/clear")

    def action_cancel_turn(self) -> None:
        """Ctrl+X: cancel whatever PyClaw is currently doing.

        If a confirmation modal (Y/N approval) is open, this dismisses it
        as "No" so a pending destructive action is not left hanging. If
        the agent is mid-response (planning, thinking, streaming, or
        running a tool), this sets the cancel event that agent/executor.py
        checks between iterations and mid-stream, so generation stops at
        the next safe point rather than running to completion.
        """
        if self._active_confirm_screen is not None:
            try:
                self._active_confirm_screen.dismiss(False)
            except Exception:  # noqa: BLE001 - dismissing a stale screen reference must never crash the app
                pass
            self.query_one("#chat-log", ChatLog).add_system_message(
                "Cancelled the pending confirmation (treated as No).", style="bold yellow"
            )
            return

        if not self._is_running:
            self.query_one("#chat-log", ChatLog).add_system_message(
                "Nothing is currently running.", style="dim italic"
            )
            return

        self._cancel_event.set()
        self._refresh_status("Cancelling...")
        self.query_one("#chat-log", ChatLog).add_system_message(
            "Cancelling current request...", style="bold yellow"
        )


def run_tui(config: Config) -> None:
    """Entry point used by main.py to launch the Textual application."""
    app = PyClawApp(config)
    app.run()
