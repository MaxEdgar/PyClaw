"""
ui/panels.py
==============

Rich renderable builders for the various TUI panels: header, sidebar
(project/model info), tool activity log, plan/task status, and status bar.

These functions return Rich renderables (Panel, Table, Text, Group) rather
than printing directly, so they can be composed into the Textual layout
in ui/tui.py or rendered standalone in a simpler Rich-console fallback
mode.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.planner import Plan
from config import Config
from ui.glyphs import glyph, safe_box

# Single source of truth for every base slash command PyClaw recognizes.
# Used by build_help_panel() below AND the live autocomplete dropdown in
# ui/tui.py, so the two can never drift out of sync with each other.
# Each entry: (command, one-line description, usage hint or None).
SLASH_COMMANDS: List[Tuple[str, str, Optional[str]]] = [
    ("/help", "Show command reference", None),
    ("/clear", "Clear the conversation and session memory", None),
    ("/history", "Show recent chat history", None),
    ("/project", "Show or change the active project root", "/project [path]"),
    ("/model", "Show or change model/backend settings", "/model [list|preset <name>|<field> <value>]"),
    ("/tools", "List all available tools", None),
    ("/memory", "Show current session memory summary", None),
    ("/skill", "Manage custom agent skills", "/skill [list|create|show <name>|delete <name>]"),
    ("/theme", "Show or change the color theme", "/theme [list|set <name>]"),
    ("/doctor", "Audit configuration and connectivity for misconfigurations", None),
    ("/todo", "Manage a persistent task list across requests", "/todo [add <text>|done <n>|list|clear]"),
    ("/agent", "Enable or disable agent planning/tool execution", "/agent [on|off|status]"),
    ("/quit", "Exit PyClaw", None),
    ("/exit", "Exit PyClaw", None),
]


def build_header_panel(config: Config, connected: bool) -> Panel:
    """Top header panel showing the PyClaw banner, model, and project."""
    status_color = "green" if connected else "red"
    status_text = "connected" if connected else "disconnected"

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left")
    table.add_row(f"[bold cyan]Model:[/bold cyan] {config.model.model_name}  "
                  f"[dim]({config.model.base_url})[/dim]  "
                  f"[{status_color}]{glyph('bullet')} {status_text}[/{status_color}]")
    table.add_row(f"[bold cyan]Project:[/bold cyan] {config.project_root}")

    return Panel(table, title="[bold]PyClaw[/bold]", border_style="cyan", expand=True, box=safe_box())


def build_plan_panel(plan: Optional[Plan]) -> Panel:
    """Render the current agent plan as a checklist panel."""
    if plan is None or not plan.steps:
        body = Text("No active plan.", style="dim")
    else:
        body = Text(plan.render_checklist())
    return Panel(body, title="[bold]Agent Plan[/bold]", border_style="magenta", box=safe_box())


def build_tool_activity_panel(recent_calls: List[Tuple[str, str]]) -> Panel:
    """Render recent tool calls as a compact activity log.

    Args:
        recent_calls: list of (tool_name, short_arg_summary) tuples, most
            recent last.
    """
    if not recent_calls:
        body = Text("No tool activity yet.", style="dim")
    else:
        lines = [f"{name}({args})" for name, args in recent_calls[-12:]]
        body = Text("\n".join(lines))
    return Panel(body, title="[bold]Tool Activity[/bold]", border_style="yellow", box=safe_box())


def build_status_panel(status: str, tokens_used: int = 0, elapsed: float = 0.0, connected: bool = True) -> Text:
    """Render the current status line as plain styled text (no Panel).

    This is intentionally NOT wrapped in a Rich Panel: the status bar lives
    inside a Textual widget that already draws its own border, so wrapping
    it in a second Panel produced a visible "double border" artifact in the
    terminal. Returning plain Text avoids that entirely while still giving
    a clear, color-coded state indicator.

    Args:
        connected: Whether the LLM server was reachable at the last check.
            When False, the status reads "Offline" regardless of `status`
            -- there is no point telling the user "Idle" (implying PyClaw
            is ready and waiting) when it cannot actually reach a model at
            all. This is re-evaluated periodically by the caller, not just
            once at startup.
    """
    dot = glyph("bullet")
    state = (status or "idle").strip().lower()

    if not connected:
        color, label = "red", "Offline (no LLM server reachable)"
    elif state == "idle":
        color, label = "green", "Idle"
    elif state in ("thinking", "planning..."):
        color, label = "yellow", "Thinking..."
    elif state.startswith("running tool"):
        color, label = "cyan", status
    elif state.startswith("cancel"):
        color, label = "yellow", status
    else:
        color, label = "yellow", status or "Working..."

    text = Text()
    text.append(f" {dot} ", style=f"bold {color}")
    text.append(label, style=f"bold {color}")
    if elapsed:
        text.append(f"   {elapsed:.1f}s", style="dim")
    if tokens_used:
        text.append(f"   |   Tokens: {tokens_used}", style="dim")
    return text


def build_sidebar(config: Config, session_summary: str) -> Panel:
    """Render the left sidebar: project info + session memory summary."""
    body = Text()
    body.append("Project\n", style="bold underline")
    body.append(f"{config.project_root}\n\n")
    body.append("Model\n", style="bold underline")
    body.append(f"{config.model.model_name}\n")
    body.append(f"temp={config.model.temperature}  ctx={config.model.context_size}\n")
    body.append(f"max_tokens={config.model.max_tokens}\n\n")
    body.append("Session\n", style="bold underline")
    body.append(session_summary)
    return Panel(body, title="[bold]Sidebar[/bold]", border_style="blue", box=safe_box())


def build_help_panel() -> Panel:
    """Render the /help slash-command reference."""
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Command")
    table.add_column("Description")
    for cmd, desc, usage in SLASH_COMMANDS:
        table.add_row(usage or cmd, desc)
    table.add_row("Ctrl+X", "Cancel the current request (or decline a pending confirmation)")
    table.add_row("Ctrl+L", "Clear the conversation (same as /clear)")
    table.add_row("Up / Down", "Recall previously submitted input (shell-style history)")
    table.add_row("Ctrl+P", "Open Textual's command palette (change theme, save screenshot, etc.)")
    return Panel(table, title="[bold]PyClaw Help[/bold]", border_style="cyan", box=safe_box())


def build_tools_panel() -> Panel:
    """Render the /tools slash-command listing of all available tools."""
    from llm.prompts import TOOL_SPECS

    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Tool")
    table.add_column("Description")
    for spec in TOOL_SPECS:
        table.add_row(spec["name"], spec["description"])
    return Panel(table, title="[bold]Available Tools[/bold]", border_style="cyan", box=safe_box())
