#!/usr/bin/env python3
"""
main.py
========

Entry point for PyClaw.

Usage:
    python main.py                      # launch the TUI in the current directory
    python main.py --project /path/to/project
    python main.py --base-url http://127.0.0.1:8080 --model qwen2.5-coder
    python main.py --no-tui             # run a simple line-based REPL instead of the Textual UI
    python main.py --config /custom/path/config.json

PyClaw connects to a local llama.cpp server (or any OpenAI-compatible
chat completion endpoint) and provides a Claude-Code-like terminal
experience for reading, searching, and editing a local codebase.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from config import Config
from llm.client import LLMClient
from rich.console import Console
from ui.glyphs import glyph

console = Console()


def _is_termux() -> bool:
    """Detect whether we're running inside Termux on Android.

    Termux's soft keyboard is tied to tapping its native terminal view, and
    Textual repainting the whole screen on every keystroke is a known
    trigger for the keyboard intermittently failing to reappear (a Termux
    limitation, not something fixable from inside a Python TUI -- see
    termux/termux-app issues #2077, #2551, and termux-packages #24534).
    The simple line-based REPL (--no-tui) uses real blocking input() calls,
    which Termux recognizes as a normal text field tap target and does not
    have this problem, so PyClaw defaults to that mode on Termux.
    """
    return "com.termux" in os.environ.get("PREFIX", "") or os.path.exists("/data/data/com.termux")


def _is_wsl() -> bool:
    """Detect whether we're running inside Windows Subsystem for Linux."""
    try:
        with open("/proc/version", "r") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def describe_environment() -> str:
    """Return a short human-readable label for the detected environment,
    used by /doctor-style diagnostics and startup messages."""
    if _is_termux():
        return "Termux (Android)"
    if _is_wsl():
        return "WSL (Windows Subsystem for Linux)"
    if sys.platform.startswith("linux"):
        return "Linux"
    if sys.platform == "darwin":
        return "macOS"
    if sys.platform.startswith("win"):
        return "Windows"
    return sys.platform


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyclaw",
        description="PyClaw -- a local AI coding assistant for Termux/Linux.",
    )
    parser.add_argument(
        "--project", "-p", default=None, help="Path to the project directory to operate on (default: current directory)."
    )
    parser.add_argument("--base-url", default=None, help="Base URL of the llama.cpp / OpenAI-compatible server.")
    parser.add_argument("--model", dest="model_name", default=None, help="Model name to report to the server.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum tokens to generate per response.")
    parser.add_argument("--context-size", type=int, default=None, help="Context window size.")
    parser.add_argument("--config", default=None, help="Path to a custom config.json file.")
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Run a simple line-based REPL instead of the full Textual TUI (useful over very limited terminals, SSH pipes, or Termux where the soft keyboard can misbehave with the full TUI).",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Force the full Textual TUI even on Termux, where --no-tui is the default due to a soft-keyboard quirk.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="DANGEROUS: disable confirmation prompts for destructive actions. Intended only for scripted/automated testing.",
    )
    return parser.parse_args(argv)


def try_auto_detect_model(config: Config, llm_client: LLMClient, announce=None) -> None:
    """If model_name is "auto" (or was never explicitly set), query the
    server for the model it actually has loaded and persist that name.

    This only runs the network call when model_name is literally "auto" --
    once a name is detected (or the user sets one explicitly via /model),
    it's treated as the explicit choice and PyClaw will not silently
    override it again on a later run.

    `announce`, if given, is called with a short status string so the
    caller (REPL or TUI) can surface what happened -- detected a name,
    couldn't reach the server, or endpoint not supported -- without this
    function needing to know whether it's printing to a console or a chat
    log widget.
    """
    if config.model.model_name != "auto":
        return

    detected = llm_client.detect_model()
    if detected:
        config.update_model(model_name=detected)
        if announce:
            announce(f"Auto-detected model: {detected}")
    else:
        if announce:
            announce(
                "Could not auto-detect a model name from the server "
                "(it may be unreachable, or its /v1/models endpoint may not be supported). "
                "PyClaw will still work -- 'auto' is sent as the model field, which most "
                "single-model local servers accept and ignore."
            )


def build_config(args: argparse.Namespace) -> Config:
    """Load config from disk, then apply any CLI overrides."""
    config = Config.load(args.config)

    if args.project:
        resolved = str(Path(args.project).expanduser().resolve())
        if not Path(resolved).is_dir():
            console.print(f"[bold red]Error:[/bold red] project path does not exist or is not a directory: {resolved}")
            sys.exit(1)
        config.project_root = resolved

    if args.base_url:
        config.model.base_url = args.base_url
    if args.model_name:
        config.model.model_name = args.model_name
    if args.temperature is not None:
        config.model.temperature = args.temperature
    if args.max_tokens is not None:
        config.model.max_tokens = args.max_tokens
    if args.context_size is not None:
        config.model.context_size = args.context_size
    if args.no_confirm:
        config.agent.require_confirmation = False

    config.save()
    return config


def print_startup_banner() -> None:
    """Brief, dependency-light startup banner for the simple REPL.

    The full Textual UI has an animated splash screen (see
    ui/tui.py:SplashScreen); the REPL is deliberately minimal/dependency-
    light (it exists specifically for constrained terminals -- see
    run_simple_repl's docstring), so its equivalent is a single styled
    print rather than an animation loop, consistent with that constraint.
    """
    console.print("[bold cyan]PyClaw[/bold cyan] [dim]-- local AI coding agent[/dim]")


def run_simple_repl(config: Config) -> None:
    """A minimal, dependency-light fallback REPL for terminals that can't
    run the full Textual UI (e.g. very small/dumb terminals over certain
    SSH or serial connections). Uses plain blocking input() for both chat
    and confirmation prompts.
    """
    from agent.executor import Executor, ExecutorCallbacks
    from agent.planner import Plan
    from memory.history import HistoryStore
    from memory.session import SessionMemory
    from memory.skills import SkillStore
    from ui import panels

    print_startup_banner()

    llm_client = LLMClient(config.model)
    history = HistoryStore()
    session = SessionMemory()
    session.set_project_root(config.project_root)
    skill_store = SkillStore()
    executor = Executor(config=config, llm_client=llm_client, history=history, session=session, skill_store=skill_store)

    connected = llm_client.health_check()
    if connected:
        try_auto_detect_model(config, llm_client, announce=lambda msg: console.print(f"[dim]{msg}[/dim]"))
        llm_client = LLMClient(config.model)  # rebuild so it carries the detected model_name
        executor.llm_client = llm_client
        executor.planner.llm_client = llm_client

    console.print(panels.build_header_panel(config, connected))
    console.print("[dim]Type your request, or /help for commands. Ctrl+C to exit.[/dim]\n")

    while True:
        try:
            user_text = console.input("[bold cyan]> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        user_text = user_text.strip()
        if not user_text:
            continue

        if user_text in ("/quit", "/exit"):
            break
        if user_text == "/help":
            console.print(panels.build_help_panel())
            continue
        if user_text == "/tools":
            console.print(panels.build_tools_panel())
            continue
        if user_text == "/clear":
            executor.reset_conversation()
            session.reset()
            history.clear()
            console.print("[dim]Cleared.[/dim]")
            continue
        if user_text == "/history":
            console.print(history.render_recent_text(20))
            continue
        if user_text == "/memory":
            console.print(session.summary_text())
            continue
        if user_text.startswith("/skill"):
            from memory.skills import InvalidSkillNameError

            skill_parts = user_text.split(maxsplit=2)
            sub = skill_parts[1] if len(skill_parts) > 1 else "list"
            if sub == "list":
                skills = skill_store.list_all()
                if not skills:
                    console.print("[dim]No skills defined yet.[/dim]")
                for s in skills:
                    console.print(f"{s.name} -- {s.description}")
            elif sub == "show" and len(skill_parts) == 3:
                try:
                    s = skill_store.load(skill_parts[2])
                    console.print(f"{s.name}\nDescription: {s.description}\nKeywords: {', '.join(s.trigger_keywords)}\n\n{s.instructions}")
                except FileNotFoundError as exc:
                    console.print(f"[bold red]{exc}[/bold red]")
            elif sub == "delete" and len(skill_parts) == 3:
                console.print("Deleted." if skill_store.delete(skill_parts[2]) else "No such skill.")
            elif sub == "create":
                console.print(
                    "[dim]Creating a skill in the REPL uses one line: "
                    "name | description | keyword1,keyword2 | instructions[/dim]"
                )
                try:
                    raw = console.input("[bold cyan]skill> [/bold cyan]")
                    name, description, keywords, instructions = [p.strip() for p in raw.split("|", 3)]
                    skill = skill_store.create(
                        name=name,
                        description=description,
                        instructions=instructions,
                        trigger_keywords=[k.strip() for k in keywords.split(",") if k.strip()],
                    )
                    console.print(f"Saved skill '{skill.name}'.")
                except (ValueError, InvalidSkillNameError) as exc:
                    console.print(f"[bold red]Could not create skill: {exc}[/bold red]")
                except FileExistsError as exc:
                    console.print(f"[bold red]{exc}[/bold red]")
            else:
                console.print("Usage: /skill [list|create|show <name>|delete <name>]")
            continue
        if user_text.startswith("/theme"):
            from ui.themes import get_theme, list_themes_text

            theme_parts = user_text.split(maxsplit=2)
            if len(theme_parts) == 1 or theme_parts[1] == "list":
                console.print(list_themes_text())
            elif theme_parts[1] == "set" and len(theme_parts) == 3:
                try:
                    theme_def = get_theme(theme_parts[2])
                    config.theme_name = theme_def.name
                    config.save()
                    console.print(
                        f"Theme set to '{theme_def.name}'. [dim](The simple REPL renders in your "
                        "terminal's own colors -- full theme rendering applies in the Textual UI.)[/dim]"
                    )
                except KeyError as exc:
                    console.print(f"[bold red]{exc}[/bold red]")
            else:
                console.print("Usage: /theme [list|set <name>]")
            continue
        if user_text == "/doctor":
            from agent.doctor import render_findings_text, run_doctor

            console.print("[dim]Running configuration and safety audit...[/dim]")
            findings = run_doctor(config, llm_client)
            console.print(render_findings_text(findings))
            continue
        if user_text.startswith("/todo"):
            todo_parts = user_text.split(maxsplit=2)
            sub = todo_parts[1] if len(todo_parts) > 1 else "list"
            if sub == "add" and len(todo_parts) > 2:
                index = session.add_todo(todo_parts[2])
                console.print(f"Added todo #{index}: {todo_parts[2]}")
            elif sub == "done" and len(todo_parts) > 2 and todo_parts[2].strip().isdigit():
                idx = int(todo_parts[2].strip())
                console.print(f"Marked todo #{idx} done." if session.mark_todo_done(idx) else f"[bold red]No todo #{idx}.[/bold red]")
            elif sub == "clear":
                session.clear_todos()
                console.print("Cleared the todo list.")
            else:
                console.print(session.render_todos_text())
            continue
        if user_text.startswith("/agent"):
            agent_parts = user_text.split(maxsplit=1)
            sub = agent_parts[1].lower() if len(agent_parts) > 1 else ""
            if sub in ("on", "enable", "enabled"):
                config.agent.planning_enabled = True
                config.save()
                console.print("Agent system enabled.")
            elif sub in ("off", "disable", "disabled"):
                config.agent.planning_enabled = False
                config.save()
                console.print(
                    "Agent system disabled. PyClaw will respond directly to every message -- "
                    "no planning, no tool execution. Use /agent on to re-enable."
                )
            else:
                state = "enabled" if config.agent.planning_enabled else "disabled"
                console.print(f"Agent system is currently {state}. Usage: /agent [on|off]")
            continue
        if user_text.startswith("/project"):
            parts = user_text.split(maxsplit=1)
            if len(parts) > 1:
                config.set_project_root(parts[1])
                session.set_project_root(config.project_root)
                console.print(f"Project root set to {config.project_root}")
            else:
                console.print(f"Current project root: {config.project_root}")
            continue
        if user_text.startswith("/model"):
            from config import MODEL_PRESETS

            def numbered_choices():
                choices = [{"kind": "preset", "name": n, **p} for n, p in MODEL_PRESETS.items()]
                choices += [{"kind": "alias", "name": n, **s} for n, s in config.model_aliases.items()]
                return choices

            def switch_to(choice):
                nonlocal llm_client
                config.update_model(base_url=choice["base_url"], model_name=choice["model_name"], api_key=choice.get("api_key"))
                llm_client = LLMClient(config.model)
                executor.llm_client = llm_client
                executor.planner.llm_client = llm_client
                console.print(f"Switched to {choice['name']} ({config.model.model_name} @ {config.model.base_url}).")

            model_parts = user_text.split(maxsplit=2)
            if len(model_parts) == 1:
                choices = numbered_choices()
                console.print(f"Current: {config.model.model_name} @ {config.model.base_url}")
                console.print("Choose with /model <number> or /model <name>:")
                for i, c in enumerate(choices, 1):
                    console.print(f"  {i}. {c['name']} [{c['kind']}] -- {c['model_name']} @ {c['base_url']}")
            elif model_parts[1].isdigit():
                choices = numbered_choices()
                idx = int(model_parts[1])
                if 1 <= idx <= len(choices):
                    switch_to(choices[idx - 1])
                else:
                    console.print(f"[bold red]No choice #{idx}. Run /model to see the numbered list (1-{len(choices)}).[/bold red]")
            elif any(c["name"].lower() == model_parts[1].lower() for c in numbered_choices()):
                match = next(c for c in numbered_choices() if c["name"].lower() == model_parts[1].lower())
                switch_to(match)
            elif model_parts[1] == "list":
                for name, preset in MODEL_PRESETS.items():
                    console.print(f"  {name}: base_url={preset['base_url']}  model_name={preset['model_name']}")
            elif model_parts[1] == "alias":
                alias_rest = model_parts[2].split(maxsplit=1) if len(model_parts) > 2 else []
                if not alias_rest:
                    if config.model_aliases:
                        for a, s in config.model_aliases.items():
                            console.print(f"  {a}: {s['model_name']} @ {s['base_url']}")
                    else:
                        console.print("[dim]No aliases saved. Usage: /model alias save|use|delete <name>[/dim]")
                else:
                    verb = alias_rest[0]
                    name = alias_rest[1].strip() if len(alias_rest) > 1 else None
                    if verb == "save" and name:
                        config.save_model_alias(name)
                        console.print(f"Saved current model as alias '{name}'.")
                    elif verb == "use" and name:
                        if config.use_model_alias(name):
                            llm_client = LLMClient(config.model)
                            executor.llm_client = llm_client
                            executor.planner.llm_client = llm_client
                            console.print(f"Switched to alias '{name}' ({config.model.model_name} @ {config.model.base_url}).")
                        else:
                            console.print(f"[bold red]No alias named '{name}'. Available: {', '.join(config.model_aliases.keys()) or '(none)'}[/bold red]")
                    elif verb == "delete" and name:
                        console.print("Deleted." if config.delete_model_alias(name) else "[bold red]No such alias.[/bold red]")
                    else:
                        console.print("Usage: /model alias save <name> | use <name> | delete <name>")
            elif model_parts[1] == "preset" and len(model_parts) == 3:
                if config.apply_preset(model_parts[2]):
                    llm_client = LLMClient(config.model)
                    executor.llm_client = llm_client
                    executor.planner.llm_client = llm_client
                    console.print(f"Switched to preset '{model_parts[2]}'.")
                else:
                    from config import MODEL_PRESETS

                    console.print(f"[bold red]Unknown preset. Available: {', '.join(MODEL_PRESETS.keys())}[/bold red]")
            elif len(model_parts) == 3:
                field, value = model_parts[1], model_parts[2]
                numeric_fields = {"temperature": float, "context_size": int, "max_tokens": int, "top_p": float}
                try:
                    if field in numeric_fields:
                        value = numeric_fields[field](value)
                    config.update_model(**{field: value})
                    llm_client = LLMClient(config.model)
                    executor.llm_client = llm_client
                    executor.planner.llm_client = llm_client
                    console.print(f"Updated {field} = {value}")
                except (ValueError, TypeError) as exc:
                    console.print(f"[bold red]Invalid value for {field}: {exc}[/bold red]")
            else:
                console.print("Usage: /model <field> <value>  |  /model list  |  /model preset <name>")
            continue

        def on_token(token: str) -> None:
            console.print(token, end="")

        def on_plan(plan: Plan) -> None:
            console.print(panels.build_plan_panel(plan))

        def on_tool_call(tool: str, args: dict) -> None:
            console.print(f"\n[yellow]{glyph('arrow')} {tool}({args})[/yellow]")

        def on_tool_result(tool: str, result: dict) -> None:
            msg = result.get("message") or result.get("error") or ""
            console.print(f"  [dim]{msg}[/dim]")

        def on_status(status: str) -> None:
            pass  # the simple REPL skips a live status line for simplicity

        console.print("[bold green]Agent:[/bold green] ", end="")
        callbacks = ExecutorCallbacks(
            on_token=on_token,
            on_plan=on_plan,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_status=on_status,
        )
        result = executor.run(user_text, callbacks=callbacks)
        if not result.final_text.strip():
            console.print("[dim](no output)[/dim]")
        console.print()


def main(argv: list = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = build_config(args)

    env_label = describe_environment()

    # On Termux, default to the simple REPL unless the user explicitly
    # forces the full TUI with --tui. This sidesteps a known Termux
    # soft-keyboard quirk where repeated full-screen repaints (which the
    # Textual TUI does on every keystroke) can cause the keyboard to stop
    # reappearing -- see describe_environment()/_is_termux() docstring.
    use_tui = args.tui or (not args.no_tui and not _is_termux())

    if not use_tui:
        if _is_termux() and not args.no_tui:
            console.print(
                f"[dim]Detected {env_label} -- using the simple REPL by default "
                "(avoids a known soft-keyboard issue with the full TUI). "
                "Pass --tui to force the full interface anyway.[/dim]"
            )
        run_simple_repl(config)
        return

    try:
        from ui.tui import run_tui

        run_tui(config)
    except ImportError as exc:
        console.print(
            f"[bold red]Could not load the Textual UI ({exc}).[/bold red] "
            "Falling back to the simple REPL. Install 'textual' for the full experience."
        )
        run_simple_repl(config)


if __name__ == "__main__":
    main()
