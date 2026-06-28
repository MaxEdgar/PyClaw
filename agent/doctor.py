"""
agent/doctor.py
=================

A self-check ("/doctor") that audits PyClaw's own configuration and
connectivity for the kinds of misconfigurations that tend to bite people
quietly -- inspired by the safety-audit step recommended for OpenClaw
deployments before exposing them to anything, adapted here to PyClaw's
much narrower (local, sandboxed, single-project) surface.

Unlike OpenClaw's audit (which checks network bindings, channel auth
tokens, and exposed gateway ports -- all relevant to an internet-facing
multi-channel assistant), PyClaw has no such surface to begin with: it
never listens on a port, never accepts inbound connections, and only
talks to the model server you configured. So this check focuses on what
actually matters for a local sandboxed tool: confirmation settings,
sandbox integrity, server reachability, and disk-state sanity.

Findings are returned as a list of DoctorFinding rather than printed
directly, so both the Textual UI and the simple REPL can render them in
their own style.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import Config

Severity = str  # "ok" | "warning" | "danger"


@dataclass
class DoctorFinding:
    severity: Severity
    title: str
    detail: str


def run_doctor(config: Config, llm_client) -> List[DoctorFinding]:
    """Run every check and return the full list of findings, in a fixed
    order (most safety-critical first) so the report reads consistently
    every time rather than in whatever order checks happened to run.
    """
    findings: List[DoctorFinding] = []

    # ------------------------------------------------------------------
    # 1. Confirmation / safety gating
    # ------------------------------------------------------------------
    if config.agent.require_confirmation:
        findings.append(
            DoctorFinding(
                "ok",
                "Destructive-action confirmation",
                "Enabled. Deletes, overwrites, and dangerous shell commands all require approval.",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                "danger",
                "Destructive-action confirmation is DISABLED",
                "require_confirmation is False -- PyClaw will delete files and run flagged shell "
                "commands without asking. This is normally only set via --no-confirm for scripted "
                "testing. If you didn't do that deliberately, fix it with: "
                "/model require_confirmation true (or edit ~/.pyclaw/config.json directly).",
            )
        )

    # ------------------------------------------------------------------
    # 2. Project root sandboxing sanity
    # ------------------------------------------------------------------
    root = Path(config.project_root).expanduser()
    home = Path.home()
    if not root.exists():
        findings.append(
            DoctorFinding(
                "warning",
                "Project root does not exist",
                f"{root} was not found. Tools that touch the project will fail until this is "
                "fixed with /project <path>.",
            )
        )
    elif root == home:
        findings.append(
            DoctorFinding(
                "warning",
                "Project root is your entire home directory",
                f"{root} is your home folder. The filesystem sandbox (see tools/filesystem.py) "
                "will allow read/write anywhere under it, including unrelated personal files, "
                "ssh keys, and other projects. Consider pointing /project at a specific project "
                "subdirectory instead.",
            )
        )
    elif str(root) in ("/", ""):
        findings.append(
            DoctorFinding(
                "danger",
                "Project root is the filesystem root",
                "project_root is set to '/'. The sandbox would allow read/write across the "
                "entire filesystem. Set a real project directory with /project <path>.",
            )
        )
    else:
        findings.append(DoctorFinding("ok", "Project root", f"{root} (sandboxed -- tools cannot read/write outside it)"))

    # ------------------------------------------------------------------
    # 3. Server reachability + model resolution
    # ------------------------------------------------------------------
    try:
        reachable = llm_client.health_check()
    except Exception:  # noqa: BLE001 - the doctor itself must never crash on a bad client
        reachable = False

    if reachable:
        findings.append(
            DoctorFinding("ok", "LLM server", f"Reachable at {config.model.base_url}.")
        )
        if config.model.model_name == "auto":
            findings.append(
                DoctorFinding(
                    "warning",
                    "Model name still 'auto'",
                    "Connected, but auto-detection hasn't resolved a concrete model name yet "
                    "(or the server's /v1/models endpoint didn't report one). This is harmless "
                    "for single-model local servers, but worth knowing if you expect to see a "
                    "specific model name in the sidebar.",
                )
            )
    else:
        findings.append(
            DoctorFinding(
                "warning",
                "LLM server unreachable",
                f"Could not reach {config.model.base_url}. PyClaw will show Offline until a "
                "model server is running there, or you switch with /model preset / /model alias use.",
            )
        )

    # ------------------------------------------------------------------
    # 4. API key hygiene
    # ------------------------------------------------------------------
    if config.model.api_key:
        findings.append(
            DoctorFinding(
                "ok",
                "API key configured",
                "An API key is set for the active backend. It is stored in plain text in "
                "~/.pyclaw/config.json (the same way most local CLI tools store credentials) -- "
                "keep that file private, and avoid committing it if your project root overlaps "
                "with a backed-up or synced location.",
            )
        )

    # ------------------------------------------------------------------
    # 5. Tool-call loop bound
    # ------------------------------------------------------------------
    if config.agent.max_tool_iterations > 50:
        findings.append(
            DoctorFinding(
                "warning",
                "max_tool_iterations is unusually high",
                f"Set to {config.agent.max_tool_iterations}. A model that gets stuck in a loop "
                "could run many tool calls (each one still individually gated by confirmation "
                "for destructive actions) before PyClaw gives up. The default of 12 is usually "
                "enough; consider lowering this unless you have a specific reason to raise it.",
            )
        )

    return findings


def render_findings_text(findings: List[DoctorFinding]) -> str:
    """Render findings as plain, readable text for the simple REPL or as
    a fallback if the TUI's richer rendering is unavailable."""
    symbol = {"ok": "[OK]", "warning": "[WARN]", "danger": "[DANGER]"}
    lines = []
    for f in findings:
        lines.append(f"{symbol.get(f.severity, '[?]')} {f.title}")
        lines.append(f"    {f.detail}")
    danger_count = sum(1 for f in findings if f.severity == "danger")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    lines.append("")
    lines.append(f"{danger_count} danger, {warning_count} warning, {len(findings) - danger_count - warning_count} ok")
    return "\n".join(lines)
