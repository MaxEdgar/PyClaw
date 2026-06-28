"""
config.py
=========

Centralized configuration management for PyClaw.

Configuration is stored as JSON on disk (default: ``~/.pyclaw/config.json``)
so that settings persist between runs without requiring environment variables
or command-line flags every time. The :class:`Config` dataclass provides typed
access to all settings, with sane defaults tuned for low-memory devices
(e.g. Termux on a Helio G99-class phone) running a local llama.cpp server.

Design goals:
    * Zero required configuration -- works out of the box against
      ``http://127.0.0.1:8080`` (the default llama.cpp server address).
    * Single source of truth: every other module reads settings through
      this class rather than re-parsing JSON itself.
    * Safe writes: config is written atomically (write to temp file, then
      rename) to avoid corrupting the file if the process is killed mid-write.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Default location for all PyClaw state (config, session memory, history).
DEFAULT_HOME = Path.home() / ".pyclaw"
DEFAULT_CONFIG_PATH = DEFAULT_HOME / "config.json"


# Common backend presets. These are starting points, not a restriction --
# PyClaw works with ANY server that implements the OpenAI-compatible
# /v1/chat/completions endpoint (local llama.cpp, Ollama, LM Studio,
# text-generation-webui, vLLM, or a hosted API). A preset just pre-fills
# base_url/model_name/api_key so switching is one command
# (`/model preset <name>`) instead of setting each field by hand.
#
# model_name is "auto" for every local-server preset rather than a
# hardcoded model like "qwen2.5-coder" -- PyClaw queries the server's
# /v1/models endpoint at connect time (see llm/client.py:detect_model())
# to discover whatever model is actually loaded, instead of assuming one.
# "auto" is kept for hosted APIs without a discoverable single loaded
# model (e.g. OpenAI) only where a concrete default is genuinely required.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "llamacpp": {
        "base_url": "http://127.0.0.1:8080",
        "model_name": "auto",
        "api_key": None,
    },
    "ollama": {
        # Ollama's OpenAI-compatible endpoint lives under /v1 on port 11434.
        "base_url": "http://127.0.0.1:11434",
        "model_name": "auto",
        "api_key": None,
    },
    "lmstudio": {
        "base_url": "http://127.0.0.1:1234",
        "model_name": "auto",
        "api_key": None,
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "model_name": "gpt-4o-mini",
        "api_key": None,  # set with: /model api_key sk-...
    },
}


@dataclass
class ModelConfig:
    """Settings related to the LLM backend (llama.cpp / OpenAI-compatible)."""

    # Base URL of the OpenAI-compatible server. llama.cpp's built-in server
    # exposes /v1/chat/completions and /v1/completions on this base.
    base_url: str = "http://127.0.0.1:8080"

    # Model identifier sent to the server and shown in the UI. Defaults to
    # "auto": PyClaw does not assume any specific model is loaded. On
    # connect, it queries the server's /v1/models endpoint and replaces
    # "auto" with whatever model name the server actually reports (see
    # llm/client.py:detect_model() and main.py/ui/tui.py's use of it). If
    # auto-detection fails (server unreachable, endpoint not supported),
    # "auto" is sent through as the literal model field, which most local
    # servers with exactly one model loaded will accept and ignore.
    model_name: str = "auto"

    # Sampling temperature. Lower = more deterministic, better for code.
    temperature: float = 0.2

    # Context window size advertised to the agent for prompt-budget
    # calculations. This should match how the server was launched
    # (e.g. `--ctx-size 8192`).
    context_size: int = 8192

    # Maximum number of tokens to generate per response.
    max_tokens: int = 1024

    # Top-p nucleus sampling.
    top_p: float = 0.9

    # Request timeout in seconds for non-streaming calls, and the
    # per-chunk read timeout for streaming calls.
    request_timeout: float = 120.0

    # Optional API key. Most local llama.cpp servers don't need one, but
    # OpenAI-compatible cloud fallbacks might.
    api_key: Optional[str] = None

    # Whether to request streaming responses (token-by-token) from the
    # server. This is the default for the live TUI experience.
    stream: bool = True


@dataclass
class AgentConfig:
    """Settings related to agent behavior (planning, tool use, safety)."""

    # Maximum number of planner -> executor -> tool loop iterations before
    # the agent gives up and reports what it has so far. Prevents infinite
    # tool-call loops on a model that gets stuck.
    max_tool_iterations: int = 12

    # Maximum number of bytes read from a single file by read_file() before
    # truncating, to keep prompts within the context budget on small models.
    max_file_read_bytes: int = 200_000

    # Whether destructive actions (delete, overwrite, risky shell commands)
    # require interactive Y/N confirmation. This should basically always be
    # True; it exists as a config switch mainly for automated testing.
    require_confirmation: bool = True

    # Directories excluded from search/indexing by default.
    excluded_dirs: tuple = (
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".pyclaw",
    )

    # Master toggle for the agent system. True (default): full agent
    # behavior is active for TASK_INTENT/TOOL_REQUEST_INTENT messages --
    # planning, tool execution, multi-step reasoning, all gated by the
    # intent classifier in agent/intent.py as usual. False: PyClaw
    # operates in direct-response mode for EVERY message regardless of
    # how it's classified -- no planning, no tool execution, no agent
    # loop, ever. See agent/executor.py:Executor._run_direct_response and
    # the /agent slash command (ui/tui.py, main.py) for the user-facing
    # control. This is a stronger statement than "skip planning for this
    # one message" (which _should_make_plan already does): with this set
    # to False, the agent cannot read files, run commands, or edit
    # anything -- it can only talk.
    planning_enabled: bool = True


@dataclass
class Config:
    """Top-level configuration object combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # The project root the agent is currently operating on. Defaults to the
    # current working directory at startup.
    project_root: str = field(default_factory=lambda: str(Path.cwd()))

    # Path this config was loaded from / will be saved to.
    config_path: str = field(default_factory=lambda: str(DEFAULT_CONFIG_PATH))

    # Name of the active color theme (see ui/themes.py for the registry).
    # Persisted so the chosen theme survives between runs, the same way
    # project_root and model settings do.
    theme_name: str = "default-dark"

    # Internal: set once the legacy-model-name migration (see
    # _migrate_stale_model_name) has run for this config file, so it never
    # re-fires and overwrites a model name the user deliberately set later.
    # Defaults to True for a brand-new Config (nothing to migrate); load()
    # only treats it as "needs migration" when reading an older file that
    # predates this field entirely (see _migrate_stale_model_name).
    _model_name_migrated: bool = True

    # Named, saved backend+model snapshots the user can switch between with
    # one word (e.g. "primary" = a strong/slow model, "fast" = a small/quick
    # one), inspired by OpenClaw's named model-slot configuration but
    # adapted to PyClaw's single-active-backend design: switching an alias
    # copies its saved {base_url, model_name, api_key} into `model` rather
    # than running multiple backends at once. See save_model_alias() and
    # use_model_alias() below, and the /model alias slash command.
    model_aliases: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config (including nested dataclasses) to a plain dict."""
        data = asdict(self)
        # tuples -> lists for clean JSON
        data["agent"]["excluded_dirs"] = list(data["agent"]["excluded_dirs"])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a Config from a dict, tolerating missing/extra keys.

        This allows older config files (missing newly-added fields) to load
        without crashing -- missing keys simply fall back to dataclass
        defaults, and unknown keys are ignored.
        """
        model_data = data.get("model", {})
        agent_data = data.get("agent", {})

        model_fields = {f for f in ModelConfig.__dataclass_fields__}
        agent_fields = {f for f in AgentConfig.__dataclass_fields__}

        model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in model_fields})
        agent_cfg = AgentConfig(**{k: v for k, v in agent_data.items() if k in agent_fields})
        if "excluded_dirs" in agent_data:
            agent_cfg.excluded_dirs = tuple(agent_data["excluded_dirs"])

        return cls(
            model=model_cfg,
            agent=agent_cfg,
            project_root=data.get("project_root", str(Path.cwd())),
            config_path=data.get("config_path", str(DEFAULT_CONFIG_PATH)),
            theme_name=data.get("theme_name", "default-dark"),
            model_aliases=data.get("model_aliases", {}),
        )

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------
    def save(self, path: Optional[str] = None) -> None:
        """Atomically persist this config to disk as JSON."""
        target = Path(path or self.config_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file in the same directory, then rename, so a
        # crash mid-write never leaves a half-written config.json behind.
        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Load config from disk, or return defaults if the file is absent
        or unreadable (corrupt JSON falls back to defaults rather than
        crashing the whole application).
        """
        target = Path(path or DEFAULT_CONFIG_PATH)
        if not target.exists():
            cfg = cls()
            cfg.config_path = str(target)
            return cfg

        try:
            with open(target, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg = cls.from_dict(data)
            cfg.config_path = str(target)
            cfg._migrate_stale_model_name(data)
            return cfg
        except (json.JSONDecodeError, OSError):
            cfg = cls()
            cfg.config_path = str(target)
            return cfg

    # Model names PyClaw itself used to write as the *default* before
    # "auto" existed, in early versions of this project. Only these exact
    # values are eligible for the one-time migration below -- anything
    # else is assumed to be a name the user deliberately set via /model or
    # a preset, and is left untouched.
    _LEGACY_DEFAULT_MODEL_NAMES = {"qwen2.5-coder", "local-model"}

    def _migrate_stale_model_name(self, raw_data: Dict[str, Any]) -> None:
        """One-time fix for configs written before model_name defaulted to
        "auto": if the saved value is exactly one of PyClaw's old hardcoded
        defaults, AND the config has no record of this migration having
        already run, reset it to "auto" and persist that, so auto-detection
        actually runs on the next connect instead of displaying a model
        name that was never real to begin with (see the bug report this
        was added for: a fresh install showing "qwen2.5-coder" despite no
        model ever having connected).

        Runs at most once per config file -- a `_model_name_migrated` flag
        is written after the first pass so this never re-fires and clobber
        a model name the user deliberately set afterwards, even if that
        name happens to also be "qwen2.5-coder" on purpose.
        """
        if raw_data.get("_model_name_migrated"):
            return
        if self.model.model_name in self._LEGACY_DEFAULT_MODEL_NAMES:
            self.model.model_name = "auto"
        self._model_name_migrated = True
        self.save()

    # ------------------------------------------------------------------
    # Convenience setters used by the /model slash command etc.
    # ------------------------------------------------------------------
    def apply_preset(self, name: str) -> bool:
        """Apply a named backend preset (see MODEL_PRESETS) and persist it.

        Returns True if the preset was found and applied, False if `name`
        is not a recognized preset (the caller should report available
        preset names in that case).
        """
        preset = MODEL_PRESETS.get(name.lower())
        if preset is None:
            return False
        for key, value in preset.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        self.save()
        return True

    def save_model_alias(self, alias: str) -> None:
        """Snapshot the CURRENTLY active model settings under a named alias
        (e.g. "primary", "fast"), so it can be switched back to later with
        use_model_alias() without re-typing base_url/model_name/api_key.

        Saving an alias with a name that already exists overwrites it --
        this is treated as "update my saved primary/fast model" rather
        than something that needs confirmation, since aliases are PyClaw's
        own convenience bookkeeping, not project files.
        """
        self.model_aliases[alias.strip().lower()] = {
            "base_url": self.model.base_url,
            "model_name": self.model.model_name,
            "api_key": self.model.api_key,
        }
        self.save()

    def use_model_alias(self, alias: str) -> bool:
        """Switch the active model to a previously saved alias.

        Returns True if the alias existed and was applied, False
        otherwise (the caller should list self.model_aliases.keys() in
        that case).
        """
        saved = self.model_aliases.get(alias.strip().lower())
        if saved is None:
            return False
        for key, value in saved.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        self.save()
        return True

    def delete_model_alias(self, alias: str) -> bool:
        """Remove a saved alias. Returns True if it existed."""
        key = alias.strip().lower()
        if key not in self.model_aliases:
            return False
        del self.model_aliases[key]
        self.save()
        return True

    def update_model(self, **kwargs: Any) -> None:
        """Update one or more ModelConfig fields and persist immediately."""
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        self.save()

    def set_project_root(self, path: str) -> None:
        """Change the active project root and persist immediately."""
        resolved = str(Path(path).expanduser().resolve())
        self.project_root = resolved
        self.save()
