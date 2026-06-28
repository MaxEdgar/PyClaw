# PyClaw

A local, terminal-based AI coding agent for Linux, WSL, and Termux. PyClaw
connects to a local or self-hosted OpenAI-compatible model server and lets
it read, search, and edit a real project on disk through a defined set of
tools, with every destructive action gated behind explicit approval.

> **Status:** Early development. Core functionality is implemented, but
> APIs and features may change as the project evolves.

This README is written as a complete, beginner-friendly tutorial. If
you've never used PyClaw (or a tool like it) before, read it top to
bottom; if you just need a specific command, jump to
[Complete command reference](#complete-command-reference).

---

## Table of contents

- [What PyClaw actually does](#what-pyclaw-actually-does)
- [Requirements](#requirements)
- [Step-by-step setup](#step-by-step-setup)
- [First launch](#first-launch)
- [Getting started: your first few minutes](#getting-started-your-first-few-minutes)
- [Complete command reference](#complete-command-reference)
- [Keyboard shortcuts](#keyboard-shortcuts)
- [Example commands with expected output](#example-commands-with-expected-output)
- [The agent toggle: full agent vs. direct-response mode](#the-agent-toggle-full-agent-vs-direct-response-mode)
- [How intent classification works](#how-intent-classification-works)
- [Skills](#skills)
- [Project Instructions (PYCLAW.md)](#project-instructions-pyclawmd)
- [Backends and switching models](#backends-and-switching-models)
- [Safety model](#safety-model)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## What PyClaw actually does

PyClaw is a program that sits between you and a language model, and gives
that model **hands**: the ability to read files, search a codebase, edit
files (with your approval), run shell commands, and use git -- inside one
project folder on your own machine.

It is not a chatbot that only talks about code. When you ask it to fix a
bug, it actually opens the relevant file, reads it, proposes a specific
change as a diff, and only writes that change after you approve it. When
you just say "hi," it answers like a person would -- no file scanning, no
multi-step planning, nothing happening behind the scenes.

The core pieces:

- **An LLM server you run yourself** (llama.cpp, Ollama, LM Studio, or a
  hosted API) -- PyClaw does not include or run a model itself.
- **PyClaw**, the program in this repository -- the interface, the safety
  rules, and the tools (read/write/search/run/git) the model is allowed
  to use.
- **You**, approving every edit and every risky action before it happens.

Everything PyClaw does to your files is sandboxed to one project
directory at a time, and nothing destructive happens without you typing
`y`.

---

## Requirements

- Python 3.11+
- `git` (for installation and the git tools)
- A model server implementing the OpenAI-compatible chat completions API:
  llama.cpp's `llama-server`, Ollama, LM Studio, vLLM,
  text-generation-webui, or a hosted equivalent.

PyClaw does not bundle a model or a model server -- you provide the
backend, PyClaw provides the agent and interface around it.

---

## Step-by-step setup

### 1. Get the code

```bash
git clone https://github.com/your-org/pyclaw.git
cd pyclaw
```

(Or unzip a downloaded copy and `cd` into that folder.)

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the installer script, which does the same three steps for you:

```bash
chmod +x install.sh
./install.sh
```

### 3. Start a model server

PyClaw needs something to talk to. The example below uses llama.cpp with
a small local model; any OpenAI-compatible server works the same way.

```bash
./llama-server -m /path/to/your-model.gguf \
    --host 127.0.0.1 --port 8080 --ctx-size 8192
```

Leave this running in its own terminal/tab -- don't close it. In a second
terminal, you can sanity-check it's alive with:

```bash
curl http://127.0.0.1:8080/health
```

If that returns a response (not "connection refused"), the server is up.

### 4. Launch PyClaw

```bash
source .venv/bin/activate   # if not already active
python main.py
```

That's the entire setup. No database, no extra services, no account
creation.

---

## First launch

The first time PyClaw runs, it creates three files under `~/.pyclaw/`:

- `config.json` -- your settings (model server address, theme, toggles)
- `session.json` -- your current task, recent files, todo list
- `history.jsonl` -- the full conversation transcript

You'll see a brief startup animation (the full Textual UI only; the
`--no-tui` REPL shows a plain text banner instead), then the main
interface with a chat box at the bottom. If no model server is reachable
yet, the status bar will say **Offline** -- that's expected and not an
error; start your server and PyClaw will detect it automatically within
about 15 seconds, or immediately if you send a message.

On **Termux**, PyClaw defaults to the simpler `--no-tui` mode automatically
(a known Android soft-keyboard quirk with the full interface -- see
[Troubleshooting](#troubleshooting)). Pass `--tui` to force the full
interface anyway.

---

## Getting started: your first few minutes

A few small things to try right away, in order:

**1. Say hello.**
```
> hi
```
PyClaw answers directly and instantly -- no scanning, no plan, no tool
calls. This is deliberate: greetings and small talk never trigger the
agent system (see [How intent classification works](#how-intent-classification-works)).

**2. Ask it to look at your project.**
```
> summarize this project
```
Now it actually reads your project structure and answers based on what's
really there.

**3. Ask it to explain a specific file.**
```
> what does main.py do?
```

**4. Try something that needs an edit.**
```
> add a docstring to the first function in utils.py
```
PyClaw will read the file, propose the change as a colored diff, and ask
`Approve patch? [Y] Yes [N] No` before touching anything.

**5. Look at what it remembers.**
```
/memory
/history
```

That's the whole loop: ask, it investigates if needed, it shows you
exactly what it wants to change, you approve or decline.

---

## Complete command reference

Every command below works in both the full Textual UI and the `--no-tui`
REPL unless noted otherwise.

| Command | What it does | When to use it |
|---|---|---|
| `/help` | Shows the command reference | You forgot a command |
| `/clear` | Clears the conversation and session memory (todos are kept -- see `/todo clear`) | Starting a fresh topic in the same project |
| `/history` | Shows recent chat history | Reviewing what was discussed earlier |
| `/project` | Shows the current project root | Confirming which folder PyClaw is operating on |
| `/project <path>` | Switches the active project root | Working on a different project without restarting |
| `/model` | Shows a numbered list of backend presets and saved aliases | Seeing what you can switch to |
| `/model <number>` | Switches to that numbered choice instantly | Quick backend switching, e.g. `/model 2` |
| `/model <name>` | Switches to a preset or alias by name | e.g. `/model ollama` or `/model primary` |
| `/model <field> <value>` | Sets one field directly (`base_url`, `model_name`, `temperature`, `context_size`, `max_tokens`, `top_p`, `api_key`) | Fine-tuning a setting without a full preset |
| `/model alias save <name>` | Saves the current backend+model under a short name | Creating a "primary"/"fast" shortcut |
| `/model alias use <name>` | Switches to a saved alias | Recalling a saved shortcut |
| `/model alias delete <name>` | Removes a saved alias | Cleaning up |
| `/tools` | Lists every tool the agent can use | Curiosity, or debugging what it's capable of |
| `/memory` | Shows the current session summary (task, recent files, todos) | Checking what PyClaw currently "remembers" |
| `/skill list` | Lists saved skills | Seeing what's taught |
| `/skill create [name]` | Starts a guided flow to teach a new skill | Teaching a reusable convention |
| `/skill show <name>` | Shows one skill's full details | Reviewing what a skill actually says |
| `/skill delete <name>` | Removes a skill | Cleaning up |
| `/theme list` | Lists color themes with their actual hex values | Picking a look |
| `/theme set <name>` | Switches theme | e.g. `/theme set pink` |
| `/doctor` | Audits your configuration and connectivity | Something feels wrong; a quick health check |
| `/todo add <text>` | Adds a persistent todo item | Tracking multi-step or multi-day work |
| `/todo list` | Shows the todo list | Checking what's left |
| `/todo done <number>` | Marks a todo item complete | Finishing an item |
| `/todo clear` | Removes all todo items | Starting a fresh task list |
| `/agent` | Shows whether the agent system is on or off | Checking current mode |
| `/agent on` | Enables full agent behavior (planning + tools) | Default mode -- back to normal |
| `/agent off` | Disables agent behavior entirely; direct-response only | You want a plain Q&A chat with no file/tool access at all |
| `/quit`, `/exit` | Exits PyClaw | Done for now |

Typing `/` alone (in the full UI) shows a live, filtered list of matching
commands as you type.

---

## Keyboard shortcuts

These work in the full Textual UI (`python main.py`, not `--no-tui`).

| Key | Action |
|---|---|
| `Ctrl+X` | Cancel the current request, or decline a pending confirmation |
| `Ctrl+L` | Clear the conversation (same as `/clear`) |
| `Ctrl+C` | Quit |
| `Ctrl+P` | Open Textual's built-in command palette (change theme, save a screenshot, etc.) |
| `Up` / `Down` | Recall previously submitted input (shell-style history) |
| `Tab` | Complete the highlighted slash-command suggestion |

---

## Example commands with expected output

**Asking a simple question (chat mode, instant, no tools):**
```
> hey
Agent: Hey! What are you working on?
```

**Asking about the project (task mode, reads real files):**
```
> what does the search tool do?

  -> grep_text(query="def search_files")
     Found 1 match(es) for 'def search_files'
  -> read_file(path="tools/search.py")
     Read 1842 bytes from tools/search.py

Agent: search_files() finds files by glob pattern (like "*.py"), matching
against filenames recursively from the project root, skipping directories
like .git and node_modules.
```

**Asking for an edit (full plan + diff approval):**
```
> fix the off-by-one error in paginate()

Agent Plan
  1. Search the project for paginate(
  2. Read the file it's defined in
  3. Identify the bug
  4. Propose a fix

  -> grep_text(query="def paginate")
  -> read_file(path="utils/pagination.py")
  -> propose_patch(path="utils/pagination.py", new_content="...")

  Patch: utils/pagination.py  (+1 / -1)
  --- a/utils/pagination.py
  +++ b/utils/pagination.py
  @@ -12,7 +12,7 @@
  -    return items[start:end]
  +    return items[start:end + 1]

  Approve patch? [Y] Yes  [N] No
  > y

Agent: Fixed the off-by-one in paginate() -- end was exclusive when it
should have included the last item on the page. Patch applied.
```

**Switching models:**
```
> /model
Current: auto @ http://127.0.0.1:8080
Choose with /model <number> or /model <name>:
  1. llamacpp [preset] -- auto @ http://127.0.0.1:8080
  2. ollama [preset] -- auto @ http://127.0.0.1:11434
  3. lmstudio [preset] -- auto @ http://127.0.0.1:1234
  4. openai [preset] -- gpt-4o-mini @ https://api.openai.com

> /model 2
Switched to ollama (auto @ http://127.0.0.1:11434) -- checking connection...
ollama: connected.
```

**Turning the agent off for a plain-chat session:**
```
> /agent off
Agent system disabled. PyClaw will respond directly to every message --
no planning, no tool execution, no file/codebase access. Use /agent on to
re-enable.

> can you refactor my auth module
Agent: I can't make changes right now -- agent mode is off. Turn it back
on with /agent on if you'd like me to look at your code.
```

---

## The agent toggle: full agent vs. direct-response mode

`/agent off` is a hard switch, separate from how individual messages are
classified (see the next section). With it off:

- **No planning.** The planning step never runs, for any message.
- **No tool execution.** PyClaw cannot read files, search the project,
  run commands, or edit anything -- even if you explicitly ask it to.
- **No multi-step reasoning.** Every message gets exactly one direct
  reply, the same as a plain chat with the model.

This is stronger than intent classification skipping planning for a
greeting: with the agent off, even an explicit "fix this bug, refactor
the whole module" request gets a plain conversational reply instead of
any real action. Turn it back on with `/agent on` whenever you want full
capability again. The setting persists across restarts until you change
it.

---

## How intent classification works

Before anything else happens, every message you send is classified into
one of four intents:

- **CHAT_INTENT** -- greetings, small talk, short acknowledgements
  ("hi", "hello", "hey", "yo", "sup", "how are you", "thanks"). Answered
  directly and instantly. No planning, no tools, no filesystem access,
  no "Agent Plan" output -- ever, for this intent.
- **TASK_INTENT** -- an explicit request to do something ("create a
  folder", "fix this bug", "build the project"). Full agent behavior is
  allowed: planning, reading files, running tools, editing code.
- **TOOL_REQUEST_INTENT** -- a specific, named action ("run pytest",
  "show me the git diff"). Tools are allowed; the planning step is
  skipped since a single named action rarely needs a multi-step plan.
- **SYSTEM_INTENT** -- a slash command. These are handled before
  reaching the agent at all.

Classification is a fast, local check -- not an extra call to the model
-- so it costs nothing and adds no delay. A short message (three words or
fewer) with no recognizable action verb defaults to CHAT_INTENT, which is
what stops something like "hi" or "yo" from accidentally being treated as
a multi-step task. This is also what fixed a real bug where simple
greetings were triggering project scanning and planning loops.

You can't disable intent classification itself (it's what keeps casual
messages fast and side-effect-free), but you can disable agent behavior
entirely for every message with `/agent off`, described above.

---

## Skills

Skills are reusable instructions you teach PyClaw once, stored as JSON
under `~/.pyclaw/skills/`, and automatically surfaced when relevant to a
request (matched by keyword overlap, not by always loading every skill
into every conversation). Full documentation, including the file format
and guidance on writing effective skills, is in
[docs/SKILLS.md](docs/SKILLS.md).

Two ready-to-use examples ship in [examples/skills/](examples/skills/) --
copy them into `~/.pyclaw/skills/` to try the feature immediately instead
of writing one from scratch first:

```bash
cp examples/skills/*.json ~/.pyclaw/skills/
```

---

## Project Instructions (PYCLAW.md)

If a project contains a `PYCLAW.md` file at its root, PyClaw reads it
automatically on every task-intent request for that project -- no command
needed. Use it for conventions everyone working in the project should
follow (code style, directories to never touch, which test runner to
use):

```markdown
# Project conventions
- Use 4-space indentation, never tabs.
- Never modify anything under vendor/.
- Run `pytest` for tests, not `unittest`.
```

This is distinct from a skill: `PYCLAW.md` is scoped to the *project* and
applies to every task-intent request, like a README meant to be committed
to the repo. A skill is scoped to *you* and only activates when its
trigger keywords match what you typed.

---

## Backends and switching models

PyClaw works with any server implementing the OpenAI-compatible
`/v1/chat/completions` endpoint -- not just one model or one vendor.

```bash
/model           # see a numbered list of presets + your saved aliases
/model 2         # switch to choice #2
/model ollama    # or switch by preset/alias name directly
```

Save your own shortcuts:

```bash
/model alias save primary   # snapshot the currently active backend+model
/model alias save fast      # ...and a different one
/model primary               # switch back with one word, any time
```

PyClaw queries the server's `/v1/models` endpoint on connect to discover
which model is actually loaded, so `model_name` does not need to be set
by hand for local single-model servers.

---

## Safety model

- All filesystem tools are sandboxed to the configured project root; a
  path that resolves outside it is rejected before anything is read or
  written.
- Overwriting an existing file, deleting a file or directory, and any
  shell command matching a destructive pattern require interactive
  approval -- a Y/N modal in the Textual UI, a blocking prompt in the
  REPL.
- File edits are always presented as a diff before being applied.
- Cancelling with `Ctrl+X` while a confirmation is pending is treated as
  a decline, not left hanging.
- No planning, tool execution, or agent loop can begin for a message
  classified as CHAT_INTENT, and none of it can run at all while
  `/agent off` is active -- both are enforced before the request ever
  reaches the model with tool information in its context.
- A skill or `PYCLAW.md` file is plain text read by the model as
  additional context. Neither can execute code or bypass the approval
  flow above.

---

## Project layout

```
pyclaw/
├── main.py                       # CLI entry point, simple REPL fallback
├── config.py                      # JSON-backed configuration (model, agent, theme settings)
├── install.sh                      # Linux installation script
├── llm/
│   ├── client.py                    # OpenAI-compatible HTTP client (streaming + non-streaming, model auto-detect)
│   └── prompts.py                    # System prompts (agent + chat-only), planner prompt, skill injection
├── tools/
│   ├── filesystem.py                  # read/write/append/delete/move/copy/list/mkdir/info
│   ├── search.py                       # search_files, grep_text, find_extensions, project_summary
│   ├── shell.py                        # run_command with timeout + safety gating
│   ├── git_tools.py                    # git status/diff/log/commit/branch
│   └── safety.py                       # dangerous-command detection, confirmation prompts
├── memory/
│   ├── session.py                       # current task, recent files, last plan, todos
│   ├── history.py                        # full chat transcript
│   └── skills.py                          # persistent user-defined skills
├── ui/
│   ├── tui.py                              # Textual application, splash screen
│   ├── panels.py                            # Rich panel builders, slash-command registry
│   ├── diff_view.py                          # unified diff rendering, patch approval
│   ├── themes.py                              # named color themes
│   └── glyphs.py                               # terminal Unicode capability detection, ASCII fallback
├── agent/
│   ├── intent.py                                # intent classification (CHAT/TASK/SYSTEM/TOOL_REQUEST)
│   ├── planner.py                                # produces a short plan via the LLM
│   ├── executor.py                                # plan → tool-call → answer loop; direct-response gate
│   ├── tool_router.py                              # parses tool-call JSON, dispatches to tools/
│   ├── project_instructions.py                      # reads an optional project-root PYCLAW.md
│   └── doctor.py                                     # /doctor: configuration and connectivity audit
├── docs/
│   └── SKILLS.md                                      # skills system documentation
├── examples/
│   └── skills/                                          # ready-to-use example skills
└── requirements.txt
```

---

## Troubleshooting

**"Offline" in the status bar.** No model server is reachable at the
configured address. Start your server, or run `/model` to switch to a
different one. PyClaw rechecks automatically every ~15 seconds.

**It shows `qwen2.5-coder` or another model name I never set.** Older
versions of PyClaw defaulted `model_name` to a hardcoded value. Current
versions default to `"auto"` and auto-detect the real model from your
server; a one-time migration fixes this automatically for existing
config files the first time you launch.

**Termux: the on-screen keyboard doesn't appear.** This is a known Termux
limitation with full-screen terminal repaints, not specific to PyClaw.
PyClaw defaults to `--no-tui` on Termux for exactly this reason; if you
forced `--tui` and hit this, drop back to the default mode.

**A box-drawing character shows as `?` on Windows.** This means your
terminal's encoding isn't UTF-8. PyClaw detects this and falls back to
plain ASCII borders/symbols automatically; if you still see `?`
characters, your terminal may be misreporting its own encoding -- try
Windows Terminal instead of the legacy `cmd.exe` console.

**Everything feels slow.** Make sure `context_size` in `/model` roughly
matches the `-c` value your model server was actually started with -- a
mismatch can cause oversized requests. Also check `/doctor` for other
configuration issues.
