# Skills

A skill is a piece of guidance you teach PyClaw once that stays available in
every future session. It's the difference between explaining your release
process in chat every single time, and teaching PyClaw the process once so
it's automatically reminded whenever a release-related request comes up.

## What a skill actually is

A skill is plain text, stored as a small JSON file under `~/.pyclaw/skills/`.
It is **not** code, and it cannot execute anything on its own. A skill is
read by the model as additional context -- exactly like an extra paragraph
of instructions -- and every action the model takes as a result still goes
through PyClaw's normal tool-call approval and safety checks (diff approval,
delete confirmation, dangerous-command confirmation). Teaching PyClaw a
skill does not grant it any new capability or bypass any safety check.

Each skill has four parts:

| Field | Purpose |
|---|---|
| `name` | A short identifier (letters, numbers, hyphens). Used as the filename and to reference the skill in commands. |
| `description` | One line describing *when* this skill applies. Shown in `/skill list`. |
| `trigger_keywords` | Words that, if present in your request, activate this skill. |
| `instructions` | The actual guidance -- what PyClaw should do or keep in mind. |

## How skills get used

PyClaw does **not** load every skill into every conversation -- that would
waste context on a small local model for no benefit. Instead, at the start
of each request, PyClaw checks your stored skills for keyword overlap with
what you just typed (case-insensitive, matching against `trigger_keywords`
and the skill's own name). Only skills that look relevant are added to that
turn's system prompt, clearly marked so the model can tell a skill apart
from its base instructions:

```
[SKILL: release-checklist]
When to use: Steps to follow before tagging a release
Run the test suite, update CHANGELOG.md, bump the version in config.py,
then tag with `git tag vX.Y.Z`.
[END SKILL: release-checklist]
```

This matching is simple keyword overlap, not semantic search -- predictable
and easy to reason about, at the cost of needing reasonably specific
trigger keywords. If a skill isn't activating when you expect, check that
your request actually contains one of its trigger words, or add more.

## Creating a skill

### In the Textual UI

```
/skill create release-checklist
```

This starts a short guided conversation -- PyClaw will ask for the
description, trigger keywords, and instructions one at a time:

```
> /skill create release-checklist
Creating skill 'release-checklist'. Describe when this skill should be used
(one line, e.g. 'Steps to follow before tagging a release'):
> Steps to follow before tagging a release
Trigger keywords, comma-separated (or leave blank):
> release, version bump, changelog
Now the actual instructions -- what should PyClaw do?
> Run the test suite, update CHANGELOG.md, bump the version in config.py,
  then tag with git tag vX.Y.Z.
Skill 'release-checklist' saved.
```

If a skill with that name already exists, PyClaw asks before overwriting it.

### In the simple REPL (`--no-tui`)

The REPL doesn't have a multi-step guided flow; instead, give all four
parts on one line, separated by `|`:

```
> /skill create
Creating a skill in the REPL uses one line:
name | description | keyword1,keyword2 | instructions
skill> release-checklist | Steps before tagging a release | release,changelog | Run tests, update CHANGELOG.md, bump version.
Saved skill 'release-checklist'.
```

## Managing skills

| Command | Effect |
|---|---|
| `/skill list` | List every saved skill with its description |
| `/skill show <name>` | Show the full details of one skill |
| `/skill delete <name>` | Remove a skill permanently |
| `/skill create [name]` | Create a new skill (or start the guided flow without a name) |

## Where skills live

Each skill is one JSON file: `~/.pyclaw/skills/<name>.json`. Since it's
plain JSON, you can also:

- Back up your skills by copying that folder.
- Share a skill with someone else by sending them the file -- they drop it
  into their own `~/.pyclaw/skills/` directory and it's immediately
  available, no import command needed.
- Edit a skill by hand in any text editor, as long as the JSON stays valid.

Example file (`~/.pyclaw/skills/release-checklist.json`):

```json
{
  "name": "release-checklist",
  "description": "Steps to follow before tagging a release",
  "trigger_keywords": ["release", "version bump", "changelog"],
  "instructions": "Run the test suite, update CHANGELOG.md, bump the version in config.py, then tag with git tag vX.Y.Z.",
  "created_at": 1719200000.0,
  "updated_at": 1719200000.0
}
```

## Writing good skills

- **Be specific in `instructions`.** "Follow good practices" gives the
  model nothing to act on. "Run `pytest`, then check that `CHANGELOG.md`
  has a new entry before tagging" is concrete enough to actually follow.
- **Pick trigger keywords you'd actually type.** If you'd say "ship this"
  rather than "release this," include "ship" as a keyword too.
- **Keep one skill per concern.** A "release-checklist" skill and a
  "code-style" skill are easier to maintain (and to tell whether they fired
  correctly) than one giant skill trying to cover everything.
- **Skills are guidance, not guarantees.** Like any instruction given to a
  language model, a skill increases the chance PyClaw follows it -- it does
  not force a specific tool call sequence. Review what PyClaw actually does
  the same way you would without a skill active, especially for destructive
  or release-affecting actions.
