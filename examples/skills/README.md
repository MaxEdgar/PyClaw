# Example skills

Two ready-to-use skills, generated through PyClaw's real skill storage code
(not hand-written JSON), so the format is guaranteed to match what
`memory/skills.py` actually expects.

- **good-commit-messages.json** — conventional-commit-style messages,
  triggers on "commit"
- **code-review-checklist.json** — a concrete review order (correctness,
  test coverage, security, then style), triggers on "review" / "pull request"

## To use them

Copy them into PyClaw's skill directory:

```bash
mkdir -p ~/.pyclaw/skills
cp examples/skills/*.json ~/.pyclaw/skills/
```

Then confirm they loaded:

```
/skill list
```

You should see both listed. From then on, any request containing one of
their trigger words ("commit this", "review this diff") will automatically
surface that skill's instructions to the model for that turn — no further
setup needed.

These are meant as a starting point, not a fixed standard — open one in a
text editor (or `/skill show <name>` then `/skill create <name>` to
overwrite it) and adjust the instructions to match how you actually want
PyClaw to behave.
