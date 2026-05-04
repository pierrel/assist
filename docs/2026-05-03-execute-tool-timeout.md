# 2026-05-03 — Execute tool needs a subprocess timeout

## Problem

The dev-agent's `execute` tool (sandbox shell wrapper) has no
internal timeout. If the agent emits a runaway shell command, the
agent loop blocks indefinitely waiting for stdout/stderr/exit.

Concretely observed 2026-05-03 while running
`test_dev_agent_planning_flow.py` with no pytest-timeout:

- Qwen3.6 dev-agent emitted `execute(command="python3 -c 'import glob;
  glob.glob(\"**/test*web*\", recursive=True)' ")` with cwd `/`
  inside the container.
- Python proceeded to walk the entire container filesystem: the
  bind-mounted assist project (large), its `.venv` (thousands of
  files), `/usr`, `/var`, `/etc`.
- After 40+ minutes the subprocess was still spinning at 99% CPU and
  producing no output the agent could act on.
- The pytest-timeout was disabled for this run (deliberately, to
  measure true per-test wall), so the test never terminated; pytest
  was killed by hand.

## Why "just add pytest-timeout back" isn't the right fix

A pytest-level timeout makes the test fail, but:

1. The same buggy tool call would happen again on the next run.
2. Other consumers of the execute tool — the web service serving real
   user threads — have no such guard. A real user could kick off the
   same shape of agent action and have their thread sit at 99% CPU
   for hours. (Eval-suite hangs are noisy; live-service hangs are
   silent.)
3. The bug is at the tool layer, not the test layer.

## Proposed fix

Add a wall-clock + inactivity timeout inside the execute tool's
subprocess management.

Suggested defaults:

- **Wall-clock cap: 120s.** Matches the longest legitimate `pip
  install` or `pytest` run we'd expect inside the sandbox. Anything
  longer than two minutes for an LLM-driven command is almost
  certainly runaway.
- **Inactivity cap: 30s.** If the subprocess produces no stdout/stderr
  for 30 seconds, kill it. Catches the glob-the-world case (which
  *does* produce output as it walks, but slowly enough to be
  uninteresting), CPU-bound infinite loops, and any "hung" subprocess.
- Return a structured tool error like
  `"command exceeded 120s wall-clock limit; terminated. partial
  output: ..."` so the agent can recover and retry with a different
  approach.

## Secondary concern: glob/grep/ls path scoping

The agent emitted `cwd="/"` for what should have been a workspace-
relative search. Worth scoping these tools so they refuse paths
outside `/workspace` (or at minimum, warn). That's a separate, smaller
fix in the file-system tool wrappers — but in the same spirit (the
small model gets path scoping wrong; the tool can guard it).

## Out of scope here

- Detecting "incorrect agent behavior" at the model level (prompt
  changes, training fixes). The fix above is a guardrail; the agent
  will still occasionally emit nonsense. We need the guardrail to
  bound the cost of nonsense.
