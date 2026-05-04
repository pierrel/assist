# CLAUDE.md

Guidance for Claude Code working in this repository.

## Two-phase development workflow

Non-trivial changes to this repo follow a **design phase** and a **coding phase**. Each phase has a named subagent team. The point is to separate "what should we build" from "build it" so the design has a chance to be wrong cheaply, and the implementation has a clear contract to satisfy.

This applies to: new middleware, agent-architecture changes, skill migrations, eval suite changes, prompt-engineering work where the small model's behavior matters, and anything that touches `assist/agent.py` or `assist/middleware/`.

It does NOT apply to: trivial fixes, doc edits, single-file refactors with no behavior change.

### Phase 1 — Design

Spawn the design team via the `Agent` tool **before writing any code**.

| Role | Subagent type | Responsibility |
| --- | --- | --- |
| **Architect** | `Plan` | Owns the implementation plan. Reads the affected modules, names the files to touch, calls out the trade-offs, and lists risks/edge cases. Output is a numbered plan with file:line references. |
| **Investigator** *(optional)* | `Explore` | When the architect needs more codebase grounding than they can do solo — finds prior patterns, locates similar middleware, surveys eval coverage. Spawn only if the plan needs concrete file references the architect doesn't yet have. |
| **Researcher** *(optional)* | `general-purpose` | When the design depends on external info (library behavior, upstream change history, API contracts not in the local tree). Skip when the question is purely internal. |

**The architect produces a written plan; the main Claude (you) reads it, redirects if anything looks wrong, then hands off to Phase 2.** Do not skip the redirect step — design agents have no memory of the user's prior corrections, so they can re-introduce mistakes the user has already pushed back on. If the plan conflicts with `auto-memory feedback` or session context, fix the plan before coding.

### Phase 2 — Coding

The main Claude (you) writes the code, runs the evals, and ships. The reviewer subagent is a peer-review gate before declaring done.

| Role | Subagent type | Responsibility |
| --- | --- | --- |
| **Implementer** | (you, the main Claude) | Writes the edits using `Edit` / `Write` against the design from Phase 1. Runs the relevant evals. |
| **Reviewer** | `general-purpose` | Reviews the uncommitted diff against the design contract. Flags bugs, missing edge cases, regressions, awkward design. Returns a numbered findings list with severity (BLOCKER / IMPORTANT / NIT) and a "ship it" or "block on these N items" bottom line. |
| **Re-tester** | (you, the main Claude) | After fixing reviewer findings, re-runs the relevant evals at higher trial count (typically N=10) to confirm stability. |

**The reviewer must run after the first complete implementation, before scaling eval trial counts.** Don't burn N=10 evals on code the reviewer hasn't seen — fix the BLOCKER and IMPORTANT items first, then re-run.

### Eval cadence

- **Baseline** (before any changes): N=3 per test on the affected suite. Establishes failure mode.
- **Post-implementation** (after coding, before review): N=3 per test. Confirms no obvious regression.
- **Post-review** (after reviewer findings are addressed): N=5 per test. Catches issues the reviewer flagged but the implementation didn't fully fix.
- **Stability** (final): N=10 per test. Pin the contract.

Treat any drop in pass rate compared to the prior step as a regression; investigate before scaling further.

### When the user gives a deadline

The user often caps eval runs at a wall-clock time ("no evals past 10:30p"). Track the deadline. Do not start a new sweep that won't finish in time. Final summaries go to the user with: files touched, trial counts, stability numbers, reviewer pushback addressed, and any documented trade-offs.

## Project conventions to remember across phases

- **No commits.** Default behavior is to leave changes uncommitted so the user can review the final diff. Confirm before any `git commit`.
- **No new docs unless asked.** Don't write tutorial docs, design docs, or README sections that weren't requested. If the user asks for documentation, mirror the existing format (Skills section in `README.md` is the template for middleware-style additions).
- **Small-model targeting.** Code is run against Qwen3-Coder-30B-A3B-Instruct-AWQ on a local vLLM instance. Prompt and tool-surface decisions optimize for this model's failure modes, not GPT-4-class behavior. When evidence is needed, run the eval rather than reasoning from training-data intuitions.
- **Eval-first contracts.** Tests in `edd/eval/` define what the system is supposed to do. When in doubt, the test wins. Don't change tests to accommodate the implementation; redesign the implementation to satisfy the existing tests.
