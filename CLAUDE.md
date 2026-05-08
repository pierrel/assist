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

## Branching strategy

Standard feature-branch flow.  Default to it for any non-trivial change.

1. **Branch off `main` per feature.**  Pick a short, kebab-case name that describes the change (verb-led when natural — `confine-research-to-references`, `bound-threads-db-growth`, `add-skill-x`).  Avoid umbrella names like `<topic>-stabilization` that don't say what's actually being done.

   **Always branch off `main` — never off another feature branch.**  Stacked PRs entangle two unrelated changes in one diff: the reviewer can't tell which lines belong to which feature, and merging one rebases the other.  If your work depends on an unmerged branch, *wait for it to merge* (or ask the user to merge it) before starting yours.  Don't optimize for "deploy preserves both fixes" — preserving prod state is the deploy step's problem, not the branching step's.

2. **Commit on the feature branch.**  Multiple commits per logically distinct change are fine; the user reviews the final diff at PR time.  `make a new commit, never amend` (per the wider Claude Code rules).  When the work is done locally, `git push -u origin <branch>` so the history is preserved even if the local clone goes away.

3. **Merge to `main` when the branch is ready.**  "Ready" means: design + reviewer phases per the Two-phase development workflow above are complete, evals pass at the cadence appropriate to scope, and any reviewer findings are addressed.  Prefer fast-forward merges (`git merge --ff-only <branch>`) so `main` stays linear; if the branch has diverged, rebase rather than create merge commits.

4. **Deploying from the feature branch is OK — but only one feature branch at a time.**  Per the user's saved preference, you may `make deploy-code` directly from the feature branch (don't wait for the merge to `main`).  But understand the trade-off: deploying a *second* feature branch will replace the first on prod — `rsync --delete` doesn't preserve unrelated changes from the previous deploy.  If multiple unmerged features need to be live simultaneously, merge the older one to `main` first so the newer feature branch can rebase onto it before deploying.

5. **Keep feature branches around briefly after merge.**  Don't immediately delete the local or remote branch — they're useful if a follow-up question or a quick revert is needed.  Garden them out periodically.

Direct commits to `main` are reserved for trivial fixes (typo, doc tweak) that wouldn't benefit from review.  Anything that touches `assist/agent.py`, `assist/middleware/`, or evals goes through a feature branch.

## Project conventions to remember across phases

- **Commits to feature branches don't need explicit confirmation; commits to `main` do.**  On a feature branch, commit and push as the work progresses — the user reviews the diff at PR time, not before each commit.  Direct commits to `main` (or anything that lands on `main` without going through a PR + review) still require the user's explicit go-ahead.
- **No new docs unless asked.** Don't write tutorial docs, design docs, or README sections that weren't requested. If the user asks for documentation, mirror the existing format (Skills section in `README.md` is the template for middleware-style additions).
- **Small-model targeting.** Code is run against Qwen3-Coder-30B-A3B-Instruct-AWQ on a local vLLM instance. Prompt and tool-surface decisions optimize for this model's failure modes, not GPT-4-class behavior. When evidence is needed, run the eval rather than reasoning from training-data intuitions.
- **Eval-first contracts.** Tests in `edd/eval/` define what the system is supposed to do. When in doubt, the test wins. Don't change tests to accommodate the implementation; redesign the implementation to satisfy the existing tests.
- **Generic skills/prompts; eval prompts probe cases the skill doesn't telegraph.** When you write or refine a skill description, prompt template, or example list, *check whether the wording is too closely aligned with the eval that validates it*. If a skill's examples mirror eval prompts almost verbatim, the eval mostly tests lexical proximity rather than the skill's actual generalization — and a high pass rate is misleading. Two consequences when you spot high alignment: (a) make the skill/prompt cover the *shape* of real user requests, not just the eval-shaped ones; (b) expand the eval suite with cases the skill's wording does **not** telegraph — that's how you actually probe generalization. The Phase 2 reviewer is the natural place to flag this; the Phase 1 architect should feed an alignment-check requirement into the example/prompt selection upfront so the reviewer catches less.
- **Skill frontmatter is YAML — avoid `: ` (colon-space) in unquoted description values.** A colon-space inside an unquoted YAML scalar starts a new mapping, breaking the skill loader (silent failure: `deepagents.middleware.skills` logs a warning and the skill is unreachable, so it never loads). Bit the calculate migration on `"stdev of these nine measurements: ..."`. Either drop the colon, rephrase to avoid `: `, or quote the whole description value. Same applies to other YAML-special sequences (`#` comments, leading `-` / `?` / `*`, balanced `[`/`{`). Run `python -c "import yaml; yaml.safe_load(open('SKILL.md').read().split('---')[1])"` after any frontmatter edit if you used punctuation that might trip the parser.
- **No real local paths in tracked files.**  Don't commit `/home/<user>/...`, real IPs/hostnames, or absolute paths from your machine — they leak operator identity, deploy topology, and (for shared repos) the existence of internal hosts.  All operator-specific paths flow through env vars defined in `.deploy.env` (production) or `.dev.env` (local), with placeholder examples in `.deploy.env.example` / `.dev.env.example` (which use generic stand-ins like `assist-prod`, `your-production-server`, `user@host:/path/to/repo.git`).  Before committing: `git ls-files | xargs grep -nE "/home/[a-z]+|/Users/[a-z]+|192\.168\."` should return nothing.  In docs that need to reference a deploy path, use `$ASSIST_THREADS_DIR/...` or `~/...`, not the literal path.  Defaults in scripts may use generic FHS conventions (`/var/lib/assist/threads`) since those aren't operator-specific, but real machine paths are off-limits.  Bit `build/ministral-14b-quantization.org` and several `docs/*.org` files before the 2026-05-08 audit; scrubbed in the same PR as this rule.
