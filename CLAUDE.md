# CLAUDE.md

Guidance for Claude Code working in this repository.  The repo's design-method
guide (exploration habits, Löwy "Righting Software" volatility decomposition,
clean-code rules) lives in `AGENTS.md`, imported here so it loads alongside
this file:

@AGENTS.md

## Three-phase development workflow

Non-trivial changes to this repo follow a **design phase**, a **coding phase**, and a **review phase**. Each phase has a named subagent team. The point is to (1) separate "what should we build" from "build it" so the design has a chance to be wrong cheaply, (2) give the implementation a clear contract to satisfy, and (3) surface anything reviewers (including automated ones like Copilot) flag once the diff is real.

This applies to: new middleware, agent-architecture changes, skill migrations, eval suite changes, prompt-engineering work where the small model's behavior matters, anything that touches `assist/agent.py` or `assist/middleware/`, the concurrency primitives (`assist/thread_queue.py`, locks, anything that runs on the asyncio event loop), or the request hot path (`manage/web/` message-submit / render).

It does NOT apply to: trivial fixes, doc edits, single-file refactors with no behavior change.

**Blast radius, not line count, sets the rigor.** A six-line change to a lock, the queue, the event loop, or the message-submit path gets the FULL three-phase treatment — design + design-review team + code-review team — even when it looks tiny, and *especially* when it's a fast fix for a regression (the rush is exactly when a subtle concurrency / hot-path change slips a light review). A 2-agent "focused review" is not enough for these. Bit us 2026-06-10: a small queue-feedback fix shipped with a light review introduced an event-loop deadlock that took the whole web server down.

### Phase 1 — Design

Spawn the design team via the `Agent` tool **before writing any code**.

| Role | Subagent type | Responsibility |
| --- | --- | --- |
| **Architect** | `Plan` | Owns the implementation plan. Reads the affected modules, names the files to touch, calls out the trade-offs, and lists risks/edge cases. Output is a numbered plan with file:line references. |
| **Investigator** *(optional)* | `Explore` | When the architect needs more codebase grounding than they can do solo — finds prior patterns, locates similar middleware, surveys eval coverage. Spawn only if the plan needs concrete file references the architect doesn't yet have. |
| **Researcher** *(optional)* | `general-purpose` | When the design depends on external info (library behavior, upstream change history, API contracts not in the local tree). Skip when the question is purely internal. |

**The architect produces a written plan; the main Claude (you) reads it, redirects if anything looks wrong, then runs the design-review team below.** Do not skip the redirect step — design agents have no memory of the user's prior corrections, so they can re-introduce mistakes the user has already pushed back on. If the plan conflicts with `auto-memory feedback` or session context, fix the plan before review.

**Design-review team.** Before handing off to coding, spawn these reviewers (each `general-purpose`, in parallel) and brief each on the plan + their specific lens. You read every report and revise the plan where reviewers' feedback is well-founded; push back in your own notes when it isn't.

| Lens | What to brief the reviewer to look for |
| --- | --- |
| **Simplicity** | Is the simplest design that meets the requirements? Are there layers, abstractions, or knobs the requirements don't justify? Would a smaller change cover the same ground? |
| **Agentic best practices** | Does the design align with current LangChain / LangGraph / `deepagents` patterns (middleware composition, state shape, sub-agent boundaries, tool surface, checkpointing)? Anything that fights the framework rather than working with it? |
| **User guidance & intention** | Does the plan match what the user actually asked for (and the spirit behind it)? Does it respect prior corrections from `auto-memory feedback` and session context? Does it introduce work the user didn't ask for? |
| **Clean interfaces** | Are the new module/function/tool boundaries cohesive? Are responsibilities well-separated? Is the public surface small and obvious? Are names accurate? |
| **Event-loop liveness** *(when the change touches `manage/web/` or anything reachable from an `async def` handler)* | Does any code that runs on the asyncio event-loop thread (a route handler, or a sync helper it calls inline) acquire a lock, do blocking file/network I/O, run a subprocess, or sleep? On single-worker uvicorn that's a FULL outage, not a slow request. Such work must be lock-free or pushed off the loop (`run_in_threadpool`). |

The architect (or the main Claude when revising the plan directly) treats the reviewers as advisors, not gates: address what's right, justify what isn't. Document any rejected suggestions briefly in the plan so the rationale survives into Phase 2.

### Phase 2 — Coding

The main Claude (you) writes the code, runs the evals, and ships. The code-review team is a peer-review gate before pushing.

| Role | Subagent type | Responsibility |
| --- | --- | --- |
| **Implementer** | (you, the main Claude) | Writes the edits using `Edit` / `Write` against the design from Phase 1. Runs the relevant evals. |
| **Re-tester** | (you, the main Claude) | After fixing review findings, re-runs the relevant evals at higher trial count (typically N=10) to confirm stability. |

**Code-review team.** Once the first complete implementation exists (tests pass at N=3, no obvious regressions), spawn these reviewers (each `general-purpose`, in parallel) and brief each on the uncommitted diff + their specific lens. You read every report, fix what's well-founded, and push back where it isn't.

| Lens | What to brief the reviewer to look for |
| --- | --- |
| **Simplicity** | Is the implementation as small as it can be while still satisfying the design? Dead branches? Premature abstractions? Knobs nobody asked for? |
| **Clean code** | Naming, function size, layering, error handling. Bugs, missing edge cases, leaks, race conditions. |
| **Event-loop liveness** *(when the diff touches `manage/web/` or anything reachable from an `async def` handler)* | Does any code on the asyncio event-loop thread (a route handler, or a sync helper it calls inline) acquire a lock, do blocking file/network I/O, run a subprocess, or sleep? On single-worker uvicorn that's a FULL outage, not a slow request. Demand a lock-free path or `run_in_threadpool`, and a test that holds the contended resource and asserts the call returns promptly. |
| **Readability** | Can someone reading this fresh in three months understand what it does and why? Where are the load-bearing comments missing? Where are comments restating obvious code? |
| **Existing patterns** | Does this follow conventions already established elsewhere in the repo (middleware shape, skill loading, eval style, threading model)? Is it reinventing something we already have? |
| **Design adherence** | Does the diff implement what the Phase-1 plan committed to? Are deviations called out and justified, or did they slip in unannounced? |
| **Refactoring opportunities** | Is there code outside the diff that this change makes easier to simplify? Note them — but do NOT widen the scope of this PR unless the user asks. |
| **Shared logic & LOC reduction** | Treat every line as a liability — each LOC is a potential bug and a maintenance cost. Actively hunt for duplicated or near-duplicated logic (within the diff AND against existing modules) that could be extracted into one shared helper, and for parallel code that should reuse an existing function instead of reimplementing it. Prioritize consolidating **copy-pasted load-bearing logic** — the subtle kind that must stay in sync (id-preserving message edits, retry tuples, tool-call pairing) — since divergence there is a latent bug. Where a smaller diff covers the same ground, say so. This is the in-PR complement to *Refactoring opportunities* (which looks outward and defers scope): here, prefer the share/extract when it cuts total LOC without fighting the design. Bit PR #138: the breaker copy-pasted loop_detection's strip-and-terminate block and re-implemented its event pairing — both shared in review, net LOC down. |

Each reviewer should return a numbered findings list with severity (BLOCKER / IMPORTANT / NIT) and a "ship it" or "block on these N items" bottom line. The code-review team must run **before** scaling eval trial counts — don't burn N=10 evals on code reviewers haven't seen. Fix BLOCKER and IMPORTANT items, then re-run.

### Phase 3 — Review (if the remote is GitHub)

After Phase 2, push the branch and verify the build succeeds. Then run a GitHub Copilot review loop.

**Setup.** Open the PR (or push to an existing one). If the build/CI doesn't pass first, fix that before requesting Copilot — there's no point reviewing a broken build.

**Requesting a Copilot review** (the magic incantation):

```bash
gh pr edit <PR#> --add-reviewer "@copilot"     # the @ prefix is required
gh api repos/<owner>/<repo>/pulls/<PR#>/requested_reviewers --jq '.users[] | {login, type}'
# confirm Copilot (login "Copilot", type "Bot") is queued
```

A push to an existing PR may also auto-trigger a Copilot review even without an explicit re-request — check the queue before re-requesting redundantly.

**Iteration loop.** Per round:

1. Wait ~10 minutes for Copilot to post the review (use `ScheduleWakeup` with `delaySeconds: 660-720` so the harness re-invokes you when it's time; don't poll).
2. Fetch the review and inline comments:
   ```bash
   gh api repos/<owner>/<repo>/pulls/<PR#>/comments
   ```
   Filter to the head commit's comments to see the latest round only.
3. Triage each comment with opinions:
   - **Address it** if the concern is real and aligned with the design — fix the code, push, move on.
   - **Push back** if it's a false positive, already-addressed, or it would push the implementation away from the Phase-1 design. Stick to the design — Copilot has no context for the trade-offs your design team weighed.
   - Watch for **re-flagged false positives**: Copilot frequently keeps the same backlog of "concerns" across rounds even after the code addresses them. Verify in the current file (`grep -n` on the named identifier) and don't re-fix.
4. After fixing, push and re-request Copilot for the next round.
5. **Resolve & reply** to comments before moving on:
   - For comments that are addressed in code, resolve the conversation thread (GraphQL `resolveReviewThread`) so the PR view stays clean.
   - For comments you're declining (false positive, scope, design-aligned trade-off), post a short reply on the thread (`POST /repos/<owner>/<repo>/pulls/<PR#>/comments/<comment_id>/replies`) explaining the rationale, then resolve.

**Cap: 7 iterations.** Copilot review converges asymptotically, not absolutely — once you've passed ~6 rounds, expect mostly re-flagged items with diminishing-returns new findings. Stop after round 7 regardless of state.

**After the loop, update the PR description.** If the review surfaced anything that warrants a *design* change (not just a code fix) — a new edge case the design didn't account for, a contract that needs to be tightened, a follow-up explicitly accepted as a v1 trade-off — call it out in the PR description under a "Design changes from review" or "Known limitations" section. Code comments alone aren't enough; the PR description is what humans read at merge time.

**Useful GraphQL for resolving threads:**

```bash
# List unresolved review threads
gh api graphql -f query='
  query { repository(owner:"<owner>", name:"<repo>") {
    pullRequest(number: <PR#>) {
      reviewThreads(first: 50) { nodes {
        id isResolved comments(first: 5) { nodes { body path line author { login } } }
      } } } } }'

# Resolve one
gh api graphql -f query='
  mutation { resolveReviewThread(input: { threadId: "<THREAD_ID>" }) {
    thread { isResolved } } }'
```

### Eval cadence

- **Baseline** (before any changes): N=3 per test on the affected suite. Establishes failure mode.
- **Post-implementation** (after coding, before code-review team): N=3 per test. Confirms no obvious regression.
- **Post-review** (after code-review team findings are addressed): N=5 per test. Catches issues the reviewers flagged but the implementation didn't fully fix.
- **Stability** (final, before push): N=10 per test. Pin the contract.

Treat any drop in pass rate compared to the prior step as a regression; investigate before scaling further.

### Testing & verification — catch the bug the review misses

These rules exist because the 2026-06-10 queue-feedback fix passed its unit tests and two reviews yet shipped two critical bugs (a message rendered in the wrong place, then an event-loop deadlock that took the server down). Both slipped because the tests checked proxies and mocked the exact risk.

- **Test the symptom, not the proxy.** When the bug is user-visible, the test must reproduce the user's actual symptom and assert it's gone — for a render bug, render the page and assert against the HTML (the element is present *and* in the right position), not an intermediate status field. Asserting `status == "queued"` is not the same as asserting the message appears where the user looks.
- **Don't mock away the risk.** Name the risky interaction a change introduces (a lock, a shared resource, a concurrency edge) and exercise it **un-mocked** in at least one test. Mocking the very thing whose timing/contention is the hazard gives false confidence — mocking the queue handle hid the lock that deadlocked the loop. For an event-loop concern, hold the contended resource in another thread and assert the call still returns promptly.
- **Verify under the failure condition, not idle.** Smoke (or `/verify`) a fix under the condition that triggered the bug, not a healthy idle path. "HTTP 200 on an idle box" can't surface a load-only deadlock; "a second prompt while a real long turn holds the queue" can. State the trigger and reproduce it.
- *(Mechanical backstop for event-loop work.)* Run the web tests or a smoke with `PYTHONASYNCIODEBUG=1` — asyncio debug mode logs any callback that blocks the loop past `loop.slow_callback_duration`, so a stall trips a warning instead of waiting to be spotted by eye.

### When the user gives a deadline

The user often caps eval runs at a wall-clock time ("no evals past 10:30p"). Track the deadline. Do not start a new sweep that won't finish in time. Final summaries go to the user with: files touched, trial counts, stability numbers, reviewer pushback addressed, Copilot rounds run, and any documented trade-offs.

## Branching strategy

Standard feature-branch flow.  Default to it for any non-trivial change.

1. **Branch off `main` per feature.**  Pick a short, kebab-case name that describes the change (verb-led when natural — `confine-research-to-references`, `bound-threads-db-growth`, `add-skill-x`).  Avoid umbrella names like `<topic>-stabilization` that don't say what's actually being done.

   **Always branch off `main` — never off another feature branch.**  Stacked PRs entangle two unrelated changes in one diff: the reviewer can't tell which lines belong to which feature, and merging one rebases the other.  If your work depends on an unmerged branch, *wait for it to merge* (or ask the user to merge it) before starting yours.  Don't optimize for "deploy preserves both fixes" — preserving prod state is the deploy step's problem, not the branching step's.

2. **Commit on the feature branch.**  Multiple commits per logically distinct change are fine; the user reviews the final diff at PR time.  `make a new commit, never amend` (per the wider Claude Code rules).  When the work is done locally, `git push -u origin <branch>` so the history is preserved even if the local clone goes away.

3. **Merge to `main` when the branch is ready.**  "Ready" means: all three phases of the Three-phase development workflow above are complete (design + design-review team feedback considered, code + code-review team findings addressed, Copilot review iteration converged or capped), evals pass at the cadence appropriate to scope.  Prefer fast-forward merges (`git merge --ff-only <branch>`) so `main` stays linear; if the branch has diverged, rebase rather than create merge commits.

4. **Deploying from the feature branch is OK — but only one feature branch at a time.**  Per the user's saved preference, you may `make deploy-code` directly from the feature branch (don't wait for the merge to `main`).  But understand the trade-off: deploying a *second* feature branch will replace the first on prod — `rsync --delete` doesn't preserve unrelated changes from the previous deploy.  If multiple unmerged features need to be live simultaneously, merge the older one to `main` first so the newer feature branch can rebase onto it before deploying.

5. **Keep feature branches around briefly after merge.**  Don't immediately delete the local or remote branch — they're useful if a follow-up question or a quick revert is needed.  Garden them out periodically.

Direct commits to `main` are reserved for trivial fixes (typo, doc tweak) that wouldn't benefit from review.  Anything that touches `assist/agent.py`, `assist/middleware/`, or evals goes through a feature branch.

## Project conventions to remember across phases

- **Commits to feature branches don't need explicit confirmation; commits to `main` do.**  On a feature branch, commit and push as the work progresses — the user reviews the diff at PR time, not before each commit.  Direct commits to `main` (or anything that lands on `main` without going through a PR + review) still require the user's explicit go-ahead.
- **No new docs unless asked.** Don't write tutorial docs, design docs, or README sections that weren't requested. If the user asks for documentation, mirror the existing format (Skills section in `README.md` is the template for middleware-style additions).
- **Small-model targeting.** Code is run against Qwen3.6-27B (Q4_K_M gguf) on a local llama.cpp instance — OpenAI-compatible, served at `ASSIST_MODEL_URL`. It is a Qwen3 *reasoning* model (emits `<think>` / `reasoning_content`), so the streaming + empty-response handling account for thinking blocks. Prompt and tool-surface decisions optimize for this model's failure modes, not GPT-4-class behavior. When evidence is needed, run the eval rather than reasoning from training-data intuitions.
- **Eval-first contracts.** Tests in `edd/eval/` define what the system is supposed to do. When in doubt, the test wins. Don't change tests to accommodate the implementation; redesign the implementation to satisfy the existing tests.
- **Generic skills/prompts; eval prompts probe cases the skill doesn't telegraph.** When you write or refine a skill description, prompt template, or example list, *check whether the wording is too closely aligned with the eval that validates it*. If a skill's examples mirror eval prompts almost verbatim, the eval mostly tests lexical proximity rather than the skill's actual generalization — and a high pass rate is misleading. Two consequences when you spot high alignment: (a) make the skill/prompt cover the *shape* of real user requests, not just the eval-shaped ones; (b) expand the eval suite with cases the skill's wording does **not** telegraph — that's how you actually probe generalization. The Phase 2 code-review team is the natural place to flag this; the Phase 1 architect (or design-review team's "clean interfaces" lens) should feed an alignment-check requirement into the example/prompt selection upfront so the code reviewers catch less.
- **Skill frontmatter is YAML — avoid `: ` (colon-space) in unquoted description values.** A colon-space inside an unquoted YAML scalar starts a new mapping, breaking the skill loader (silent failure: `deepagents.middleware.skills` logs a warning and the skill is unreachable, so it never loads). Bit the calculate migration on `"stdev of these nine measurements: ..."`. Either drop the colon, rephrase to avoid `: `, or quote the whole description value. Same applies to other YAML-special sequences (`#` comments, leading `-` / `?` / `*`, balanced `[`/`{`). Run `python -c "import yaml; yaml.safe_load(open('SKILL.md').read().split('---')[1])"` after any frontmatter edit if you used punctuation that might trip the parser.
- **No real local paths in tracked files.**  Don't commit `/home/<user>/...`, real IPs/hostnames, or absolute paths from your machine — they leak operator identity, deploy topology, and (for shared repos) the existence of internal hosts.  All operator-specific paths flow through env vars defined in `.deploy.env` (production) or `.dev.env` (local), with placeholder examples in `.deploy.env.example` / `.dev.env.example` (which use generic stand-ins like `assist-prod`, `your-production-server`, `user@host:/path/to/repo.git`).  Before committing: `git ls-files | xargs grep -nE "/home/[a-z]+|/Users/[a-z]+|192\.168\."` should return nothing.  In docs that need to reference a deploy path, use `$ASSIST_THREADS_DIR/...` or `~/...`, not the literal path.  Defaults in scripts may use generic FHS conventions (`/var/lib/assist/threads`) since those aren't operator-specific, but real machine paths are off-limits.  Bit `build/ministral-14b-quantization.org` and several `docs/*.org` files before the 2026-05-08 audit; scrubbed in the same PR as this rule.
- **No real user/family PII in tracked or published artifacts.**  Never put real personal details — a child's or spouse's name, ages, birthdays, addresses — into source, **test fixtures**, prompt examples, eval queries, design docs, commit messages, or **GitHub PR descriptions / review comments**.  Invent synthetic personas (e.g. `Sam`, `Jordan`, a made-up age/date) for fixtures and examples from the start.  Before committing, grep the diff for real names/ages/identifying details.  Bit `edd/eval/test_context_agent.py` (a son's name + age, a spouse's name + birthdays), a `research_instructions.txt.j2` example (`kids_watch.md`), an eval query, a design doc, and PR #120's public description — which had also reached deployed prod code and `main`'s history; scrubbed 2026-06-01.  When PII has already leaked, scrubbing current files isn't enough: it persists in git history (usually on `main`, not just the branch) and in deployed code, so the cleanup needs a history rewrite + redeploy.
