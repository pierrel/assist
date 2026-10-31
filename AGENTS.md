# Assist agent guide

# Exploration guidelines
Always read the README file before doing any work to understand project structure, patterns, and conventions.
# Agent guidance philosophy and architecture

This is the durable philosophy for Assist prompting. The model should see the
smallest useful instruction at the point where it can act on it. Code and tools
enforce real boundaries; prompting explains how to make good choices within
them.

| Layer | Responsibility |
|---|---|
| Deterministic tools and guards | Authorization, safety, real effects, and trusted-state boundaries. |
| System prompt | Short, product-wide invariants: user intent, grounding, evidence, determinism, ownership, and truthful results. |
| Skill description | When a capability applies. |
| Loaded skill | How to perform that capability: tools, workers, sequencing, storage, formats, and checks. |
| Evals | Measure behavior, reveal general patterns, and prevent a fixture from becoming the design. |

Tools, skills, and their runtime capability travel together. Expose a
capability's tools and instructions only when its backing is available; make
optional instructions conditional without hiding an otherwise usable skill.
Review tool availability and skill text together when their wiring changes.

Apply this placement rule when changing model-facing behavior:

1. Put only general principles with value across capabilities in the system
   prompt. Grounding and research may appear there as epistemic requirements,
   but their procedure remains in their skills. Do not add other skill names,
   routing chains, worker call sequences, report formats, or one-eval exceptions
   to the core.
2. Put *when to load* a capability in its skill description. Put *what to do
   after it is loaded* in the body. Keep one concept with one owner; do not
   duplicate a worker's handoff or give two visible paths for the same kind of
   file or state.
3. Try general skill-description/body guidance before a system-prompt rider or
   a guard. Add a core rule only when it expresses a useful principle beyond
   the failing capability. If no such principle exists, record the failure
   rather than teaching the prompt a fixture.
4. Treat failed probes as evidence. Compare a small, representative candidate
   panel with its matched baseline, preserve adjacent behavior, and scale trial
   counts only after review. Keep natural outcome acceptance separate from
   explicit capability probes. Do not stack competing priority instructions to
   force one eval through.

## Steering the small model: prefer guidance over middleware
Code is run against Qwen3.6-27B (Q4_K_M gguf) on a local llama.cpp instance — OpenAI-compatible, served at `ASSIST_MODEL_URL`. It is a Qwen3 *reasoning* model (emits `<think>` / `reasoning_content`), so the streaming + empty-response handling account for thinking blocks. Prompt and tool-surface decisions optimize for this model's failure modes, not GPT-4-class behavior. When evidence is needed, run the eval rather than reasoning from training-data intuitions.

We shape its behavior primarily by **instructing it** (skills, prompts), not by wrapping it in deterministic middleware. Middleware guardrails are a last resort, not a first reflex.

1. **Guidance first; middleware only when guidance provably can't.** When the model misbehaves, fix it by changing what we tell it (a skill / prompt) before adding a middleware guard. Reach for middleware only after eval evidence shows guidance can't carry it — and even then, weigh it. A guard that has to re-do the model's task, or inspect state the model already has, is usually the wrong tool.
   - Worked example: the org mid-section insertion bug. A deterministic `edit_file` guard was leaky (blind to anything outside the edit's `old_string`) and amounted to "doing the agent's job for it"; an eval-driven **skill** rewrite took it from 0/5 to ~4/5 with no middleware. See `docs/2026-06-03-org-insertion-mid-section.org`.
2. **System prompts state general principles; skills state procedures.** Apart from grounding and research lifecycle guidance, put only guidance with value across several capabilities in the system prompt. Do not restate, boost, or tailor it to one skill or eval. When skill catalog/body guidance fails, first identify the broader outcome or principle that skill supports and test adjacent behavior too. If no valuable general principle is clear, ask Pierre; if neither side can identify one, reconsider whether making that eval pass is worth the added guidance.
3. **The instruction's shape matters more than its content.** This model follows a **checkable constraint on its tool arguments** far better than prose describing the goal. Prefer "set `old_string` to exactly one heading line" over "insert at the right place." Six prose phrasings of the same rule failed (0/3); the arg-shape constraint moved it.

# Eval-driven behavior changes (process)
When changing how the model behaves (a skill, a prompt, a tool surface):

1. **Reproduce the failure in an eval FIRST.** Use a realistic fixture — match production scale/shape; small, clean fixtures often pass when the real failure needs a large/messy file (we saw synthetic fixtures pass 34/34 while the real ~365-line file failed every time). The eval is the bar. Reasoning about what the small model "should" do is unreliable — **measure, don't argue.**
2. **Change one thing at a time and measure (isolate-to-learn).** Add candidate instructions/riders individually, eval each, and combine only if no single one suffices. The goal is to learn *which* instruction the model actually obeys, not just to ship a fix.
3. **Review for generality; don't overfit to the one case.** A fix tuned to the exact reproduction is a non-fix. Stress-test the rule against the range of real request shapes (an agent reviewing for generality is worth a pass).
4. **Chasing a residual: 100%-or-revert.** When adding an extra rider to close the last failures, keep it only if it fully clears them; otherwise revert to the simpler version that already worked.
5. **Evals are not a pass/fail gate.** A partial pass-rate is expected and fine for evals (unlike unit tests). Don't mask a known-partial eval with `xfail` to keep a suite green — let it report its real rate.
6. **Run capability coverage for model-facing edits.** Before changing a system
   prompt, skill, model-visible tool description, or other behavior-modifying
   guidance, inventory every capability it mentions, moves, removes, or
   rewords. Run the corresponding natural and/or explicit capability evals in
   the candidate safety panel, including a capability whose instruction was
   removed. Treat framework capabilities such as `write_todos` the same way as
   named skills, and list the selected rows and results in the PR description.
7. **Survey, compare, and scale by default.** Unless Pierre explicitly
   overrides it, survey the affected and adjacent behavioral evals before a
   model-facing change. The survey may be representative rather than
   exhaustive, but must name the selected rows and why they cover the changed
   behavior. Run the matched pre-change baseline at N=3, candidate testing at
   N=3, confidence at N=5, and final stability at N=10. Do not advance after a
   regression: iterate generally, then re-run the affected stage. New natural
   behavioral rows must reach at least 70% by the final stage; deterministic or
   explicit capability contracts must pass completely. Put a baseline-versus-
   candidate table with every stage and pass count in the PR description.

# Skill and eval contracts

- **Eval-first contracts.** Tests in `edd/eval/` define what the system is supposed to do. When in doubt, the test wins. Don't change tests to accommodate the implementation; redesign the implementation to satisfy the existing tests.

- **Generic skills/prompts; eval prompts probe cases the skill doesn't telegraph.** When you write or refine a skill description, prompt template, or example list, *check whether the wording is too closely aligned with the eval that validates it*. If a skill's examples mirror eval prompts almost verbatim, the eval mostly tests lexical proximity rather than the skill's actual generalization — and a high pass rate is misleading. Two consequences when you spot high alignment: (a) make the skill/prompt cover the *shape* of real user requests, not just the eval-shaped ones; (b) expand the eval suite with cases the skill's wording does **not** telegraph — that's how you actually probe generalization.

- **Skill frontmatter is YAML — avoid `: ` (colon-space) in unquoted description values.** A colon-space inside an unquoted YAML scalar starts a new mapping, breaking the skill loader (silent failure: `deepagents.middleware.skills` logs a warning and the skill is unreachable, so it never loads). Bit the calculate migration on `"stdev of these nine measurements: ..."`. Either drop the colon, rephrase to avoid `: `, or quote the whole description value. Same applies to other YAML-special sequences (`#` comments, leading `-` / `?` / `*`, balanced `[`/`{`). Run `python -c "import yaml; yaml.safe_load(open('SKILL.md').read().split('---')[1])"` after any frontmatter edit if you used punctuation that might trip the parser.

# Testing guidelines
1. Always run python tests (`pytest`) whenever any python files are modified and iterate on any failures until tests pass.
2. Always run the emacs linter (`eldev lint` or `eldev lint -f [file]`) whenever any elisp files are modified.
3. Always run tests (`eldev test` from the assist/emacs directory) whenever elisp files are modified.
4. Always run any new elisp functions for correct functionality (using `eldev eval [expression]`) and iterate on any failures until all tests pass.
5. **ALWAYS MOCK the research subagent in evals — unless the eval specifically tests research/search behavior.** Real search hits the self-hosted SearXNG, which rate-limits under a burst (the whole suite grinds for hours + trips "search unavailable" false-negatives). Only a *few crucial* evals do real search (`test_research_agent`, `test_research_reliability`); every *other* eval that happens to trigger research must stub the research subagent to return canned findings immediately — use the shared `stub_research_subagent` helper (generalized from `test_context_agent`'s `_stub_subagent`). Mocking also adds determinism (canned findings vs stochastic search). See `docs/2026-07-06-research-eval-mocking.org`. When you DO run a real-search eval, run it slowly/serially and grep the log for rate-limit messages (`_SEARCH_UNAVAILABLE` / "search unavailable") so a throttled run isn't mistaken for a real failure.

## Failure-condition verification

Load-bearing paths include locks, `THREAD_QUEUE`, the single-worker uvicorn
event loop, and the message-submit/render hot path (`manage/web/threads.py`
`_process_message`). After a layout or packaging refactor, smoke
`python -m manage.web`, the service entry point.

Code reached inline from an async handler must not block the single-worker
uvicorn event loop with locks, file/network I/O, subprocesses or sleeps. Use a
lock-free path or move blocking work off the loop with `run_in_threadpool`;
exercise actual contention and assert the handler returns promptly.

- **Test the symptom, not the proxy.** When the bug is user-visible, the test must reproduce the user's actual symptom and assert it's gone — for a render bug, render the page and assert against the HTML (the element is present *and* in the right position), not an intermediate status field. Asserting `status == "queued"` is not the same as asserting the message appears where the user looks.
- **Don't mock away the risk.** Name the risky interaction a change introduces (a lock, a shared resource, a concurrency edge) and exercise it **un-mocked** in at least one test. Mocking the very thing whose timing/contention is the hazard gives false confidence — mocking the queue handle hid the lock that deadlocked the loop. For an event-loop concern, hold the contended resource in another thread and assert the call still returns promptly.
- **Verify under the failure condition, not idle.** Smoke (or `/verify`) a fix under the condition that triggered the bug, not a healthy idle path. "HTTP 200 on an idle box" can't surface a load-only deadlock; "a second prompt while a real long turn holds the queue" can. State the trigger and reproduce it.
- *(Mechanical backstop for event-loop work.)* Run the web tests or a smoke with `PYTHONASYNCIODEBUG=1` — asyncio debug mode logs any callback that blocks the loop past `loop.slow_callback_duration`, so a stall trips a warning instead of waiting to be spotted by eye.

# Documentation guidelines
README files are always updated as new user-facing functionality is added or modified. Do not clutter the top-level README file(s) - always add sections as needed.

Do not add tutorial docs, design docs or README sections that were not requested. For requested documentation, mirror the existing format.

# Configuration & repository hygiene
Host-specific values — absolute paths inside someone's home directory, deploy hosts, service-specific identifiers — must never land in committed files. They go in `.dev.env` (local development) or `.deploy.env` (production); both are gitignored. Use the existing patterns:

1. **Env vars in scripts.** Scripts read paths from env (e.g. `ASSIST_THREADS_DIR`, `DEPLOY_PATH`, `SERVICE_NAME`). Refuse to start with `: "${VAR:?explanation}"` rather than embedding a default that ships your machine layout to others. See `scripts/vacuum-prod-db.sh` for the pattern.

2. **Makefile passes env over ssh.** Targets that run remote scripts pass the values from `.deploy.env` through ssh, e.g.:
   ```make
   vacuum-now:
       @ssh $(DEPLOY_HOST) \
           ASSIST_THREADS_DIR=$(ASSIST_THREADS_DIR) \
           SERVICE_NAME=$(SERVICE_NAME) \
           '$(DEPLOY_PATH)/scripts/vacuum-prod-db.sh'
   ```
   See `deploy-service` and `vacuum-now` for the canonical pattern.

3. **Examples / doc snippets.** Use angle-bracket placeholders (`<DEPLOY_PATH>`, `<ASSIST_THREADS_DIR>`) where a literal path would otherwise appear. Operators substitute their own values when they read the file.

Before committing a script or example, grep your diff for your own `/home/<you>/...` and any production hostnames; if anything matches, lift it into env first.

- **No real user/family PII in tracked or published artifacts.**  Never put real personal details — a child's or spouse's name, ages, birthdays, addresses — into source, **test fixtures**, prompt examples, eval queries, design docs, commit messages, or **GitHub PR descriptions / review comments**.  Invent synthetic personas (e.g. `Sam`, `Jordan`, a made-up age/date) for fixtures and examples from the start.  Before committing, grep the diff for real names/ages/identifying details.  Bit `edd/eval/test_context_agent.py` (a son's name + age, a spouse's name + birthdays), a `research_instructions.txt.j2` example (`kids_watch.md`), an eval query, a design doc, and PR #120's public description — which had also reached deployed prod code and `main`'s history; scrubbed 2026-06-01.  When PII has already leaked, scrubbing current files isn't enough: it persists in git history (usually on `main`, not just the branch) and in deployed code, so the cleanup needs a history rewrite + redeploy.

# Runtime deployment facts

Production runs the copy under `$DEPLOY_PATH`; a source merge alone does not
update it. `make deploy-code` copies the code, and runtime changes need the
appropriate restart and verification. Verify installed source identity and
service health after deployment.

Caveat for in-repo skills specifically: the skill list is cached per session in the thread's checkpoint, so an **existing** thread won't pick up a newly-discovered domain skill even after deploy — a new chat will. New threads work immediately.
