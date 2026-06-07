# Exploration guidelines
Always read the README file before doing any work to understand project structure, patterns, and conventions.
# Design guideline
1. **Prime Directive: never design against the requirements**
   Design the architecture to *absorb change*, not to mirror today’s use cases. ([InfoQ][1])

2. **Decompose by volatility, not by function/domain**
   Identify likely areas of change and make them your building blocks; avoid functional/domain decomposition. ([InfoQ][1], [idesign.net][2])

3. **Encapsulate volatility (“vaults”)**
   Each component/service contains a specific kind of change so requirement changes don’t ripple across the system. ([InfoQ][1], [idesign.net][2])

4. **Compose behavior from stable building blocks**
   Satisfy use cases by integrating services; the architecture stays static as use cases vary (“Composable Design,” “There Is No Feature”). ([PagePlace][3])

5. **Use the standard roles to classify components**
   Structure around Clients, Managers, Engines, Resource Access, Resources (+ Utilities) to reflect common volatilities. ([PagePlace][3], [idesign.net][4])

6. **Keep business logic out of clients**
   Clients are the most volatile; push rules and workflows into Managers/Engines. ([InfoQ][1], [Bookey][5])

7. **Managers orchestrate, Engines decide**
   Managers handle workflows/sequencing; Engines implement rules/algorithms. Mind the Managers-to-Engines ratio. ([PagePlace][3], [Bookey][5])

8. **Isolate resource access as its own volatility**
   Separate *how* you access resources from *what* the resources are; both can change. ([PagePlace][3], [se-radio.net][6])

9. **Prefer closed (or semi-closed) architectures**
   Constrain who can call whom to control coupling and keep independence/stability. ([PagePlace][3], [Bookey][5])

10. **Layer consistently**
    Typical layers: Client → Business Logic (Managers/Engines) → ResourceAccess → Resource, with a Utilities bar. ([PagePlace][3])

11. **Classify responsibilities with the “Four Questions”**
    Use Löwy’s classification questions to decide the right role for a capability. ([PagePlace][3])

12. **Smallest useful set of services (\~order of 10)**
    Aim for the minimal number of composable services that cover the *core* use cases. ([InfoQ][1])

13. **Distinguish volatility vs. variability; map axes of volatility**
    Hunt for true *volatility* (business change) vs. code-level variability; use explicit axes to find it. ([PagePlace][3], [InformIT][7])

14. **Keep the architecture static across versions**
    Contain change inside components; the high-level structure shouldn’t churn. ([PagePlace][3])

15. **Resist functional/domain decomposition**
    It creates either tiny fragments with high integration cost or blobs with high internal complexity and pollutes clients. ([InfoQ][1])

16. **Design for communication and onboarding**
    Using common roles/semantics accelerates comprehension and keeps teams aligned. ([idesign.net][2], [PagePlace][3])

17. **Validate the system design early**
    Separate architecture (fast, upfront) from detailed design (during development); avoid “big upfront *detailed* design.” ([idesign.net][2], [PagePlace][3])

18. **Be microservice-agnostic**
    Service granularity follows volatility and composition, not a “microservices first” rule. ([PagePlace][3])

19. **Name and structure components explicitly**
    Clear names and semantics (“What’s in a Name”) enforce intent and make designs reviewable. ([PagePlace][3])

20. **Strive for symmetry**
    Keep analogous parts shaped the same; symmetrical structures reduce accidental complexity. ([PagePlace][3])

If you want, I can map these to your LangGraph agent platform with concrete examples of “Managers vs. Engines vs. ResourceAccess” nodes and how to keep clients thin.

[1]: https://www.infoq.com/articles/book-review-righting-software/ "Q&A on the Book Righting Software - InfoQ"
[2]: https://idesign.net/assets/documents/IDesign-Method-Management-Overview.pdf "Microsoft Word - IDesign-Method-Management-Overview.docx"
[3]: https://api.pageplace.de/preview/DT0400.9780136523987_A41316644/preview-9780136523987_A41316644.pdf "Righting Software"
[4]: https://www.idesign.net/Training/Architect-Master-Class?utm_source=chatgpt.com "Architect's Master Class"
[5]: https://cdn.bookey.app/files/pdf/book/en/righting-software.pdf?utm_source=chatgpt.com "Righting Software PDF"
[6]: https://se-radio.net/2020/04/episode-407-juval-lowy-on-righting-software/?utm_source=chatgpt.com "SE Radio 407: Juval Löwy on Righting Software"
[7]: https://www.informit.com/articles/article.aspx?p=2995357&seqNum=3&utm_source=chatgpt.com "Identifying Volatility | Software System Decomposition - InformIT"


# Coding guidelines
Always follow these guidelines when writing code:
1. Use intention-revealing names
   — Pronounceable, searchable, no encodings; name by purpose.
2. Keep functions small
   — Few lines, minimal branches.
3. Functions do one thing
   — No mixed responsibilities; extract until each has a single concern.
4. Minimize parameters
   — Prefer 0–2 args; avoid boolean flags and long parameter lists.
5. Separate command from query (CQS)
   — A routine either changes state or returns information, not both.
6. Avoid side effects; prefer immutability
   — Eliminate hidden temporal coupling; limit shared mutable state.
7. Make the code communicate
   — Self-documenting structure; comments only for “why,” not “what/how.”
8. Format for readability
   — Consistent layout, sensible spacing/indentation, meaningful vertical ordering.
9. Eliminate duplication (DRY)
   — Factor out repeats; duplication breeds divergence.
10. Prefer polymorphism/composition over conditionals
    — Replace big switch/if chains with strategy, state, or polymorphic dispatch.
11. Law of Demeter (“don’t talk to strangers”)
    — Avoid train-wreck call chains; interact with immediate collaborators.
12. Tell, don’t ask
    — Expose behaviors, not data; choose objects vs. data structures deliberately.
13. Handle errors with exceptions, not return codes
    — Narrow try/catch blocks; add context; avoid swallowed errors.
14. Isolate external boundaries
    — Wrap third-party APIs; keep dependencies at the edges.
15. Single Responsibility Principle (SRP)
    — One reason to change per module/class.
16. Open–Closed Principle (OCP)
    — Open for extension, closed for modification.
17. Liskov Substitution Principle (LSP)
    — Subtypes must be substitutable for their base types without surprises.
18. Interface Segregation Principle (ISP)
    — Many small, focused interfaces over one “fat” interface.
19. Dependency Inversion Principle (DIP)
    — Depend on abstractions, not concretions; invert source-code dependencies.
20. Simple, emergent design + keep it clean
    — Follows Kent Beck’s four rules (passes tests, no duplication, expresses intent, minimal elements) and the Boy Scout Rule: leave the code cleaner than you found it.

# Steering the small model: prefer guidance over middleware
This project runs a small local model (Qwen3.6-27B). We shape its behavior primarily by **instructing it** (skills, prompts), not by wrapping it in deterministic middleware. Middleware guardrails are a last resort, not a first reflex.

1. **Guidance first; middleware only when guidance provably can't.** When the model misbehaves, fix it by changing what we tell it (a skill / prompt) before adding a middleware guard. Reach for middleware only after eval evidence shows guidance can't carry it — and even then, weigh it. A guard that has to re-do the model's task, or inspect state the model already has, is usually the wrong tool.
   - Worked example: the org mid-section insertion bug. A deterministic `edit_file` guard was leaky (blind to anything outside the edit's `old_string`) and amounted to "doing the agent's job for it"; an eval-driven **skill** rewrite took it from 0/5 to ~4/5 with no middleware. See `docs/2026-06-03-org-insertion-mid-section.org`.
2. **The instruction's shape matters more than its content.** This model follows a **checkable constraint on its tool arguments** far better than prose describing the goal. Prefer "set `old_string` to exactly one heading line" over "insert at the right place." Six prose phrasings of the same rule failed (0/3); the arg-shape constraint moved it.

# Eval-driven behavior changes (process)
When changing how the model behaves (a skill, a prompt, a tool surface):

1. **Reproduce the failure in an eval FIRST.** Use a realistic fixture — match production scale/shape; small, clean fixtures often pass when the real failure needs a large/messy file (we saw synthetic fixtures pass 34/34 while the real ~365-line file failed every time). The eval is the bar. Reasoning about what the small model "should" do is unreliable — **measure, don't argue.**
2. **Change one thing at a time and measure (isolate-to-learn).** Add candidate instructions/riders individually, eval each, and combine only if no single one suffices. The goal is to learn *which* instruction the model actually obeys, not just to ship a fix.
3. **Review for generality; don't overfit to the one case.** A fix tuned to the exact reproduction is a non-fix. Stress-test the rule against the range of real request shapes (an agent reviewing for generality is worth a pass).
4. **Chasing a residual: 100%-or-revert.** When adding an extra rider to close the last failures, keep it only if it fully clears them; otherwise revert to the simpler version that already worked.
5. **Evals are not a pass/fail gate.** A partial pass-rate is expected and fine for evals (unlike unit tests). Don't mask a known-partial eval with `xfail` to keep a suite green — let it report its real rate.

# Testing guidelines
1. Always run python tests (`pytest`) whenever any python files are modified and iterate on any failures until tests pass.
2. Always run the emacs linter (`eldev lint` or `eldev lint -f [file]`) whenever any elisp files are modified.
3. Always run tests (`eldev test` from the assist/emacs directory) whenever elisp files are modified.
4. Always run any new elisp functions for correct functionality (using `eldev eval [expression]`) and iterate on any failures until all tests pass.

# Documentation guidelines
README files are always updated as new user-facing functionality is added or modified. Do not clutter the top-level README file(s) - always add sections as needed.

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

# Deploying after a merge
Whenever something lands on `main` — merged by Claude **or** by the user — deploy the latest `main` to prod as the next step: `make deploy-code` then `make restart`. A merge alone changes nothing live; prod runs the rsync'd copy under `$DEPLOY_PATH`, so runtime changes (agent, middleware, skills, sandbox image) have no effect until deployed. After deploying, verify the running code actually contains the change (e.g. `grep` the deployed file for a new symbol) and that the service is healthy.

Do not leave `main` ahead of prod silently: a "merged but not deployed" gap is invisible until a live thread hits the missing behavior. (This bit us: in-repo domain skills + the `elisp` skill were merged but undeployed, so a thread couldn't see a committed in-repo skill — prod had no discovery code.)

Caveat for in-repo skills specifically: the skill list is cached per session in the thread's checkpoint, so an **existing** thread won't pick up a newly-discovered domain skill even after deploy — a new chat will. New threads work immediately.
