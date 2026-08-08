---
name: complex-request
description: Use when a request has several distinct deliverables, independently useful workstreams, files, or dependent stages. Typical requests include preparing website copy, a release checklist, and a support announcement; comparing options before making an implementation plan; or revising a budget before reconciling its totals.
---

# Complex request workflow

## First action after loading

Make the next tool call only the planned `start_async_task` calls. For an entirely
context, external-research, or critique request, start that specialist directly.
Otherwise, start one `delegate-agent` per independent, currently unblocked outcome.
Do not inspect inputs or start unplanned context, research, or other tasks first;
known input paths belong in the self-contained task brief.

1. List the requested outcomes with `write_todos` after launch when it helps
   supervision. Use one
   TODO per independently useful, verifiable result and note real dependencies.
   TODOs are a planning aid; task status and workspace evidence are the truth, and
   imperfect bookkeeping must not prevent useful work.
2. Choose an owner for each outcome:
   - When the entire request is pure context, external research, or critique,
     send it directly to that specialist.
   - Keep one outcome with several production and verification steps together.
   - For two or more requested outcomes, start one `delegate-agent` per outcome;
     each delegate may call specialists for its own grounding.
3. Start every independent delegate in the same turn. An exact input path is enough
   context for a delegate that can read it; do not pre-read or summarize known inputs.
   If an outcome needs an earlier result or changes the same workspace state, start it
   only after checking its prerequisite.
4. Make every subagent request explicit and self-contained. Use labeled fields so
   nothing is implicit: `Outcome`, `Inputs`, `Constraints / non-goals`, `Verify`,
   and `Return`. State:
   - the exact outcome and target path, if any;
   - the inputs and relevant facts it may use;
   - constraints and non-goals;
   - how to verify completion;
   - the evidence or concise result to return.

   A subagent receives no parent conversation history. Never use shorthand such as
   "handle the second item" or assume it can infer omitted context.
5. After each completion wake, check the exact task. Reconcile the TODOs with the
   checked result and workspace evidence, then start newly unblocked work. Leave
   pending siblings alone; their own completion will wake you, so never poll or do
   their work yourself. A failed or timed-out outcome stays failed: do not retry it,
   replace it yourself, or start its dependents unless the user asks you to try again.
6. Before finishing, compare the requested outcomes with actual results. Correct any
   missed or stale TODO state, verify each outcome, and report what completed, what
   remains usable, and what is blocked.
