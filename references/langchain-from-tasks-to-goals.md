# From tasks to goals — review and Assist incorporation notes

- **Source:** https://www.langchain.com/blog/from-tasks-to-goals
- **Author:** Harrison Chase (LangChain CEO / co-founder)
- **Series / category:** Harrison's In the Loop
- **Published:** 2026-11-16 (last modified 2026-11-16)
- **Read time:** ~5 min
- **Reviewed for:** the Assist project (local-first, small-LLM-tuned assistant)
- **Reviewed:** 2026-11-27
- **Related post linked from the article:** [Goal engineering](https://www.langchain.com/blog/goal-engineering)

---

## 1. Summary

As models get more capable, the useful unit you hand an agent is shifting from a
**task** ("write this email", "fix this bug") to a **goal** ("keep my
commitments on track", "maintain a blog"). That is not just a bigger prompt. A
goal delegates two decisions the user no longer makes: *what work to do* and
*when to do it*. The user's role moves from specifying each action to setting
direction, defining boundaries, and reviewing progress.

Under the hood, "tasks → goals" is really two practical shifts:

- From specifying an **action** to specifying an **outcome**.
- From a **bounded assignment** to an **ongoing responsibility**.

The post walks through six consequences, each ending in a "practical
consideration":

1. **The agent chooses the tasks.** The user states a goal; the agent
   deconstructs it into tasks, does some, updates the task list, and repeats.
   The agent must be good at brainstorming *varied* tasks (not re-suggesting the
   same ones). You need a visible task list (a to-do) plus prompting for creative
   task generation.

2. **Tool calls are far less known.** For a task you can predict the tools
   (email → gmail/calendar). For a goal you often can't predict which tools will
   be needed. You need more control over what an agent *can and cannot* call.

3. **An inbox to ask questions.** A task runs with the user in the loop, so
   follow-up questions are easy. A goal may run in the background and need
   clarification later — it needs a channel to ask a human (Slack, a dedicated
   UI).

4. **A way to get feedback.** The agent needs to know how it's doing — via a
   metric it can track, or explicit human feedback. Metrics are easier to track
   automatically but should never be blindly optimized, so explicit feedback may
   be best. Design goals so feedback is easy to give.

5. **Ongoing work needs triggers (and stopping rules).** Background goal work is
   not always running — it wakes on a trigger: heartbeats, recurring cron,
   one-off cron/reminder, or an event stream. It equally needs a way to *stop,
   pause, or ignore* triggers; pinging the same question every 30 minutes is
   terrible UX. "Ongoing responsibility does not mean always find something to
   do. Sometimes doing nothing is the right action."

6. **Memory lets feedback compound.** In a task world the user re-supplies context
   each time, so memory matters less. With an ongoing responsibility the agent
   must carry decisions and feedback forward — e.g. rejecting a blog topic should
   shape future topic selection, not just the current draft.

**Conclusion:** giving an agent a goal is delegating the decisions of *what* work
to do and *when*. Your role shifts from specifying each task to setting
direction, defining boundaries, and reviewing progress.

---

## 2. How this maps onto Assist

The useful framing for Assist is that it is **already most of the way to the goal
paradigm** — most of the post's six building blocks already exist as shipped
capabilities. The real work is closing a few specific gaps, mostly in prompting
and one new HITL mechanism, not in new infrastructure.

| Post building block | Assist already has | Where |
|---|---|---|
| Agent chooses tasks; visible task list | `write_todos` tool; planning/`plan→test→implement` flow; task decomposition for complex/multi-part work | `write_todos`; `assist/skills/dev`; `assist/main_skills/complex-request`; `assist/main_skills/orchestrate-repeated-work` |
| Triggers (cron / event) | Recurring prompts; inbound-message subscriptions | `assist/skills/schedule`; `assist/skills/subscribe-events` |
| Stopping rule ("do nothing is OK") | `quiet()` — a routine run with nothing to report adds no "new" badge | `quiet()` tool |
| Memory that compounds feedback | Two-tier memory re-read every turn: cross-thread `AGENTS.md` + per-thread `/agent/memory.md` (goal, decisions, corrections, progress, blockers); checkpoint refresh | `assist/middleware/memory_middleware.py`; `assist/main_guidance_skills/maintain-thread-state` |
| Tool control | Skill-gated tools (hidden until the owning skill loads), egress allowlist with per-thread exact host/port grants, read-only enforcer, per-turn Docker sandbox | `assist/middleware/skills_middleware.py`, egress store, `read_only_enforcer.py`, sandbox |
| Asking a human (HITL) | Approval cards (email/SMS/egress/browser), `notify()` phone alert, web + phone inbox | approval/`notify()` infrastructure |

So the post validates the direction Assist is already built in. Below are the
gaps, ordered by value-per-effort. Per the project's own philosophy
(`AGENTS.md`), these are framed **guidance-first** (skills/prompts before
middleware), to be proven on a small representative eval panel before anything
touches the core prompt, and to be measured rather than argued.

---

## 3. Recommendations for Assist (prioritized)

### R1 — Strengthen the stopping rule on scheduled / event goal runs (lowest cost, high value)
**Blog tie:** §5 — ongoing work needs *stopping* rules; "sometimes doing nothing
is the right action"; don't ping the same question every trigger.

Assist already has the triggers (`schedule`, `subscribe-events`) and the
"do nothing" primitive (`quiet()`), but the *prompt* that a scheduled goal run
executes does not consistently tell the agent to first decide whether action is
needed. The `subscribe-events` skill already models this well — its template ends
with "always state your decision every time — either propose a reply or say
explicitly that no action is needed — so a message never vanishes silently."

**Do:**
- Codify that same "decide-then-act-or-quiet" contract for *scheduled* goal runs
  (extend `assist/skills/schedule` guidance, or a small companion), so each wake
  begins with an explicit act-vs-nothing decision and ends with `quiet()` when
  nothing changed.
- Add a de-duplication rule: if the agent already asked a question on a prior run
  and it is still unanswered, it should *not* re-ask — it should stay quiet (or
  note the open question in `/agent/memory.md`) instead of re-pinging.

**Prove:** a small eval panel where a scheduled goal run fires N times with no
new information; assert the agent acts at most once and stays `quiet()` the rest,
and does not repeat an unanswered question. This is a prompt/skill change — no
middleware, no core-prompt edit.

### R2 — A first-class "pursue a goal" loop (the core gap)
**Blog tie:** §1 — the agent chooses the tasks, deconstructs a goal, and iterates
toward it with a visible, *varied* task list.

Today the "goal" lives implicitly in `/agent/memory.md`, and `write_todos` is
used opportunistically ("after launch when it helps supervision"). There is no
named loop that says: *here is the goal → decompose into tasks → do a slice →
update the list → decide whether to continue or check in*.

**Do:** add a supervisor-only skill, e.g. `assist/main_skills/pursue-goal/`,
that, when the user states an ongoing goal rather than a one-off task:
1. Records the goal + its boundaries in `/agent/memory.md` (the goal, what "done"
   or "on track" means, and the stopping/feedback rules from R1/R4).
2. Decomposes it into a `write_todos` list — one independently verifiable result
   per TODO, real dependencies noted (reuse the `complex-request` convention).
3. Explicitly prompts for *varied* next-task brainstorming so it does not
   re-suggest the same task run after run (the post calls this out by name).
4. Each run does a slice, updates the todos, writes progress/decisions to
   `/agent/memory.md`, then applies the R1 stopping rule.

**Keep it out of the system prompt.** Per `AGENTS.md`, core orchestration
behavior needs repeated natural-eval evidence before it is dependable — so this
is a *loaded skill* (layer 4), not a core-prompt rule. **Prove:** an eval where a
multi-run goal makes forward progress across wakes, updates its todo list, and
does not stall or repeat tasks.

### R2a — Composing with the existing decomposition skills (`complex-request`, `orchestrate-repeated-work`)
The new skill must not compete with the existing decomposition skills; it composes
on top of them. The distinction: `complex-request` and `orchestrate-repeated-work`
decompose one bounded request *across agents within a single work episode* (fan
out `delegate-agent`s via `start_async_task`, reconcile on completion wakes,
synthesize, done). Their lifecycle closes inside one piece of work. `pursue-goal`
spans *episodes over time* (which slice now, when to act, when to stop, what
carries between wakes). The goal skill is the outer loop that selects a slice and,
when a slice is complex or repetitive, hands execution to the existing skills.

Mechanisms:

1. **Descriptions split on the temporal discriminator.** `pursue-goal`'s
   description keys on ongoing phrasing ("keep X current", "watch over Y", "get me
   to Z over the next few weeks"); the decomposition skills keep keying on
   single-request-with-many-parts. When a request matches both (e.g. "weekly
   vendor contract check into one review"), `pursue-goal` owns the outer frame and
   the decomposition skill owns the slice — the goal skill's body says so
   explicitly so routing is deterministic.
2. **One owner per state.** Durable goal state (the goal, what "on track" means,
   decisions, open questions, next-slice hint) lives in `/agent/memory.md`, owned
   by `pursue-goal`. The live per-episode outcome list lives in `write_todos`,
   owned by whichever decomposition skill executes the slice. The goal skill never
   keeps a second live todo list; the decomposition skills never touch the durable
   goal state — they return their slice result and the goal skill records it.
3. **The goal skill references, not re-derives.** Its body says: if this slice
   matches `complex-request` or `orchestrate-repeated-work`, load and follow that
   skill for the slice — do not restate its launch/reconcile rules here (the
   "one visible path per kind of state" rule). Trivial slices that need no
   delegate are worked inline by the goal skill; the decomposition skills are not
   mandatory per slice.
4. **Completion semantics and child boundaries.** (a) After a slice completes, the
   episode ends in a checkpoint (progress note, goal state updated), never a "goal
   complete" claim — the goal stays open until its stopping rule says otherwise.
   (b) A delegate receives only the bounded slice brief — the goal's
   open-endedness never leaks into the child, which is the self-contained-brief
   contract (`Outcome`, `Inputs`, `Constraints`, `Verify`, `Return`) those skills
   already enforce.

**Worked example.** "Every week, check each vendor's contract and roll them into
one review." `pursue-goal` loads (ongoing), records goal state, and sets the weekly
trigger via the `schedule` skill. Each weekly wake: the goal skill picks the slice
("this week's vendor check") → it matches `orchestrate-repeated-work` → load it →
group the vendors, evidence-only delegates, main synthesizes this week's review →
progress note to goal state → episode done, next wake later. User feedback ("drop
vendor X, they left") → goal-state correction (R4) → next week's slice excludes
them.

**Anti-conflict rules to encode in the body:**
- Big is not ongoing: a large one-shot multi-part request is `complex-request`,
  not a goal.
- A one-shot batch over a list is `orchestrate-repeated-work` alone.
- Never re-decompose a slice that is already running; never poll (the
  decomposition skills already forbid polling).

**Prove (routing):** a panel of natural requests — one-shot multi-part (→
`complex-request` only), one-shot repeated batch (→ `orchestrate-repeated-work`
only), ongoing single-item (→ goal + schedule), ongoing repeated (→ goal outer +
decomposition slice), plus "big but one-shot" distractors that must NOT load
`pursue-goal`. Run it as a matched baseline per the staged-eval rule (N=3 → 3 → 5
→ 10) and assert the decomposition skills' adjacent behavior is preserved when
`pursue-goal` is absent.

### R3 — Let a background goal run ask a clarification question and wait
**Blog tie:** §3 — a background agent needs a way to ask a human and get an
answer back.

Assist's HITL today is *approval* of a concrete effect (send email, egress
grant, browser action) — the user approves/edits/rejects. There is no first-class
"I have a question, here are the options, resume when I answer" proposal type
for a background/scheduled run. That is the genuine new capability in the post.

**Do:** extend the existing approval-card infrastructure with a `clarification`
proposal kind (distinct from effect approvals): the background run parks with a
question (and, where useful, a small set of options), it surfaces through the
existing phone `notify()` + web inbox, and the run resumes on the user's answer
via the same resume path the other cards use. Reuse the card/token/decision
machinery rather than inventing a new transport. This also dovetails with R1: an
unanswered clarification is the reason a later trigger should stay quiet.

**Note:** this is the only recommendation that is real infrastructure (a new
proposal kind + resume wiring). Scope it to the existing card surface and keep the
native (Hermes) path out, consistent with how native egress is currently held.

### R4 — Make feedback→memory compounding explicit
**Blog tie:** §6 — rejecting a blog topic should shape future topic selection, not
just the current draft; memory is what lets feedback compound across runs.

Assist already re-reads `/agent/memory.md` every turn and has `maintain-thread-state`
to checkpoint decisions/progress — but the *loop* "when the user rejects or
corrects me, persist that so future runs inherit it" is implicit.

**Do:** in the goal skill (R2) and `maintain-thread-state`, make the rule
explicit: when the user rejects, edits, or corrects something during a goal run,
write a one-line decision/correction to `/agent/memory.md` (e.g. "topic X rejected
on <date> — do not re-propose") so the next wake reads it. This is the difference
between a goal that improves and one that re-asks. Guidance-only change.

### R5 — Per-goal tool policy (reuse, don't rebuild)
**Blog tie:** §2 — for goals, tool calls are less known, so you need more control
over what an agent can/can't call.

Assist already has the strongest control surface: skill-gated tools, the egress
allowlist with per-thread exact host/port grants, read-only enforcement, and the
sandbox. For goal work the gap is *stating* a tool boundary for a given goal.

**Do:** let the goal brief in `/agent/memory.md` name an allowed-tools / allowed-
destinations policy for that goal, and have the agent self-restrain to it. Back it
with the existing egress grants (a goal that needs a new host still raises the
normal exact-host approval card — no new mechanism). This is mostly documentation
of existing controls plus a self-restraint prompt; measure that the agent actually
respects the stated boundary.

### R6 — A runtime progress/feedback loop (optional, later)
**Blog tie:** §4 — the agent needs to know how it's doing; design goals so
feedback is easy to give; don't blindly optimize a metric.

Assist's evals and the private judged-capture/judge machinery are for development
and calibration, not for the *user* to rate a live goal's progress. A lightweight
version: each goal run appends a short "progress vs. goal" note, and the user can
affirm or correct it (via the R3 clarification surface); the correction feeds R4.
Keep it minimal — a progress line the user can react to, not a dashboard. Defer
until R2/R3/R4 are proven; this is the most likely to become over-engineering if
shipped early.

---

### Suggested sequencing
1. **R1** (stopping rule) — prompt-only, small, immediately improves the existing
   `schedule`/`subscribe-events` + `quiet()` machinery.
2. **R4** (feedback→memory) — prompt-only, small, compounds with R1/R2.
3. **R2** (`pursue-goal` skill) — the core loop; needs an eval panel.
4. **R3** (clarification HITL) — the one real new mechanism; build after R2 shows
   where questions actually arise.
5. **R5** (per-goal tool policy) — mostly documenting existing controls.
6. **R6** (progress/feedback loop) — optional, last.

Each step is guidance-first, proven on a representative eval panel (baseline
N=3 → candidate N=3 → confidence N=5 → final N=10 per the project's staged-eval
rule), and nothing here should go into the system prompt unless it states a
principle with value beyond a single capability.

---

## 4. Full blog post

> The article body below is reproduced from the page (author Harrison Chase).
> Images are referenced by their original CDN URLs.

As agents become more powerful, we're moving from giving them tasks to giving them goals.

Tasks are:

- Write this email
- Draft this blog
- Fix this bug

Goals are:

- **Reduce my coordination burden while keeping important commitments on track**
- Maintain a blog
- Make this product successful

Under the hood, this move from tasks to goals consists of two more practical shifts:

- From specifying an action to specifying an outcome
- From a bounded assignment to an ongoing responsibility

![goal vs task agent](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4bc3faf9d01d8d8dd3ee_goal%20vs%20task%20agent.jpeg)

This is changes what an agent does, what it needs, how its built, and we interact with it.

## The agent chooses the tasks

Previously, the user would specify tasks. Now, the user specifies a goal, and the agent iterates towards that goal by deconstructing into tasks, doing some of them, updating the task list, etc.

![Goal focused agent](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4c9cd95fcbca0837d5b5_Goal%20focused%20agent.jpeg)

The agent needs to be good at brainstorming and selecting different tasks, which requires new context engineering and prompting strategies. It needs to be somewhat creative - and varied - in brainstorming potential tasks (if it keeps on suggesting the same tasks over and over again, that's not great).

***Practical consideration: you need some way to see what tasks the agent is doing (e.g. a to-do list), as well as prompting for the agent to be creative in coming up with tasks***

## Tool calls are way less known

When you give an agent task, you largely have an idea of what tools it will call. When you give it a goal - you don't really know what tools it will call, and you may not even know what tools it will need to call.

E.g.

- If you ask an agent to send an email (task), you know it needs the gmail tool, maybe a calendar tool.
- If you ask an agent to **reduce my coordination burden while keeping important commitments on track** (goal), you don't know when it will need a email, text, search tool, a browser tool, or possibly other tools.

![Task vs goal workflow](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4ce24c1e354fe29d25cb_Task%20vs%20goal%20workflow.jpeg)

***Practical consideration: you need more controls over what tools an agent can or cannot call***

## Inbox to ask questions

When you give an agent a task, you're naturally in the loop. If you don't provide all the information up front, its easy for the agent to ask follow up questions in chat.

When you give an agent a goal - it may be running in the background and need to get clarification on certain points. It needs a way to ask questions to humans - whether through Slack or a dedicated UI.

![Inbox to ask questions](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4d0b4506aba208c3c30d_Inbox%20to%20ask%20questions.jpeg)

***Practical consideration: you need a mechanism for a background agent to ask you for clarification***

## Way to get feedback

The agent needs to know how its doing as it progresses towards a goal. This could either be by checking in on a metric to see if it is improving, or by getting explicit feedback from a human on how its doing. The more closely tied to a metric, the better in some sense (easy for it to track automatically!) but its never wise to blindly optimize a metric, so explicit human feedback may be the best thing. This is why its important to [design goals](https://www.langchain.com/blog/goal-engineering) in a way where its easy to give feedback.

You can have both metrics for the agent to drive towards, and a way to get feedback from users. Define how the agent will assess progress, and give people a way to correct its interpretation of success.

![Metric and HITL](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4d30c513ebf77b71a5d5_Metric%20and%20HITL.jpeg)

***Practical consideration: you need to design goals in a way that is aligned with feedback, and also set up a mechanism where the agent gets feedback***

## Ongoing work needs triggers (and stopping rules)

When you give agents tasks, they run until its completed.

When you give them a goal, they need to work in the background. They are not always running - they wake up at times to do some work. How do they wake up?

Some methods we see:

- Heartbeats every X minutes
- Recurring cron
- One-off cron (reminder)
- Listening to event stream

That being said, it also needs a way to stop or pause or ignore these triggers. If it gets a heartbeat every thirty minutes, and pings you the same question every time - that's a terrible UX.

![Ongoing agent work](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4d78b39ab9ccff902c96_Ongoing%20agent%20work.jpeg)

Ongoing responsibility does not mean the agent should always find something to do. Sometimes doing nothing is the right action.

***Practical consideration: you need crons and event triggers to wake the agent, and prompting for it to decide when to act.***

## Memory lets feedback compound

While memory sounds really cool and has captured the imagination of developers for the past few years, in reality it has not been that needed in a task based world (unless you are doing the same task repeatedly).

With individual tasks, the user can supply the relevant context each time. As you delegate an ongoing responsibility, more of that burden shifts to the agent. It needs to carry forward decisions and feedback without requiring you to repeat them.

E.g: rejecting a blog topic should inform future topic selection, not just the current draft.

![Multi agent memory](https://cdn.prod.website-files.com/65c81e88c254bb0f97633a71/6afb4db73c4228258fef40d4_Multi-agent%20memory.jpeg)

***Practical considerations: The agent needs to retain and use relevant decisions and feedback across runs.***

## In Conclusion

Giving an agent a goal isn't simply giving it a bigger prompt. You're delegating decisions about what work to do and when to do it. Your role shifts from specifying each task to setting direction, defining boundaries, and reviewing progress.
