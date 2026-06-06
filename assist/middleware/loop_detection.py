"""Middleware that detects and breaks tool-call loops.

Catches the failure mode where the model gets stuck repeating the same
(or near-same) tool call against the same error: e.g. a sequence of
``write_file`` calls to slightly different filenames after the first
one returns "Cannot write to ... because it already exists".

Detection runs in ``after_model``. When the latest AI message's tool
calls would extend a loop pattern visible in the completed history,
those tool calls are stripped and the AI message content is replaced
with a short terminal summary. The agent loop then ends naturally
because the AI message carries no tool calls.

Six patterns are recognised (A-F). A-D are general and on by default;
E and F are opt-in (off unless their threshold is set) and used by the
research flow — see ``agent.py``.

A. Same tool + same normalised error, repeated >=
   ``error_repeat_threshold`` times in a row. Errors are normalised
   so varying paths/IDs/numbers don't hide the repetition.

B. Same tool + same args, repeated >= ``args_repeat_threshold`` times
   in a row. Catches the model calling the same tool with identical
   arguments back-to-back regardless of result.

C. Same tool + >= ``distinct_args_threshold`` distinct arg sets within
   the last ``distinct_args_window`` tool calls, where the tool has
   mutating side effects and at least one of those calls errored.
   Catches the filename-mutation pattern (``write_file foo.py`` →
   error → ``foo2.py`` → error → ``foo_new.py``) even when individual
   error strings differ. Read-only tools (``read_file``, ``ls``,
   ``grep``, etc.) are exempt because distinct-args usage is the
   normal shape of legitimate exploration. Embedder-declared
   ``exploration_tools`` (e.g. emacsos's ``eval_elisp``) are not exempt
   but get a HIGHER breadth threshold
   (``_EXPLORATION_DISTINCT_ARGS_THRESHOLD``); they remain fully subject
   to A and B.

D. >= ``http_failure_threshold`` consecutive tool results whose bodies
   look like an HTTP 4xx/5xx page (bot-detection, rate-limit, captcha),
   regardless of args. Catches a fetch loop across distinct URLs that
   all return error pages — A/B/C miss it because the body is HTML, not
   a Python error, and the args differ.

E. Sheer VOLUME: >= ``volume_threshold`` calls to a single tool in
   ``volume_tools`` within the window, regardless of args or errors
   (counted per-tool). Catches over-use of a *successful* tool — the
   research over-search runaway. Off unless ``volume_threshold`` > 0.

F. Per-subagent ``task`` RE-DISPATCH: the same subagent dispatched >=
   ``subagent_dispatch_threshold`` times within the window. Catches an
   orchestrator re-dispatching the same sub-agent. Off unless
   ``subagent_dispatch_threshold`` > 0.

Threshold semantics (all patterns): the counts above are over the
*completed* history (calls whose results are back), and ``after_model``
only strips when the latest, not-yet-run message would *extend* the
pattern.  So a threshold of N effectively allows up to N completed calls
and strips the (N+1)th attempt — read the thresholds as "max allowed
completed calls", not "intervene the instant the Nth is requested".
"""

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_ERROR_PREFIXES = (
    "error:",
    "cannot write",
    "cannot edit",
    "cannot read",
    "failed to",
    "exception:",
    "traceback",
)

_PATH_RE = re.compile(r"(/[\w./\\-]+)")
_ID_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")

# Tools without mutating side effects. Pattern C (distinct-args thrash) does
# not fire for these because reading multiple distinct files / running
# multiple distinct queries is the normal shape of exploration, not a loop.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "ls",
    "glob",
    "grep",
    "read_url",
    "search_internet",
})

# Pattern C distinct-args threshold for tools an embedder marks as
# "exploration" (e.g. emacsos's `eval_elisp`, which probes a live emacs and
# is the agent's primary inspection tool).  Such tools legitimately fire
# many distinct forms, and the small local model fumbles the API as it goes
# — so a few distinct erroring probes are normal trial-and-error, not yet a
# loop.  Exploration tools get this HIGHER breadth threshold than ordinary
# mutating tools (default 3), so legitimate exploration isn't terminated;
# but they are NOT exempt — a sustained flail of this many distinct erroring
# forms is still caught (faster + with a clearer message than the recursion
# limit).  They ALSO stay fully subject to Patterns A/B (repetition), which
# is why `_mutating_only` below intentionally keeps using `_READ_ONLY_TOOLS`
# only.  Tune against the live exploration shape in the evals.
_EXPLORATION_DISTINCT_ARGS_THRESHOLD = 6


def _looks_like_error(content: str) -> bool:
    head = content.lstrip().lower()[:120]
    return any(head.startswith(p) for p in _ERROR_PREFIXES)


# Patterns that mean "you've gathered ENOUGH", not "you're broken".  When one
# fires the agent has usable state (search results / a subagent's result), so
# ending the turn with a canned stub destroys the synthesis it was about to
# write — the stub message is a false promise.  For these, instead of
# strip-to-END we strip the over-limit tool calls, leave _FINALIZE_NUDGE as the
# assistant turn, and jump back to the model for ONE synthesis turn.  A SECOND
# firing in the same turn (model ignored the nudge and kept going) falls through
# to the normal hard strip-to-END, so the runaway bound still holds.  The
# error/stuck patterns (A–D) are NOT here — for them, ending is correct.
# See docs/2026-06-05-loop-detection-audit.org.
_FINALIZE_PATTERNS = frozenset({"tool-volume"})
_FINALIZE_NUDGE = (
    "Search budget reached: I have gathered enough and will now write the "
    "complete final answer from the results I already have, without searching "
    "or fetching again."
)


def _already_finalized_this_turn(messages: list) -> bool:
    """True if a finalize-nudge was already injected in the current turn.

    Used to fall through to the hard stop on a SECOND firing (the model
    searched again after being nudged), preserving the runaway bound WITHOUT
    instance state — so the middleware stays stateless / rollback-safe.  Bounded
    to the current turn (``_current_turn_slice``) so a nudge from a prior turn
    can't suppress this turn's finalize."""
    for msg in _current_turn_slice(messages):
        if isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else ""
            if _FINALIZE_NUDGE in content:
                return True
    return False


def _normalise_error(content: str) -> str:
    s = content.strip()[:200].lower()
    s = _PATH_RE.sub("<path>", s)
    s = _ID_RE.sub("<id>", s)
    s = _NUMBER_RE.sub("<n>", s)
    return s


def _normalise_args(args: Any) -> str:
    try:
        s = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        s = repr(args)
    if len(s) > 4000:
        s = s[:4000]
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# HTTP-failure markers commonly present in tool RESULT bodies when a
# fetch tool gets a 4xx page (bot-detection, rate-limit, captcha, etc.).
# Sites often return their failure UI as a 200-or-403 HTML body — the
# tool happily returns that body as a "successful" result, so neither
# `_looks_like_error` (Python-style errors) nor Patterns A/B/C (which
# need args/error repetition) catch the loop.  Pattern D (below) uses
# `_looks_like_http_failure` to count consecutive 4xx-shaped responses
# regardless of args, which is what catches the 2026-05-30 casio runaway
# (fetch -> 403, fetch -> 403, … across distinct watch-product URLs).
_HTTP_ERROR_MARKERS: tuple[str, ...] = (
    " 401 ", " 403 ", " 404 ", " 429 ", " 500 ", " 502 ", " 503 ",
    "forbidden", "not found", "unauthorized",
    "rate limit", "rate-limit", "rate limited",
    "access denied", "request blocked", "captcha",
    "too many requests", "blocked by",
)


def _looks_like_http_failure(content: str) -> bool:
    """True if the tool result content suggests an HTTP 4xx/5xx response.

    Many sites return their bot-detection or rate-limit UI as the
    response body even on 4xx — `_looks_like_error` won't catch these
    because the body is HTML, not a Python-style error string.  This
    scans the first ~1KB for HTTP-failure markers; deliberately
    case-insensitive and substring-based to tolerate varied page
    layouts.  False negatives are fine (Patterns A/B/C still apply);
    false positives are the thing to avoid, which is why the markers
    are biased toward strings that don't show up in a normal article
    body (status-code numerals padded with spaces, common 4xx/5xx
    error phrases)."""
    head = content.lower()[:1000]
    return any(m in head for m in _HTTP_ERROR_MARKERS)


def _current_turn_slice(messages: list) -> list:
    """Return messages from the most recent ``HumanMessage`` onward.

    Loop detection is per-turn: tool calls from a completed prior turn
    must not contribute to the current turn's loop assessment, or every
    new user message starts in the looping state described in
    ``docs/loop-misfire.md``.

    If no ``HumanMessage`` is present (synthetic test fixtures, harness
    that hasn't injected one yet) the full list is returned — preserving
    the historical behavior of every caller that doesn't lead with one.

    Note: the slice is bounded by the *most recent* ``HumanMessage``,
    not "the one ``HumanMessage`` per turn".  If two ``HumanMessage``s
    appear back-to-back (user typing twice before the model replies),
    only the later one is the boundary — harmless, since loop detection
    cares about tool-call events and ``HumanMessage`` carries none.
    Sub-agent calls in deepagents go through the ``task`` tool and
    surface in the parent stream as ``AIMessage(tool_calls=...)`` +
    ``ToolMessage`` — they do NOT inject fresh ``HumanMessage``
    instances mid-turn, so the boundary stays one-to-one with user
    turns.  If a future change starts injecting ``HumanMessage``s
    mid-turn (e.g. HITL resume), this slice would over-truncate;
    revisit then.
    """
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return messages[idx:]
    return messages


def _extract_events(messages: list, window: int) -> list[dict]:
    """Collect recent (AIMessage tool_call, matching ToolMessage) pairs.

    Each event is ``{tool_name, args_sig, result_content, is_error,
    completed}``. ``completed`` is False for tool calls without a
    matching ToolMessage yet (i.e. the most-recent AI message before
    the tool node has run).

    Bounded to the current user turn — see ``_current_turn_slice``.
    """
    messages = _current_turn_slice(messages)
    tool_msgs: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            tool_msgs[msg.tool_call_id] = msg

    events: list[dict] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in (getattr(msg, "tool_calls", None) or []):
            tc_id = tc.get("id")
            tm = tool_msgs.get(tc_id)
            args = tc.get("args") or {}
            # For `task` calls, record which subagent was dispatched so
            # Pattern F can cap per-subagent re-dispatch.  deepagents emits
            # `subagent_type`; the small model sometimes uses agent/name.
            subagent = (args.get("subagent_type")
                        or args.get("agent")
                        or args.get("name") or "") if (tc.get("name") == "task") else ""
            if tm is not None:
                content = str(tm.content) if tm.content is not None else ""
                is_error = (
                    getattr(tm, "status", None) == "error"
                    or _looks_like_error(content)
                )
                events.append({
                    "tool_name": tc.get("name") or "",
                    "args_sig": _normalise_args(args),
                    "subagent": subagent,
                    "result_content": content,
                    "is_error": is_error,
                    # Pattern D signal: HTTP-failure-shaped body even when
                    # the tool didn't surface it as a Python-style error.
                    "http_failure": _looks_like_http_failure(content),
                    "completed": True,
                })
            else:
                events.append({
                    "tool_name": tc.get("name") or "",
                    "args_sig": _normalise_args(args),
                    "subagent": subagent,
                    "result_content": "",
                    "is_error": False,
                    "http_failure": False,
                    "completed": False,
                })

    return events[-window:]


def _detect_loop(
    completed_events: list[dict],
    error_repeat_threshold: int,
    args_repeat_threshold: int,
    distinct_args_threshold: int,
    distinct_args_window: int,
    exploration_tools: frozenset[str] = frozenset(),
    exploration_args_threshold: int = _EXPLORATION_DISTINCT_ARGS_THRESHOLD,
    http_failure_threshold: int = 4,
    volume_threshold: int = 0,
    volume_tools: frozenset[str] = frozenset(),
    subagent_dispatch_threshold: int = 0,
) -> dict | None:
    """Return loop-detection info or ``None`` if no loop.

    Result keys:
      pattern     -- "same-tool-same-error" | "same-tool-same-args" |
                     "distinct-args-thrash" | "http-failure-streak" |
                     "tool-volume" | "subagent-redispatch"
      reason      -- short human-readable string for logs
      tools       -- set of looping tool names
      run_length  -- length of the trailing run (or distinct-arg count)
      subagents   -- (subagent-redispatch only) set of over-threshold
                     subagent names; used by after_model to strip only a
                     latest call that re-dispatches one of them
    """

    if not completed_events:
        return None

    # Pattern A walks the trailing run of "mutating" tool events;
    # read-only events (ls, read_file, grep, glob, read_url,
    # search_internet) are TRANSPARENT to the walk — neither extending
    # the run nor breaking it.  Rationale: a read-only call between two
    # failing mutating calls is "let me check what's there" — the agent
    # is observing state but not changing approach.  Treating that as
    # progress-resetting would let "[ls, write_file_err] x N" loops
    # escape detection (which is exactly the 2026-05-16 winged-horse-
    # flag thread).
    #
    # Pattern B walks ALL completed events including read-only ones.
    # Same-tool-same-args repetition is a loop regardless of read-only
    # status — there's no "exploration" justification for hitting the
    # same URL or running the same search query 3 times in a row.
    # The 2026-05-30 runaway issued the same `read_url(F-91W product
    # page)` ~1000 times under a sub-research-agent because the prior
    # mutating-only filter made Pattern B blind to it.
    #
    # NOTE: this intentionally filters on `_READ_ONLY_TOOLS` ONLY — NOT the
    # caller's `exploration_tools`.  Exploration tools (eg. eval_elisp) get a
    # relaxed *Pattern C* breadth threshold below, but they must stay
    # "mutating" here so Pattern A still catches a genuine repetition loop
    # in them (same error repeated).  Folding `exploration_tools` into this
    # filter would disable A for them and remove their only remaining loop
    # protection.
    def _mutating_only(events):
        return [e for e in events if e["tool_name"] not in _READ_ONLY_TOOLS]

    mutating_events = _mutating_only(completed_events)

    # Pattern A: trailing run of same mutating tool + same normalised error.
    run_tool = None
    run_err = None
    run_len = 0
    for e in reversed(mutating_events):
        if not e["is_error"]:
            break
        err_sig = _normalise_error(e["result_content"])
        if run_tool is None:
            run_tool = e["tool_name"]
            run_err = err_sig
            run_len = 1
        elif e["tool_name"] == run_tool and err_sig == run_err:
            run_len += 1
        else:
            break
    if run_tool and run_len >= error_repeat_threshold:
        return {
            "pattern": "same-tool-same-error",
            "reason": f"same-tool-same-error: {run_tool} x{run_len}",
            "tools": {run_tool},
            "run_length": run_len,
        }

    # Pattern B: trailing run of same tool + same args.  Walks ALL
    # completed events.  A non-matching event of a DIFFERENT read-only
    # tool is transparent (skipped) — so `[write_file_X, ls,
    # write_file_X, ls, write_file_X]` still registers as 3 write_file
    # repetitions (the 2026-05-16 winged-horse-flag case).  Same-tool
    # different-args BREAKS the run — `[read_url(A), read_url(B),
    # read_url(A)]` is exploration across URLs, not a loop.  Same-tool
    # same-args extends regardless of read-only category — catches the
    # 2026-05-30 sub-research-agent runaway that issued the same
    # `read_url(F-91W)` ~1000 times in a row.
    # Pattern B is computed TWO ways and we take whichever finds the longer
    # trailing run.  The naive single-pass (anchor on the very last event)
    # misses the case where the trailing event is a read-only call of a
    # DIFFERENT tool than the mutating loop just before it — e.g.
    # ``[write_X, ls, write_X, ls, write_X, ls]``.  Anchored on ``ls`` the
    # ``write_X`` break would terminate the run at length 1.  The second
    # pass advances past trailing read-only events of a different tool to
    # anchor on the latest mutating event, recovering the loop.  Both
    # passes still respect "same-tool same-args extends regardless of
    # read-only category," so the F-91W ``read_url(URL) x1000`` case is
    # also still caught.
    def _trailing_run(skip_trailing_read_only: bool) -> tuple:
        run_tool = None
        run_args = None
        run_len = 0
        for e in reversed(completed_events):
            if run_tool is None:
                if skip_trailing_read_only and e["tool_name"] in _READ_ONLY_TOOLS:
                    continue
                run_tool = e["tool_name"]
                run_args = e["args_sig"]
                run_len = 1
            elif e["tool_name"] == run_tool and e["args_sig"] == run_args:
                run_len += 1
            elif e["tool_name"] != run_tool and e["tool_name"] in _READ_ONLY_TOOLS:
                # Different-tool read-only event: transparent — skip without
                # extending or breaking the run.  Same-tool different-args
                # falls through to the break below.
                continue
            else:
                break
        return run_tool, run_args, run_len

    a_tool, a_args, a_len = _trailing_run(skip_trailing_read_only=False)
    b_tool, b_args, b_len = _trailing_run(skip_trailing_read_only=True)
    if b_len > a_len:
        run_tool, run_args, run_len = b_tool, b_args, b_len
    else:
        run_tool, run_args, run_len = a_tool, a_args, a_len
    if run_tool and run_len >= args_repeat_threshold:
        return {
            "pattern": "same-tool-same-args",
            "reason": f"same-tool-same-args: {run_tool} x{run_len}",
            "tools": {run_tool},
            "run_length": run_len,
        }

    # Pattern C: distinct-args thrash within recent window.
    # Only fires for mutating tools that have errored at least once in the
    # window — pure exploration (3 distinct read_file paths, etc.) is not a
    # loop signal.
    recent = completed_events[-distinct_args_window:]
    by_tool: dict[str, set[str]] = {}
    tool_errored: dict[str, bool] = {}
    for e in recent:
        if e["tool_name"] in _READ_ONLY_TOOLS:
            continue
        by_tool.setdefault(e["tool_name"], set()).add(e["args_sig"])
        if e["is_error"]:
            tool_errored[e["tool_name"]] = True
    for tool_name, sigs in by_tool.items():
        # Exploration tools (eg. eval_elisp) probe many distinct forms
        # legitimately, so they get a higher breadth threshold — but still a
        # finite one, so a sustained flail is caught.
        threshold = (exploration_args_threshold
                     if tool_name in exploration_tools
                     else distinct_args_threshold)
        if len(sigs) >= threshold and tool_errored.get(tool_name):
            return {
                "pattern": "distinct-args-thrash",
                "reason": (f"distinct-args-thrash: {tool_name} "
                           f"{len(sigs)} distinct in {len(recent)}"),
                "tools": {tool_name},
                "run_length": len(sigs),
            }

    # Pattern D: trailing run of consecutive tool calls whose RESULT
    # bodies look like an HTTP failure (4xx/5xx page), regardless of args.
    # Catches the 2026-05-30 casio runaway: the agent emitted ~9,000
    # fetch_url calls across distinct watch-product URLs on casio.com,
    # all returning 403 bot-detection pages.  Patterns A/B/C all
    # short-circuit because the result body is HTML (not a Python error),
    # the args differ across calls (different URLs), and the model never
    # repeats the same arg twice.  `_looks_like_http_failure` picks up
    # the 4xx body markers; consecutive failures across ANY args trigger.
    http_fail_len = 0
    http_fail_tool: str | None = None
    for e in reversed(completed_events):
        if e.get("http_failure"):
            http_fail_len += 1
            # Use the LATEST failing tool as the loop's name (most recent
            # is what the model is currently trying); if the streak spans
            # multiple tool names (e.g. fetch_url + search_internet both
            # 4xx'd), report the latest.
            if http_fail_tool is None:
                http_fail_tool = e["tool_name"]
        else:
            break
    if http_fail_tool and http_fail_len >= http_failure_threshold:
        return {
            "pattern": "http-failure-streak",
            "reason": (f"http-failure-streak: {http_fail_tool} "
                       f"x{http_fail_len} (4xx/5xx-shaped responses)"),
            "tools": {http_fail_tool},
            "run_length": http_fail_len,
        }

    # Pattern F: per-subagent `task` RE-DISPATCH.  The research orchestrator
    # is meant to dispatch each subagent (research / critique / fact-check)
    # at most once; the small model re-dispatches the research-agent
    # repeatedly (prod: 3x for one query), multiplying the inner search
    # volume that Pattern E caps only per-agent.  Off unless
    # subagent_dispatch_threshold > 0.  Keyed BY subagent, so dispatching
    # three DIFFERENT subagents once each never trips it.  Returns the set
    # of over-threshold subagents; after_model strips only a latest call
    # that re-dispatches one of them.
    if subagent_dispatch_threshold > 0:
        sub_counts = Counter(
            e.get("subagent") for e in completed_events
            if e["tool_name"] == "task" and e.get("subagent")
        )
        over = {s for s, n in sub_counts.items()
                if n >= subagent_dispatch_threshold}
        if over:
            return {
                "pattern": "subagent-redispatch",
                "reason": (f"subagent-redispatch: "
                           + ", ".join(f"{s} x{sub_counts[s]}" for s in sorted(over))),
                "tools": {"task"},
                "subagents": over,
                "run_length": max(sub_counts[s] for s in over),
            }

    # Pattern E: sheer VOLUME of one tool within the window, regardless of
    # args or errors.  Off unless volume_threshold > 0 AND the tool is in
    # volume_tools (enabled on the research flow — see _RESEARCH_VOLUME_TOOLS
    # in agent.py, which caps search_internet + read_url — where the small
    # model otherwise searches/reads dozens of times under "conduct thorough
    # research"; observed: 50+ search calls for one trivial query).  This is
    # a HIGHER, looser bound than Pattern C: distinct-query exploration is
    # the *normal* shape of research, but calling one capped tool
    # volume_threshold+ times is a runaway no matter how varied the args.
    #
    # The CALLER decides which tools to cap via volume_tools.  Important:
    # because firing strips the WHOLE latest AI message's tool calls, only
    # cap tools on agents that don't batch a capped tool with a call you
    # must keep (e.g. a write).  The research flow caps read_url on the
    # write-less subagents but NOT on the report-writing orchestrator,
    # precisely to avoid stripping a read+write batch.  Comes last so the
    # args/error patterns (with better terminal messages) win when they
    # also apply.
    if volume_threshold > 0 and volume_tools and completed_events:
        capped = Counter(
            e["tool_name"] for e in completed_events
            if e["tool_name"] in volume_tools
        )
        # Report EVERY over-threshold tool, not just the most frequent.  The
        # cap is per-tool, and `after_model` only intervenes when the latest
        # call extends a tool in `tools` — so if two capped tools both exceed
        # the threshold (e.g. search_internet AND read_url on the research
        # agent) and we named only the busiest, a latest call to the *other*
        # over-threshold tool would slip through uncapped.
        over = {t for t, n in capped.items() if n >= volume_threshold}
        if over:
            return {
                "pattern": "tool-volume",
                "reason": ("tool-volume: "
                           + ", ".join(f"{t} x{capped[t]}" for t in sorted(over))
                           + f" within last {len(completed_events)} calls"),
                "tools": over,
                "run_length": max(capped[t] for t in over),
            }

    return None


def _last_error_excerpt(
    messages: list, tools: set[str], max_chars: int = 160
) -> str | None:
    """Most recent error content for any tool in ``tools``, trimmed.

    Bounded to the current user turn — see ``_current_turn_slice``.
    A stale error from a completed prior turn must not be quoted in
    the current turn's terminal message.
    """
    messages = _current_turn_slice(messages)
    tool_msgs: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            tool_msgs[msg.tool_call_id] = msg

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc.get("name") not in tools:
                continue
            tm = tool_msgs.get(tc.get("id"))
            if tm is None:
                continue
            content = str(tm.content) if tm.content is not None else ""
            if not _looks_like_error(content):
                continue
            excerpt = content.strip().splitlines()[0]
            if len(excerpt) > max_chars:
                excerpt = excerpt[: max_chars - 1].rstrip() + "…"
            return excerpt
    return None


def _compose_terminal_message(detection: dict, messages: list) -> str:
    """User-facing message in the agent's voice for the stripped AI message.

    With a known successful artifact, we close out cleanly. Without one,
    we describe what was being repeated so the model on the next turn
    has a concrete hint about what *not* to retry, and ask the user for
    direction.
    """
    looping_tools = detection["tools"]
    artifact = _last_successful_artifact(messages)

    if artifact:
        return (
            f"I've saved the output to `{artifact}`. "
            "Let me know if you'd like changes or follow-up."
        )

    tool_list = ", ".join(f"`{t}`" for t in sorted(looping_tools))
    pattern = detection["pattern"]

    if pattern == "same-tool-same-error":
        excerpt = _last_error_excerpt(messages, looping_tools)
        excerpt_clause = f' (the error was: "{excerpt}")' if excerpt else ""
        return (
            f"I wasn't able to make progress — I kept hitting the same error "
            f"from {tool_list}{excerpt_clause}. I won't retry that approach. "
            "Could you give me more direction on how to proceed?"
        )

    if pattern == "same-tool-same-args":
        return (
            f"I kept making the same {tool_list} call and wasn't getting new "
            "information. I won't repeat it. Could you tell me how you'd "
            "like to proceed?"
        )

    if pattern == "tool-volume":
        # A graceful "enough" — not an error.  The agent has gathered
        # plenty; tell it to finalize with what it has rather than asking
        # the user for direction (there's nothing wrong, just over-effort).
        return (
            f"I've already used {tool_list} enough times to answer this. "
            "I'll stop gathering and write up what I have now."
        )

    if pattern == "subagent-redispatch":
        # Graceful: the orchestrator already has this sub-agent's result;
        # tell it to use what it has rather than re-dispatching.
        return (
            "I've already gathered that sub-agent's result. I'll finalize "
            "with what I have rather than dispatching it again."
        )

    # distinct-args-thrash
    excerpt = _last_error_excerpt(messages, looping_tools)
    excerpt_clause = f' (most recent issue: "{excerpt}")' if excerpt else ""
    return (
        f"I kept calling {tool_list} with different inputs and couldn't "
        f"settle on one{excerpt_clause}. I won't keep trying variations. "
        "Could you tell me how you'd like to proceed?"
    )


def _last_successful_artifact(messages: list) -> str | None:
    """Most recent successful ``write_file``/``edit_file`` path, if any.

    Bounded to the current user turn — see ``_current_turn_slice``.
    Without this bound, a successful write from a prior turn surfaces
    in the current turn's terminal message, producing the byte-identical
    canned reply documented in ``docs/loop-misfire.md``.
    """
    messages = _current_turn_slice(messages)
    tool_msgs: dict[str, ToolMessage] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            tool_msgs[msg.tool_call_id] = msg

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        for tc in (getattr(msg, "tool_calls", None) or []):
            if tc.get("name") not in ("write_file", "edit_file"):
                continue
            tm = tool_msgs.get(tc.get("id"))
            if tm is None:
                continue
            content = str(tm.content) if tm.content is not None else ""
            if _looks_like_error(content):
                continue
            path = (tc.get("args") or {}).get("file_path")
            if path:
                return path
    return None


class LoopDetectionMiddleware(AgentMiddleware):
    """Detect and break tool-call loops by stripping the offending tool calls.

    On detection the latest AI message's looping tool calls are removed
    and its content is replaced with a short terminal summary that
    cites the most recent successful artifact (if any). The agent
    loop ends because no tool calls remain to dispatch.

    The middleware is stateless: every check is performed by inspecting
    the message tail, so it composes safely with checkpointing and
    rollback.

    Args:
        window: Number of recent tool-call events to consider.
        error_repeat_threshold: Same-tool / same-normalised-error
            repetitions in a row that constitute a loop. Default 2.
        args_repeat_threshold: Same-tool / same-args repetitions in a
            row that constitute a loop. Default 3.
        distinct_args_threshold: Distinct arg-sets for a single
            *mutating* tool within ``distinct_args_window`` that
            constitute a loop, provided at least one of those calls
            errored. Read-only tools (see ``_READ_ONLY_TOOLS``) are
            exempt. Default 3.
        distinct_args_window: Sliding window for the distinct-args
            check. Default 10.
        exploration_tools: Tool names whose distinct-args *breadth* gets a
            higher Pattern-C threshold
            (``_EXPLORATION_DISTINCT_ARGS_THRESHOLD``) because probing many
            distinct forms is their normal
            shape (e.g. emacsos's ``eval_elisp``).  They are NOT exempt from
            Pattern C (a sustained flail is still caught) and remain fully
            subject to Patterns A/B (repetition).  Default ``None`` → empty,
            so the dev/code agent is unaffected; an embedder opts in per
            agent.
        http_failure_threshold: Number of consecutive trailing tool calls
            with an HTTP-failure-shaped body (4xx/5xx markers, see
            ``_looks_like_http_failure``) that constitute a loop, regardless
            of args.  Catches the case where the model iterates through
            distinct URLs that all return 403 / captcha / rate-limit pages
            — Patterns A/B/C all miss this because the result body is HTML,
            not a Python error, and the args differ across calls.  Default 4.
        volume_threshold: Pattern E. Max calls to any single tool in
            ``volume_tools`` within the window before intervening, regardless
            of args/errors. Catches sheer over-use of a successful tool
            (e.g. the research over-search runaway). 0 disables (default), so
            non-research agents are unaffected.
        volume_tools: Tool names the Pattern-E volume cap applies to. Default
            ``None`` → empty (cap inert even if volume_threshold > 0). Cap
            only tools on agents that don't batch a capped tool with a call
            you must keep (firing strips the whole AI message — see Pattern E).
        subagent_dispatch_threshold: Pattern F. Max times the same subagent
            may be dispatched via ``task`` within the window before a further
            re-dispatch of it is stripped. 1 = each subagent at most once.
            0 disables (default). Counts per-subagent, so dispatching several
            *different* subagents once each never trips it.
    """

    def __init__(
        self,
        window: int = 12,
        error_repeat_threshold: int = 2,
        args_repeat_threshold: int = 3,
        distinct_args_threshold: int = 3,
        distinct_args_window: int = 10,
        exploration_tools: frozenset[str] | None = None,
        http_failure_threshold: int = 4,
        volume_threshold: int = 0,
        volume_tools: frozenset[str] | None = None,
        subagent_dispatch_threshold: int = 0,
    ):
        super().__init__()
        self.window = window
        self.error_repeat_threshold = error_repeat_threshold
        self.args_repeat_threshold = args_repeat_threshold
        self.distinct_args_threshold = distinct_args_threshold
        self.distinct_args_window = distinct_args_window
        # Normalise to frozenset so a caller passing a mutable set still
        # yields an immutable attribute (matches the type annotation).
        self.exploration_tools = frozenset(exploration_tools or ())
        self.http_failure_threshold = http_failure_threshold
        # Pattern E volume cap.  0 disables it (default) so non-research
        # agents are unaffected; the research flow opts in.  volume_tools
        # scopes which tools it caps (the research flow caps search_internet
        # + read_url on its subagents).  See Pattern E in _detect_loop.
        self.volume_threshold = volume_threshold
        self.volume_tools = frozenset(volume_tools or ())
        # Pattern F per-subagent re-dispatch cap.  0 disables it (default);
        # the research orchestrator opts in.  See Pattern F in _detect_loop.
        self.subagent_dispatch_threshold = subagent_dispatch_threshold
        self.tools = []
        self._intervention_count = 0

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        if not getattr(last, "tool_calls", None):
            return None

        events = _extract_events(messages, window=self.window)
        completed = [e for e in events if e["completed"]]

        detection = _detect_loop(
            completed,
            error_repeat_threshold=self.error_repeat_threshold,
            args_repeat_threshold=self.args_repeat_threshold,
            distinct_args_threshold=self.distinct_args_threshold,
            distinct_args_window=self.distinct_args_window,
            exploration_tools=self.exploration_tools,
            http_failure_threshold=self.http_failure_threshold,
            volume_threshold=self.volume_threshold,
            volume_tools=self.volume_tools,
            subagent_dispatch_threshold=self.subagent_dispatch_threshold,
        )
        if detection is None:
            return None

        looping_tools = detection["tools"]

        # Only act if the latest AI message's tool calls would extend the
        # loop. Otherwise the model may already be breaking out.
        last_call_names = {tc.get("name") for tc in last.tool_calls}
        if last_call_names.isdisjoint(looping_tools):
            logger.info(
                "LoopDetection: pattern matched (%s) but latest tool calls "
                "(%s) do not extend it — letting model continue.",
                detection["reason"],
                sorted(last_call_names),
            )
            return None

        # Pattern F is subagent-specific: only intervene if the latest
        # message RE-dispatches an already-used subagent.  A `task` call to
        # a fresh subagent (the normal research -> critique -> fact-check
        # progression) must pass through untouched.
        if detection["pattern"] == "subagent-redispatch":
            latest_subagents = {
                (tc.get("args") or {}).get("subagent_type")
                or (tc.get("args") or {}).get("agent")
                or (tc.get("args") or {}).get("name")
                for tc in last.tool_calls if tc.get("name") == "task"
            }
            if latest_subagents.isdisjoint(detection["subagents"]):
                logger.info(
                    "LoopDetection: subagent-redispatch matched (%s) but "
                    "latest dispatches a fresh subagent (%s) — continuing.",
                    detection["reason"], sorted(s for s in latest_subagents if s),
                )
                return None

        # ── "Enough"-family finalize path (E tool-volume; A–D fall through) ──
        # The agent has usable state; don't kill the turn with a stub (that
        # destroys the synthesis it was about to write).  Strip the over-limit
        # tool calls, leave a finalize nudge as the assistant turn, and jump
        # back to the model for ONE synthesis turn.  A SECOND firing this turn
        # (model ignored the nudge) is excluded by _already_finalized_this_turn
        # and falls through to the hard stop below — the runaway bound holds.
        if (detection["pattern"] in _FINALIZE_PATTERNS
                and not _already_finalized_this_turn(messages)):
            self._intervention_count += 1
            if detection["pattern"] == "tool-volume":
                logger.warning(
                    "LoopDetection: tool-volume cap fired (%s).  This cap "
                    "should rarely trigger — investigate an upstream cause "
                    "(prompt drift, poor/empty tool results, or a loop the "
                    "other patterns missed).", detection["reason"],
                )
            logger.warning(
                "LoopDetection: finalize-nudge #%d — pattern=%s tools=%s "
                "run_length=%d stripped_calls=%d; jumping to model for one "
                "synthesis turn (hard-stops if it keeps going).",
                self._intervention_count, detection["pattern"],
                sorted(looping_tools), detection["run_length"],
                len(last.tool_calls),
            )
            new_last = last.model_copy() if hasattr(last, "model_copy") else last.copy()
            new_last.tool_calls = []
            new_last.content = _FINALIZE_NUDGE
            if hasattr(new_last, "additional_kwargs"):
                ak = dict(getattr(new_last, "additional_kwargs", {}) or {})
                if ak.get("tool_calls"):
                    ak["tool_calls"] = []
                new_last.additional_kwargs = ak
            # jump_to:"model" re-enters the model node for the synthesis turn
            # (the after_model→model edge is wired by the can_jump_to decorator
            # above); jump_to is an EphemeralValue, auto-cleared after the step.
            return {"messages": [new_last], "jump_to": "model"}

        artifact = _last_successful_artifact(messages)
        terminal_content = _compose_terminal_message(detection, messages)
        self._intervention_count += 1

        preview = terminal_content.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:159] + "…"

        logger.warning(
            "LoopDetection: intervention #%d — pattern=%s tools=%s "
            "run_length=%d stripped_calls=%d artifact=%s terminal=%r",
            self._intervention_count,
            detection["pattern"],
            sorted(looping_tools),
            detection["run_length"],
            len(last.tool_calls),
            artifact or "<none>",
            preview,
        )

        # The volume cap is a deterministic BACKSTOP, not the primary bound on
        # research effort — the 2026-06-03 ablation showed the dominant levers
        # are the orchestrator delegating search and the focused-research
        # prompt, with this cap only holding the margin.  So if it actually
        # fires, the likely real cause is upstream: a prompt regression, a
        # search tool returning poor/empty results (so the model keeps
        # retrying), or a loop the args/error patterns missed.  Flag it loudly
        # so a firing prompts an investigation rather than being silently
        # absorbed as "working as intended".
        if detection["pattern"] == "tool-volume":
            logger.warning(
                "LoopDetection: tool-volume backstop fired (%s).  This cap "
                "should rarely trigger — investigate an upstream cause for the "
                "over-use (prompt drift, poor/empty tool results, or a loop the "
                "other patterns missed), don't rely on this cap as the bound.",
                detection["reason"],
            )

        new_last = last.model_copy() if hasattr(last, "model_copy") else last.copy()
        new_last.tool_calls = []
        new_last.content = terminal_content

        # Strip any raw OpenAI-format tool calls so the checkpointer sees
        # consistent state.
        if hasattr(new_last, "additional_kwargs"):
            ak = dict(getattr(new_last, "additional_kwargs", {}) or {})
            if ak.get("tool_calls"):
                ak["tool_calls"] = []
            new_last.additional_kwargs = ak

        # Return ONLY the modified last message, not the whole list.  The
        # `messages` channel reducer (`add_messages`) replaces by `.id`, and
        # `new_last` keeps the model's id (model_copy preserves it), so this
        # replaces the last AIMessage in place.  Returning `messages[:-1]`
        # too would re-append every id-less HumanMessage/ToolMessage (the
        # checkpointer deserializes them with id=None) as duplicates — see
        # docs/2026-06-05-middleware-message-duplication.org.
        return {"messages": [new_last]}
