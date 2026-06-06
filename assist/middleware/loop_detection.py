"""Middleware that detects and breaks *exact-repeat* tool-call loops.

Catches the failure mode where the model gets stuck repeating the same
tool call — either against the same error, or with identical args — e.g.
a ``write_file`` to the same path that keeps returning "Cannot write …
because it already exists", or the same ``read_url(URL)`` issued back to
back hundreds of times.

Detection runs in ``after_model``. When the latest AI message's tool
calls would extend a loop pattern visible in the completed history, those
tool calls are stripped and the AI message content is replaced with a
short terminal summary. The agent loop then ends naturally because the AI
message carries no tool calls.

Two patterns are recognised — both are "you are repeating yourself
verbatim", the only loops worth stopping deterministically:

A. Same *mutating* tool + same normalised error, repeated >=
   ``error_repeat_threshold`` times in a row. Errors are normalised so
   varying paths/IDs/numbers don't hide the repetition. Read-only tools
   (``_READ_ONLY_TOOLS``) are transparent to this walk — a repeated
   read-only error is NOT caught by A (reading the same thing and getting
   the same error is left to B's same-args check), since a read-only call
   between two failing writes is "let me check what's there", not a loop.

B. Same tool + same args, repeated >= ``args_repeat_threshold`` times in
   a row. Catches the model calling the same tool with identical
   arguments back-to-back regardless of result. Applies to read-only tools
   too (e.g. the same ``read_url(URL)`` hundreds of times).

Everything else — distinct-arg exploration, http-failure streaks, sheer
tool volume, sub-agent re-dispatch — is intentionally NOT caught here.
A few extra research hops are normal and far less harmful than yanking a
working agent into a confusing half-finished state; the runaway backstop
for those is the per-agent ``recursion_limit`` (see ``agent.py``) and the
per-call LLM timeout, not this middleware.  (This is a deliberate rollback
of the earlier six-pattern design — see
``docs/2026-06-05-loop-detection-audit.org``.)

Threshold semantics: the counts above are over the *completed* history
(calls whose results are back), and ``after_model`` only strips when the
latest, not-yet-run message would *extend* the pattern.  So a threshold of
N effectively allows up to N completed calls and strips the (N+1)th
attempt — read the thresholds as "max allowed completed calls", not
"intervene the instant the Nth is requested".
"""

import hashlib
import json
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
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

# Read-only tools.  Pattern A walks only the trailing run of *mutating*
# tools (read-only calls are transparent — a `ls`/`read_file` between two
# failing writes is "let me check what's there", not a change of approach).
# Pattern B walks all tools but treats a different-tool read-only event as
# transparent so an interleaved `[write_X, ls, write_X, ls, write_X]` still
# registers as repeated write_X.
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "ls",
    "glob",
    "grep",
    "read_url",
    "search_internet",
})


def _looks_like_error(content: str) -> bool:
    head = content.lstrip().lower()[:120]
    return any(head.startswith(p) for p in _ERROR_PREFIXES)


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
            if tm is not None:
                content = str(tm.content) if tm.content is not None else ""
                is_error = (
                    getattr(tm, "status", None) == "error"
                    or _looks_like_error(content)
                )
                events.append({
                    "tool_name": tc.get("name") or "",
                    "args_sig": _normalise_args(args),
                    "result_content": content,
                    "is_error": is_error,
                    "completed": True,
                })
            else:
                events.append({
                    "tool_name": tc.get("name") or "",
                    "args_sig": _normalise_args(args),
                    "result_content": "",
                    "is_error": False,
                    "completed": False,
                })

    return events[-window:]


def _detect_loop(
    completed_events: list[dict],
    error_repeat_threshold: int,
    args_repeat_threshold: int,
) -> dict | None:
    """Return loop-detection info or ``None`` if no loop.

    Result keys:
      pattern     -- "same-tool-same-error" | "same-tool-same-args"
      reason      -- short human-readable string for logs
      tools       -- set of looping tool names
      run_length  -- length of the trailing run
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
    # page)` ~1000 times under a sub-research-agent.
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

    a_tool, _a_args, a_len = _trailing_run(skip_trailing_read_only=False)
    b_tool, _b_args, b_len = _trailing_run(skip_trailing_read_only=True)
    if b_len > a_len:
        run_tool, run_len = b_tool, b_len
    else:
        run_tool, run_len = a_tool, a_len
    if run_tool and run_len >= args_repeat_threshold:
        return {
            "pattern": "same-tool-same-args",
            "reason": f"same-tool-same-args: {run_tool} x{run_len}",
            "tools": {run_tool},
            "run_length": run_len,
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

    # same-tool-same-args
    return (
        f"I kept making the same {tool_list} call and wasn't getting new "
        "information. I won't repeat it. Could you tell me how you'd "
        "like to proceed?"
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
    """Detect and break *exact-repeat* tool-call loops by stripping them.

    On detection the latest AI message's looping tool calls are removed
    and its content is replaced with a short terminal summary that
    cites the most recent successful artifact (if any). The agent
    loop ends because no tool calls remain to dispatch.

    Only the two exact-repetition patterns are caught (same-tool-same-error
    and same-tool-same-args).  Non-repetition runaways (lots of distinct
    research hops, fetch streaks, sheer volume) are intentionally allowed to
    run to the per-agent ``recursion_limit`` — a few extra hops beats a
    confusing artificial stop.  See the module docstring.

    The middleware is stateless: every check is performed by inspecting
    the message tail, so it composes safely with checkpointing and
    rollback.

    Args:
        window: Number of recent tool-call events to consider.
        error_repeat_threshold: Same-tool / same-normalised-error
            repetitions in a row that constitute a loop. Default 2.
        args_repeat_threshold: Same-tool / same-args repetitions in a
            row that constitute a loop. Default 3.
    """

    def __init__(
        self,
        window: int = 12,
        error_repeat_threshold: int = 2,
        args_repeat_threshold: int = 3,
    ):
        super().__init__()
        self.window = window
        self.error_repeat_threshold = error_repeat_threshold
        self.args_repeat_threshold = args_repeat_threshold
        self.tools = []
        self._intervention_count = 0

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
        )
        if detection is None:
            return None

        looping_tools = detection["tools"]

        # Only act if the latest AI message's tool calls would extend the
        # loop. Otherwise the model may already be breaking out.
        # `or ""` matches _extract_events' convention and keeps the set
        # all-strings, so the sorted() in the log below can't TypeError on a
        # tool_call with a missing/None name.
        last_call_names = {(tc.get("name") or "") for tc in last.tool_calls}
        if last_call_names.isdisjoint(looping_tools):
            logger.info(
                "LoopDetection: pattern matched (%s) but latest tool calls "
                "(%s) do not extend it — letting model continue.",
                detection["reason"],
                sorted(last_call_names),
            )
            return None

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
