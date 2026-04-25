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

Three patterns are recognised:

A. Same tool + same normalised error, repeated >=
   ``error_repeat_threshold`` times in a row. Errors are normalised
   so varying paths/IDs/numbers don't hide the repetition.

B. Same tool + same args, repeated >= ``args_repeat_threshold`` times
   in a row. Catches the model calling the same tool with identical
   arguments back-to-back regardless of result.

C. Same tool + >= ``distinct_args_threshold`` distinct arg sets within
   the last ``distinct_args_window`` tool calls. Catches the
   filename-mutation pattern even when individual error strings
   differ.
"""

import hashlib
import json
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
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


def _extract_events(messages: list, window: int) -> list[dict]:
    """Collect recent (AIMessage tool_call, matching ToolMessage) pairs.

    Each event is ``{tool_name, args_sig, result_content, is_error,
    completed}``. ``completed`` is False for tool calls without a
    matching ToolMessage yet (i.e. the most-recent AI message before
    the tool node has run).
    """
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
    distinct_args_threshold: int,
    distinct_args_window: int,
) -> dict | None:
    """Return loop-detection info or ``None`` if no loop.

    Result keys:
      pattern     -- "same-tool-same-error" | "same-tool-same-args" |
                     "distinct-args-thrash"
      reason      -- short human-readable string for logs
      tools       -- set of looping tool names
      run_length  -- length of the trailing run (or distinct-arg count)
    """

    if not completed_events:
        return None

    # Pattern A: trailing run of same tool + same normalised error.
    run_tool = None
    run_err = None
    run_len = 0
    for e in reversed(completed_events):
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

    # Pattern B: trailing run of same tool + same args.
    run_tool = None
    run_args = None
    run_len = 0
    for e in reversed(completed_events):
        if run_tool is None:
            run_tool = e["tool_name"]
            run_args = e["args_sig"]
            run_len = 1
        elif e["tool_name"] == run_tool and e["args_sig"] == run_args:
            run_len += 1
        else:
            break
    if run_tool and run_len >= args_repeat_threshold:
        return {
            "pattern": "same-tool-same-args",
            "reason": f"same-tool-same-args: {run_tool} x{run_len}",
            "tools": {run_tool},
            "run_length": run_len,
        }

    # Pattern C: distinct-args thrash within recent window.
    recent = completed_events[-distinct_args_window:]
    by_tool: dict[str, set[str]] = {}
    for e in recent:
        by_tool.setdefault(e["tool_name"], set()).add(e["args_sig"])
    for tool_name, sigs in by_tool.items():
        if len(sigs) >= distinct_args_threshold:
            return {
                "pattern": "distinct-args-thrash",
                "reason": (f"distinct-args-thrash: {tool_name} "
                           f"{len(sigs)} distinct in {len(recent)}"),
                "tools": {tool_name},
                "run_length": len(sigs),
            }

    return None


def _last_error_excerpt(
    messages: list, tools: set[str], max_chars: int = 160
) -> str | None:
    """Most recent error content for any tool in ``tools``, trimmed."""
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

    # distinct-args-thrash
    excerpt = _last_error_excerpt(messages, looping_tools)
    excerpt_clause = f' (most recent issue: "{excerpt}")' if excerpt else ""
    return (
        f"I kept calling {tool_list} with different inputs and couldn't "
        f"settle on one{excerpt_clause}. I won't keep trying variations. "
        "Could you tell me how you'd like to proceed?"
    )


def _last_successful_artifact(messages: list) -> str | None:
    """Most recent successful ``write_file``/``edit_file`` path, if any."""
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
        distinct_args_threshold: Distinct arg-sets for a single tool
            within ``distinct_args_window`` that constitute a loop.
            Default 3.
        distinct_args_window: Sliding window for the distinct-args
            check. Default 10.
    """

    def __init__(
        self,
        window: int = 12,
        error_repeat_threshold: int = 2,
        args_repeat_threshold: int = 3,
        distinct_args_threshold: int = 3,
        distinct_args_window: int = 10,
    ):
        super().__init__()
        self.window = window
        self.error_repeat_threshold = error_repeat_threshold
        self.args_repeat_threshold = args_repeat_threshold
        self.distinct_args_threshold = distinct_args_threshold
        self.distinct_args_window = distinct_args_window
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
            distinct_args_threshold=self.distinct_args_threshold,
            distinct_args_window=self.distinct_args_window,
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

        return {"messages": messages[:-1] + [new_last]}
