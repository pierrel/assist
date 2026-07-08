"""Break a same-URL ``read_url`` RE-READ loop.

The third, orthogonal ``read_url`` bound, alongside ``UrlProvenanceMiddleware`` (refuses
a FABRICATED first fetch — but can't see a re-read: the URL is "seen" after read #1) and
``loop_detection`` (consecutive repeats only). A page's content is unchanged between
fetches, so re-reading the SAME normalized URL past a few times can NEVER return new
information — it is a runaway. The 2026-07-07 peptides incident re-read one PubMed URL
*78 times* under search-unavailable (see docs/2026-07-07-read-url-reread-breaker.org).

On the call that follows ``max_reads`` completed fetches of the same URL in this RUN's
history, refuse with a corrective ToolMessage (status=error) — FINALIZE, don't kill: the model is told to
answer from what it has (or say it can't and stop), instead of looping. That's the
volume-cap-destroys-output lesson (a runaway must be nudged to finalize, not terminated
with an empty stub).

Unambiguous signal (a normalized-URL count, reusing ``url_provenance.normalize_url`` so the
two guards agree on "the same URL"), so it stays a coarse guard, not a fuzzy heuristic.
"""
import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from assist.middleware.loop_detection import _extract_events
from assist.middleware.url_provenance import normalize_url

logger = logging.getLogger(__name__)

_READ_TOOL = "read_url"
_DEFAULT_MAX_READS = 3


def _read_url_arg(tool_call: dict) -> str:
    """The raw URL a read_url tool call targets, or "" if none — the one extraction
    path (the incoming request AND each history event go through it; callers normalize
    for counting and reuse the raw form in the refusal text). ``args``-or-``arguments``
    mirrors url_provenance (normalized AIMessage.tool_calls use ``args``; the raw
    OpenAI shape uses ``arguments``)."""
    if tool_call.get("name") != _READ_TOOL:
        return ""
    args: Any = tool_call.get("args") or tool_call.get("arguments") or {}
    return args.get("url", "") if isinstance(args, dict) else ""


def _prior_read_count(messages: list, target: str) -> int:
    """How many COMPLETED reads of the same normalized URL are in this run's history —
    read_url calls that already have their ToolMessage result.

    Completed-only matters: at ``wrap_tool_call`` time the state already contains the
    AIMessage carrying the call being executed, so counting raw calls would include the
    CURRENT call (shifting the threshold by one) and any parallel same-URL siblings in
    that message (refusing them ALL, including the very first fetch — a guard blocking
    legitimate work). A refusal's corrective ToolMessage also pairs, so once the cap is
    hit the count stays at/over it and later retries stay refused.

    The pairing is ``loop_detection._extract_events`` — the SAME call→result loop
    loop_detection and the search-unavailable breaker count with, so all three
    history-counting guards share one implementation (its ``completed`` flag IS the
    completed-only semantics above; it also bounds the count to the current turn)."""
    n = 0
    for e in _extract_events(messages, window=None):
        if not e["completed"]:
            continue
        url = _read_url_arg({"name": e["tool_name"], "args": e["args"]})
        if url and normalize_url(url) == target:
            n += 1
    return n


class ReadUrlRereadBreaker(AgentMiddleware):
    """Refuse a ``read_url`` once the same URL has been fetched ``max_reads`` times this run.

    The count is bounded to the current turn (via ``_extract_events``' turn slice) —
    a no-op on the single-run research/fact-check sub-agents it's wired on, and the
    right behavior if it's ever attached to a longer-lived agent.

    Corrective, not turn-ending (mirrors UrlProvenanceMiddleware): the refused call returns
    an error ToolMessage telling the model to stop re-reading and answer. Stateless across
    turns except an intervention counter for logging."""

    def __init__(self, max_reads: int = _DEFAULT_MAX_READS) -> None:
        super().__init__()
        self.tools = []
        self._max_reads = max_reads
        self._intervention_count = 0

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], "ToolMessage | Command"],
    ) -> "ToolMessage | Command":
        url = _read_url_arg(request.tool_call)
        if not url:
            return handler(request)
        target = normalize_url(url)

        state = request.state or {}
        messages = state.get("messages", []) if isinstance(state, dict) \
            else getattr(state, "messages", [])
        if _prior_read_count(messages, target) < self._max_reads:
            return handler(request)

        self._intervention_count += 1
        logger.warning(
            "ReadUrlRereadBreaker: refused read_url(%s) — already read %d times this run "
            "(intervention #%d)", url, self._max_reads, self._intervention_count)
        return ToolMessage(
            content=(
                f"You have already read {url} {self._max_reads} times in this task. Its "
                f"content does not change between reads, so reading it again returns exactly "
                f"the same thing. Do NOT read it again. Answer from what you already have; if "
                f"the page does not contain what you need, say so plainly and stop — do not "
                f"keep retrying the same URL."),
            tool_call_id=request.tool_call.get("id", ""),
            name=_READ_TOOL,
            status="error",
        )
