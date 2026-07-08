"""Break a same-URL ``read_url`` RE-READ loop.

The third, orthogonal ``read_url`` bound, alongside ``UrlProvenanceMiddleware`` (refuses
a FABRICATED first fetch — but can't see a re-read: the URL is "seen" after read #1) and
``loop_detection`` (consecutive repeats only). A page's content is unchanged between
fetches, so re-reading the SAME normalized URL past a few times can NEVER return new
information — it is a runaway. The 2026-07-07 peptides incident re-read one PubMed URL
*78 times* under search-unavailable (see docs/2026-07-07-read-url-reread-breaker.org).

On the read that would exceed ``max_reads`` fetches of a URL already read this turn, refuse
with a corrective ToolMessage (status=error) — FINALIZE, don't kill: the model is told to
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
from langchain_core.messages import AIMessage, ToolMessage

from assist.middleware.url_provenance import normalize_url

logger = logging.getLogger(__name__)

_READ_TOOL = "read_url"
_DEFAULT_MAX_READS = 3


def _read_url_target(tool_call: dict) -> str:
    """The normalized URL a read_url tool call targets, or "" if none.
    ``args``-or-``arguments`` mirrors url_provenance (normalized AIMessage.tool_calls use
    ``args``; the raw OpenAI shape uses ``arguments``)."""
    if tool_call.get("name") != _READ_TOOL:
        return ""
    args: Any = tool_call.get("args") or tool_call.get("arguments") or {}
    url = args.get("url", "") if isinstance(args, dict) else ""
    return normalize_url(url) if url else ""


def _prior_read_count(messages: list, target: str) -> int:
    """How many times read_url was already CALLED with the same normalized URL this turn."""
    n = 0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for tc in (getattr(m, "tool_calls", None) or []):
            if _read_url_target(tc) == target:
                n += 1
    return n


class ReadUrlRereadBreaker(AgentMiddleware):
    """Refuse a ``read_url`` once the same URL has been fetched ``max_reads`` times this turn.

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
        handler: Callable[[ToolCallRequest], "ToolMessage"],
    ) -> "ToolMessage":
        target = _read_url_target(request.tool_call)
        if not target:
            return handler(request)

        state = request.state or {}
        messages = state.get("messages", []) if isinstance(state, dict) \
            else getattr(state, "messages", [])
        if _prior_read_count(messages, target) < self._max_reads:
            return handler(request)

        self._intervention_count += 1
        url = (request.tool_call.get("args") or request.tool_call.get("arguments") or {}).get("url", target)
        logger.warning(
            "ReadUrlRereadBreaker: refused read_url(%s) — already read %d times this turn "
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
