"""Structural backstop for research URL provenance.

The research searcher (``sub_research.txt.j2``) fabricates canonical URLs it
never saw — building ``casio.com/.../product.<model>/`` and store-search paths
from memory and fetching them (the 2026-06-24 threads.db dead-URL flood; one
thread held 101 GB of such a runaway). Guidance reduces this a lot (a worked
example + read-cap took guessing from ~86 to 0-3 per turn) but cannot guarantee
zero — the small model still slips. This middleware makes a fabricated fetch
*unreachable*: ``read_url`` is refused unless the URL it's given already appears
somewhere earlier in the conversation.

"Already appears earlier" — not "came from a search result" — is deliberate: a
provenanced URL is one the agent could have *copied* rather than *invented*. That
set is every URL textually present in a prior message: search-result URLs, a URL
the user put in the question, and links inside a page the agent already fetched
(legitimate link-following). Only a URL that appears NOWHERE prior — a pure
fabrication — is rejected. This keeps the guard a coarse, unambiguous bound (a
substring/membership check on prior text, not a fuzzy "looks invented"
heuristic), and it does not end the turn: it returns a corrective tool result so
the model retries with a real URL.

Scope: the research SEARCHER only (it owns both ``search_internet`` and
``read_url``). NOT the fact-check agent — that subagent re-fetches URLs cited in
a report it was handed, with no search results of its own, so the same check
would reject every (legitimate) fetch.
"""
import logging
import re
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

_READ_TOOL = "read_url"
# URLs as they appear in tool results / text. Stop at whitespace and the
# delimiters that bracket a URL in JSON-ish search output ("..."), markdown, or
# prose so the captured token is the bare URL.
_URL_RE = re.compile(r'https?://[^\s"\'<>)\]}]+')
# Cap how many available URLs the correction lists — enough to redirect, not so
# many it bloats the context the model must re-read.
_MAX_LISTED = 8


def normalize_url(url: str) -> str:
    """Canonical form for provenance comparison: lowercase scheme+host, drop the
    fragment and a single trailing slash. Tolerates junk (returns it stripped).

    Single source of truth for "the same URL" — the provenance eval imports this
    so the guard and the eval can't drift on what counts as a match."""
    from urllib.parse import urlsplit, urlunsplit
    try:
        p = urlsplit(url.strip())
        if not p.scheme:
            return url.strip().rstrip("/")
        host = (p.hostname or "").lower()
        netloc = host + (f":{p.port}" if p.port else "")
        return urlunsplit((p.scheme.lower(), netloc, p.path.rstrip("/"), p.query, ""))
    except ValueError:
        return url.strip().rstrip("/")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _seen_urls(messages: list) -> set[str]:
    """Every URL that textually appears in the message content so far, normalized.

    Scans ``.content`` only — a ``read_url`` call's URL lives in the AIMessage's
    ``tool_calls``, not its content, so the URL under check is never counted as
    its own provenance."""
    seen: set[str] = set()
    for m in messages:
        for raw in _URL_RE.findall(_message_text(m)):
            seen.add(normalize_url(raw))
    return seen


def _correction(allowed: set[str]) -> str:
    listed = sorted(allowed)[:_MAX_LISTED]
    urls = "\n".join(f"- {u}" for u in listed) if listed else "(none yet — run search_internet first)"
    return (
        "That URL was not fetched: it does not appear in any search result, the "
        "question, or a page you already read, so it is a guess and would be a "
        "dead link. Do NOT type URLs from memory. Read one of the URLs your "
        "search returned, or run search_internet to find the page:\n"
        f"{urls}"
    )


class UrlProvenanceMiddleware(AgentMiddleware):
    """Refuse a ``read_url`` whose URL appears nowhere earlier in the turn.

    Corrective, not turn-ending: a rejected call returns an error ToolMessage
    listing the URLs the agent may actually read, so it retries with a real one.
    Stateless across turns except an intervention counter for logging."""

    def __init__(self) -> None:
        super().__init__()
        self.tools = []
        self._intervention_count = 0

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], "ToolMessage | Command"],
    ) -> "ToolMessage | Command":
        tool_call = request.tool_call
        if tool_call.get("name", "") != _READ_TOOL:
            return handler(request)

        args: Any = tool_call.get("args") or tool_call.get("arguments") or {}
        url = args.get("url", "") if isinstance(args, dict) else ""
        if not url:
            return handler(request)

        state = request.state or {}
        messages = state.get("messages", []) if isinstance(state, dict) \
            else getattr(state, "messages", [])
        allowed = _seen_urls(messages)
        if normalize_url(url) in allowed:
            return handler(request)

        self._intervention_count += 1
        logger.warning(
            "UrlProvenanceGuard: rejected read_url(%s) — not seen in prior "
            "messages (intervention #%d, %d allowed urls)",
            url, self._intervention_count, len(allowed),
        )
        return ToolMessage(
            content=_correction(allowed),
            tool_call_id=tool_call.get("id", ""),
            name=_READ_TOOL,
            status="error",
        )
