"""Structural backstop for research URL provenance.

The research searcher (``sub_research.txt.j2``) fabricates canonical URLs it
never saw — building ``casio.com/.../product.<model>/`` and store-search paths
from memory and fetching them (the 2026-06-24 threads.db dead-URL flood; one
thread held 101 GB of such a runaway). Guidance reduces this a lot (a worked
example + read-cap took guessing from ~86 to 0-3 per turn) but cannot guarantee
zero — the small model still slips. This middleware makes a fabricated fetch
*unreachable*: ``read_url`` is refused unless the URL it's given already appears
somewhere earlier in the conversation.

A provenanced URL is one the agent could have *copied* rather than *invented*:
one textually present in a TOOL RESULT or the USER's message — search-result
URLs, links inside a page the agent already fetched (legitimate link-following),
or a URL the user pasted in the question. The model's OWN prior text does NOT
count (else it launders a fabricated URL by writing it into its reasoning, then
fetching it — see ``_seen_urls``). Only a URL that appears in no tool/user
message — a pure fabrication — is rejected. This keeps the guard a coarse,
unambiguous bound (a substring/membership check on tool+user text, not a fuzzy
"looks invented" heuristic), and it does not end the turn: it returns a
corrective tool result so the model retries with a real URL.

Scope: the research SEARCHER and the FACT-CHECKER — the two sub-agents whose
``read_url`` should only ever hit a URL already in context (the searcher's own
search results; the fact-checker's cited URLs, which live in the report
``ToolMessage`` it is handed). NOT the orchestrator: guarding its ``read_url``
risks re-dispatch thrash (a rejected fetch → re-dispatch research → more
searches); its runaway is bounded by ``recursion_limit`` instead.
"""
import logging
import re
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

_READ_TOOL = "read_url"
# URLs as they appear in tool results / text. Stop at whitespace and the quote
# delimiters that bracket a URL in the search output (Python-repr ``'...'`` /
# JSON ``"..."``). We deliberately DO NOT stop at ``)`` / ``]`` / ``}``: those are
# valid URL characters (e.g. Wikipedia ``.../Mercury_(element)``) and the model
# copies the FULL url verbatim as the read_url arg — truncating it here would
# store a different string in the seen-set and wrongly reject the legit fetch.
# Over-including a trailing bracket from prose only ever fails OPEN (the arg has
# no trailing bracket, so it just doesn't match), never over-rejects.
_URL_RE = re.compile(r'https?://[^\s"\'<>]+')
# Trailing sentence punctuation to trim off the ORIGINAL before it's surfaced in
# the correction message — so a URL captured from prose ("…see …/page.") is
# listed as a clean, fetchable "…/page". Excludes brackets ``)]}`` on purpose:
# those can be valid URL characters (Wikipedia ``…/Mercury_(element)``), so
# stripping them would list a broken URL. (Membership uses normalize_url, which
# is stricter; this only cleans the human/model-facing display.)
_DISPLAY_STRIP = ".,;:!?"
# Cap how many available URLs the correction lists — enough to redirect, not so
# many it bloats the context the model must re-read.
_MAX_LISTED = 8
# Trailing chars stripped before comparison — sentence punctuation and brackets
# that cling to a URL captured from prose. Stripped from BOTH the seen-set and
# the checked URL (via normalize_url), so they stay consistent: a search result's
# ``…/Mercury_(element)`` and the model's copied ``…/Mercury_(element)`` both
# reduce to the same key, and a prose ``…/page.`` matches the clean ``…/page``.
_TRAILING = ".,;:!?)]}'\""


def normalize_url(url: str) -> str:
    """Canonical form for provenance comparison: lowercase scheme+host, drop the
    fragment, a trailing slash, and trailing sentence punctuation/brackets.
    Tolerates junk (returns it stripped).

    Single source of truth for "the same URL" — the provenance eval imports this
    so the guard and the eval can't drift on what counts as a match."""
    s = url.strip().rstrip(_TRAILING)
    try:
        p = urlsplit(s)
        if not p.scheme:
            return s.rstrip("/")
        host = (p.hostname or "").lower()
        netloc = host + (f":{p.port}" if p.port else "")
        rebuilt = urlunsplit((p.scheme.lower(), netloc, p.path.rstrip("/"), p.query, ""))
        return rebuilt.rstrip(_TRAILING)
    except ValueError:
        return s.rstrip("/")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _seen_urls(messages: list) -> dict[str, str]:
    """Every URL in a TOOL RESULT or the USER's message: a map from the normalized
    form (the membership key) to the ORIGINAL as it appeared (first seen). The
    normalized keys are the sources the model cannot fabricate; the originals feed
    the correction message so it never surfaces a normalize_url-truncated form
    (e.g. a stripped trailing paren on a Wikipedia URL).

    Scans ``ToolMessage`` content (search results + pages the agent already
    fetched, so legitimate link-following is preserved) and ``HumanMessage``
    content (a URL the user pasted in the question). It deliberately EXCLUDES the
    model's own ``AIMessage`` content: otherwise the model launders a fabricated
    URL by writing it into its reasoning first, then fetching it — observed on
    Qwen3.6 (14 of 24 fetches slipped through this way against the provenance
    eval). A copied-from-search URL is still allowed because it also appears in
    the search ``ToolMessage``; only a URL invented in the model's own text and
    present in no tool/user message is rejected."""
    seen: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, (HumanMessage, ToolMessage)):
            continue
        for raw in _URL_RE.findall(_message_text(m)):
            seen.setdefault(normalize_url(raw), _display_url(raw))
    return seen


def _display_url(raw: str) -> str:
    """The captured URL trimmed for the correction message so it's fetchable:
    drop trailing sentence punctuation, plus a trailing closing bracket ONLY when
    it's UNBALANCED — a prose wrapper like ``(…/page)`` — while a balanced one is
    a valid URL character (Wikipedia ``…/Mercury_(element)``) and is kept."""
    u = raw.rstrip(_DISPLAY_STRIP)
    closers = {")": "(", "]": "[", "}": "{"}
    while u and u[-1] in closers and u.count(u[-1]) > u.count(closers[u[-1]]):
        u = u[:-1].rstrip(_DISPLAY_STRIP)
    return u


def _correction(allowed: dict[str, str]) -> str:
    listed = sorted(allowed.values())[:_MAX_LISTED]
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
