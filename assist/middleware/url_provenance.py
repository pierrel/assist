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
one textually present in a SEARCH RESULT, a non-offloaded ``read_file`` report, or
the USER's message. Three channels are deliberately NOT provenance sources, because
their content is attacker-derivable: the model's OWN prior text (else it launders a
fabricated URL by writing it into its reasoning, then fetching it — see
``_seen_urls``); ``read_url`` PAGE CONTENT (the untrusted channel — an injected page
can't name an exfil URL and have a follow-up read of it pass, a read-becomes-write);
and a ``read_file``/``grep`` of OFFLOADED read_url content (under
/large_tool_results/) — the SAME page content laundered through a file. So a URL
sourced only from fetched-page content, direct or offloaded, cannot be fetched; to
reach a new page the agent re-searches. A URL sourced no other way — a pure
fabrication, or one planted in a page — is rejected. This keeps the guard a coarse,
unambiguous bound (a substring/membership check on trusted text, not a fuzzy "looks
invented/malicious" heuristic), and it does not end the turn: it returns a corrective
tool result so the model retries with a real URL. RESIDUAL: a model that FOLLOWS a
multi-step injection to ``write_file`` an arbitrary URL then ``read_file`` it can
self-launder — gated by behavioral injection-resistance, not this guard.

Scope: the two ``read_url``-bearing research agents — the SEARCHER and the
FACT-CHECKER. Each should only ever fetch a URL already trusted in its context: the
searcher's own search results; the fact-checker's cited URLs (which reach it via the
report ``read_file``/initial message it is handed). The V2 orchestrator holds no
``read_url`` (it delegates all fetching), so the guard doesn't run there — an earlier
version guarded it after thread 20260708090812 fabricated URLs under
search-unavailable and 404-looped ~90 min. Guidance now tells the orchestrator to STOP
(not re-dispatch) when it has no sources, so removing it there is safe.
See docs/2026-07-08-research-reread-runaway.org.
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
# File-read tools whose result can surface OFFLOADED read_url page content: a large
# read_url result is written to /large_tool_results/<id> and the model greps/read_files
# it back (ToolResultToFileMiddleware). That read is still read_url page content — the
# untrusted channel, laundered through a file — so URLs in it must stay untrusted.
_FILE_READ_TOOLS = {"read_file", "grep"}
_OFFLOAD_MARK = "large_tool_results"   # the /large_tool_results/ offload path prefix
# The LOCATION args of read_file/grep (NOT the grep `pattern`, which is a search term):
# a marker here means the read TARGETS offloaded content.
_OFFLOAD_PATH_KEYS = ("path", "file_path", "glob")


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


def _reads_offloaded(tool_call: dict, content: str) -> bool:
    """True if a ``read_file``/``grep`` (whose originating call IS known) touched
    OFFLOADED read_url content — the same untrusted page content laundered through a
    file. Two ways it shows up: the read's TARGET PATH is under /large_tool_results/
    (checked on the LOCATION args only — path/file_path/glob, not the grep ``pattern``),
    OR the result names an offloaded file (grep's default ``files_with_matches`` lists
    matching paths, so a grep-all that hit the offload dir shows it). Over-matches only
    make the model re-search — never trust more."""
    args = tool_call.get("args") or {}
    if any(_OFFLOAD_MARK in str(args.get(k, "")) for k in _OFFLOAD_PATH_KEYS):
        return True
    return _OFFLOAD_MARK in (content or "")


def _is_untrusted_result(m: ToolMessage, calls_by_id: dict) -> bool:
    """A ToolMessage whose content must NOT provenance URLs — the untrusted channels:
    read_url page content directly, or a ``read_file``/``grep`` of OFFLOADED read_url
    content. Fails CLOSED: a file read whose originating ``tool_call`` is missing from
    the message list (e.g. summarization dropped it) is treated as untrusted — we can't
    verify its target wasn't the offload dir, so we don't provenance its URLs."""
    name = getattr(m, "name", None)
    if name == _READ_TOOL:                       # direct read_url page content
        return True
    if name in _FILE_READ_TOOLS:                 # a file read — verify its target
        tc = calls_by_id.get(getattr(m, "tool_call_id", None))
        if tc is None:                           # unknown target -> fail CLOSED
            return True
        return _reads_offloaded(tc, _message_text(m))
    return False


def _seen_urls(messages: list) -> dict[str, str]:
    """Every URL in a TRUSTED tool result or the USER's message: a map from the
    normalized form (the membership key) to the ORIGINAL as it appeared (first seen).
    The normalized keys are the sources the model cannot fabricate; the originals
    feed the correction message so it never surfaces a normalize_url-truncated form
    (e.g. a stripped trailing paren on a Wikipedia URL).

    Scans ``HumanMessage`` content (a URL the user pasted) and TRUSTED ``ToolMessage``
    content (search results, non-offloaded ``read_file`` reports). Three channels are
    EXCLUDED because their content is attacker-derivable: (1) the model's own
    ``AIMessage`` — else it launders a fabricated URL by writing it into its reasoning
    first, then fetching it (observed on Qwen3.6, 14 of 24 fetches slipped through this
    way); (2) ``read_url`` results — arbitrary, possibly attacker-controlled page
    content; and (3) a ``read_file``/``grep`` of OFFLOADED read_url content (under
    /large_tool_results/) — the SAME page content laundered through a file
    (ToolResultToFileMiddleware writes big pages there and the model greps them). So a
    URL appearing only inside fetched-page content — direct or offloaded — never
    becomes trusted; to reach a new page the agent re-searches (the searcher prompt
    requires that). RESIDUAL: a model that FOLLOWS a multi-step injection to
    ``write_file`` an arbitrary URL then ``read_file`` it can still self-launder —
    that path is gated by the model's behavioral injection-resistance, not this guard."""
    # Map tool_call_id -> the call, so a file read can be traced to its TARGET path
    # (to tell an offloaded-read_url read from a legitimate report read).
    calls_by_id: dict[str, dict] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            cid = tc.get("id")
            if cid:
                calls_by_id[cid] = tc
    seen: dict[str, str] = {}
    for m in messages:
        if not isinstance(m, (HumanMessage, ToolMessage)):
            continue
        if isinstance(m, ToolMessage) and _is_untrusted_result(m, calls_by_id):
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
    # No sourced URLs in context yet. Shared by both read_url agents: the searcher HAS
    # search_internet (so "search first" is the right nudge if it simply hasn't searched),
    # while the fact-checker does NOT (an empty list there means search already came back
    # empty/unavailable, so stop). Cover both without inspecting tool availability:
    # search-if-you-can, otherwise stop — never keep guessing.
    if not allowed:
        return (
            "That URL was a guess: it appears in no search result, the question, or any "
            "trusted source in your context, so it would be a dead link. You have no "
            "sourced URLs to read. Do NOT type URLs from memory or keep trying different "
            "guesses. If you can run search_internet, do that first to find real URLs; if "
            "search has already come back empty or unavailable, report that you could not "
            "find reliable sources for this and stop."
        )
    listed = sorted(allowed.values())[:_MAX_LISTED]
    urls = "\n".join(f"- {u}" for u in listed)
    return (
        "That URL was not fetched: it does not appear in any search result, the "
        "question, or a trusted source in your context, so it is a guess and would be a "
        "dead link. Do NOT type URLs from memory. Read one of the URLs already in "
        "context instead:\n"
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
