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
one textually present in a SEARCH RESULT, a non-offloaded ``read_file``/``grep``
result, or the USER's message. Four channels are deliberately NOT provenance sources, because
their content is attacker-derivable: the model's OWN prior text (else it launders a
fabricated URL by writing it into its reasoning, then fetching it — see
``_seen_urls``); ``read_url`` PAGE CONTENT (the untrusted channel — an injected page
can't name an exfil URL and have a follow-up read of it pass, a read-becomes-write);
``execute`` (shell) OUTPUT (arbitrary bytes — a cat'd download, a large log the agent
re-reads from its real-fs offload — never a URL source); and a ``read_file``/``grep``
of OFFLOADED untrusted content (read_url page, execute output), written under
``…/large_tool_results/untrusted-`` (``/large_tool_results/…`` in state, or
``/tmp/large_tool_results/…`` for a real-fs offload) — the SAME content laundered
through a file.

SAME-HOST NAVIGATION (2026-07-21): a URL that literally appeared in fetched-page
content is fetchable IF its host is already trusted (a user URL or search result was
on that host). BOTH conditions are required — this is what lets ``read_url`` navigate
WITHIN a site the user pointed at (the "explore a website, find and download a file"
flow) without re-searching every next page, while keeping the two attacks the guard
exists for blocked: (1) a FABRICATED same-host path (the searcher inventing
``casio.com/products/<model>`` — the dead-URL flood) never appears on any page, so
the page-content requirement refuses it; (2) a CROSS-HOST jump from page content
(SSRF to 169.254.169.254 / an internal service, or a secret-bearing exfil URL to
``evil.example/leak?d=…``) has no matching trusted host, so the same-host requirement
refuses it — and the attacker can't statically author a link carrying the victim's
runtime secret anyway. Only *more pages a trusted site actually links to* become
reachable. A URL that is neither trusted-sourced nor a same-host page link is still
rejected. This keeps the guard a coarse,
unambiguous bound (a substring/membership check on trusted text, not a fuzzy "looks
invented/malicious" heuristic), and it does not end the turn: it returns a corrective
tool result so the model retries with a real URL. RESIDUAL: a model that FOLLOWS a
multi-step injection to ``write_file`` an arbitrary URL then ``read_file`` it — or to
``execute``-copy the real-fs untrusted offload into a workspace path (shedding the
``untrusted-`` marker) then ``read_file`` the copy — can self-launder; gated by
behavioral injection-resistance, not this guard. A narrower same-class residual: a
same-host URL in a LARGE ``execute`` log gets offloaded to ``untrusted-`` and, read
back via ``read_file``/``grep``, counts as page content (the ``untrusted-`` marker
can't tell a read_url offload from an execute one) — so it is navigable if its host is
already trusted. Direct shell output is closed by construction (``_is_page_content``);
this multi-step offload-then-file-read path is not, and shares the same behavioral gate.

Scope: every ``read_url``-bearing agent — the research SEARCHER and FACT-CHECKER
sub-agents, and the MAIN agent's navigation ``read_url`` (added 2026-07-21 for the
explore-website flow, whose trusted channel is the user's URLs plus same-host page
links). Each should only ever fetch a URL already trusted in its context: the
searcher's own search results; the fact-checker's cited URLs (which reach it via the
report ``read_file``/initial message it is handed); the main agent's user-supplied
URL and pages on that same host. The V2 research orchestrator holds no
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

from assist.middleware.loop_detection import _messages_from_state
from assist.middleware.tool_result_to_file import UNTRUSTED_OFFLOAD_MARK

logger = logging.getLogger(__name__)

_READ_TOOL = "read_url"
# Shell output is an arbitrary, attacker-influenceable channel (a downloaded file
# cat'd, a large `execute` log the agent re-reads from its real-fs offload) — it is
# NOT a URL-provenance source. It was never a documented trusted channel; excluding
# it by construction means moving the `execute` offload onto the shell-readable fs
# (docs/2026-07-22-offload-to-real-fs.org) can't launder URLs into the trusted set.
_SHELL_TOOL = "execute"
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
# File-read tools whose result can surface OFFLOADED untrusted content: a large read_url/
# execute result is written to …/large_tool_results/untrusted-<id> and the model greps/
# read_files it back (ToolResultToFileMiddleware). That read is still the untrusted content,
# laundered through a file — so URLs in it must stay untrusted.
_FILE_READ_TOOLS = {"read_file", "grep"}
# UNTRUSTED offloads (read_url/execute) are written at .../large_tool_results/untrusted-<id>
# by ToolResultToFileMiddleware, which sets the untrusted-ness AT WRITE TIME (where it knows
# the tool). We key on that exact fragment (imported, so writer + guard can't drift): a bare
# prose mention of "large_tool_results" can't match it, and a normal (future-trusted) offload
# at .../large_tool_results/<id> is NOT excluded — no mixing of trusted/untrusted.
_OFFLOAD_MARK = UNTRUSTED_OFFLOAD_MARK
# The LOCATION args of read_file/grep (NOT the grep `pattern`, which is a search term):
# a marker here means the read TARGETS offloaded content.
_OFFLOAD_PATH_KEYS = ("path", "file_path", "glob")


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url.strip().rstrip(_TRAILING)).hostname or "").lower()
    except ValueError:
        return ""


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
    OFFLOADED untrusted content (read_url page, execute output) — the same untrusted page content laundered through a
    file. Two ways it shows up: the read's TARGET PATH contains ``large_tool_results/untrusted-``
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
    read_url page content directly, ``execute`` (shell) output, or a ``read_file``/
    ``grep`` of OFFLOADED untrusted content (read_url page OR execute output — both land
    under ``…/large_tool_results/untrusted-``). Fails CLOSED: a file read whose originating ``tool_call`` is missing from
    the message list (e.g. summarization dropped it) is treated as untrusted — we can't
    verify its target wasn't the offload dir, so we don't provenance its URLs."""
    name = getattr(m, "name", None)
    if name in (_READ_TOOL, _SHELL_TOOL):        # direct untrusted output: read_url page / shell
        return True
    if name in _FILE_READ_TOOLS:                 # a file read — verify its target
        tc = calls_by_id.get(getattr(m, "tool_call_id", None))
        if tc is None:                           # unknown target -> fail CLOSED
            return True
        return _reads_offloaded(tc, _message_text(m))
    return False


def _is_page_content(m: ToolMessage, calls_by_id: dict) -> bool:
    """Genuine FETCHED-PAGE content — the subset of untrusted results whose URLs a real
    page actually surfaced, so a same-host member is NAVIGABLE (see _page_content_urls /
    wrap_tool_call). That is direct ``read_url`` and a ``read_file``/``grep`` of an
    offloaded page — but NOT ``execute`` (shell) output. This split from
    ``_is_untrusted_result`` is load-bearing: that predicate has OPPOSITE polarity in its
    two callers — ``_seen_urls`` EXCLUDES it (untrusted → not a provenance source),
    ``_page_content_urls`` INCLUDES it (page content → navigable if same-host) — and shell
    output must be excluded by BOTH. Untrusted it is; a fetched page it is not, so a URL
    printed by ``execute`` must not become navigable just because its host is trusted."""
    return (_is_untrusted_result(m, calls_by_id)
            and getattr(m, "name", None) != _SHELL_TOOL)


def _seen_urls(messages: list) -> dict[str, str]:
    """Every URL in a TRUSTED tool result or the USER's message: a map from the
    normalized form (the membership key) to the ORIGINAL as it appeared (first seen).
    The normalized keys are the sources the model cannot fabricate; the originals
    feed the correction message so it never surfaces a normalize_url-truncated form
    (e.g. a stripped trailing paren on a Wikipedia URL).

    Scans ``HumanMessage`` content (a URL the user pasted) and TRUSTED ``ToolMessage``
    content (search results, non-offloaded ``read_file``/``grep`` results). Four channels
    are EXCLUDED because their content is attacker-derivable: (1) the model's own
    ``AIMessage`` — else it launders a fabricated URL by writing it into its reasoning
    first, then fetching it (observed on Qwen3.6, 14 of 24 fetches slipped through this
    way); (2) ``read_url`` results — arbitrary, possibly attacker-controlled page
    content; (3) ``execute`` (shell) output — arbitrary bytes (a cat'd download, a log the
    agent re-reads from its real-fs offload), never a URL source; and (4) a
    ``read_file``/``grep`` of OFFLOADED untrusted content (read_url page, execute output)
    under ``…/large_tool_results/untrusted-`` — the SAME content laundered through
    a file (ToolResultToFileMiddleware writes it there and the model greps it). So a
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


def _page_content_urls(messages: list) -> set[str]:
    """Normalized URLs that literally appeared in FETCHED-PAGE content — the
    untrusted read_url channel, direct or offloaded (see ``_is_page_content``).
    NOT the exact inverse of ``_seen_urls``: ``execute`` (shell) output is in
    NEITHER set — untrusted, so not trusted, but also not a page, so not navigable.
    These are the links a page actually surfaced; a fabricated path never appears
    here. Same-host members are navigable (see wrap_tool_call) — that admits
    following a real link on an already-trusted site while still blocking a
    fabricated same-host path (the dead-URL flood) and an exfil URL the model
    invents (or one printed into shell output on a trusted host)."""
    calls_by_id: dict[str, dict] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            cid = tc.get("id")
            if cid:
                calls_by_id[cid] = tc
    urls: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and _is_page_content(m, calls_by_id):
            for raw in _URL_RE.findall(_message_text(m)):
                urls.add(normalize_url(raw))
    return urls


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

        messages = _messages_from_state(request)
        allowed = _seen_urls(messages)
        if normalize_url(url) in allowed:
            return handler(request)
        # Same-host navigation: a link the fetched page ACTUALLY surfaced, on a
        # host already trusted, is fetchable — this is what lets read_url
        # navigate WITHIN a site the user pointed at (find a manual, follow
        # Support → the download). BOTH conditions are load-bearing:
        #  - in page content (not just host-matched): a fabricated same-host
        #    path never appeared on any page, so the dead-URL flood the guard
        #    exists to stop stays blocked — the model can only follow links a
        #    page really emitted, not invent canonical URLs.
        #  - same host (not just page-present): a page linking cross-host
        #    (169.254.169.254, an internal service, evil.example/leak?d=secret)
        #    has no matching trusted host, so SSRF + secret-bearing exfil stay
        #    blocked — the jump to a NEW host from page content is refused.
        nurl = normalize_url(url)
        host = _host_of(url)
        if (host and host in {_host_of(u) for u in allowed}
                and nurl in _page_content_urls(messages)):
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
