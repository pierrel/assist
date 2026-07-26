"""Structural backstop for research URL provenance.

The research searcher (``sub_research.txt.j2``) fabricates canonical URLs it
never saw — building ``casio.com/.../product.<model>/`` and store-search paths
from memory and fetching them (the 2026-06-24 threads.db dead-URL flood; one
thread held 101 GB of such a runaway). Guidance reduces this a lot (a worked
example + read-cap took guessing from ~86 to 0-3 per turn) but cannot guarantee
zero — the small model still slips. This middleware makes a fabricated fetch
*unreachable*: ``read_url`` is refused unless the URL is provenanced by a trusted
message or an admission-frozen delegate seed.

A provenanced URL is one the agent could have *copied* rather than *invented*:
one textually present in a SEARCH RESULT, a non-offloaded ``read_file``/``grep``
result outside delegate mode, or the USER's message. A delegate is the exception: its HumanMessage is
the main model's brief, so the web host supplies only brief-form URLs canonically
matched to owner-authored Runs already accepted when the child Run is admitted;
the brief may remove owner userinfo but cannot add or change it.
Five channels are deliberately NOT provenance sources, because
their content is attacker-derivable: the model's OWN prior text (else it launders a
fabricated URL by writing it into its reasoning, then fetching it — see
``_seen_urls``); ``read_url`` PAGE CONTENT (the untrusted channel — an injected page
can't name an exfil URL and have a follow-up read of it pass, a read-becomes-write);
``execute`` (shell) OUTPUT (arbitrary bytes — a cat'd download, a large log the agent
re-reads from its real-fs offload — never a URL source); and a ``read_file``/``grep``
of OFFLOADED untrusted content, written under
the ``…/large_tool_results/untrusted-{data,page}-`` family (in state, or
``/tmp/large_tool_results/…`` for a real-fs offload) — the SAME content laundered
through a file; and async task-management results, whose descriptions and child
results are model-authored data. Read-url offloads use the narrower
``untrusted-page-`` marker: they remain excluded as authority but can supply only
same-host link evidence. Shell and child results use the generic marker and cannot.
Delegate graphs exclude a sixth channel: their synchronous ``task`` specialist
results, because the delegate authored those briefs and could ask a specialist to
echo an invented URL. The legacy main retains trusted synchronous ``task`` results so
research-specialist citations can reach its navigation tool; the V2 research
orchestrator itself has no ``read_url``.

SAME-HOST NAVIGATION (2026-07-21): a normalized destination that appeared in
fetched-page content is fetchable IF its host is already trusted (a user URL or search
result was on that host). A credentialed fetch additionally requires that exact
userinfo from a trusted URL on the same origin (scheme, host, and effective port);
page content can prove a path, never credentials. These conditions let ``read_url`` navigate
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
untrusted namespace) then ``read_file`` the copy — can self-launder; gated by
behavioral injection-resistance, not this guard.

Scope: every ``read_url``-bearing agent — the research SEARCHER and FACT-CHECKER
sub-agents, and the main/delegate navigation ``read_url`` (added 2026-07-21 for the
explore-website flow, whose trusted channel is the user's URLs plus same-host page
links). Each should only ever fetch a URL already trusted in its context: the
searcher's own search results; the fact-checker's cited URLs (which reach it via the
report ``read_file``/initial message it is handed); the main agent's user-supplied
URL and pages on that same host. Delegate graphs use admission-frozen,
brief-intersected user URL seeds and ignore model-authored briefs as authority.
The V2 research orchestrator holds no
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
from langgraph.config import get_config
from langgraph.types import Command

from assist.middleware.loop_detection import _messages_from_state
from assist.middleware.tool_result_to_file import (
    UNTRUSTED_OFFLOAD_MARK,
    UNTRUSTED_PAGE_OFFLOAD_MARK,
)

logger = logging.getLogger(__name__)

_READ_TOOL = "read_url"
_SYNC_SUBAGENT_TOOL = "task"
_ASYNC_TASK_TOOLS = {
    "start_async_task",
    "check_async_task",
    "update_async_task",
    "cancel_async_task",
    "list_async_tasks",
}
# Per-run trusted URL seeds for a delegate. The web composition root freezes the
# intersection of its brief and already-accepted owner Runs; the model cannot write
# the durable Run field or RunnableConfig.
# Delegate HumanMessages are model-authored briefs, so they are deliberately not
# provenance sources.
DELEGATE_USER_URLS_KEY = "assist_delegate_user_urls"
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
# File-read tools whose result can surface OFFLOADED untrusted content: a large read_url,
# execute, or child task result is written to a …/large_tool_results/untrusted-*
# namespace and the model greps/
# read_files it back (ToolResultToFileMiddleware). That read is still the untrusted content,
# laundered through a file — so URLs in it must stay untrusted.
_FILE_READ_TOOLS = {"read_file", "grep"}
# UNTRUSTED offloads use disjoint .../large_tool_results/untrusted-data-* and
# .../large_tool_results/untrusted-page-* namespaces.
# by ToolResultToFileMiddleware, which sets the untrusted-ness AT WRITE TIME (where it knows
# the tool). We key on that exact fragment (imported, so writer + guard can't drift): a bare
# prose mention of "large_tool_results" can't match it, and a normal (future-trusted) offload
# at .../large_tool_results/<id> is NOT excluded — no mixing of trusted/untrusted.
_OFFLOAD_MARK = UNTRUSTED_OFFLOAD_MARK
_PAGE_OFFLOAD_MARK = UNTRUSTED_PAGE_OFFLOAD_MARK
# The LOCATION args of read_file/grep (NOT the grep `pattern`, which is a search term):
# a marker here means the read TARGETS offloaded content.
_OFFLOAD_PATH_KEYS = ("path", "file_path", "glob")


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url.strip().rstrip(_TRAILING)).hostname or "").lower()
    except ValueError:
        return ""


def _origin_of(url: str) -> tuple[str, str, int | None]:
    """Lowercase HTTP(S) origin with default ports made explicit."""
    try:
        parsed = urlsplit(url.strip().rstrip(_TRAILING))
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return "", "", None
    if port is None:
        port = {"http": 80, "https": 443}.get(scheme)
    return scheme, host, port


def normalize_url(url: str) -> str:
    """Canonical form for provenance comparison: lowercase scheme+host; drop
    userinfo, the fragment, a trailing slash, and trailing punctuation/brackets.
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


def url_userinfo(url: str) -> str | None:
    """Return exact raw URL userinfo, or ``None`` when the authority has none."""
    try:
        netloc = urlsplit(url.strip().rstrip(_TRAILING)).netloc
    except ValueError:
        return None
    return netloc.rpartition("@")[0] if "@" in netloc else None


def _userinfo_allowed(requested: str, provenanced: str) -> bool:
    """A fetch may drop provenanced credentials, never add or change them."""
    requested_userinfo = url_userinfo(requested)
    return (requested_userinfo is None
            or requested_userinfo == url_userinfo(provenanced))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def urls_in_text(text: str) -> tuple[str, ...]:
    """Return clean HTTP(S) URLs literally present in ``text``, in order."""
    return tuple(_display_url(raw) for raw in _URL_RE.findall(text))


def _is_offload_path(value: Any, marker: str) -> bool:
    path = str(value)
    return (path.startswith(marker)
            or path.startswith(f"/{marker}")
            or path.startswith(f"/tmp/{marker}"))


def _reads_offloaded(tool_call: dict, content: str) -> bool:
    """True if a ``read_file``/``grep`` (whose originating call IS known) touched
    OFFLOADED untrusted content (read_url page, execute output, or child output) —
    the same untrusted content laundered through a file. Two ways it shows up: the read's TARGET PATH contains an untrusted namespace
    (checked on the LOCATION args only — path/file_path/glob, not the grep ``pattern``),
    OR the result names an offloaded file (grep's default ``files_with_matches`` lists
    matching paths, so a grep-all that hit the offload dir shows it). Over-matches only
    make the model re-search — never trust more."""
    args = tool_call.get("args") or {}
    if any(_is_offload_path(args.get(k, ""), marker)
           for k in _OFFLOAD_PATH_KEYS
           for marker in (_OFFLOAD_MARK, _PAGE_OFFLOAD_MARK)):
        return True
    return any(marker in (content or "")
               for marker in (_OFFLOAD_MARK, _PAGE_OFFLOAD_MARK))


def _reads_page_offload(tool_call: dict) -> bool:
    """True only for a reread of content originally returned by ``read_url``."""
    args = tool_call.get("args") or {}
    return any(_is_offload_path(args.get(k, ""), _PAGE_OFFLOAD_MARK)
               for k in _OFFLOAD_PATH_KEYS)


def _is_untrusted_result(
        m: ToolMessage,
        calls_by_id: dict,
        *,
        trust_task_results: bool = True,
) -> bool:
    """A ToolMessage whose content must NOT provenance URLs — the untrusted channels:
    read_url page content directly, ``execute`` (shell) output, async task-management
    output, or a delegate's synchronous specialist and file-read output when
    task-result trust is disabled. Other roles also exclude a ``read_file``/``grep``
    of OFFLOADED untrusted content (read_url page or execute/child output). Fails
    CLOSED: a file read whose originating ``tool_call`` is missing from
    the message list (e.g. summarization dropped it) is treated as untrusted — we can't
    verify its target wasn't the offload dir, so we don't provenance its URLs."""
    name = getattr(m, "name", None)
    if name in {_READ_TOOL, _SHELL_TOOL, *_ASYNC_TASK_TOOLS}:
        return True
    if name == _SYNC_SUBAGENT_TOOL and not trust_task_results:
        return True
    if name in _FILE_READ_TOOLS:                 # a file read — verify its target
        if not trust_task_results:
            # Delegate specialists save reports in the same filesystem namespace as
            # owner files. Their origin is not recoverable from a later file read, so
            # delegate file results cannot mint URL authority. User URL seeds and
            # page-marked same-host navigation remain available separately.
            return True
        tc = calls_by_id.get(getattr(m, "tool_call_id", None))
        if tc is None:                           # unknown target -> fail CLOSED
            return True
        return _reads_offloaded(tc, _message_text(m))
    return False


def _is_page_content(m: ToolMessage, calls_by_id: dict) -> bool:
    """Content allowed to contribute same-host navigation paths.

    Direct ``read_url`` output qualifies. A ``read_file``/``grep`` qualifies only
    when its target has the page-specific offload marker written by the ``read_url``
    middleware. Shell and child-task results never qualify. This classifier is separate from
    ``_is_untrusted_result``: `_seen_urls` uses that broader predicate to exclude trust
    sources, while `_page_content_urls` uses only this predicate for navigation paths."""
    name = getattr(m, "name", None)
    if name == _READ_TOOL:
        return True
    if name in _FILE_READ_TOOLS:
        tc = calls_by_id.get(getattr(m, "tool_call_id", None))
        return tc is not None and _reads_page_offload(tc)
    return False


def _record_url(urls: dict[str, list[str]], raw: str) -> None:
    """Retain every credential variant for one normalized URL in encounter order."""
    variants = urls.setdefault(normalize_url(raw), [])
    if raw not in variants:
        variants.append(raw)


def _seen_urls(messages: list, *, trust_human_messages: bool = True,
               trust_task_results: bool = True,
               ) -> dict[str, list[str]]:
    """Every URL in a TRUSTED tool result or the USER's message: a map from the
    normalized form (the membership key) to every ORIGINAL credential variant seen.
    The normalized keys are the sources the model cannot fabricate; the originals
    feed the correction message so it never surfaces a normalize_url-truncated form
    (e.g. a stripped trailing paren on a Wikipedia URL).

    Normally scans ``HumanMessage`` content (a URL the user pasted) and TRUSTED ``ToolMessage``
    content (search results, non-offloaded ``read_file``/``grep`` results). Five channels
    are always EXCLUDED because their content is attacker-derivable: (1) the model's own
    ``AIMessage`` — else it launders a fabricated URL by writing it into its reasoning
    first, then fetching it (observed on Qwen3.6, 14 of 24 fetches slipped through this
    way); (2) ``read_url`` results — arbitrary, possibly attacker-controlled page
    content; (3) ``execute`` (shell) output — arbitrary bytes (a cat'd download, a log the
    agent re-reads from its real-fs offload), never a URL source; (4) async
    task-management results — model-authored briefs and child output; and (5) a
    ``read_file``/``grep`` of OFFLOADED untrusted content (read_url page,
    execute output, or child result)
    under an ``…/large_tool_results/untrusted-{data,page}-`` namespace — the SAME content laundered through
    a file (ToolResultToFileMiddleware writes it there and the model greps it). So a
    URL appearing only inside fetched-page content — direct or offloaded — never
    becomes trusted authority. A real same-host page link is separately navigable;
    a new host requires another trusted source such as search. Delegate mode also excludes its synchronous specialist results,
    sets ``trust_human_messages=False``, and adds only
    admission-frozen, brief-intersected owner URL seeds later. RESIDUAL: a model that FOLLOWS a multi-step injection to
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
    seen: dict[str, list[str]] = {}
    for m in messages:
        if not isinstance(m, (HumanMessage, ToolMessage)):
            continue
        if isinstance(m, HumanMessage) and not trust_human_messages:
            continue
        if isinstance(m, ToolMessage) and _is_untrusted_result(
                m, calls_by_id, trust_task_results=trust_task_results):
            continue
        for url in urls_in_text(_message_text(m)):
            _record_url(seen, url)
    return seen


def _page_content_urls(messages: list) -> dict[str, list[str]]:
    """Normalized-to-original URL variants from page-compatible content: direct
    ``read_url`` output or a page-marked read_url offload reread (see
    ``_is_page_content``). NOT the exact inverse of ``_seen_urls``: shell and child-task
    output, direct or offloaded, are in NEITHER set.
    These are the links a page actually surfaced; a fabricated path never appears
    here. Credentialless same-host members are navigable (see wrap_tool_call);
    credentialed requests also need exact trusted same-origin userinfo. This admits
    following a real link on an already-trusted site while still blocking a
    fabricated same-host path (the dead-URL flood), page-introduced credentials,
    and an exfil URL the model invents (or one printed into shell output on a
    trusted host)."""
    calls_by_id: dict[str, dict] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            cid = tc.get("id")
            if cid:
                calls_by_id[cid] = tc
    urls: dict[str, list[str]] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and _is_page_content(m, calls_by_id):
            for url in urls_in_text(_message_text(m)):
                _record_url(urls, url)
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


def _correction(allowed: dict[str, list[str]]) -> str:
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
    listed = sorted(variants[0] for variants in allowed.values())[:_MAX_LISTED]
    urls = "\n".join(f"- {u}" for u in listed)
    return (
        "That URL was not fetched: it does not appear in any search result, the "
        "question, or a trusted source in your context, so it is a guess and would be a "
        "dead link. Do NOT type URLs from memory. Read one of the URLs already in "
        "context instead:\n"
        f"{urls}"
    )


class UrlProvenanceMiddleware(AgentMiddleware):
    """Refuse ``read_url`` outside trusted messages or configured delegate seeds.

    Corrective, not turn-ending: a rejected call returns an error ToolMessage
    listing the URLs the agent may actually read, so it retries with a real one.
    ``trust_human_messages=False`` and ``trust_task_results=False`` form the delegate
    boundary: model-authored task briefs, synchronous specialist results, and later
    file reads are ignored; only admission-frozen, brief-intersected owner URL seeds
    supplied from the durable child Run can replace them. Page-marked same-host links
    remain navigation evidence but never cross-host authority.
    Stateless across turns except an intervention counter for logging."""

    def __init__(self, *, trust_human_messages: bool = True,
                 trust_task_results: bool = True) -> None:
        super().__init__()
        self.tools = []
        self._trust_human_messages = trust_human_messages
        self._trust_task_results = trust_task_results
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
        allowed = _seen_urls(
            messages,
            trust_human_messages=self._trust_human_messages,
            trust_task_results=self._trust_task_results,
        )
        configurable = (get_config() or {}).get("configurable") or {}
        seeds = configurable.get(DELEGATE_USER_URLS_KEY, ())
        if isinstance(seeds, (list, tuple)):
            for raw in seeds:
                if isinstance(raw, str):
                    _record_url(allowed, _display_url(raw))
        nurl = normalize_url(url)
        provenanced = allowed.get(nurl, ())
        if any(_userinfo_allowed(url, raw) for raw in provenanced):
            return handler(request)
        # Same-host navigation: a path the fetched page ACTUALLY surfaced, on a
        # host already trusted, is fetchable without credentials. Credentialed
        # navigation additionally needs trusted same-origin userinfo. This lets
        # read_url navigate WITHIN a site the user pointed at (find a manual,
        # follow Support → the download). All conditions are load-bearing:
        #  - in page content (not just host-matched): a fabricated same-host
        #    path never appeared on any page, so the dead-URL flood the guard
        #    exists to stop stays blocked — the model can only follow links a
        #    page really emitted, not invent canonical URLs.
        #  - same host (not just page-present): a page linking cross-host
        #    (169.254.169.254, an internal service, evil.example/leak?d=secret)
        #    has no matching trusted host, so SSRF + secret-bearing exfil stay
        #    blocked — the jump to a NEW host from page content is refused.
        #  - trusted same-origin userinfo for a credentialed request: page
        #    content proves only the destination path and cannot introduce
        #    Basic Auth credentials or move them to HTTP / another port.
        host = _host_of(url)
        page_urls = _page_content_urls(messages).get(nurl, ())
        trusted_host_urls = [
            raw for variants in allowed.values() for raw in variants
            if _host_of(raw) == host]
        requested_userinfo = url_userinfo(url)
        userinfo_allowed = (requested_userinfo is None or any(
            _origin_of(url) == _origin_of(raw) and _userinfo_allowed(url, raw)
            for raw in trusted_host_urls))
        if host and page_urls and trusted_host_urls and userinfo_allowed:
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
