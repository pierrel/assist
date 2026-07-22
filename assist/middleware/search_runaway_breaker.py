"""Break a ``search_internet`` RUNAWAY — repeated / empty-hammering / high-volume
searching against a HEALTHY backend.

The fourth history-counting guard on the research searcher, orthogonal to the other three:
``loop_detection`` (consecutive exact-repeats only), ``SearchUnavailableBreaker`` (keys on
the backend-DOWN constant), and ``ReadUrlRereadBreaker`` (read_url, not search). The gap
this fills is the 2026-07-18 coding-toy runaway: the searcher issued THREE obscure
product-spec queries ~80 times each, ALL returning healthy-empty ``"[]"`` — cycling A/B/C
(so consecutive-repeat detection never fired) and returning real (empty) results (so the
unavailable breaker never fired), 346 searches in 33 min. See docs (search-runaway).

Four bounds, all keyed on the tool's OWN emitted constant (``_SEARCH_EMPTY_GUIDANCE``) and
call counts — never on result CONTENT, so injected search snippets can't forge an empty or
reset a counter. Decision order in ``wrap_tool_call`` (first match wins), counting only
COMPLETED ``search_internet`` events in the current turn (via ``loop_detection._extract_events``):

  1. total completed searches >= TOTAL_CAP        -> refuse-finalize (stop, answer now)
  2. trailing run of empties >= CONSECUTIVE_EMPTY  -> refuse-finalize (nothing is landing)
  3. same normalized query completed >= SAME_QUERY -> refuse-steer (vary the query)
  4. this query already returned empty this turn   -> short-circuit: serve the cached empty
                                                       guidance WITHOUT hitting the backend
  else -> real backend call.

Return-status convention (consistent with the sibling breakers): a *refused* call returns a
``status="error"`` ToolMessage (FINALIZE-not-kill — the model is told to answer from what it
has, not terminated with an empty stub); the *short-circuit* (4) returns a NON-error
ToolMessage byte-identical to a real healthy-empty, so history-counting and the model's
interpretation don't diverge from a backend empty. Both always set ``tool_call_id`` so the
call→result pairing (and thus the counters, and the OpenAI tool-call contract) hold.

Scope and honest residuals:
  * "Turn" == one searcher-subagent dispatch (``_extract_events`` bounds to the current
    turn); a no-op on the single-run research agents this is wired on. Cross-dispatch
    search budget is NOT bounded here — the searcher's own recursion_limit (~25 supersteps,
    NOT the orchestrator's 300 — deepagents doesn't forward it) is that backstop.
  * Completed-only counting can't see the in-flight siblings of ONE concurrent batch, so
    the first batch of identical queries is un-counted; the effective ceiling is
    ``TOTAL_CAP + one batch width`` (observed batch width ~3), not a hard 50. From the 2nd
    superstep on, the first batch's empties are committed, so decision 4 short-circuits the
    repeats. (A strict in-batch bound is stateless-feasible — count same-query siblings
    preceding this call's id in the current AIMessage — but is a deliberate v1 deferral:
    the observed width is single-digit.)
  * TOTAL_CAP is per-dispatch. Legit research is ~11 searches/dispatch (a broad comparison
    is spread across many dispatches, ~11 each), so 50 has ~4.5x headroom and can't clip
    real breadth. TOTAL_CAP is also the ONLY guard on Tavily spend for a distinct-query
    loop against a THROTTLED backend (results come back real, dodging both empty and
    unavailable detection).
"""
import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from assist.middleware.loop_detection import _extract_events, _messages_from_state
from assist.tools import _SEARCH_EMPTY_GUIDANCE

logger = logging.getLogger(__name__)

_SEARCH_TOOL = "search_internet"

# Hardcoded like ReadUrlRereadBreaker's _DEFAULT_MAX_READS (repeat-cap sibling) — these are
# coarse real bounds, not per-deploy dials. SAME_QUERY_CAP: identical completed queries
# allowed before the next is refused. TOTAL_CAP: completed searches/dispatch. CONSECUTIVE_
# EMPTY_CAP: trailing all-empty run (catches distinct-reword hammering that dodges the
# same-query cap; a run resets on any non-empty result, so TOTAL_CAP is its real backstop).
_SAME_QUERY_CAP = 3
_TOTAL_CAP = 50
_CONSECUTIVE_EMPTY_CAP = 5

# Shared by the two finalize refusals (total-cap and consecutive-empty) — one string, not two
# near-identical ones (the copy-paste-breeds-divergence lesson).
_FINALIZE_STEER = (
    "Stop searching now and write your answer from what you have already gathered. "
    "Further searching this turn will not help — if you could not find something after "
    "the searches so far, say plainly that it is not available and answer from what you have."
)


def _search_query(tool_call: dict) -> str:
    """The raw ``query`` a search_internet call targets, or "" if none. ``args``-or-
    ``arguments`` mirrors the sibling extractors (normalized AIMessage.tool_calls use
    ``args``; the raw OpenAI shape uses ``arguments``). ``max_results`` is deliberately NOT
    part of query identity, so varying it can't evade the same-query cap."""
    if tool_call.get("name") != _SEARCH_TOOL:
        return ""
    args: Any = tool_call.get("args") or tool_call.get("arguments") or {}
    return args.get("query", "") if isinstance(args, dict) else ""


def _normalize_query(query) -> str:
    """Query identity for the same-query cap: lowercased, stripped, internal whitespace
    collapsed. An UNAMBIGUOUS normalizer only — no fuzzy/token-set matching, which would
    risk clipping legitimately-distinct queries (coarse real bounds over ambiguous signals).
    Type-safe: ANY non-string ``query`` (``None``, a ``list``, an ``int`` — the small model
    emits malformed tool args, cf. the ``glob(path="/")`` runaway) normalizes to ``""``, so
    malformed calls group together and the caps still bound them instead of the normalizer
    crashing on ``.lower()``. Local to this module, mirroring ReadUrlRereadBreaker's local
    ``_read_url_arg``. American spelling (matches ``normalize_url``)."""
    q = query if isinstance(query, str) else ""
    return " ".join(q.lower().split())


def _completed_searches(messages: list) -> list[dict]:
    """The completed ``search_internet`` events in this turn's history (each has its
    ToolMessage result), via the shared ``loop_detection._extract_events`` pairing —
    completed-only so the CURRENT in-flight call (and its concurrent siblings) isn't counted."""
    return [e for e in _extract_events(messages, window=None)
            if e["completed"] and e["tool_name"] == _SEARCH_TOOL]


def _trailing_empty_run(events: list[dict]) -> int:
    """Length of the trailing run of completed searches whose result is the empty-guidance
    constant. Distinct queries count (that's what catches reword-hammering); the run resets
    on any non-empty result (so TOTAL_CAP is the real backstop against reset-interleaving)."""
    n = 0
    for e in reversed(events):
        if e["result_content"] == _SEARCH_EMPTY_GUIDANCE:
            n += 1
        else:
            break
    return n


class SearchRunawayBreakerMiddleware(AgentMiddleware):
    """Bound a ``search_internet`` runaway (repetition, empty-hammering, volume) on a healthy
    backend. Stateless across turns except an intervention counter for logging; all bounds
    read the current turn's committed history. Named for what it does — the flat TOTAL_CAP
    covers non-looping runaways too, so "runaway" not "loop"."""

    def __init__(self) -> None:
        super().__init__()
        self.tools = []
        self._intervention_count = 0

    def _refuse(self, request: ToolCallRequest, content: str, reason: str) -> ToolMessage:
        self._intervention_count += 1
        logger.warning("SearchRunawayBreaker: refused search_internet — %s (intervention #%d)",
                       reason, self._intervention_count)
        return ToolMessage(content=content,
                           tool_call_id=request.tool_call.get("id", ""),
                           name=_SEARCH_TOOL,
                           status="error")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], "ToolMessage | Command"],
    ) -> "ToolMessage | Command":
        # Gate on the tool NAME, not on a truthy query — a malformed search_internet(query="")
        # or (query=None) must still be COUNTED (else it bypasses every bound, keyed on the
        # empty query; the small model emits malformed args). Only a genuinely different tool
        # passes straight through.
        if request.tool_call.get("name") != _SEARCH_TOOL:
            return handler(request)

        query = _search_query(request.tool_call)
        events = _completed_searches(_messages_from_state(request))
        target = _normalize_query(query)
        same = [e for e in events if _normalize_query(_search_query(
            {"name": e["tool_name"], "args": e["args"]})) == target]

        # 1. hard volume ceiling (per dispatch)
        if len(events) >= _TOTAL_CAP:
            return self._refuse(request, _FINALIZE_STEER,
                                f"{len(events)} searches this turn >= total cap {_TOTAL_CAP}")
        # 2. nothing is landing — a trailing run of empties (distinct queries allowed)
        if _trailing_empty_run(events) >= _CONSECUTIVE_EMPTY_CAP:
            return self._refuse(request, _FINALIZE_STEER,
                                f">= {_CONSECUTIVE_EMPTY_CAP} consecutive empty searches")
        # 3. this exact query is being repeated (any position; loop_detection catches only
        #    the CONSECUTIVE case, so this covers the A/B/C-cycling shape it misses)
        if len(same) >= _SAME_QUERY_CAP:
            return self._refuse(
                request,
                (f"You have already run this exact search {_SAME_QUERY_CAP} times this "
                 f"turn — re-running an identical query returns the same thing. Try a "
                 f"distinctly different query, or answer from what you have."),
                f"same query x{len(same)} >= cap {_SAME_QUERY_CAP}")
        # 4. this query already came back empty — serve the cached empty, no backend hit
        if any(e["result_content"] == _SEARCH_EMPTY_GUIDANCE for e in same):
            self._intervention_count += 1
            logger.warning("SearchRunawayBreaker: short-circuit search_internet(%r) — already "
                           "empty this turn (intervention #%d)", query, self._intervention_count)
            return ToolMessage(content=_SEARCH_EMPTY_GUIDANCE,
                               tool_call_id=request.tool_call.get("id", ""),
                               name=_SEARCH_TOOL)
        return handler(request)
