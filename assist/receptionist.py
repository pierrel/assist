"""The receptionist — a phone-call router agent (voice-call assistant, P0.5;
docs/2026-07-21-voice-call-tech-design.org §4).

A caller reaches this agent first; it helps them get to the right thread (or start
a new one) and hands off. It is a *vanilla LangChain* agent (``create_agent``),
NOT a deepagent — deliberately, so it physically lacks the filesystem/execute/task
surface and can never read a thread's CONTENTS. It sees only the read-only
``ThreadCatalog`` (topic + state per thread), enforced by construction: a
mechanical test asserts the bound tool set is exactly the three tools below.

The three tools take injected callbacks (the ``notify_tools`` mold — no web
imports): ``on_open(tid)`` binds the session to an existing thread; ``on_new(domain,
first_message)`` creates one AND binds the session to it. Both run synchronously on
the caller's session thread and NEVER raise into the agent loop — a failure comes
back as a spoken directive.
"""
from __future__ import annotations


def _resolve(selector, entries):
    """Resolve an open_thread selector against the current listing. Returns
    ``(entry, ambiguous)``: an exact hit → ``(entry, [])``; no match →
    ``(None, [])``; several substring matches → ``(None, [those matches])`` so the
    tool can ask which one."""
    s = str(selector or "").strip()
    if not s:
        return None, []
    if s.isdigit():                                  # "2" — a number from the list
        i = int(s)
        return (entries[i - 1], []) if 1 <= i <= len(entries) else (None, [])
    for e in entries:                                # an exact thread id
        if e.id == s:
            return e, []
    low = s.lower()                                  # words from the topic
    matches = [e for e in entries if low in e.description.lower()]
    if len(matches) == 1:
        return matches[0], []
    return None, matches                             # 0 → miss; >1 → ambiguous


def _match_domain(name, domains):
    """Match a spoken domain name to the configured domain labels. Returns
    ``(domain, ambiguous)`` mirroring ``_resolve``: an exact match wins outright;
    otherwise the substring matches (either way, to absorb loose phrasing like "the
    garden project thread") — exactly one → ``(that, [])``, none → ``(None, [])``,
    several → ``(None, [those])`` so ``new_thread`` asks WHICH rather than silently
    picking the first (overlapping labels like Home/Homework must not misroute)."""
    n = str(name or "").strip().lower()
    if not n:
        return None, []
    for d in domains:
        if d.lower() == n:
            return d, []
    matches = [d for d in domains if n in d.lower() or d.lower() in n]
    if len(matches) == 1:
        return matches[0], []
    return None, matches


def receptionist_tools(catalog, domains, on_open, on_new) -> list:
    """The three receptionist tools, closing over the read-only ``catalog``, the
    ``domains`` labels, and the injected callbacks. Session-scoped: build a fresh
    set per call — ``_last`` (the listing ``open_thread`` resolves numbers against)
    and ``_bound`` (the thread already connected this session) are per-session
    state."""
    _last: list = []
    _bound: list = []   # size-1 holder of the tid this session is connected to (idempotent open)

    def list_threads() -> str:
        """List the caller's existing conversation threads so you can help them
        choose. Returns a short numbered list of topics and their state. Use this
        ONLY when the caller is unsure which thread they want — if they've already
        told you, just open it."""
        entries = catalog.entries()[:20]
        _last.clear()
        _last.extend(entries)
        if not entries:
            return "There are no threads yet — we can start a new one."
        lines = []
        for i, e in enumerate(entries, 1):
            # urgent and a busy stage are orthogonal (a thread can be urgent AND
            # processing) — show both so the caller doesn't lose either signal.
            marks = (["urgent"] if e.urgent else []) + \
                    ([] if e.status == "ready" else [e.status])
            tag = f" ({', '.join(marks)})" if marks else ""
            lines.append(f"{i}. {e.description}{tag}")
        return "\n".join(lines)

    def open_thread(selector: str) -> str:
        """Connect the caller to an existing thread. ``selector`` can be its number
        from the list or a few words from its topic ('the trip one'). Call this as
        soon as you know which thread the caller wants — you do not need to list
        first if they already told you."""
        # A number refers to the list the caller actually heard (_last, itself
        # capped at 20); a topic or id must reach ANY thread, so resolve those
        # against the FULL catalog — never the capped listing, or older threads
        # become unopenable by name once the caller has listed.
        stripped = str(selector or "").strip()
        entries = (_last or catalog.entries()) if stripped.isdigit() else catalog.entries()
        entry, ambiguous = _resolve(selector, entries)
        if entry is None:
            if ambiguous:
                names = " or ".join(f"the {m.description} one" for m in ambiguous[:3])
                return f"I've got a few that match — did you mean {names}?"
            return ("I couldn't tell which thread you meant. Tell me its number or "
                    "a few words from its topic.")
        if _bound and _bound[-1] == entry.id:
            # Idempotent: the model sometimes fires open_thread twice for one intent —
            # don't bind (and speak the connect) again for the thread we're already on.
            return f"You're already connected to the {entry.description} thread."
        try:
            on_open(entry.id)
        except Exception:
            return "Sorry — I couldn't connect you to that one. Want to try another?"
        _bound[:] = [entry.id]   # only the current binding matters; keep it size-1
        return f"Connecting you to the {entry.description} thread."

    def new_thread(domain: str, first_message: str) -> str:
        """Start a NEW thread. ``domain`` is which of the user's projects it belongs
        to (ask in one short question if they didn't say; if there's only one
        project, just use it). ``first_message`` is what the caller wants to do, in
        their own words. Only call this once you have both."""
        matched, ambiguous = _match_domain(domain, domains)
        if matched is None:
            if ambiguous:
                return ("Which project should this go under — "
                        + " or ".join(ambiguous[:3]) + "?")
            if len(domains) == 1:
                matched = domains[0]
            elif not domains:
                return "There aren't any projects set up to start a thread in."
            else:
                return ("Which project should this go under? The options are "
                        + ", ".join(domains) + ".")
        try:
            on_new(matched, first_message)
        except Exception:
            return "Sorry — I couldn't start that thread. Want to try again?"
        return f"Starting a new thread in {matched} — go ahead."

    return [list_threads, open_thread, new_thread]


def create_receptionist(tools, domains):
    """Build the receptionist as a VANILLA LangChain agent (never a deepagent — a
    deepagent would drag in the filesystem/execute/task surface this router must
    not have; the vanilla loop's tool set is exactly ``tools``).

    ``domains`` (the same labels passed to ``receptionist_tools``) are rendered into
    the system prompt so the router knows the user's projects up front — a real
    receptionist knows the departments — and can NAME them when eliciting which
    project a new thread belongs to, instead of asking blindly.

    The three protective middleware are load-bearing on a phone call, not polish:
    EmptyResponseRecovery (a bare empty/<think>-only AIMessage would end the turn in
    dead air), ModelRetry (ChatOpenAI is built max_retries=0), BadRequestRetry (the
    400 class). The model is built thinking-OFF with per-phase timeouts (a thinking
    router would burn its latency budget on <think> tokens the caller never hears; a
    hung first token is unbounded dead air)."""
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    from assist.agent import _make_retry_middleware
    from assist.middleware.bad_request_retry import BadRequestRetryMiddleware
    from assist.middleware.empty_response_recovery import EmptyResponseRecoveryMiddleware
    from assist.model_manager import select_assistant_model
    from assist.promptable import base_prompt_for

    return create_agent(
        select_assistant_model(0.1),                # enable_thinking=False default + timeouts
        tools,
        system_prompt=base_prompt_for("receptionist_system.md.j2", domains=domains),
        middleware=[_make_retry_middleware(),
                    BadRequestRetryMiddleware(),
                    EmptyResponseRecoveryMiddleware()],
        checkpointer=InMemorySaver(),               # scoped to the one call
    )
