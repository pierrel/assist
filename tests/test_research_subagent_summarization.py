"""The research subagents must summarize EARLY.

deepagents auto-installs a SummarizationMiddleware on every dict-spec
subagent that fires at a FIXED 0.85 fraction of the subagent model's
``profile["max_input_tokens"]``.  At the orchestrator's real ~131k window
that's ~111k — but the local Qwen3-27B bogs down at ~30k, so the research
subagents' context sails through the slow zone before compaction and the
research-agent "hangs".

``create_research_agent`` fixes this by handing the subagents a model that
*reports* a smaller window (``RESEARCH_SUBAGENT_WINDOW``) so the same
auto-summarizer fires at ~51k, bounding their working set under the slow
zone, while the orchestrator keeps its full window.

Two layers of test:

1. *Wiring* (model-free): spy on the summarization factory the auto-stack
   calls (``deepagents.graph.create_summarization_middleware``) and assert
   which window each summarizer is built with.  No LLM — ``create_deep_agent``
   only compiles the graph.

2. *Bounding* (search-free, the symptom): build the configured summarizer and
   feed it REAL bulk content (read_url-style tool results grown past the
   trigger), then exercise the un-mocked decision the live agent makes —
   ``_should_summarize`` — to prove the reduced window actually FIRES in the
   ~55k danger zone where the slow model bogs, whereas the pre-fix full window
   would have sailed on toward ~111k.  This validates the runtime behavior the
   live research eval can't (its heavy search traffic rate-limits SearXNG).
"""
import tempfile
from unittest.mock import patch

import deepagents.graph as dg
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from assist.agent import RESEARCH_SUBAGENT_WINDOW, create_research_agent

ORCHESTRATOR_WINDOW = 131_000  # the real prod n_ctx (roadmap.org:328)


def _stub_model(window):
    """A real ChatOpenAI (so model_copy/profile behave exactly as in prod)
    pointed at an unroutable URL — it is never called."""
    m = ChatOpenAI(model="stub", base_url="http://127.0.0.1:9/v1", api_key="stub")
    m.profile = {"max_input_tokens": window}
    return m


def _build_and_record():
    """Build a research agent with a full-window orchestrator model, spying on
    the summarization factory to capture the window each summarizer is built
    with.  The spy sees FIVE summarizers: the 3 reduced-window research
    subagents, plus deepagents' auto general-purpose subagent and the main
    agent — both at the orchestrator's full window.  Returns (windows_seen,
    orchestrator_model)."""
    seen = []
    real = dg.create_summarization_middleware

    def spy(model, backend, *args, **kwargs):
        profile = getattr(model, "profile", None) or {}
        seen.append(profile.get("max_input_tokens"))
        return real(model, backend, *args, **kwargs)

    orchestrator = _stub_model(ORCHESTRATOR_WINDOW)
    with tempfile.TemporaryDirectory() as d, \
            patch.object(dg, "create_summarization_middleware", spy):
        create_research_agent(orchestrator, d)
    return seen, orchestrator


def test_each_research_subagent_summarizes_at_reduced_window():
    """All three dict-spec subagents (research/critique/fact-check) get a
    summarizer built against RESEARCH_SUBAGENT_WINDOW — the symptom-cause of
    the hang was these firing at the orchestrator's ~111k instead."""
    seen, _ = _build_and_record()
    assert seen.count(RESEARCH_SUBAGENT_WINDOW) == 3, (
        f"expected 3 subagent summarizers at {RESEARCH_SUBAGENT_WINDOW}, "
        f"got windows {seen}"
    )


def test_orchestrator_keeps_full_window():
    """The orchestrator's own summarizer is untouched — it works fine at 0.85
    of its real window (it only reads URLs).  A regression here would mean the
    reduced window leaked onto the main agent."""
    seen, _ = _build_and_record()
    assert ORCHESTRATOR_WINDOW in seen, (
        f"orchestrator summarizer should see the full {ORCHESTRATOR_WINDOW}, "
        f"got windows {seen}"
    )


def test_orchestrator_model_profile_not_mutated():
    """model_copy() must isolate the subagents' profile — the orchestrator
    model the caller passed in keeps its full window after the build (a shared
    mutation would make the MAIN agent summarize far too early)."""
    _, orchestrator = _build_and_record()
    assert orchestrator.profile == {"max_input_tokens": ORCHESTRATOR_WINDOW}


def test_reduced_window_fires_earlier_and_is_not_raised():
    """The reduced window must fire strictly earlier than the orchestrator's
    AND must not be raised back toward it.  Eval-tuning the window DOWN (toward
    ~40k if stalls persist) is expected; raising it above the 60k design
    ceiling — creeping the summarization trigger back toward the ~111k that
    hung — is the regression this guards."""
    subagent_trigger = int(0.85 * RESEARCH_SUBAGENT_WINDOW)
    orchestrator_trigger = int(0.85 * ORCHESTRATOR_WINDOW)
    assert subagent_trigger < orchestrator_trigger, "must fire earlier than orchestrator"
    # Assert on the WINDOW itself, not the derived trigger: a trigger bound
    # (e.g. <= 60_000) would let the window creep to ~70k since 0.85*70k≈59.5k.
    assert RESEARCH_SUBAGENT_WINDOW <= 60_000, (
        f"RESEARCH_SUBAGENT_WINDOW {RESEARCH_SUBAGENT_WINDOW} raised above the "
        "60k ceiling — that creeps the summarization trigger back toward the hang"
    )


# --- Bounding (search-free): does the configured window actually fire? --------
# The wiring tests prove the summarizer is BUILT at the reduced window.  These
# prove the CONSEQUENCE with real injected bulk content: the configured
# summarizer fires at ~51k (the slow zone), where the pre-fix full window would
# not — the exact before/after the live research eval can't measure because its
# search traffic rate-limits SearXNG.

# A context size in the danger zone: past the reduced-window trigger
# (0.85*60k≈51k) but far below the orchestrator's (0.85*131k≈111k) — exactly
# where the hang's context kept growing on the slow model.
_DANGER_ZONE_TOKENS = 55_000


def _subagent_summarizer(window):
    """The summarization middleware a research subagent actually gets, built the
    way create_research_agent builds it: a model REPORTING `window` fed to the
    deepagents factory, which derives its 0.85 fraction trigger off that
    profile.  The backend is unused by the trigger decision."""
    return create_summarization_middleware(_stub_model(window), StateBackend())


def _bulk_conversation(token_counter, target_tokens):
    """A realistic research-subagent conversation — read_url tool calls plus
    large page-content results — grown until the middleware's OWN counter
    reports >= target_tokens.  This is genuine injected bulk content (counted by
    the real counter), not a hand-fed token number."""
    msgs = [HumanMessage(content="Research: persistent emacs sessions over tramp/ssh.")]
    page = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 120
    i = 0
    while token_counter(msgs) < target_tokens:
        msgs.append(AIMessage(content="", tool_calls=[
            {"name": "read_url", "args": {"url": f"https://example.com/doc{i}"}, "id": f"call{i}"}]))
        msgs.append(ToolMessage(content=page, tool_call_id=f"call{i}"))
        i += 1
    return msgs


def test_fires_when_bulk_context_exceeds_reduced_trigger():
    """Real ~55k bulk context DOES trip the reduced-window summarizer — the
    bounding kicks in before the slow zone runs away.  Mirrors exactly the
    decision wrap_model_call makes (count tokens, then _should_summarize)."""
    mw = _subagent_summarizer(RESEARCH_SUBAGENT_WINDOW)
    msgs = _bulk_conversation(mw.token_counter, _DANGER_ZONE_TOKENS)
    total = mw.token_counter(msgs)
    assert total >= _DANGER_ZONE_TOKENS
    assert mw._should_summarize(msgs, total) is True, (
        f"{total} tokens should trigger summarization at the "
        f"{RESEARCH_SUBAGENT_WINDOW} window (threshold {int(0.85 * RESEARCH_SUBAGENT_WINDOW)})"
    )


def test_pre_fix_full_window_would_not_fire_at_same_size():
    """The before/after that IS the fix: the SAME ~55k bulk context the reduced
    window compacts would sail past unsummarized under the orchestrator's full
    window — that unbounded growth is the hang."""
    reduced = _subagent_summarizer(RESEARCH_SUBAGENT_WINDOW)
    msgs = _bulk_conversation(reduced.token_counter, _DANGER_ZONE_TOKENS)
    total = reduced.token_counter(msgs)
    full = _subagent_summarizer(ORCHESTRATOR_WINDOW)
    assert full._should_summarize(msgs, total) is False, (
        f"{total} tokens must NOT trigger at the full {ORCHESTRATOR_WINDOW} window "
        f"(threshold {int(0.85 * ORCHESTRATOR_WINDOW)}) — that's the pre-fix hang"
    )


def test_does_not_fire_below_reduced_trigger():
    """A modest research turn well under the trigger is NOT compacted — the fix
    bounds runaway context without over-summarizing small turns (which would
    cost breadth and an extra LLM call)."""
    mw = _subagent_summarizer(RESEARCH_SUBAGENT_WINDOW)
    msgs = _bulk_conversation(mw.token_counter, 30_000)
    total = mw.token_counter(msgs)
    assert total < int(0.85 * RESEARCH_SUBAGENT_WINDOW), f"setup overshoot: {total}"
    assert mw._should_summarize(msgs, total) is False
