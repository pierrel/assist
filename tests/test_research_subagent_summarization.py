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

These tests pin that wiring model-free by spying on the summarization
factory the auto-stack calls (``deepagents.graph.create_summarization_
middleware``).  No LLM is invoked — ``create_deep_agent`` only compiles the
graph; it never calls the model.
"""
import tempfile
from unittest.mock import patch

import deepagents.graph as dg
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
