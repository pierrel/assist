"""Stable, bounded decisions for the optional private-state maintenance skill."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from assist import frequency


def test_decision_is_stable_across_retries_and_store_restarts(tmp_path):
    (tmp_path / "thread-1").mkdir()
    first = frequency.FrequencyDecisionStore(str(tmp_path)).decide(
        "thread-1", "run-1", "thread-checkpoint", 0.25)
    second = frequency.FrequencyDecisionStore(str(tmp_path)).decide(
        "thread-1", "run-1", "thread-checkpoint", 0.25)
    assert second == first
    saved = json.loads((tmp_path / "thread-1" / "frequency-decisions.json").read_text())
    assert saved["run-1:thread-checkpoint"]["should_run"] is first.should_run


def test_changed_probability_cannot_resample_a_run(tmp_path):
    (tmp_path / "thread-1").mkdir()
    store = frequency.FrequencyDecisionStore(str(tmp_path))
    store.decide("thread-1", "run-1", "thread-checkpoint", 0.25)
    with pytest.raises(ValueError, match="reuse"):
        store.decide("thread-1", "run-1", "thread-checkpoint", 0.5)


def test_changed_policy_cannot_resample_a_run(tmp_path):
    (tmp_path / "thread-1").mkdir()
    store = frequency.FrequencyDecisionStore(str(tmp_path))
    store.decide("thread-1", "run-1", "thread-checkpoint", 0.25)
    with pytest.raises(ValueError, match="another policy"):
        store.decide("thread-1", "run-1", "another-policy", 0.25)


@pytest.mark.parametrize("policy, probability", [
    ("../escape", 0.25), ("Thread Checkpoint", 0.25),
    ("thread-checkpoint", -0.1), ("thread-checkpoint", 1.1),
])
def test_invalid_policy_or_probability_is_rejected(tmp_path, policy, probability):
    (tmp_path / "thread-1").mkdir()
    with pytest.raises(ValueError):
        frequency.FrequencyDecisionStore(str(tmp_path)).decide(
            "thread-1", "run-1", policy, probability)


def test_tool_requires_an_ordinary_web_run(tmp_path, monkeypatch):
    monkeypatch.setattr(frequency, "get_config", lambda: {"configurable": {}})
    tool = frequency.frequency_tools(frequency.FrequencyDecisionStore(str(tmp_path)))[0]
    assert "only during an ordinary web run" in tool("thread-checkpoint", 0.25)


def test_tool_reuses_the_persisted_answer(tmp_path, monkeypatch):
    (tmp_path / "thread-1").mkdir()
    config = {"configurable": {
        "thread_id": "thread-1", frequency.FREQUENCY_RUN_ID_KEY: "run-1"}}
    monkeypatch.setattr(frequency, "get_config", lambda: config)
    tool = frequency.frequency_tools(frequency.FrequencyDecisionStore(str(tmp_path)))[0]
    first = tool("thread-checkpoint", 0.25)
    second = tool("thread-checkpoint", 0.25)
    assert second == first


def test_tool_rejects_other_policy_names(tmp_path, monkeypatch):
    config = {"configurable": {
        "thread_id": "thread-1", frequency.FREQUENCY_RUN_ID_KEY: "run-1"}}
    monkeypatch.setattr(frequency, "get_config", lambda: config)
    tool = frequency.frequency_tools(frequency.FrequencyDecisionStore(str(tmp_path)))[0]
    assert "thread-checkpoint" in tool("another-policy", 0.25)


@pytest.mark.parametrize("mode, origin, sender, assistant_id, expected", [
    ("turn", None, None, "general-agent", {frequency.FREQUENCY_RUN_ID_KEY: "work-1"}),
    ("turn", "task-completion", None, "general-agent", None),
    ("turn", "system", None, "general-agent", None),
    ("turn", None, "+15551234567", "general-agent", None),
    ("task", None, None, "general-agent", None),
    ("turn", None, None, "context-agent", None),
])
def test_frequency_identity_is_limited_to_visible_main_runs(
        mode, origin, sender, assistant_id, expected):
    from manage.web.threads import _frequency_configurable

    run = SimpleNamespace(mode=mode, origin=origin, work_id="work-1")
    assert _frequency_configurable(
        run, sender=sender, assistant_id=assistant_id) == expected


def test_ordinary_web_profile_exposes_the_maintenance_tool():
    """Only the normal web composition supplies a durable Run ID for it."""
    import manage.web.state  # noqa: F401 - initializes the web composition root
    from assist.thread_manager import _web_tools

    assert "should_run_maintenance" in {
        getattr(tool, "name", getattr(tool, "__name__", "")) for tool in _web_tools}
