"""Unit tests for the agent-facing subscription tools (thread-scoped, never-raise)."""
import os

import pytest

import assist.events.tools as tools_mod
from assist.events.store import SubscriptionStore


@pytest.fixture
def tools(tmp_path, monkeypatch):
    os.makedirs(tmp_path / "t1")
    store = SubscriptionStore(str(tmp_path))
    monkeypatch.setattr(tools_mod, "get_config", lambda: {"configurable": {"thread_id": "t1"}})
    fns = {f.__name__: f for f in tools_mod.subscription_tools(store)}
    return fns, store


def test_create_and_list(tools):
    fns, store = tools
    out = fns["create_subscription"](r"^\+1555", "from {sender}: {text}\nreply nicely")
    assert "Subscribed" in out
    assert len(store.for_thread("t1")) == 1
    assert "sender ~" in fns["list_subscriptions"]()


def test_create_rejects_bad_regexp(tools):
    fns, store = tools
    out = fns["create_subscription"](r"(unclosed", "t")
    assert "doesn't compile" in out
    assert "regexp" in out.lower()          # points at the regexp skill
    assert store.for_thread("t1") == []      # nothing stored


def test_create_rejects_empty_template(tools):
    fns, _ = tools
    assert "empty" in fns["create_subscription"](r".*", "   ").lower()


def test_modify_sparse(tools):
    fns, store = tools
    fns["create_subscription"](r"^\+1555", "orig template")
    sid = store.for_thread("t1")[0].id
    out = fns["modify_subscription"](sid, template="new template")
    assert "Updated" in out
    s = store.for_thread("t1")[0]
    assert s.template == "new template" and s.sender_regexp == r"^\+1555"  # regexp untouched


def test_modify_unknown_id(tools):
    fns, _ = tools
    assert "No subscription" in fns["modify_subscription"]("nope", template="x")


def test_pause_resume_delete(tools):
    fns, store = tools
    fns["create_subscription"](r".*", "t")
    sid = store.for_thread("t1")[0].id
    assert "Paused" in fns["pause_subscription"](sid)
    assert store.for_thread("t1")[0].enabled is False
    assert "Resumed" in fns["resume_subscription"](sid)
    assert "Deleted" in fns["delete_subscription"](sid)
    assert store.for_thread("t1") == []


def test_no_active_thread(tools, monkeypatch):
    fns, _ = tools
    monkeypatch.setattr(tools_mod, "get_config", lambda: {"configurable": {}})
    assert "no active thread" in fns["create_subscription"](r".*", "t").lower()
