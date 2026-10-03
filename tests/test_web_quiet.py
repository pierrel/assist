"""A quiet turn skips only its own ordinary new marker."""

import os

import pytest

from assist.events import quiet as quiet_module
from assist.run_service import Run
from manage import web
from manage.web import state, threads


@pytest.fixture
def quiet_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    (tmp_path / "one").mkdir()
    state._UNSEEN.clear()
    yield tmp_path
    state._UNSEEN.clear()


def _running():
    run = threads._runs().create("one", "general-agent", "check")
    return threads._runs().claim("one", run.id)


def test_quiet_marks_exact_running_run_and_skips_ordinary_new(quiet_root, monkeypatch):
    run = _running()
    monkeypatch.setattr(quiet_module, "get_config", lambda: {
        "configurable": {"thread_id": "one", "web_run_id": run.id}})
    tool = quiet_module.quiet_tools(state._request_quiet)[0]
    assert "will not add a new badge" in tool()
    assert threads._quiet_ready(run)
    state._set_status("one", "ready", mark_unseen=not threads._quiet_ready(run))
    assert not state._has_unseen_response("one")
    assert not os.path.exists(state._unseen_path("one"))
    persisted = threads._runs().get("one", run.id)
    assert persisted.quiet_requested
    assert Run.from_dict(persisted.to_dict()).quiet_requested
    assert Run.from_dict(run.to_dict()).quiet_requested is False


def test_quiet_cannot_mark_other_or_finished_run(quiet_root):
    run = _running()
    assert not threads._runs().request_quiet("one", "other")
    assert not threads._runs().request_quiet("other-thread", run.id)
    threads._runs().transition("one", run.id, "success")
    assert not threads._runs().request_quiet("one", run.id)
    assert not threads._quiet_ready(run)


def test_quiet_does_not_clear_earlier_unread_or_approval(quiet_root):
    state._mark_unseen_response("one")
    run = _running()
    assert threads._runs().request_quiet("one", run.id)
    state._set_status("one", "ready", mark_unseen=not threads._quiet_ready(run))
    assert state._has_unseen_response("one")
    state._clear_unseen_response("one")
    state._set_status("one", "awaiting_approval", mark_unseen=False)
    assert state._has_unseen_response("one")


def test_quiet_survives_fair_pause_but_not_next_work(quiet_root):
    first = _running()
    assert threads._runs().request_quiet("one", first.id)
    threads._runs().transition("one", first.id, "interrupted")
    successor = threads._runs().create("one", "general-agent", None,
                                       work_id=first.work_id, resume=True)
    successor = threads._runs().claim("one", successor.id)
    assert threads._quiet_ready(successor)
    later = _running()
    assert not threads._quiet_ready(later)


def test_triage_approval_and_child_runs_cannot_request_quiet(quiet_root):
    triage = threads._runs().create("one", "general-agent", "sms", sender="+15551234567")
    triage = threads._runs().claim("one", triage.id)
    assert not threads._runs().request_quiet("one", triage.id)
    approval = threads._runs().create("one", "general-agent", None,
                                      resume_decision={"type": "approve"})
    approval = threads._runs().claim("one", approval.id)
    assert not threads._runs().request_quiet("one", approval.id)
    child = threads._runs().create("sub-one", "research-agent", "task", mode="child",
                                  parent_thread_id="one", parent_run_id="parent",
                                  dispatch_key="child:one")
    child = threads._runs().claim("sub-one", child.id)
    assert not threads._runs().request_quiet("sub-one", child.id)


def test_invalid_persisted_quiet_type_fails_closed(quiet_root):
    run = _running().to_dict()
    run["quiet_requested"] = "yes"
    with pytest.raises(ValueError, match="quiet"):
        Run.from_dict(run)
