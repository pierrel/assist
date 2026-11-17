"""Hermes client identity and Run projection without constructing a Deep graph."""
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from assist.pi_preview import PiPreviewPolicy
from assist.thread_engine import read_thread_engine
from manage.web import phone_api, state, threads

_REAL_DISPATCH_PENDING_AFTER = threads._dispatch_pending_after


@pytest.fixture
def hermes_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads, "DOMAINS", [])
    monkeypatch.setattr(state, "DOMAINS", [])
    policy = PiPreviewPolicy(tmp_path, engine="hermes")
    monkeypatch.setattr(threads, "HERMES_PREVIEW", policy)
    monkeypatch.setattr(state, "HERMES_PREVIEW", policy)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: None)
    monkeypatch.setattr(threads.MANAGER, "get", lambda *_args, **_kwargs:
                        pytest.fail("Native clients must not construct a Deep graph"))
    state.DESCRIPTION_CACHE.clear()
    yield policy
    threads._RUN_SERVICES_BY_ROOT.pop(str(tmp_path), None)
    state.DESCRIPTION_CACHE.clear()


def test_hermes_creation_is_operator_gated_and_keeps_immutable_identity(hermes_threads):
    client = TestClient(threads.app)
    assert client.post("/threads", data={"engine": "hermes"}).status_code == 503
    hermes_threads.set_enabled(True)
    response = client.post("/threads", data={"engine": "hermes"}, follow_redirects=False)
    assert response.status_code == 303
    tid = response.headers["location"].split("/")[-1]
    assert read_thread_engine(threads.MANAGER.thread_dir(tid)).name == "hermes"
    assert Path(threads.MANAGER.thread_default_working_dir(tid)).is_dir()
    hermes_threads.set_enabled(False)
    assert client.get(f"/thread/{tid}").status_code == 200
    assert client.post(f"/thread/{tid}/message", data={"text": "Revise the note"}).status_code == 503


def test_resumed_native_turn_projects_one_reply_in_browser_and_phone(hermes_threads):
    tid = threads.MANAGER.reserve_visible("hermes")
    threads._create_empty_pi_workspace(tid)
    service = threads._runs()
    first = threads._create_run(tid, "Make a blue note")
    service.claim(tid, first.id)
    service.transition(tid, first.id, "interrupted")
    successor = threads._create_run(tid, None, resume=True, work_id=first.work_id,
                                    user_event_id=first.user_event_id, pending_text=first.text)
    service.claim(tid, successor.id)
    service.transition(tid, successor.id, "success", result="The note is blue.")
    records = threads._hermes_messages(tid)
    assert [(m["role"], m["content"]) for m in records] == [
        ("user", "Make a blue note"), ("assistant", "The note is blue.")]
    page = TestClient(threads.app).get(f"/thread/{tid}")
    assert page.status_code == 200
    assert page.text.count("The note is blue.") == 1
    history, more, checkpoint = phone_api._thread_history(tid, None)
    assert history == records and not more and not checkpoint
    cursor = phone_api._message_id(tid, history[-1], 2)
    assert phone_api._thread_history(tid, cursor)[0] == history[:1]
    with pytest.raises(HTTPException, match="Deep Agents"):
        threads._require_deep_thread(tid)


@pytest.mark.parametrize("status", ["running", "interrupted"])
def test_native_process_loss_ends_run_without_replay(hermes_threads, status, monkeypatch):
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, "Keep the existing note")
    service = threads._runs()
    service.claim(tid, run.id)
    if status == "interrupted":
        service.transition(tid, run.id, "interrupted")
    monkeypatch.setattr(threads._HERMES_RUNTIME, "run_slice", lambda *_a, **_kw:
                        pytest.fail("Lost native work must not replay"))
    threads._execute_hermes_run(service.get(tid, run.id), user_priority=False)
    assert service.get(tid, run.id).status == ("cancelled" if status == "interrupted" else "error")
    assert "not replayed" in service.get(tid, run.id).error


def test_resume_setup_failure_stops_native_before_slot_release(hermes_threads, monkeypatch):
    from assist.thread_queue import ThreadAffinityQueue
    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, None, resume=True, pending_text="Keep the note")
    events = []
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(threads._HERMES_RUNTIME, "stop", lambda _tid: events.append("stopped"))
    def released(_tid):
        assert events == ["stopped"]
        events.append("released")
    monkeypatch.setattr(threads._HERMES_RUNTIME, "release_slice", released)
    def broken_git(*_args, **_kwargs):
        raise OSError("fixture Git acquisition failed")
    monkeypatch.setattr(threads.GitLifecycle, "acquire", broken_git)
    threads._execute_hermes_run(run, user_priority=False)
    assert events == ["stopped", "released"]
    assert threads._runs().get(tid, run.id).status == "error"


def test_queued_native_turn_is_one_browser_bubble_but_remains_in_phone_history(hermes_threads):
    tid = threads.MANAGER.reserve_visible("hermes")
    repeated = "Repeat this exact queued note"
    first = threads._create_run(tid, repeated)
    service = threads._runs()
    service.claim(tid, first.id)
    service.transition(tid, first.id, "success", result="First note received.")
    threads._create_run(tid, repeated)
    page = TestClient(threads.app).get(f"/thread/{tid}")
    assert page.status_code == 200
    assert page.text.count(repeated) == 2
    assert page.text.count('font-weight:normal;">queued</span>') == 1
    history = phone_api._thread_history(tid, None)[0]
    assert sum(m["role"] == "user" and m["content"] == repeated for m in history) == 2


def test_native_lost_head_notifies_accepted_follower(hermes_threads, monkeypatch):
    tid = threads.MANAGER.reserve_visible("hermes")
    head = threads._create_run(tid, "First accepted turn")
    threads._runs().claim(tid, head.id)
    follower = threads._create_run(tid, "Next accepted turn")
    submitted = []
    monkeypatch.setattr(threads, "_dispatch_pending_after", _REAL_DISPATCH_PENDING_AFTER)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, thread_id, **kwargs: submitted.append((run_id, thread_id)))
    threads._execute_hermes_run(threads._runs().get(tid, head.id), user_priority=False)
    assert submitted == [(follower.id, tid)]
    assert threads._runs().get(tid, head.id).status == "error"


@pytest.mark.parametrize("same_work,stop_fails", [(True, False), (False, False), (True, True)])
def test_queue_admission_failure_stops_only_matching_work_before_terminal_or_dispatch(
        hermes_threads, monkeypatch, same_work, stop_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    from assist.thread_queue import QueueWaitTimeout
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, None, resume=True, pending_text="Retained work")
    runtime = HermesRuntimeManager()
    runtime.turns[tid] = SimpleNamespace(work_id=run.work_id if same_work else "unrelated-work", paused=True)
    events = []
    def stopped(_tid):
        assert threads._runs().get(tid, run.id).status == "pending"
        events.append("stop")
        if stop_fails:
            raise RuntimeError("fixture teardown unconfirmed")
        runtime.turns.pop(tid)
    runtime.stop = stopped
    @contextmanager
    def unavailable(*_args, **_kwargs):
        raise QueueWaitTimeout("fixture queue wait expired")
        yield
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", runtime)
    monkeypatch.setattr(threads, "THREAD_QUEUE", SimpleNamespace(acquire=unavailable))
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: events.append("dispatch"))
    if stop_fails:
        with pytest.raises(RuntimeError, match="teardown unconfirmed"):
            threads._execute_hermes_run(run, user_priority=False)
        assert threads._runs().get(tid, run.id).status == "pending"
        assert events == ["stop"]
    else:
        threads._execute_hermes_run(run, user_priority=False)
        assert threads._runs().get(tid, run.id).status == "error"
        assert events == (["stop", "dispatch"] if same_work else ["dispatch"])
        assert (tid in runtime.turns) == (not same_work)


def test_native_claim_between_render_steps_keeps_one_visible_user_message(hermes_threads, monkeypatch):
    tid = threads.MANAGER.reserve_visible("hermes")
    text = "Claim this accepted note during rendering"
    run = threads._create_run(tid, text)
    service = threads._runs()
    original = service.peek
    snapshots = []
    def claim_after_snapshot(thread_id):
        snapshot = original(thread_id)
        snapshots.append(snapshot)
        if len(snapshots) == 1:
            service.claim(tid, run.id)
        return snapshot
    monkeypatch.setattr(service, "peek", claim_after_snapshot)
    page = TestClient(threads.app).get(f"/thread/{tid}")
    assert page.status_code == 200
    assert page.text.count(text) == 1
    assert len(snapshots) == 1
