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
    monkeypatch.setattr(threads._HERMES_RUNTIME, "stop_work",
                        lambda _tid, _work: (events.append("stopped"), True)[1])
    def released(_tid):
        assert events == ["stopped"]
        events.append("released")
    monkeypatch.setattr(threads._HERMES_RUNTIME, "release_slice", released)
    def broken_git(*_args, **_kwargs):
        raise OSError("fixture Git acquisition failed")
    monkeypatch.setattr(threads.GitLifecycle, "acquire", broken_git)
    threads._execute_hermes_run(run, user_priority=False)
    assert events == ["stopped", "stopped"]
    assert threads.THREAD_QUEUE.current_handle() is None
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
        assert threads._runs().get(tid, run.id).status == "running"
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
    monkeypatch.setattr(threads, "THREAD_QUEUE", SimpleNamespace(acquire=unavailable, current_handle=lambda: None))
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: events.append("dispatch"))
    if stop_fails:
        with pytest.raises(RuntimeError, match="teardown unconfirmed"):
            threads._execute_hermes_run(run, user_priority=False)
        assert threads._runs().get(tid, run.id).status == "running"
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


def test_failed_parking_proof_then_verified_stop_releases_model_slot(hermes_threads, monkeypatch, tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    from assist.thread_queue import ThreadAffinityQueue, ThreadPauseRequested
    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, None, resume=True, pending_text="Retained native work")
    stopped = []
    container = SimpleNamespace()
    manager = HermesRuntimeManager(sandbox_manager=SimpleNamespace(
        cleanup_verified=lambda _work, generation: stopped.append(generation)))
    control = tmp_path / "private-control"
    control.mkdir()
    manager.turns[tid] = SimpleNamespace(work_id=run.work_id, work_dir="fixture", paused=True,
        backend=SimpleNamespace(container=container), control_dir=control,
        relay=SimpleNamespace(parked=False, connections={},
                              stop_admission=lambda: None, close=lambda: None))
    def pause(*_args, **_kwargs):
        raise ThreadPauseRequested()
    manager.run_slice = pause
    queue = ThreadAffinityQueue()
    @contextmanager
    def git_scope(*_args, **_kwargs):
        from contextlib import nullcontext
        yield SimpleNamespace(preserve_on_error=lambda _tz: nullcontext())
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", manager)
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", git_scope)
    threads._execute_hermes_run(run, user_priority=False)
    assert stopped == [container] and tid not in manager.turns
    assert threads._runs().get(tid, run.id).status == "error"
    assert threads._runs().get(tid, run.id).error == "Hermes parking is unconfirmed"
    assert queue.current_handle() is None, "verified teardown left the global model holder stranded"
    with queue.acquire("competitor", wait_timeout_s=0.01):
        pass


@pytest.mark.parametrize("successor_status", ["pending", "running", "interrupted", "success"])
def test_superseded_native_slice_notification_cannot_recover_or_dispatch_followers(
        hermes_threads, monkeypatch, successor_status):
    tid = threads.MANAGER.reserve_visible("hermes")
    service = threads._runs()
    first = threads._create_run(tid, "Keep native work")
    service.claim(tid, first.id)
    service.transition(tid, first.id, "interrupted")
    successor = threads._create_run(tid, None, resume=True, work_id=first.work_id,
                                    pending_text=first.text)
    if successor_status != "pending":
        service.claim(tid, successor.id)
        service.transition(tid, successor.id, successor_status)
    threads._create_run(tid, "Accepted next user turn")
    threads._set_status(tid, "paused")
    monkeypatch.setattr(threads, "_dispatch_pending_after",
                        lambda *_args: pytest.fail("A stale notification must not dispatch"))
    threads._execute_hermes_run(service.get(tid, first.id), user_priority=False)
    assert service.get(tid, first.id).status == "interrupted"
    assert service.get(tid, successor.id).status == successor_status
    assert threads._get_status(tid)["stage"] == "paused"


@pytest.mark.parametrize("retained", [False, True])
def test_duplicate_running_native_notification_does_not_interrupt_live_work(
        hermes_threads, monkeypatch, retained):
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    from assist.thread_queue import ThreadAffinityQueue
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, "Live native work")
    service = threads._runs()
    service.claim(tid, run.id)
    runtime = HermesRuntimeManager()
    if retained:
        runtime.turns[tid] = SimpleNamespace(work_id=run.work_id)
    queue = ThreadAffinityQueue()
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", runtime)
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads, "_dispatch_pending_after",
                        lambda *_args: pytest.fail("Live work must not recover"))
    if retained:
        threads._execute_hermes_run(service.get(tid, run.id), user_priority=False)
    else:
        with queue.acquire(tid):
            threads._execute_hermes_run(service.get(tid, run.id), user_priority=False)
    assert service.get(tid, run.id).status == "running"


def test_native_pause_commits_one_notification_without_finally_redispatch(hermes_threads, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from assist.thread_queue import ThreadAffinityQueue, ThreadPauseRequested
    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, None, resume=True, pending_text="Keep native work")
    events = []
    def pause(*_args, **_kwargs):
        raise ThreadPauseRequested()
    @contextmanager
    def git_scope(*_args, **_kwargs):
        from contextlib import nullcontext
        yield SimpleNamespace(preserve_on_error=lambda _tz: nullcontext())
    monkeypatch.setattr(threads, "THREAD_QUEUE", ThreadAffinityQueue())
    monkeypatch.setattr(threads._HERMES_RUNTIME, "run_slice", pause)
    monkeypatch.setattr(threads._HERMES_RUNTIME, "release_slice", lambda _tid: None)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", git_scope)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "reserve", lambda *_args: "ticket")
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "commit", lambda ticket: events.append(ticket))
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: events.append("duplicate"))
    threads._execute_hermes_run(run, user_priority=False)
    assert events == ["ticket"]
    assert threads._runs().get(tid, run.id).status == "interrupted"


def test_duplicate_recovery_does_not_wait_on_runtime_teardown_lock(hermes_threads, monkeypatch):
    import threading
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, "Still retained")
    threads._runs().claim(tid, run.id)
    runtime = HermesRuntimeManager()
    runtime.turns[tid] = SimpleNamespace(work_id=run.work_id)
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", runtime)
    done = threading.Event()
    errors = []
    def dispatch():
        try:
            threads._execute_hermes_run(threads._runs().get(tid, run.id), user_priority=False)
        except Exception as error:
            errors.append(error)
        finally:
            done.set()
    with runtime.lock:
        duplicate = threading.Thread(target=dispatch)
        duplicate.start()
        completed_while_teardown_locked = done.wait(1)
    duplicate.join(2)
    assert completed_while_teardown_locked, "admission waited on the runtime teardown lock"
    assert errors == []
    assert threads._runs().get(tid, run.id).status == "running"


def test_native_git_changes_keep_diff_and_merge_without_unsupported_review(
        hermes_threads, monkeypatch):
    from assist.domain_manager import Change
    from types import SimpleNamespace

    tid = threads.MANAGER.reserve_visible("hermes")
    threads._create_empty_pi_workspace(tid)
    monkeypatch.setattr(threads, "_review_base", lambda _tid: "fixture-base")
    monkeypatch.setattr(threads, "_get_domain_manager", lambda _tid: SimpleNamespace(
        main_diff=lambda _base: [Change(path="note.txt", diff="+blue note")]))
    page = TestClient(threads.app).get(f"/thread/{tid}")
    assert page.status_code == 200
    assert "blue note" in page.text
    assert f'action="/thread/{tid}/merge"' in page.text
    assert f'href="/thread/{tid}/review"' not in page.text
    assert TestClient(threads.app).get(f"/thread/{tid}/review").status_code == 409


def test_merge_invalid_engine_keeps_existing_unavailable_response(hermes_threads):
    tid = threads.MANAGER.reserve_visible("hermes")
    marker = Path(threads.MANAGER.thread_dir(tid)) / "engine.json"
    marker.write_text('{"engine":"invalid"}')
    response = TestClient(threads.app, raise_server_exceptions=False).post(f"/thread/{tid}/merge")
    assert response.status_code == 409
    assert response.json()["detail"] == "Thread engine is unavailable"


def test_recovered_relay_drain_failure_releases_exact_model_slot(
        hermes_threads, monkeypatch, tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    from assist.sandbox_manager import SandboxManager
    from assist.thread_queue import ThreadAffinityQueue
    from assist import sandbox_expiry

    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    run = threads._create_run(tid, None, resume=True, pending_text="Retained native work")
    work_dir = str(tmp_path / "private-work")
    control = tmp_path / "private-control"
    control.mkdir()
    stopped, drains = [], []
    container = SimpleNamespace(id="private-generation", kill=lambda: stopped.append(True),
                                wait=lambda **_kwargs: None)
    monkeypatch.setattr(SandboxManager, "_containers", {work_dir: container})
    monkeypatch.setattr(SandboxManager, "_generation_owners", {})
    monkeypatch.setattr(SandboxManager, "_egress_client_ips", {})
    monkeypatch.setattr(sandbox_expiry, "cancel", lambda _generation: None)

    def close():
        drains.append(True)
        if len(drains) == 1:
            raise RuntimeError("fixture provider drain unconfirmed")

    manager = HermesRuntimeManager()
    manager.turns[tid] = SimpleNamespace(work_id=run.work_id, work_dir=work_dir, paused=True,
        backend=SimpleNamespace(container=container), control_dir=control,
        relay=SimpleNamespace(stop_admission=lambda: None, close=close))

    def failed_slice(*_args, **_kwargs):
        raise RuntimeError("fixture native work failed")

    manager.run_slice = failed_slice
    queue = ThreadAffinityQueue()

    @contextmanager
    def git_scope(*_args, **_kwargs):
        from contextlib import nullcontext
        yield SimpleNamespace(preserve_on_error=lambda _tz: nullcontext())

    monkeypatch.setattr(threads, "_HERMES_RUNTIME", manager)
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", git_scope)
    threads._execute_hermes_run(run, user_priority=False)
    assert stopped == [True] and len(drains) == 2
    assert tid not in manager.turns
    assert threads._runs().get(tid, run.id).status == "error"
    assert queue.current_handle() is None
    with queue.acquire("competitor", wait_timeout_s=0.01):
        pass


@pytest.mark.parametrize("drain_fails", [False, True])
def test_first_message_cleanup_failure_preserves_running_head_and_pending_follower(
        hermes_threads, monkeypatch, tmp_path, drain_fails):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    from assist.sandbox_manager import SandboxManager
    from assist.thread_queue import ThreadAffinityQueue
    from manage.web.run_stream import RunStreamJournal

    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    first = threads._create_run(tid, "Create the blue note")
    follower = threads._create_run(tid, "Make it green afterward")
    threads._set_status(tid, "initializing", pending_message=first.text, pending_run_id=first.id)
    work_dir = str(tmp_path / "private-work")
    control = tmp_path / "private-control"
    control.mkdir()

    def unconfirmed():
        raise RuntimeError("fixture native stop unconfirmed")

    def drain(*, drain_egress=True):
        assert not drain_egress, "failed cleanup must not retain the independent model slot"
        if drain_fails:
            raise RuntimeError("fixture provider drain unconfirmed")

    container = SimpleNamespace(id="private-generation", kill=unconfirmed)
    monkeypatch.setattr(SandboxManager, "_containers", {work_dir: container})
    manager = HermesRuntimeManager()
    manager.turns[tid] = SimpleNamespace(work_id=first.work_id, work_dir=work_dir, paused=False,
        backend=SimpleNamespace(container=container), control_dir=control,
        relay=SimpleNamespace(stop_admission=lambda: None, close=drain))
    manager.run_slice = lambda *_args, **_kwargs: unconfirmed()
    queue = ThreadAffinityQueue()
    journal = RunStreamJournal()
    for run in (first, follower):
        assert journal.reserve(tid, run.work_id)
        assert journal.activate(tid, run.work_id)
    dispatched = []

    @contextmanager
    def git_scope(*_args, **_kwargs):
        yield SimpleNamespace(prepare=lambda _tz: None)

    monkeypatch.setattr(threads, "_HERMES_RUNTIME", manager)
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads, "RUN_STREAMS", journal)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", git_scope)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: dispatched.append(True))
    errors = []
    try:
        threads._initialize_thread(tid, first.id, None)
    except RuntimeError as error:
        errors.append(str(error))
    if drain_fails:
        assert queue.current_handle().thread_id == tid
        assert queue.current_handle().expired
    else:
        assert queue.current_handle() is None
        with queue.acquire("unrelated", wait_timeout_s=0.01):
            pass
    assert tid in manager.turns
    assert SandboxManager.current_container(work_dir) is container
    assert [run.status for run in threads._runs().list(tid)] == ["running", "pending"]
    assert errors == ["fixture provider drain unconfirmed" if drain_fails
                      else "fixture native stop unconfirmed"]
    assert not journal.read(tid, first.work_id)["terminal"]
    assert not journal.read(tid, follower.work_id)["terminal"]
    assert threads._get_status(tid)["stage"] == "processing"
    assert not dispatched


@pytest.mark.parametrize("original_claimed", [False, True])
def test_stale_pending_notification_timeout_cannot_stop_claimed_live_work(
        hermes_threads, monkeypatch, original_claimed):
    import threading
    from types import SimpleNamespace
    from assist.hermes_runtime import HermesRuntimeManager
    from assist.thread_queue import ThreadAffinityQueue

    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    stale = threads._create_run(tid, "Keep working on the note")
    queue = ThreadAffinityQueue(wait_timeout_s=0.02)
    runtime = HermesRuntimeManager()
    runtime.turns[tid] = SimpleNamespace(work_id=stale.work_id)
    stops, dispatches, errors = [], [], []
    runtime.stop = lambda _tid: stops.append(_tid)
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", runtime)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda *_args: dispatches.append(True))

    def duplicate():
        try:
            threads._execute_hermes_run(stale, user_priority=False)
        except Exception as error:
            errors.append(error)

    with queue.acquire(tid) as original:
        if original_claimed:
            threads._runs().claim(tid, stale.id)
        worker = threading.Thread(target=duplicate)
        worker.start()
        worker.join(2)
        assert not worker.is_alive()
        assert queue.current_handle() is original
        assert not stops, "a stale queued notification stopped the live native invocation"
        assert threads._runs().get(tid, stale.id).status == ("running" if original_claimed else "pending")
        assert not dispatches and not errors


def test_stale_pending_notification_that_acquires_slot_does_not_release_native_work(
        hermes_threads, monkeypatch):
    from assist.thread_queue import ThreadAffinityQueue
    hermes_threads.set_enabled(True)
    tid = threads.MANAGER.reserve_visible("hermes")
    stale = threads._create_run(tid, "Original note")
    threads._runs().claim(tid, stale.id)
    threads._runs().transition(tid, stale.id, "interrupted")
    queue = ThreadAffinityQueue()
    monkeypatch.setattr(threads, "THREAD_QUEUE", queue)
    monkeypatch.setattr(threads._HERMES_RUNTIME, "release_slice",
                        lambda _tid: pytest.fail("Unclaimed notification cannot release native work"))
    monkeypatch.setattr(threads, "_dispatch_pending_after",
                        lambda *_args: pytest.fail("Unclaimed notification cannot dispatch followers"))
    threads._execute_hermes_run(stale, user_priority=False)
    assert threads._runs().get(tid, stale.id).status == "interrupted"
    assert queue.current_handle() is None
