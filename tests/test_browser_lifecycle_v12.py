"""Direct-message admission and crash recovery use durable browser proofs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event

import pytest

from assist.browser import authority, manager as browser
from assist.run_service import InvalidRunTransition, RunService
from assist.egress.store import EgressRequest, EgressStore, request_key
from manage.web import phone_api, threads


@pytest.fixture
def admitted(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    runs = RunService(str(tmp_path))
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(threads, "_get_status", lambda _tid: {})
    monkeypatch.setattr(threads, "_is_pi_thread", lambda _tid: False)
    monkeypatch.setattr(threads, "_queue_browser_revocation", lambda _tid: None)
    monkeypatch.setattr(threads, "_mark_pending", lambda *_args: None)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote", lambda _tid: None)
    monkeypatch.setattr(threads.THREAD_QUEUE, "promote", lambda _tid: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda _tid: None)
    monkeypatch.setattr(browser, "configured_directory", lambda: str(tmp_path))
    monkeypatch.setattr(threads, "configured_directory", lambda: str(tmp_path))
    return tmp_path, runs


def test_held_commit_between_create_and_map_publication_denies_old_startup(
        monkeypatch, admitted):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit a public site", user_origin=True)
    session = browser.BrowserSession("t", old.id, str(root), str(root), None)
    started, release = Event(), Event()
    killed = []

    def create(_request, **_kwargs):
        started.set()
        assert release.wait(3)
        return {"generation": "old-generation", "ip": "172.20.0.2"}

    monkeypatch.setattr(browser, "_launch_sidecar_bounded", create)
    monkeypatch.setattr(browser, "_kill_container", killed.append)
    monkeypatch.setattr(browser, "_load_egress_allowlist", lambda: [])
    with ThreadPoolExecutor(max_workers=1) as pool:
        opening = pool.submit(session.command, "open", url="https://example.com/")
        assert started.wait(3)
        held, _ = threads._accept_message_run("t", "Now read another page")
        assert held.status == "revocation_pending"
        assert held.browser_reset_run_id == old.id
        release.set()
        with pytest.raises(browser.BrowserUnavailable, match="newer user message"):
            opening.result(timeout=3)
    assert killed == ["old-generation"]
    with authority.fence(str(root), "t") as state:
        assert state.lease["owner_run_id"] == old.id
        assert state.lease["generations"] == []
    assert browser.browser_records(str(root), "t") == {}


def test_first_open_deadline_write_cannot_overwrite_direct_held_commit(
        monkeypatch, admitted):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit a public site", user_origin=True)
    session = browser.BrowserSession(
        "t", old.id, str(root), str(root), None, run_service=runs)
    entered, release, admission_entered = Event(), Event(), Event()
    bind = RunService.bind_browser_deadline

    def stalled_bind(service, tid, run_id):
        entered.set()
        assert release.wait(3)
        return bind(service, tid, run_id)

    monkeypatch.setattr(RunService, "bind_browser_deadline", stalled_bind)
    monkeypatch.setattr(session, "_begin_lease", lambda: (_ for _ in ()).throw(
        browser.BrowserUnavailable("stop before Docker")))

    def submit_direct():
        admission_entered.set()
        return threads._accept_message_run("t", "Read another page")

    with ThreadPoolExecutor(max_workers=2) as pool:
        opening = pool.submit(session.command, "open", url="https://example.com/")
        assert entered.wait(3)
        admission = pool.submit(submit_direct)
        assert admission_entered.wait(3)
        assert not admission.done()
        release.set()
        with pytest.raises(browser.BrowserUnavailable, match="stop before Docker"):
            opening.result(timeout=3)
        held, _ = admission.result(timeout=3)
    assert held.status == "revocation_pending"
    assert runs.get("t", old.id).browser_deadline_ns is not None
    assert runs.get("t", held.id).status == "revocation_pending"


def test_crash_after_held_commit_failed_kill_stays_held_then_recovers(
        monkeypatch, admitted):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
        state.add_generation(old.id, "old-generation")
    first, _ = threads._accept_message_run("t", "Read a public page")
    second, _ = threads._accept_message_run("t", "Read another page")
    assert first.status == second.status == "revocation_pending"
    assert first.browser_reset_run_id == second.browser_reset_run_id == old.id

    # Process-local session state vanished after the held journal commit.
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: None)
    monkeypatch.setattr(browser, "_bounded_cli",
                        lambda args, **_kwargs: (
                            b"old-generation\n" if "--all" in args else b""))
    kills = []

    def stop(generation):
        kills.append(generation)
        if len(kills) == 1:
            raise browser.BrowserUnavailable("Docker kill failed")

    monkeypatch.setattr(browser, "_kill_container_confirmed", stop)
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda _tid: None)
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", first.id).status == "revocation_pending"
    assert runs.get("t", second.id).status == "revocation_pending"
    assert "unconfirmed" in runs.get("t", first.id).error
    with authority.fence(str(root), "t") as state:
        assert state.lease["owner_run_id"] == old.id

    assert threads._drain_held_browser_events("t") is True
    assert [runs.get("t", item.id).status for item in (first, second)] == [
        "pending", "revocation_pending"]
    assert threads._drain_held_browser_events("t") is True
    assert kills == ["old-generation", "old-generation", "old-generation"]
    assert [runs.get("t", item.id).status for item in (first, second)] == [
        "pending", "pending"]
    with authority.fence(str(root), "t") as state:
        assert state.lease is None


def test_marked_clean_thread_admits_text_without_global_docker_scan(
        monkeypatch, admitted):
    _, runs = admitted
    monkeypatch.setattr(browser.BrowserManager, "reap_orphans",
                        lambda *_args: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("Docker unavailable")))
    run, _ = threads._accept_message_run("t", "Hello")
    assert run.status == "pending"
    assert runs.get("t", run.id).status == "pending"


def test_browser_tool_waits_for_successful_global_orphan_reconciliation(
        monkeypatch, admitted):
    root, runs = admitted
    monkeypatch.setattr(browser.BrowserManager, "_reconciled_roots", set())
    attempts = []

    def scan(_root):
        attempts.append(1)
        if len(attempts) == 1:
            raise browser.BrowserUnavailable("orphan Docker scan failed")

    monkeypatch.setattr(browser.BrowserManager, "reap_orphans", scan)
    assert browser.BrowserManager.reconcile_startup(str(root)) is False
    assert browser.BrowserManager.ready_for_browser(str(root), "t") is False
    text, _ = threads._accept_message_run("t", "Hello")
    assert text.status == "pending"
    assert runs.get("t", text.id).status == "pending"
    assert attempts == [1]
    browser.BrowserManager._retry_reconciliation(str(root))
    assert browser.BrowserManager.ready_for_browser(str(root), "t") is True
    assert attempts == [1, 1]


def test_promoted_pending_is_redispatched_after_notification_failure(
        monkeypatch, admitted):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
    held, _ = threads._accept_message_run("t", "Read public status")
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs: b"")
    dispatches = []

    def dispatch(_tid):
        dispatches.append(1)
        if len(dispatches) == 1:
            raise RuntimeError("scheduler notification failed")

    monkeypatch.setattr(threads, "_dispatch_pending_after", dispatch)
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda _tid: None)
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", held.id).status == "pending"
    assert threads._drain_held_browser_events("t") is True
    assert runs.get("t", held.id).status == "pending"
    assert dispatches == [1, 1]


def test_promoted_a_notification_precedes_failed_b_reset(monkeypatch, admitted):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
    first, _ = threads._accept_message_run("t", "Read public status")
    second, _ = threads._accept_message_run("t", "Read another page")
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs: b"")
    confirm = browser.BrowserManager.confirm_owner_stopped
    confirmations, notifications = [], []

    def stop(*args):
        confirmations.append(1)
        if len(confirmations) == 2:
            raise browser.BrowserUnavailable("B reset unavailable")
        return confirm(*args)

    def notify(_tid):
        notifications.append(1)
        if len(notifications) == 1:
            raise RuntimeError("scheduler notification failed")

    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped", stop)
    monkeypatch.setattr(threads, "_dispatch_pending_after", notify)
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda _tid: None)
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", first.id).status == "pending"
    assert runs.get("t", second.id).status == "revocation_pending"
    assert threads._drain_held_browser_events("t") is False
    assert notifications == [1, 1]  # A recovered before B's failed reset
    assert runs.get("t", second.id).status == "revocation_pending"
    assert threads._drain_held_browser_events("t") is True
    assert notifications == [1, 1, 1]
    assert runs.get("t", second.id).status == "pending"


@pytest.mark.parametrize("step", ["resume", "queue"])
def test_promoted_pending_retries_each_scheduler_notification_step(
        monkeypatch, admitted, step):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
    held, _ = threads._accept_message_run("t", "Read public status")
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs: b"")
    target = (threads._RESUME_SCHEDULER if step == "resume"
              else threads.THREAD_QUEUE)
    attempts, dispatched = [], []

    def fail_once(_tid):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("scheduler notification failed")

    monkeypatch.setattr(target, "promote", fail_once)
    monkeypatch.setattr(threads, "_dispatch_pending_after", dispatched.append)
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda _tid: None)
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", held.id).status == "pending"
    assert threads._drain_held_browser_events("t") is True
    assert attempts == [1, 1]
    assert dispatched == ["t"]


def test_reset_worker_requeues_after_one_held_event_for_fairness(
        monkeypatch, admitted):
    root, runs = admitted
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
    held = [threads._accept_message_run("t", f"Message {index}")[0]
            for index in range(3)]
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    monkeypatch.setattr(browser, "_bounded_cli", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda _tid: None)
    queued = []
    monkeypatch.setattr(threads, "_queue_browser_revocation", queued.append)
    assert threads._drain_held_browser_events("t") is True
    assert [runs.get("t", item.id).status for item in held] == [
        "pending", "revocation_pending", "revocation_pending"]
    assert queued == ["t"]


def _phone_held_context(monkeypatch, admitted):
    root, runs = admitted
    monkeypatch.setattr(phone_api, "_thread_dir", lambda _tid: str(root / "t"))
    monkeypatch.setattr(phone_api.state, "_get_status", lambda _tid: {"stage": "ready"})
    monkeypatch.setattr(threads, "_set_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(browser.BrowserManager, "current_session", lambda _tid: None)
    monkeypatch.setattr(browser, "_bounded_cli",
                        lambda args, **_kwargs: b"old-generation\n"
                        if "--all" in args else b"")
    queued, dispatched, finished = [], [], []
    monkeypatch.setattr(threads, "_queue_browser_revocation", queued.append)
    monkeypatch.setattr(threads, "_dispatch_pending_after",
                        lambda tid, *_args: dispatched.append(tid))
    monkeypatch.setattr(phone_api.RUN_STREAMS, "finish",
                        lambda tid, work_id: finished.append((tid, work_id)))
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda _tid: None)
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
        state.add_generation(old.id, "old-generation")
    held, _ = threads._accept_message_run("t", "Read public status")
    assert held.status == "revocation_pending"
    return root, runs, held, queued, dispatched, finished


def test_held_phone_delete_receipt_survives_failed_reset_and_retry(
        monkeypatch, admitted):
    root, runs, held, queued, dispatched, finished = _phone_held_context(
        monkeypatch, admitted)
    follower, _ = threads._accept_message_run("t", "Then read a second page")
    kills = []

    def kill(generation):
        kills.append(generation)
        if len(kills) == 1:
            raise browser.BrowserUnavailable("Docker kill failed")

    monkeypatch.setattr(browser, "_kill_container_confirmed", kill)
    code, response = phone_api._cancel_logical_run("t", held.id)
    assert code == 202 and response["outcome"] == "cancelling"
    receipt = runs.get("t", held.id)
    assert receipt.status == "cancelled" and receipt.browser_cancel_reset
    assert receipt.cancel_cleanup == "pending"
    assert phone_api._logical_status("t", held.id)["status"] == "cancelling"
    assert dispatched == [] and finished == []

    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", held.id).cancel_cleanup == "pending"
    assert runs.get("t", follower.id).status == "revocation_pending"
    assert dispatched == [] and finished == []
    retry, _ = phone_api._cancel_logical_run("t", held.id)
    assert retry == 202 and queued

    assert threads._drain_held_browser_events("t") is True
    assert kills == ["old-generation", "old-generation"]
    assert runs.get("t", held.id).cancel_cleanup == "complete"
    assert phone_api._logical_status("t", held.id)["status"] == "cancelled"
    assert finished == [("t", held.work_id)]
    assert runs.get("t", follower.id).status == "revocation_pending"
    assert threads._drain_held_browser_events("t") is True
    assert runs.get("t", follower.id).status == "pending"
    assert dispatched
    code, response = phone_api._cancel_logical_run("t", held.id)
    assert code == 200 and response["outcome"] == "cancelled"


def test_held_phone_cancel_wins_promotion_race_under_short_fence(
        monkeypatch, admitted):
    _, runs, held, _, _, _ = _phone_held_context(monkeypatch, admitted)
    entered, release = Event(), Event()
    confirm = browser.BrowserManager.confirm_owner_stopped

    def paused_confirm(*args):
        entered.set()
        assert release.wait(3)
        return confirm(*args)

    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        paused_confirm)
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _gen: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        resetting = pool.submit(threads._drain_held_browser_events, "t")
        assert entered.wait(3)
        code, _ = phone_api._cancel_logical_run("t", held.id)
        assert code == 202
        release.set()
        assert resetting.result(timeout=3) is True
    receipt = runs.get("t", held.id)
    assert receipt.status == "cancelled" and receipt.cancel_cleanup == "complete"
    assert phone_api._logical_status("t", held.id)["status"] == "cancelled"


def test_held_phone_cancel_receipt_precedes_follower_notification_retry(
        monkeypatch, admitted):
    _, runs, held, _, _, _ = _phone_held_context(monkeypatch, admitted)
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _gen: None)
    attempts = []

    def notify(_tid, *_args):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("scheduler temporarily unavailable")

    monkeypatch.setattr(threads, "_dispatch_pending_after", notify)
    assert phone_api._cancel_logical_run("t", held.id)[0] == 202
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", held.id).cancel_cleanup == "complete"
    assert phone_api._logical_status("t", held.id)["status"] == "cancelled"
    assert threads._drain_held_browser_events("t") is True
    assert attempts == [1, 1]


def test_cancelled_a_notification_precedes_failed_b_reset(monkeypatch, admitted):
    _, runs, held, _, _, _ = _phone_held_context(monkeypatch, admitted)
    second, _ = threads._accept_message_run("t", "Read a second report")
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _gen: None)
    confirm = browser.BrowserManager.confirm_owner_stopped
    confirmations, notifications = [], []

    def stop(*args):
        confirmations.append(1)
        if len(confirmations) == 2:
            raise browser.BrowserUnavailable("B reset unavailable")
        return confirm(*args)

    def notify(_tid, *_args):
        notifications.append(1)
        if len(notifications) == 1:
            raise RuntimeError("scheduler notification failed")

    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped", stop)
    monkeypatch.setattr(threads, "_dispatch_pending_after", notify)
    assert phone_api._cancel_logical_run("t", held.id)[0] == 202
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", held.id).cancel_cleanup == "complete"
    assert runs.get("t", second.id).status == "revocation_pending"
    assert threads._drain_held_browser_events("t") is False
    assert notifications == [1, 1]
    assert runs.get("t", second.id).status == "revocation_pending"


@pytest.mark.parametrize("source", ["schedule", "egress-resolution"])
def test_synthetic_follower_waits_for_cancelled_browser_receipt(
        monkeypatch, admitted, source):
    root, runs, held, _, dispatched, _ = _phone_held_context(
        monkeypatch, admitted)
    kills = []

    def kill(generation):
        kills.append(generation)
        if len(kills) == 1:
            raise browser.BrowserUnavailable("Docker kill failed")

    monkeypatch.setattr(browser, "_kill_container_confirmed", kill)
    assert phone_api._cancel_logical_run("t", held.id)[0] == 202
    assert threads._drain_held_browser_events("t") is False
    monkeypatch.setattr(threads, "_require_deep_thread", lambda _tid: None)
    monkeypatch.setattr(threads, "_last_visible_user_timezone", lambda _tid: None)
    monkeypatch.setattr(threads, "_is_pi_thread", lambda _tid: (_ for _ in ()).throw(
        AssertionError("follower reached execution before browser reset")))
    monkeypatch.setattr(threads, "_create_run", lambda tid, text, **kwargs: runs.create(
        tid, "general-agent", text, origin=kwargs.get("origin")))
    if source == "schedule":
        threads._scheduled_dispatch("t", "Check the weather", None)
    else:
        store = EgressStore(str(root / "egress"))
        store.add_pending(EgressRequest(
            host="example.com", port=443, task="Check the report", origin_tid="t",
            created_at=datetime.now(timezone.utc).isoformat()))
        store.resolve(request_key("t", "example.com", 443), "hour")
        monkeypatch.setattr(threads, "EGRESS_STORE", store)
        monkeypatch.setattr(threads, "_mark_urgent", lambda _tid: None)
        threads._dispatch_egress_resolution("t")
    follower = runs.list("t")[-1]
    assert follower.origin == "system" and follower.status == "pending"
    assert dispatched == []
    with pytest.raises(InvalidRunTransition, match="browser safety reset"):
        runs.claim("t", follower.id)
    assert threads._drain_held_browser_events("t") is True
    assert runs.get("t", held.id).cancel_cleanup == "complete"
    assert dispatched
    assert runs.claim("t", follower.id).status == "running"


def test_phone_delete_disappearing_thread_is_404(monkeypatch, admitted):
    root, runs, held, _, _, _ = _phone_held_context(monkeypatch, admitted)
    from contextlib import contextmanager
    from fastapi import HTTPException

    @contextmanager
    def gone(*_args):
        (root / "t").rename(root / "gone")
        raise RuntimeError("browser thread directory unavailable")
        yield

    monkeypatch.setattr(threads.browser_authority, "fence", gone)
    with pytest.raises(HTTPException) as error:
        phone_api._cancel_logical_run("t", held.id)
    assert error.value.status_code == 404


def test_held_phone_promotion_wins_before_delete(monkeypatch, admitted):
    _, runs, held, _, _, _ = _phone_held_context(monkeypatch, admitted)
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _gen: None)
    assert threads._drain_held_browser_events("t") is True
    assert runs.get("t", held.id).status == "pending"
    code, result = phone_api._cancel_logical_run("t", held.id)
    assert code == 200 and result["outcome"] == "cancelled"
    assert runs.get("t", held.id).cancel_cleanup == "complete"
    assert not runs.get("t", held.id).browser_cancel_reset


def test_legacy_unmarked_thread_stays_held_until_migration_scan(monkeypatch, admitted):
    root, runs = admitted
    (root / "t" / authority.STATE_FILE).unlink()
    held, _ = threads._accept_message_run("t", "Read a public page")
    assert held.status == "revocation_pending"
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "reap_orphans",
                        lambda *_args: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("Docker scan failed")))
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda _tid: None)
    assert threads._drain_held_browser_events("t") is False
    assert runs.get("t", held.id).status == "revocation_pending"

    def reconciled(root_dir):
        with authority.fence(root_dir, "t") as state:
            state.mark_covered()

    monkeypatch.setattr(browser.BrowserManager, "reap_orphans", reconciled)
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        lambda *_args: False)
    assert threads._drain_held_browser_events("t") is True
    assert runs.get("t", held.id).status == "pending"


def test_stalled_scoped_docker_scan_does_not_hold_global_user_admission(
        monkeypatch, admitted):
    root, runs = admitted
    (root / "u").mkdir()
    authority.mark_new_thread(str(root), "u")
    old = runs.create("t", "general-agent", "Visit a public site", user_origin=True)
    with authority.fence(str(root), "t") as state:
        state.begin(old.id, old.admission_sequence)
    held, _ = threads._accept_message_run("t", "Read a different page")
    assert held.status == "revocation_pending"
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: None)
    entered, release = Event(), Event()

    def stalled_scan(_root, tid, owner):
        assert (tid, owner) == ("t", old.id)
        entered.set()
        assert release.wait(3)
        with authority.fence(str(root), "t") as state:
            state.clear_reconciled(old.id)
        return False

    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        stalled_scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        resetting = pool.submit(threads._drain_held_browser_events, "t")
        assert entered.wait(3)
        other = pool.submit(threads._accept_message_run, "u", "Hello")
        admitted_u, _ = other.result(timeout=0.5)
        assert admitted_u.status == "pending"
        release.set()
        assert resetting.result(timeout=3) is True


def test_new_held_work_is_requeued_if_first_reset_fails_after_retry_wake(
        monkeypatch):
    tid = "browser-reset-requeue-test"
    entered, release, second = Event(), Event(), Event()
    calls = []

    def reset(_tid):
        assert _tid == tid
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
            return False
        second.set()
        return True

    monkeypatch.setattr(threads, "_drain_held_browser_events", reset)
    threads._queue_browser_revocation(tid)
    assert entered.wait(3)
    # Models a newer held event or a retry timer firing while the first
    # bounded reset is still in progress.
    threads._queue_browser_revocation(tid)
    release.set()
    assert second.wait(3)
    assert len(calls) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_real_approval_resolution_creates_no_internal_browser_authority(
        monkeypatch, admitted, mixed):
    root, runs = admitted
    direct = runs.create("t", "general-agent", "Visit host.docker.internal",
                         user_origin=True)
    store = EgressStore(str(root / "egress"))
    hosts = ["one.example.com", "two.example.com"] if mixed else ["one.example.com"]
    for host in hosts:
        store.add_pending(EgressRequest(
            host=host, port=443, task=direct.text, origin_tid="t",
            created_at=datetime.now(timezone.utc).isoformat()))
        store.resolve(request_key("t", host, 443), "hour")
    monkeypatch.setattr(threads, "EGRESS_STORE", store)
    monkeypatch.setattr(threads, "_require_deep_thread", lambda _tid: None)
    monkeypatch.setattr(threads, "_execute_run", lambda *_args: None)
    monkeypatch.setattr(threads, "_mark_urgent", lambda _tid: None)
    threads._dispatch_egress_resolution("t")
    synthetic = runs.list("t")[-1]
    assert synthetic.origin == "system"
    assert "host.docker.internal" in synthetic.text
    assert threads._browser_user_request(synthetic, runs.list("t")) is None
    session = browser.BrowserSession("t", synthetic.id, str(root), str(root), None)
    monkeypatch.setattr(browser, "_load_egress_allowlist",
                        lambda: ["host.docker.internal"])
    with pytest.raises(browser.BrowserUnavailable, match="fresh user request"):
        session._ensure_mode("http://host.docker.internal/")
