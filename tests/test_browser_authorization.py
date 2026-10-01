"""Exact browser generation and held-event admission boundaries."""
import json
import shutil
from types import SimpleNamespace
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock

import pytest

from assist.browser import manager as browser
from assist.browser import authority
from assist.run_service import RunService
from manage.web import threads
from manage.web.threads import _browser_user_request


@pytest.mark.parametrize("sequence,valid", [(True, False), (False, False), (1, True)])
def test_persisted_browser_lease_requires_numeric_sequence(tmp_path, sequence, valid):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / authority.STATE_FILE).write_text(json.dumps({
        "covered": True,
        "lease": {"owner_run_id": "run", "sequence": sequence, "generations": []},
    }), encoding="utf-8")
    if valid:
        with authority.fence(str(tmp_path), "t") as state:
            assert state.lease["sequence"] == 1
    else:
        with pytest.raises(RuntimeError, match="browser lease is invalid"):
            with authority.fence(str(tmp_path), "t"):
                pass


@pytest.mark.parametrize("url", [
    "http://@host.docker.internal/",
    "http://user:@host.docker.internal/",
    "http://host.docker.internal:0/",
])
def test_manager_rejects_empty_userinfo_and_port_zero(url):
    with pytest.raises(browser.BrowserUnavailable, match="HTTP"):
        browser._url_target(url)


def test_browser_request_is_scoped_to_active_user_work(tmp_path):
    (tmp_path / "t").mkdir()
    runs = RunService(str(tmp_path))
    internal = runs.create("t", "general-agent", "Visit host.docker.internal",
                           user_origin=True)
    public = runs.create("t", "general-agent", "Read example.com",
                         work_id=internal.work_id, user_origin=True)
    assert _browser_user_request(public, runs.list("t")).text == "Read example.com"
    resumed_public = runs.create("t", "general-agent", None,
                                 work_id=public.work_id, resume=True,
                                 user_event_id=public.user_event_id)
    assert _browser_user_request(resumed_public, runs.list("t")).event_id == public.id
    # Approval-created synthetic work has copied task text but no user-origin
    # authorization, whether its batch is mixed or contains one request.
    synthetic = runs.create("t", "general-agent", internal.text,
                            origin="system", work_id=internal.work_id)
    assert _browser_user_request(synthetic, runs.list("t")) is None
    resume_internal = runs.create("t", "general-agent", None,
                                  work_id=internal.work_id, resume=True,
                                  user_event_id=internal.user_event_id)
    # A reused work_id has an intervening public request, so even this
    # successor cannot borrow the older internal request.
    assert _browser_user_request(resume_internal, runs.list("t")) is None
    same_timestamp = [replace(entry, created_at=internal.created_at)
                      for entry in runs.list("t")]
    assert _browser_user_request(resume_internal, same_timestamp) is None


def test_copied_text_and_approval_origin_cannot_mint_internal_authority(tmp_path):
    (tmp_path / "t").mkdir()
    runs = RunService(str(tmp_path))
    direct = runs.create("t", "general-agent", "Visit host.docker.internal",
                         user_origin=True)
    copied = runs.create("t", "general-agent", direct.text,
                         work_id="unrelated")
    assert _browser_user_request(copied, runs.list("t")) is None
    for text in [direct.text, "Read example.com and visit host.docker.internal"]:
        approval = runs.create("t", "general-agent", text, origin="system",
                               work_id=direct.work_id)
        assert _browser_user_request(approval, runs.list("t")) is None
    resumed = runs.create("t", "general-agent", None, work_id=direct.work_id,
                          resume=True, user_event_id=direct.user_event_id)
    assert _browser_user_request(resumed, runs.list("t")).event_id == direct.id


def test_direct_owner_events_have_durable_order_and_held_state(tmp_path):
    (tmp_path / "t").mkdir()
    runs = RunService(str(tmp_path))
    first = runs.create("t", "general-agent", "Visit host.docker.internal",
                        user_origin=True)
    held = runs.create("t", "general-agent", "Read public status",
                       user_origin=True, revocation_pending=True)
    third = runs.create("t", "general-agent", "Visit host.docker.internal",
                        user_origin=True, revocation_pending=True)
    assert [run.admission_sequence for run in (first, held, third)] == [1, 2, 3]
    assert held.status == third.status == "revocation_pending"
    assert runs.list("t")[1].status == "revocation_pending"
    promoted = runs.transition("t", held.id, "pending",
                               browser_reset_notice=True)
    assert promoted.admission_sequence == 2
    assert promoted.browser_reset_notice is True
    assert runs.get("t", third.id).status == "revocation_pending"


def test_held_event_fences_old_browser_tool_before_worker_runs(tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    runs = RunService(str(tmp_path))
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    session = browser.BrowserSession(
        "t", old.id, str(tmp_path), str(tmp_path),
        browser.BrowserUserRequest(old.id, old.work_id, old.text,
                                   old.admission_sequence), work_id=old.work_id,
        sandbox_generation="old-generation", run_service=runs)
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.2", "old-generation")
    with authority.fence(str(tmp_path), "t") as state:
        state.begin(old.id, old.admission_sequence)
        state.add_generation(old.id, "old-generation")
    runs.create("t", "general-agent", "Read a public page",
                user_origin=True, revocation_pending=True)
    with pytest.raises(browser.BrowserUnavailable, match="newer user message"):
        session.command("observe", page_id="old-page")


def test_held_events_promote_without_cross_work_consent_rebind(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    runs = RunService(str(tmp_path))
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    first = runs.create("t", "general-agent", "Read a public page",
                        user_origin=True, revocation_pending=True)
    second = runs.create("t", "general-agent", "Yes, please visit host.docker.internal",
                         user_origin=True, revocation_pending=True)
    session = browser.BrowserSession(
        "t", old.id, str(tmp_path), str(tmp_path),
        browser.BrowserUserRequest(old.id, old.work_id, old.text,
                                   old.admission_sequence), work_id=old.work_id,
        sandbox_generation="old-generation", run_service=runs)
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.2", "old-generation")
    resets = []

    def revoke():
        resets.append(session.identity is not None)
        session.identity = None
        session.user_request = None
        return resets[-1]

    monkeypatch.setattr(session, "revoke_for_promotion", revoke)
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: session)
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        lambda *_args: False)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote", lambda _tid: None)
    monkeypatch.setattr(threads.THREAD_QUEUE, "promote", lambda _tid: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda _tid: None)
    monkeypatch.setattr(threads, "_queue_browser_revocation", lambda _tid: None)
    threads._drain_held_browser_events("t")
    assert [runs.get("t", run.id).status for run in (first, second)] == [
        "pending", "revocation_pending"]
    threads._drain_held_browser_events("t")
    assert resets == [True, False]
    assert [runs.get("t", run.id).status for run in (first, second)] == [
        "pending", "pending"]
    assert runs.get("t", first.id).browser_reset_notice is True
    assert session.user_request is None


def test_failed_reset_keeps_message_held_with_retry(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    runs = RunService(str(tmp_path))
    held = runs.create("t", "general-agent", "Read a public page",
                       user_origin=True, revocation_pending=True)
    session = browser.BrowserSession("t", held.id, str(tmp_path), str(tmp_path), None)
    monkeypatch.setattr(session, "revoke_for_promotion",
                        lambda: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("kill failed")))
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: session)
    scheduled = []
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda tid: scheduled.append(tid))
    threads._drain_held_browser_events("t")
    record = runs.get("t", held.id)
    assert record.status == "revocation_pending"
    assert record.revocation_retry_at is not None
    assert "unconfirmed" in record.error
    assert scheduled == ["t"]


def test_failed_rebind_does_not_promote_held_event(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    runs = RunService(str(tmp_path))
    held = runs.create("t", "general-agent", "Visit host.docker.internal",
                       user_origin=True, revocation_pending=True)
    session = browser.BrowserSession("t", held.id, str(tmp_path), str(tmp_path), None)
    monkeypatch.setattr(session, "revoke_for_promotion", lambda: True)
    monkeypatch.setattr(session, "rebind_user_request",
                        lambda _request: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("session closed")))
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: session)
    scheduled = []
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda tid: scheduled.append(tid))
    threads._drain_held_browser_events("t")
    assert runs.get("t", held.id).status == "revocation_pending"
    assert scheduled == ["t"]


def test_inflight_internal_command_finishes_before_held_event_promotes(
        monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    runs = RunService(str(tmp_path))
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    session = browser.BrowserSession(
        "t", old.id, str(tmp_path), str(tmp_path),
        browser.BrowserUserRequest(old.id, old.work_id, old.text,
                                   old.admission_sequence), work_id=old.work_id,
        sandbox_generation="old-generation", run_service=runs)
    session.identity = browser._ContainerIdentity(
        str(tmp_path), "172.20.0.2", "old-generation")
    with authority.fence(str(tmp_path), "t") as state:
        state.begin(old.id, old.admission_sequence)
        state.add_generation(old.id, "old-generation")
    entered, release, revoked = Event(), Event(), Event()

    def inflight(_generation, _command, **_kwargs):
        entered.set()
        assert release.wait(3)
        return b'{"result": {"snapshot": "old page"}}'

    def revoke():
        session.identity = None
        session.user_request = None
        revoked.set()
        return True

    monkeypatch.setattr(browser, "_docker_exec", inflight)
    monkeypatch.setattr(session, "revoke_for_promotion", revoke)
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: session)
    def confirmed_stop(*_args):
        with authority.fence(str(tmp_path), "t") as state:
            state.clear_reconciled(old.id)
        return False
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        confirmed_stop)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote", lambda _tid: None)
    monkeypatch.setattr(threads.THREAD_QUEUE, "promote", lambda _tid: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda _tid: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        command = pool.submit(session.command, "observe", page_id="old")
        assert entered.wait(3)
        held = runs.create("t", "general-agent", "Read a public page",
                           user_origin=True, revocation_pending=True)
        drain = pool.submit(threads._drain_held_browser_events, "t")
        assert not revoked.wait(0.1)
        assert runs.get("t", held.id).status == "revocation_pending"
        release.set()
        assert command.result(timeout=3)["result"]["snapshot"] == "old page"
        drain.result(timeout=3)
    assert revoked.is_set()
    assert runs.get("t", held.id).status == "pending"


def test_held_event_stays_queued_while_browser_startup_holds_thread_gate(
        monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    runs = RunService(str(tmp_path))
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    session = browser.BrowserSession(
        "t", old.id, str(tmp_path), str(tmp_path),
        browser.BrowserUserRequest(old.id, old.work_id, old.text,
                                   old.admission_sequence), work_id=old.work_id,
        sandbox_generation="generation", run_service=runs)
    with authority.fence(str(tmp_path), "t") as state:
        state.begin(old.id, old.admission_sequence)
        state.add_generation(old.id, "generation")
    entered, release = Event(), Event()
    real_gate = RLock()

    class ShortWaitGate:
        def acquire(self, timeout):
            assert timeout == 20
            return real_gate.acquire(timeout=0.1)

        def release(self):
            real_gate.release()

    def stalled_start(_argv, **kwargs):
        timeout = kwargs["timeout"]
        assert timeout <= 20
        entered.set()
        assert release.wait(3)
        raise browser.BrowserUnavailable("startup transport timed out")

    monkeypatch.setattr(browser.BrowserManager, "thread_gate",
                        classmethod(lambda _cls, _tid: ShortWaitGate()))
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: session)
    monkeypatch.setattr(browser, "configured_directory", lambda: str(tmp_path))
    monkeypatch.setattr(browser.SandboxManager, "current_container",
                        lambda _work_dir: SimpleNamespace(id="generation"))
    monkeypatch.setattr(browser.SandboxManager, "_egress_client_ips", {
        str(tmp_path): (str(tmp_path), "172.20.0.9", "generation")})
    monkeypatch.setattr(browser, "_kill_container_confirmed", lambda _gen: None)
    monkeypatch.setattr(browser, "forget_client", lambda *_args: None)
    monkeypatch.setattr(browser, "disarm_sandbox_browser", lambda *_args: None)
    monkeypatch.setattr(browser, "_bounded_cli", stalled_start)
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    scheduled = []
    monkeypatch.setattr(threads, "_schedule_browser_revocation_retry",
                        lambda tid: scheduled.append(tid))
    with ThreadPoolExecutor(max_workers=1) as pool:
        opening = pool.submit(session.command, "open",
                              url="http://host.docker.internal/")
        assert entered.wait(3)
        held = runs.create("t", "general-agent", "Read a public page",
                           user_origin=True, revocation_pending=True)
        threads._drain_held_browser_events("t")
        assert runs.get("t", held.id).status == "revocation_pending"
        assert scheduled == ["t"]
        release.set()
        with pytest.raises(browser.BrowserUnavailable, match="startup failed"):
            opening.result(timeout=3)


def test_recovery_confirms_saved_owner_without_session(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    runs = RunService(str(tmp_path))
    old = runs.create("t", "general-agent", "Visit host.docker.internal",
                      user_origin=True)
    held = runs.create("t", "general-agent", "Read a public page",
                       user_origin=True, revocation_pending=True,
                       browser_reset_run_id=old.id)
    confirmed = []
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(browser.BrowserManager, "current_session",
                        lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        lambda root, tid, owner: confirmed.append((root, tid, owner)))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "promote", lambda _tid: None)
    monkeypatch.setattr(threads.THREAD_QUEUE, "promote", lambda _tid: None)
    monkeypatch.setattr(threads, "_dispatch_pending_after", lambda _tid: None)
    threads._drain_held_browser_events("t")
    assert confirmed == [(str(tmp_path), "t", old.id)]
    assert runs.get("t", held.id).status == "pending"


def test_thread_deletion_stops_browser_before_erasing_run_journal(
        monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    runs = RunService(str(tmp_path))
    held = runs.create("t", "general-agent", "Read a public page",
                       user_origin=True, revocation_pending=True)
    calls = []
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads._PI_RUNTIME, "retire", lambda _tid: None)
    monkeypatch.setattr(threads.RUN_STREAMS, "mark_thread_gone", lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "cleanup",
                        lambda _tid: calls.append("browser"))
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        lambda *_args: calls.append("proof"))
    monkeypatch.setattr(threads.MANAGER, "_hard_delete_after_browser_stop",
                        lambda _tid, on_delete=None: calls.append("journal"))
    threads._delete_thread_and_children("t")
    assert calls == ["browser", "proof", "journal"]
    assert runs.get("t", held.id).status == "revocation_pending"


def test_failed_browser_stop_prevents_thread_deletion(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    calls = []
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads._PI_RUNTIME, "retire", lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "cleanup",
                        lambda _tid: (_ for _ in ()).throw(
                            browser.BrowserUnavailable("stop unconfirmed")))
    monkeypatch.setattr(threads.MANAGER, "_hard_delete_after_browser_stop",
                        lambda *_args, **_kwargs: calls.append("journal"))
    with pytest.raises(browser.BrowserUnavailable, match="stop unconfirmed"):
        threads._delete_thread_and_children("t")
    assert calls == []


def test_deletion_gate_rejects_late_browser_registration(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    entered, release = Event(), Event()
    monkeypatch.setattr(threads, "_runs", lambda: RunService(str(tmp_path)))
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads._PI_RUNTIME, "retire", lambda _tid: None)
    monkeypatch.setattr(threads.RUN_STREAMS, "mark_thread_gone", lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "cleanup", lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        lambda *_args: None)

    def erase(_tid, on_delete=None):
        entered.set()
        assert release.wait(3)
        shutil.rmtree(tmp_path / "t")

    monkeypatch.setattr(threads.MANAGER, "_hard_delete_after_browser_stop", erase)
    session = browser.BrowserSession(
        "t", "late-run", str(tmp_path), str(tmp_path), None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(threads._delete_thread_and_children, "t")
        assert entered.wait(3)
        registering = pool.submit(browser.BrowserManager.register, session)
        assert not registering.done()
        release.set()
        deleting.result(timeout=3)
        with pytest.raises(browser.BrowserUnavailable, match="no longer exists"):
            registering.result(timeout=3)


def test_deletion_waits_for_pending_managed_generation(monkeypatch, tmp_path):
    (tmp_path / "t").mkdir()
    authority.mark_new_thread(str(tmp_path), "t")
    started = Event()
    calls = []
    monkeypatch.setattr(threads, "_runs", lambda: RunService(str(tmp_path)))
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(threads._PI_RUNTIME, "retire", lambda _tid: started.set())
    monkeypatch.setattr(threads.RUN_STREAMS, "mark_thread_gone", lambda _tid: None)
    monkeypatch.setattr(browser.BrowserManager, "cleanup",
                        lambda _tid: calls.append("cleanup"))
    monkeypatch.setattr(browser.BrowserManager, "confirm_owner_stopped",
                        lambda *_args: calls.append("stop-proof"))
    monkeypatch.setattr(threads.MANAGER, "_hard_delete_after_browser_stop",
                        lambda _tid, on_delete=None: (
                            calls.append("delete"), shutil.rmtree(tmp_path / "t")))
    with ThreadPoolExecutor(max_workers=1) as pool:
        with authority.generation_fence(str(tmp_path), "t"):
            deleting = pool.submit(threads._delete_thread_and_children, "t")
            assert started.wait(3)
            assert not deleting.done()
            assert calls == []
        deleting.result(timeout=3)
    assert calls == ["cleanup", "stop-proof", "delete"]
