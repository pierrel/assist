"""SMS route authentication, durable archive/Run admission and reply approval.

Storage and admission run for real; _execute_run is stubbed so these tests
need neither a model, sandbox container nor a real SMS transport.
"""
import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient
from tests.approval_helpers import checkpoint_chat

from manage import web
from manage.web import threads
from manage.web.state import _set_status


@pytest.fixture
def client(tmp_path, monkeypatch):
    tdir = tmp_path / "t-sub"
    tdir.mkdir()
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    from assist.events.model import Subscription
    from assist.events.store import SubscriptionStore
    store = SubscriptionStore(str(tmp_path))
    store.add(Subscription("subscription", "t-sub", ".*", "from {sender}: {text}", "2026-10-20T00:00:00+00:00"))
    monkeypatch.setattr(threads, "SUBSCRIPTION_STORE", store)
    monkeypatch.setenv("ASSIST_SMS_ARCHIVE_DIR", str(tmp_path / "sms-archive"))
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(tmp_path / "thread-workspaces"))
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda rid, tid: threads._execute_run(rid, tid))
    return TestClient(web.app)


def _body(mid="abc123", sender="+15551234567", text="hi"):
    return {"message_id": mid, "sender": sender, "text": text, "occurred_at": "2026-10-21T00:00:00+00:00"}


def _reply_chat(text):
    """A route-only fake with the exact gate now required by shared admission."""
    return checkpoint_chat(SimpleNamespace(
        thread_id="t-sub", pending_reply=lambda: {"text": text},
        pending_actions=lambda: [{"name": "send_reply", "args": {"text": text}}]))


def test_inbound_503_when_secret_unset(client, monkeypatch):
    monkeypatch.delenv("ASSIST_SMS_SECRET", raising=False)
    assert client.post("/inbound/sms", json=_body()).status_code == 503


def test_inbound_401_bad_secret(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    r = client.post("/inbound/sms", json=_body(), headers={"X-Assist-SMS-Secret": "wrong"})
    assert r.status_code == 401
    r2 = client.post("/inbound/sms", json=_body())  # missing header
    assert r2.status_code == 401


def test_inbound_accepts_and_dispatches(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    calls = []
    def execute(run_id, tid):
        run = threads._runs().get(tid, run_id)
        calls.append((run.sender, run.text))
    monkeypatch.setattr(threads, "_execute_run", execute)
    r = client.post("/inbound/sms", json=_body(), headers={"X-Assist-SMS-Secret": "s3cret"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    assert calls == [("+15551234567", "from +15551234567: hi")]


def test_inbound_dedup_same_message_id(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    calls = []
    monkeypatch.setattr(threads, "_execute_run", lambda *_: calls.append(1))
    h = {"X-Assist-SMS-Secret": "s3cret"}
    first = client.post("/inbound/sms", json=_body(mid="dup1"), headers=h)
    second = client.post("/inbound/sms", json=_body(mid="dup1"), headers=h)
    assert first.json()["status"] == "accepted"
    assert second.json()["status"] == "duplicate"
    assert len(calls) == 1                       # the duplicate did NOT re-dispatch


def test_archive_acceptance_interruption_recovers_same_run_once(client, monkeypatch):
    from assist.events.sms_archive import configured_archive
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    original = threads._create_run
    monkeypatch.setattr(threads, "_create_run", lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")))
    h = {"X-Assist-SMS-Secret": "s3cret"}
    assert client.post("/inbound/sms", json=_body(), headers=h).status_code == 503
    pending = configured_archive().read("abc123")
    assert pending["handling"] == "routed"
    monkeypatch.setattr(threads, "_create_run", original)
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    assert client.post("/inbound/sms", json=_body(), headers=h).status_code == 200
    assert client.post("/inbound/sms", json=_body(), headers=h).json()["status"] == "duplicate"
    assert len(queued) == 1 and len(threads._runs().list("t-sub")) == 1
    assert configured_archive().read("abc123")["handling"] == "routed"


def test_interruption_after_run_creation_does_not_create_second_run(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    original = threads._create_run
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("interrupted after ordinary Run commit")
    monkeypatch.setattr(threads, "_create_run", crash)
    h = {"X-Assist-SMS-Secret": "s3cret"}
    assert client.post("/inbound/sms", json=_body(), headers=h).status_code == 503
    first = threads._runs().list("t-sub")[0]
    monkeypatch.setattr(threads, "_create_run", original)
    monkeypatch.setattr(threads, "_execute_run", lambda *a: None)
    assert client.post("/inbound/sms", json=_body(), headers=h).status_code == 200
    assert [run.id for run in threads._runs().list("t-sub")] == [first.id]


def test_unmatched_archived_and_outgoing_receipts_never_dispatch(client, monkeypatch):
    from assist.events.sms_archive import configured_archive
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    monkeypatch.setattr(threads.SUBSCRIPTION_STORE, "route", lambda _, **kwargs: None)
    monkeypatch.setattr(threads, "_execute_run", lambda *a: pytest.fail("unmatched receipt dispatched"))
    h = {"X-Assist-SMS-Secret": "s3cret"}
    assert client.post("/inbound/sms", json=_body(), headers=h).status_code == 200
    assert configured_archive().read("abc123")["handling"] == "archive_only"
    body = {"message_id": "outgoing1", "number": "+15551234567", "text": 'exact "quote"', "status": "sent"}
    assert client.post("/sms/receipts", json=body, headers=h).json()["status"] == "sent"
    assert client.post("/sms/receipts", json={**body, "status": "unknown"}, headers=h).json()["status"] == "sent"
    assert client.post("/sms/receipts", json={**body, "text": "changed"}, headers=h).status_code == 409
    assert client.post("/sms/receipts", json=body).status_code == 401
    assert configured_archive().read("outgoing1")["text"] == 'exact "quote"'


def test_sms_payload_bounded_and_archive_not_in_triage(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    assert client.post("/inbound/sms", content=b"x" * (2 * 1024 * 1024 + 1),
                       headers={"X-Assist-SMS-Secret": "s3cret"}).status_code == 413
    from assist import thread_manager as tm
    assert {f.__name__ for f in tm._web_tools} >= {"search_sms_messages", "read_sms_message"}
    assert not {"search_sms_messages", "read_sms_message"} & {f.__name__ for f in tm._web_triage_tools}


def test_inbound_400_bad_message_id(client, monkeypatch):
    monkeypatch.setenv("ASSIST_SMS_SECRET", "s3cret")
    r = client.post("/inbound/sms", json=_body(mid="../etc/passwd"),
                    headers={"X-Assist-SMS-Secret": "s3cret"})
    assert r.status_code == 400


def test_dispatch_no_matching_subscription_is_noop(client, monkeypatch):
    seen = []
    monkeypatch.setattr(threads.SUBSCRIPTION_STORE, "route", lambda sender: None)
    monkeypatch.setattr(threads, "_process_message", lambda *a, **k: seen.append(1))
    threads._dispatch_event("+1999", "hi")
    assert seen == []                            # no subscription → no turn


def test_reply_decision_gated_on_awaiting_approval(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: _reply_chat("draft"))
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    # not awaiting → 409
    _set_status("t-sub", "ready")
    assert client.post("/thread/t-sub/reply/approve").status_code == 409
    # awaiting → 303 + resume queued with an approve decision + the stored sender
    _set_status("t-sub", "awaiting_approval", pending_reply="draft", pending_sender="+1555")
    token = threads.reply_approval_preview("t-sub")["token"]
    r = client.post("/thread/t-sub/reply/approve", data={"token": token}, follow_redirects=False)
    assert r.status_code == 303
    assert len(queued) == 1
    run = threads._runs().get("t-sub", queued[0][0])
    assert queued[0][1] == "t-sub"
    assert run.sender == "+1555" and run.resume_decision["type"] == "approve"
    assert run.resume_decision["reply_review"]["text"] == "draft"
    assert run.resume_decision["reply_review"]["sender"] == "+1555"


def test_reply_decision_bad_verb(client, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    _set_status("t-sub", "awaiting_approval", pending_reply="d", pending_sender="+1")
    assert client.post("/thread/t-sub/reply/nonsense").status_code == 400


def test_dispatch_calls_process_message_with_rendered_template(client, monkeypatch):
    from assist.events.model import Subscription
    sub = Subscription(id="s", thread_id="t-sub", sender_regexp=".*", template="from {sender}: {text}")
    monkeypatch.setattr(threads.SUBSCRIPTION_STORE, "route", lambda sender: sub)
    calls = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: calls.append(a))
    threads._dispatch_event("+1555", "hello")
    assert len(calls) == 1                                # supersede now lives in _process_message
    run = threads._runs().get("t-sub", calls[0][0])
    assert calls[0][1] == "t-sub" and "hello" in run.text
    assert run.sender == "+1555"


def test_triage_tools_exclude_host_effect_tools():
    # The untrusted-SMS triage turn must NOT get the host-effect config tools (schedule/
    # subscription) — only the HITL-gated reply. Normal turns keep the config tools.
    from assist import thread_manager as tm
    triage = {getattr(f, "__name__", "") for f in tm._web_triage_tools}
    normal = {getattr(f, "__name__", "") for f in tm._web_tools}
    assert "send_reply" in triage
    assert "send_email" not in triage
    assert "create_subscription" not in triage and "create_schedule" not in triage
    assert "delete_subscription" not in triage
    assert "create_subscription" in normal and "send_reply" not in normal
    assert "send_email" in normal
    assert "get_location" in normal and "get_location" not in triage
    assert "send_email" in tm._web_interrupt_on
    assert "send_reply" in tm._web_triage_interrupt_on


def test_reply_approve_refuses_superseded_draft(client, monkeypatch):
    from manage.web.state import _set_status
    monkeypatch.setattr(threads, "_existing_thread_dir", lambda tid: str(tid))
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: _reply_chat("NEW draft"))
    queued = []
    monkeypatch.setattr(threads, "_process_message", lambda *a, **k: queued.append(1))
    _set_status("t-sub", "awaiting_approval", pending_reply="NEW draft", pending_sender="+1")
    # user approves the OLD draft they saw → mismatch → 409, nothing queued
    r = client.post("/thread/t-sub/reply/approve", data={"seen": "OLD draft"})
    assert r.status_code == 409 and queued == []
    # approving the current draft goes through
    token = threads.reply_approval_preview("t-sub")["token"]
    r2 = client.post("/thread/t-sub/reply/approve", data={"seen": "NEW draft", "token": token},
                     follow_redirects=False)
    assert r2.status_code == 303 and len(queued) == 1


def test_email_approval_requires_its_token_and_exact_review(client, monkeypatch):
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    _set_status("t-sub", "awaiting_approval", pending_email_to="a@example.test",
                pending_email_subject="Subject", pending_email_body="Body",
                pending_email_token="approval-token")

    token = threads.email_approval_preview(threads._get_status("t-sub"))["token"]
    missing = client.post("/thread/t-sub/email/approve")
    assert missing.status_code == 409
    stale = client.post("/thread/t-sub/email/approve", data={
        "token": token, "seen_to": "other@example.test",
        "seen_subject": "Subject", "seen_body": "Body"})
    assert stale.status_code == 409
    changed = client.post("/thread/t-sub/email/approve", data={
        "token": token, "to": "other@example.test", "subject": "Subject",
        "body": "Body", "seen_to": "a@example.test", "seen_subject": "Subject",
        "seen_body": "Body"})
    assert changed.status_code == 409
    approved = client.post("/thread/t-sub/email/approve", data={
        "token": token, "to": "a@example.test", "subject": "Subject",
        "body": "Body", "seen_to": "a@example.test",
        "seen_subject": "Subject", "seen_body": "Body"}, follow_redirects=False)
    assert approved.status_code == 303 and len(queued) == 1
    run = threads._runs().get("t-sub", queued[0][0])
    assert run.resume_decision == {"type": "approve",
        "email_review_identity": list(threads.email_identity() or ("", ""))}


def test_email_unicode_token_refuses_without_consuming_proposal(client):
    _set_status("t-sub", "awaiting_approval", pending_email_to="a@example.test",
                pending_email_subject="Subject", pending_email_body="Body",
                pending_email_token="approval-token")
    before = threads._get_status("t-sub")

    response = client.post("/thread/t-sub/email/reject", data={"token": "é"})

    assert response.status_code == 409
    assert threads._get_status("t-sub") == before
    assert threads._runs().list("t-sub") == []


def test_email_edit_rewrites_only_user_editable_fields(client, monkeypatch):
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    monkeypatch.setattr(web.MANAGER, "get", lambda tid, **k: object())
    queued = []
    monkeypatch.setattr(threads, "_execute_run", lambda *a: queued.append(a))
    _set_status("t-sub", "awaiting_approval", pending_email_to="a@example.test",
                pending_email_subject="Subject", pending_email_body="Body",
                pending_email_token="approval-token")

    token = threads.email_approval_preview(threads._get_status("t-sub"))["token"]
    edited = client.post("/thread/t-sub/email/edit", data={
        "token": token, "to": "b@example.test", "subject": "Edited",
        "body": "Edited body"}, follow_redirects=False)

    assert edited.status_code == 303 and len(queued) == 1
    run = threads._runs().get("t-sub", queued[0][0])
    assert run.resume_decision == {
        "type": "edit", "edited_action": {"name": "send_email", "args": {
            "to": "b@example.test", "subject": "Edited", "body": "Edited body"}},
        "email_review_identity": list(threads.email_identity() or ("", ""))}


def test_message_post_refuses_while_email_is_awaiting_approval(client):
    _set_status("t-sub", "awaiting_approval", pending_email_token="approval-token")

    response = client.post("/thread/t-sub/message", data={"text": "new message"})

    assert response.status_code == 409


def test_email_approval_card_renders_full_message(client, monkeypatch):
    monkeypatch.setattr(
        web.MANAGER, "get", lambda tid, sandbox_backend=None, **k:
        type("C", (), {"get_messages": lambda self: [],
                        "pending_reply": lambda self: None})())
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda tid: "Thread")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    _set_status("t-sub", "awaiting_approval", pending_email_to="to@example.test",
                pending_email_subject="Subject", pending_email_body="A full\nmessage",
                pending_email_token="approval-token")

    page = client.get("/thread/t-sub").text

    assert "Email awaiting your approval" in page
    assert "Assistant &lt;assistant@example.test&gt;" in page
    assert "oversight@example.test" in page
    assert "A full\nmessage" in page
    token = threads.email_approval_preview(threads._get_status("t-sub"))["token"]
    assert f'name="token" value="{token}"' in page


@pytest.mark.parametrize("decision", ["approve", "edit"])
@pytest.mark.parametrize("setting", ["EMAIL_FROM_ADDRESS", "EMAIL_ALWAYS_CC"])
def test_browser_email_rejects_identity_changed_since_render(client, monkeypatch, decision, setting):
    import re
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    monkeypatch.setattr(web.MANAGER, "get", lambda *a, **k: type("C", (), {
        "get_messages": lambda self: [], "pending_reply": lambda self: None})())
    monkeypatch.setattr(threads, "get_cached_description", lambda tid: "Thread")
    _set_status("t-sub", "awaiting_approval", pending_email_token="stored-token",
                pending_email_to="to@example.test", pending_email_subject="Subject",
                pending_email_body="Body")
    before = threads._get_status("t-sub")
    page = client.get("/thread/t-sub").text
    token = re.search(r'name="token" value="([A-Za-z0-9_-]+)"', page).group(1)
    monkeypatch.setenv(setting, "changed@example.test")

    response = client.post(f"/thread/t-sub/email/{decision}", data={
        "token": token, "to": "to@example.test", "subject": "Subject", "body": "Body",
        "seen_to": "to@example.test", "seen_subject": "Subject", "seen_body": "Body"})

    assert response.status_code == 409
    assert threads._get_status("t-sub") == before
    assert threads._runs().list("t-sub") == []


@pytest.mark.parametrize('sender', ['T-Mobile', 'CITY BANK', ''])
def test_incoming_nonnumeric_or_unknown_sender_keeps_archive_and_triage(client, monkeypatch, sender):
    from assist.events.sms_archive import configured_archive
    monkeypatch.setenv('ASSIST_SMS_SECRET', 's3cret')
    observed = []
    def execute(run_id, tid): observed.append(threads._runs().get(tid, run_id).sender)
    monkeypatch.setattr(threads, '_execute_run', execute)
    response = client.post('/inbound/sms', json=_body(sender=sender, text=''), headers={'X-Assist-SMS-Secret': 's3cret'})
    assert response.status_code == 200
    assert configured_archive().read('abc123')['number'] == sender
    assert observed == [sender or '(unknown SMS sender)']
    assert bool(observed[0])  # Never falls through to normal full-authority profile.



def test_paused_incoming_queues_behind_its_serial_resume(client, monkeypatch):
    monkeypatch.setenv('ASSIST_SMS_SECRET', 's3cret')
    monkeypatch.setattr(threads, '_get_status', lambda tid: {'stage':'paused'})
    queued=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER, 'submit', lambda *args: queued.append(args))
    monkeypatch.setattr(threads, '_execute_run', lambda *args: pytest.fail('paused checkpoint bypassed scheduler'))
    response=client.post('/inbound/sms',json=_body(),headers={'X-Assist-SMS-Secret':'s3cret'})
    assert response.status_code == 200 and len(queued) == 1
    assert threads._runs().get('t-sub',queued[0][0]).sender == '+15551234567'


def test_replayed_admission_cannot_recover_an_actively_running_run(client,monkeypatch):
    from assist.events.sms_archive import SmsArchive
    monkeypatch.setenv('ASSIST_SMS_SECRET','s3cret')
    original=threads._create_run
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('crash after Run commit')
    monkeypatch.setattr(threads,'_create_run',crash)
    headers={'X-Assist-SMS-Secret':'s3cret'}
    assert client.post('/inbound/sms',json=_body(),headers=headers).status_code==503
    run=threads._runs().list('t-sub')[0]
    threads._runs().claim('t-sub',run.id)
    monkeypatch.setattr(threads,'_create_run',original)
    queued=[];monkeypatch.setattr(threads._RESUME_SCHEDULER,'submit',lambda *args:queued.append(args))
    monkeypatch.setattr(threads,'_recover_run',lambda *args,**kwargs:pytest.fail('live recovery interrupted'))
    assert client.post('/inbound/sms',json=_body(),headers=headers).status_code==200
    assert threads._runs().get('t-sub',run.id).status=='running'
    assert queued==[]
    threads._runs().transition('t-sub',run.id,'success')
    threads._execute_run(run.id, 't-sub')
    assert threads._runs().get('t-sub',run.id).status=='success'


def test_lost_ack_retry_after_destination_deletion_is_archived_only(client, monkeypatch):
    import shutil
    from pathlib import Path
    from assist.events.sms_archive import configured_archive
    monkeypatch.setenv('ASSIST_SMS_SECRET','s3cret')
    monkeypatch.setattr(threads,'_execute_run',lambda *args: None)
    headers={'X-Assist-SMS-Secret':'s3cret'}
    assert client.post('/inbound/sms',json=_body(),headers=headers).status_code==200
    before=configured_archive().read('abc123')
    assert len(threads._runs().list('t-sub'))==1
    destination=Path(threads.MANAGER.thread_dir('t-sub'))
    shutil.rmtree(destination)  # Same storage removal as permanent thread deletion.
    original=threads._create_run
    created=[]
    def create(*args,**kwargs):
        created.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(threads,'_create_run',create)
    response=client.post('/inbound/sms',json=_body(),headers=headers)
    assert response.status_code==200 and response.json()['status']=='duplicate'
    assert configured_archive().read('abc123')==before
    assert not destination.exists() and created==[]


@pytest.mark.parametrize('failure',['root-missing','permission','io','dangling-link','file'])
def test_missing_or_invalid_destination_storage_is_not_silently_acknowledged(client,monkeypatch,failure):
    import os
    import errno
    import shutil
    from pathlib import Path
    monkeypatch.setenv('ASSIST_SMS_SECRET','s3cret')
    monkeypatch.setattr(threads,'_execute_run',lambda *args: None)
    headers={'X-Assist-SMS-Secret':'s3cret'}
    assert client.post('/inbound/sms',json=_body(),headers=headers).status_code==200
    destination=Path(threads.MANAGER.thread_dir('t-sub'))
    if failure in {'root-missing','dangling-link','file'}:
        shutil.rmtree(destination)
    if failure=='dangling-link':destination.symlink_to(destination.parent/'absent-target')
    elif failure=='file':destination.write_text('not a directory')
    elif failure=='root-missing':
        original=os.stat
        def unavailable(path,*args,**kwargs):
            if os.fspath(path)==threads.MANAGER.root_dir:raise FileNotFoundError(errno.ENOENT,'synthetic root missing')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(os,'stat',unavailable)
    else:
        original=os.lstat
        def unavailable(path,*args,**kwargs):
            if os.fspath(path)==str(destination):
                raise OSError(errno.EACCES if failure=='permission' else errno.EIO,'synthetic storage failure')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(os,'lstat',unavailable)
    response=client.post('/inbound/sms',json=_body(),headers=headers)
    assert response.status_code==503


@pytest.mark.parametrize('source_time',['malformed-time','2026-10-28T00:00:00',''])
def test_unusable_incoming_time_archives_only_and_retries_same_identity(client,monkeypatch,source_time):
    from assist.events.sms_archive import configured_archive
    monkeypatch.setenv('ASSIST_SMS_SECRET','s3cret')
    monkeypatch.setattr(threads,'_execute_run',lambda *args: pytest.fail('invalid time dispatched'))
    body={**_body(mid='unusable-time',text='exact synthetic history'), 'occurred_at':source_time}
    headers={'X-Assist-SMS-Secret':'s3cret'}
    response=client.post('/inbound/sms',json=body,headers=headers)
    assert response.status_code==200 and response.json()['status']=='accepted'
    saved=configured_archive().read('unusable-time')
    assert saved['text']=='exact synthetic history' and saved['handling']=='archive_only'
    assert threads._runs().list('t-sub')==[]
    response=client.post('/inbound/sms',json=body,headers=headers)
    assert response.status_code==200 and response.json()['status']=='duplicate'
    assert configured_archive().read('unusable-time')==saved
    assert threads._runs().list('t-sub')==[]
