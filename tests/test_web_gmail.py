"""Real web approval admission/worker checks, isolated from mail and local models."""
from tests.approval_helpers import checkpoint_chat
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from manage import web
from manage.web import threads
from manage.web.state import _get_status, _set_status


@pytest.fixture
def client(tmp_path,monkeypatch):
    (tmp_path/"mail-thread").mkdir()
    # Match new-thread browser authority before the mocked resume scheduler runs.
    from assist.browser.authority import mark_new_thread
    mark_new_thread(str(tmp_path), "mail-thread")
    monkeypatch.setattr(web.MANAGER,"root_dir",str(tmp_path))
    monkeypatch.setattr(web.MANAGER,"thread_dir",lambda tid:str(tmp_path/tid))
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:SimpleNamespace(
        get_messages=lambda:[],pending_reply=lambda:None))
    monkeypatch.setattr(threads,"get_cached_description",lambda _:"Mail")
    return TestClient(web.app)


def _pending(error="", *, proposal=False):
    proposal_run = None
    if proposal:
        proposal_run = threads._create_run("mail-thread", "Original Gmail request")
        threads._runs().claim("mail-thread", proposal_run.id)
        proposal_run = threads._runs().transition("mail-thread", proposal_run.id, "awaiting_approval")
    _set_status("mail-thread","awaiting_approval",pending_gmail_token="exact-token",
                pending_gmail_run_id=proposal_run.id if proposal_run else None,
                pending_gmail_action={"name":"email_delete","args":{"message_ids":["abc123"]}},
                pending_gmail_interrupt_id="gmail-interrupt",
                pending_gmail_messages=[{"id":"abc123","from":"Sender <sender@example.test>",
                                         "to":"reader@example.test","subject":"Receipt", "date":"2026-10-01",
                                         "body":"Full <script>alert(1)</script>\nbody",
                                         "url":"https://mail.google.com/mail/u/0/#all/abc123"}],
                pending_gmail_error=error)
    return proposal_run


def test_full_web_preview_escaped_and_exact_token(client):
    _pending()
    page=client.get("/thread/mail-thread").text
    assert "Move to Trash awaiting your approval" in page
    assert "Sender &lt;sender@example.test&gt;" in page
    assert "Full &lt;script&gt;alert(1)&lt;/script&gt;\nbody" in page
    assert 'name="token" value="exact-token"' in page
    assert "https://mail.google.com/mail/u/0/#all/abc123" in page
    assert '<script>alert(1)</script>' not in page


def test_one_resume_for_concurrent_web_phone_and_replay(client):
    _pending(proposal=True)
    def decide(_):
        return threads.gmail_decision_core("mail-thread","approve","exact-token")
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies=list(pool.map(decide,range(2)))
    assert replies[0][0].id == replies[1][0].id
    assert sorted(replayed for _,replayed in replies)==[False,True]
    runs=threads._runs().list("mail-thread")
    assert len(runs)==2 and runs[-1].resume_decision=={"type":"approve","approval_interrupt_id":"gmail-interrupt"}
    assert runs[-1].approval_id == "exact-token"
    assert threads._runs().approval("mail-thread", "exact-token").accepted_run_id == runs[-1].id
    assert not _get_status("mail-thread").get("pending_gmail_token")
    with pytest.raises(HTTPException) as failure:
        threads.gmail_decision_core("mail-thread","reject","exact-token")
    assert failure.value.status_code==409
    # An old approval replay cannot consume a new proposal.
    _pending(proposal=True)
    _set_status("mail-thread","awaiting_approval",**{
        **{k:v for k,v in _get_status("mail-thread").items() if k!="stage"},
        "pending_gmail_token":"successor-token"})
    assert threads.gmail_decision_core("mail-thread","approve","exact-token")[1]
    assert _get_status("mail-thread")["pending_gmail_token"]=="successor-token"


def test_incomplete_preview_can_only_reject(client,monkeypatch):
    _pending(error="Complete preview unavailable",proposal=True)
    executed=[]
    monkeypatch.setattr(threads,"_execute_run",lambda *a:executed.append(a))
    page=client.get("/thread/mail-thread").text
    assert "Complete preview unavailable" in page and "Approve move" not in page
    assert client.post("/thread/mail-thread/gmail/approve",data={"token":"exact-token"}).status_code==409
    result=client.post("/thread/mail-thread/gmail/reject",data={"token":"exact-token"},follow_redirects=False)
    assert result.status_code==303 and len(executed)==1
    assert threads._runs().list("mail-thread")[-1].resume_decision=={"type":"reject","approval_interrupt_id":"gmail-interrupt"}


def test_new_message_refused_before_durable_admission_and_stale_decisions(client):
    _pending()
    assert client.post("/thread/mail-thread/message",data={"text":"Another request"}).status_code==409
    assert threads._runs().list("mail-thread")==[]
    for decision in ("approve","reject"):
        assert client.post(f"/thread/mail-thread/gmail/{decision}",data={"token":"old-token"}).status_code==409
    assert _get_status("mail-thread")["pending_gmail_token"]=="exact-token"


@pytest.mark.parametrize("preview_write_recovery", [None, "reject", "restart"])
def test_worker_gmail_proposal_and_approval_resume(client,monkeypatch,preview_write_recovery):
    class Chat:
        agent=None
        on_queue_state=None
        thread_id="mail-thread"
        pending=False
        decisions=[]
        messages=[]
        def message(self,text):
            self.messages.append(text)
            if text=="Archive this":
                # Real admission while the model turn is still processing.
                self.follower=threads._accept_message_run("mail-thread","Followup")[0]
                self.pending=True
                return "Approval required"
            return "Followup acknowledged"
        def pending_reply(self):
            return None
        def pending_action(self,name):
            if self.pending and name=="email_archive":
                return {"message_ids":["abc123"]}
            return None
        def pending_action_interrupt_id(self,name,args=None):
            return "gmail-interrupt" if self.pending_action(name) is not None else None
        def pending_actions(self):
            args=self.pending_action("email_archive")
            return [{"name":"email_archive","args":args}] if args else []
        def resume_actions(self,decisions):
            self.decisions.extend(decisions)
            self.pending=False
            return "Archived"
        def get_messages(self):
            return [{"role":"user","content":"Archive this"},{"role":"assistant","content":"Approval required"}]
        def get_raw_messages(self):
            return self.get_messages()
    chat=Chat()
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:checkpoint_chat(chat, k.get("configurable")))
    def touch(*args):
        if chat.pending:
            raise OSError("synthetic timestamp write failure")
    monkeypatch.setattr(web.MANAGER,"touch",touch)
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    scheduled=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,"submit",lambda *a,**k:scheduled.append(a))
    # Drive the real browser reset synchronously after this mocked turn closes.
    monkeypatch.setattr(threads,"_queue_browser_revocation",lambda *_:None)
    def no_sidecars(command, **_kwargs):
        assert command[:4] == ["docker", "ps", "--all", "--no-trunc"]
        return b""
    monkeypatch.setattr("assist.browser.manager._bounded_cli",no_sidecars)
    monkeypatch.setattr("assist.browser.manager.configured_directory",lambda:None)
    import threading
    preview_started=threading.Event()
    release_preview=threading.Event()
    def preview(action):
        preview_started.set()
        assert release_preview.wait(2)
        return [{"id":"abc123","body":"Full booking", "subject":"Receipt"}]
    monkeypatch.setattr(threads,"gmail_action_preview",preview)
    original_set_status = threads._set_status
    fail_publication = preview_write_recovery is not None
    def publish(tid, stage, **fields):
        nonlocal fail_publication
        if fail_publication and fields.get("pending_gmail_preview_pending") is False:
            fail_publication = False
            raise OSError("Synthetic atomic preview publication failure")
        return original_set_status(tid, stage, **fields)
    monkeypatch.setattr(threads, "_set_status", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        request=pool.submit(client.post,"/thread/mail-thread/message",
                            data={"text":"Archive this"},follow_redirects=False)
        try:
            assert preview_started.wait(2)
            queued=threads._create_run("mail-thread","Followup",origin="continuation")
            continuation=pool.submit(threads._execute_run,queued.id,"mail-thread")
            assert threads.THREAD_QUEUE.peek_holder() is None
            continuation.result(2)
            assert threads._runs().get("mail-thread",queued.id).status=="pending"
            loading=_get_status("mail-thread")
            assert loading["pending_gmail_preview_pending"]
            with pytest.raises(HTTPException):
                threads.gmail_decision_core("mail-thread","approve",loading["pending_gmail_token"])
        finally:
            release_preview.set()
        assert request.result(2).status_code==303
        continuation.result(2)
    assert threads._runs().get("mail-thread",chat.follower.id).status=="revocation_pending"
    assert threads._drain_held_browser_events("mail-thread")
    status=_get_status("mail-thread")
    assert status["stage"]=="awaiting_approval"
    assert status["pending_gmail_action"]["name"]=="email_archive"
    if preview_write_recovery:
        assert status["pending_gmail_preview_pending"] and status["pending_gmail_messages"] == []
        assert status["pending_gmail_token"] == loading["pending_gmail_token"]
        assert status["pending_gmail_interrupt_id"] == "gmail-interrupt"
        assert threads._runs().get("mail-thread", status["pending_gmail_run_id"]).status == "awaiting_approval"
        with pytest.raises(HTTPException):
            threads.gmail_decision_core("mail-thread", "approve", status["pending_gmail_token"])
    if preview_write_recovery == "restart":
        from manage.web import state
        monkeypatch.setattr(web.MANAGER, "list", lambda: ["mail-thread"])
        state._recover_interrupted_threads()
        assert (status["pending_gmail_run_id"], "mail-thread") in scheduled
        with threads.THREAD_QUEUE.acquire("other-thread"):
            threads._execute_run(status["pending_gmail_run_id"], "mail-thread")
        recovered = _get_status("mail-thread")
        assert recovered["pending_gmail_token"] == threads.approval_status("mail-thread")["pending_gmail_token"]
        assert recovered["pending_gmail_token"] != loading["pending_gmail_token"]
        assert recovered["pending_gmail_messages"][0]["body"] == "Full booking"
        assert not recovered["pending_gmail_preview_pending"]
        status = recovered
    if preview_write_recovery:
        # The complete preview committed before the failed status projection.
        status = threads.approval_status("mail-thread")
        assert status["pending_gmail_messages"][0]["body"] == "Full booking"
        assert not status["pending_gmail_preview_pending"]
    # A system continuation stays pending rather than entering the interrupted graph.
    assert threads._runs().get("mail-thread",queued.id).status=="pending"
    # Avoid executing that artificial continuation after the tested approval resumes.
    threads._runs().transition("mail-thread",queued.id,"cancelled")
    choice = "reject" if preview_write_recovery == "reject" else "approve"
    successor,replayed=threads.gmail_decision_core("mail-thread",choice,status["pending_gmail_token"])
    assert not replayed
    # A previously admitted user Run cannot enter the paused graph after token consumption.
    threads._execute_run(chat.follower.id,"mail-thread")
    assert threads._runs().get("mail-thread",chat.follower.id).status=="pending"
    assert chat.messages==["Archive this"]
    threads._dispatch_pending_after("mail-thread")
    assert scheduled[-1]==(successor.id,"mail-thread")
    threads._execute_run(successor.id,"mail-thread")
    assert chat.decisions==[{"type":choice}]
    proposal=threads._runs().get("mail-thread",status["pending_gmail_run_id"])
    successor=threads._runs().get("mail-thread",successor.id)
    assert successor.work_id==proposal.work_id and successor.status=="success"
    from manage.web.phone_api import _logical_status
    projection=_logical_status("mail-thread",proposal.id)
    assert projection["physical_run_id"]==successor.id and projection["status"]=="success"
    threads._execute_run(chat.follower.id,"mail-thread")
    threads._execute_run(chat.follower.id,"mail-thread")
    assert chat.messages==["Archive this","Followup"]
    assert threads._runs().get("mail-thread",chat.follower.id).status=="success"
    assert _get_status("mail-thread")["stage"]=="ready"


def test_contended_approval_admission_does_not_block_event_loop(client,monkeypatch):
    import asyncio
    import threading
    import httpx
    _pending(proposal=True)
    started=threading.Event()
    original=threads.gmail_decision_core
    def core(*args):
        started.set()
        return original(*args)
    monkeypatch.setattr(threads,"gmail_decision_core",core)
    monkeypatch.setattr(threads,"_execute_run",lambda *a:None)
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app),base_url="http://test") as http:
            threads._RUN_ADMISSION_LOCK.acquire()
            task=asyncio.create_task(http.post("/thread/mail-thread/gmail/approve",data={"token":"exact-token"}))
            try:
                assert await asyncio.to_thread(started.wait,1)
                response=await asyncio.wait_for(http.get("/nonexistent"),timeout=0.5)
                assert response.status_code==404
            finally:
                threads._RUN_ADMISSION_LOCK.release()
            assert (await asyncio.wait_for(task,2)).status_code==303
    asyncio.run(check())


def test_resume_admission_survives_status_write_failure(client,monkeypatch):
    _pending(proposal=True)
    original=threads._set_status
    fail=True
    def set_status(*args,**kwargs):
        nonlocal fail
        if fail:
            fail=False
            raise OSError("synthetic disk failure")
        return original(*args,**kwargs)
    monkeypatch.setattr(threads,"_set_status",set_status)
    with pytest.raises(OSError):
        threads.gmail_decision_core("mail-thread","approve","exact-token")
    calls=[]
    monkeypatch.setattr(threads,"_execute_run",lambda *a:calls.append(a))
    response=client.post("/thread/mail-thread/gmail/approve",data={"token":"exact-token"},follow_redirects=False)
    assert response.status_code==303 and len(calls)==1
    assert len(threads._runs().list("mail-thread"))==2


def test_reject_during_preview_preserves_successor_status_and_context(client,monkeypatch):
    import threading
    started=threading.Event()
    release=threading.Event()
    class Chat:
        pending=False
        agent=None
        def message(self,text):
            self.pending=True
            return "Proposal"
        def pending_reply(self):
            return None
        def pending_action(self,name):
            return {"message_ids":["abc123"]} if self.pending and name=="email_delete" else None
        def pending_action_interrupt_id(self,name,args=None):
            return "gmail-interrupt" if self.pending_action(name) is not None else None
        def pending_actions(self):
            return [{"name":"email_delete","args":{"message_ids":["abc123"]}}] if self.pending else []
        def resume_actions(self,decisions):
            assert decisions==[{"type":"reject"}]
            self.pending=False
            return "Rejected"
        def get_messages(self):
            return [{"role":"user","content":"Trash this"},{"role":"assistant","content":"Proposal"}]
    chat=Chat()
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:checkpoint_chat(chat, k.get("configurable")))
    monkeypatch.setattr(web.MANAGER,"touch",lambda *a:None)
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    def preview(action):
        started.set()
        assert release.wait(2)
        return [{"id":"abc123","body":"Complete preview"}]
    monkeypatch.setattr(threads,"gmail_action_preview",preview)
    proposal=threads._create_run("mail-thread","Trash this")
    context={"claimed":[],"run_id":"a-new-successor-context"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(threads._execute_run,proposal.id,"mail-thread")
        try:
            assert started.wait(2)
            assert threads.THREAD_QUEUE.peek_holder() is None
            status=_get_status("mail-thread")
            successor,_=threads.gmail_decision_core("mail-thread","reject",status["pending_gmail_token"])
            threads._execute_run(successor.id,"mail-thread")
            assert _get_status("mail-thread")["stage"]=="ready"
            threads._TURN_INTERJECTION["mail-thread"]=context
            # Persist a new same-thread slice as a paused successor would.
            with threads.THREAD_QUEUE.acquire("mail-thread",accumulated_active_ms=2000) as later_handle:
                pass
        finally:
            release.set()
        pending.result(2)
    try:
        assert _get_status("mail-thread")["stage"]=="ready"
        assert threads._TURN_INTERJECTION["mail-thread"] is context
        assert threads.THREAD_QUEUE.pop_hold(later_handle) >= 2000
        assert threads._runs().get("mail-thread",proposal.id).status=="awaiting_approval"
        assert threads._runs().get("mail-thread",successor.id).status=="success"
    finally:
        threads._TURN_INTERJECTION.pop("mail-thread",None)


def test_delayed_queue_owner_cannot_drain_a_newer_same_thread_hold():
    from assist.thread_queue import ThreadAffinityQueue
    queue=ThreadAffinityQueue()
    with queue.acquire("same-thread",accumulated_active_ms=1000) as old:
        pass
    with queue.acquire("same-thread",accumulated_active_ms=2000) as successor:
        pass
    assert queue.pop_hold(old)>=1000
    assert queue.pop_hold(successor)>=2000
    assert queue.pop_hold(old)==0
    assert queue.pop_hold(successor)==0


def test_unicode_approval_token_is_stale_without_consuming_proposal(client):
    _pending()
    response=client.post("/thread/mail-thread/gmail/approve",data={"token":"é"})
    assert response.status_code==409
    assert threads._runs().list("mail-thread")==[]
    assert _get_status("mail-thread")["pending_gmail_token"]=="exact-token"


@pytest.mark.parametrize("decision",["approve","reject","edit"])
@pytest.mark.parametrize("card",["gmail","email"])
def test_reply_endpoint_cannot_consume_other_action_proposal(client,monkeypatch,decision,card):
    if card=="gmail":
        _pending(error="Preparing complete message preview; approve only after it loads.")
    else:
        _set_status("mail-thread","awaiting_approval",pending_email_token="exact-token",
                    pending_email_to="reader@example.test",pending_email_subject="Receipt",pending_email_body="Full body")
    executed=[]
    monkeypatch.setattr(threads,"_execute_run",lambda *a:executed.append(a))
    response=client.post(f"/thread/mail-thread/reply/{decision}",follow_redirects=False)
    assert response.status_code==409
    assert executed==[] and threads._runs().list("mail-thread")==[]
    assert _get_status("mail-thread")[f"pending_{card}_token"]=="exact-token"


@pytest.mark.parametrize("gmail_key,other_reply,gmail_pending,stale,wait,recover",[
    (False,False,True,False,False,False),(False,True,True,False,False,False),
    (True,True,True,False,False,False),(True,True,False,False,False,False),
    (True,False,True,True,False,False),(True,False,True,True,True,False),
    (True,False,True,False,False,True),
    (True,False,True,False,False,"restart"),
    (True,False,True,False,False,"changed-interrupt"),
])
def test_worker_binds_gmail_decision_to_its_dispatch_kind(client,monkeypatch,gmail_key,other_reply,gmail_pending,stale,wait,recover):
    class Chat:
        agent=None
        pending=True
        ids=["abc123"]
        interrupt_id="gmail-interrupt"
        decisions=[]
        def pending_reply(self):
            return {"text":"Separate unseen reply"} if self.pending and other_reply else None
        def pending_action(self,name):
            return {"message_ids":self.ids} if self.pending and gmail_pending and name=="email_archive" else None
        def pending_action_interrupt_id(self,name,args=None):
            return self.interrupt_id if self.pending_action(name) is not None else None
        def pending_actions(self):
            return ([{"name":"email_archive","args":{"message_ids":self.ids}}]
                    + ([{"name":"send_reply","args":{"text":"Separate unseen reply"}}]
                       if other_reply else []))
        def resume_reply(self,decision):
            pytest.fail("Gmail approval cannot authorize a different reply")
        def resume_actions(self,decisions):
            self.decisions=decisions
            self.pending=False
            return "Archived"
        def get_messages(self):
            return []
        def get_raw_messages(self):
            return []
    chat=Chat()
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:checkpoint_chat(chat, k.get("configurable")))
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    monkeypatch.setattr(threads,"gmail_action_preview",lambda _: [{"id":"abc123","body":"Complete preview"}])
    scheduled=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,"submit",lambda *a,**k:scheduled.append(a))
    proposal=threads._create_run("mail-thread","Original proposal")
    threads._runs().claim("mail-thread",proposal.id)
    threads._runs().transition("mail-thread",proposal.id,"awaiting_approval")
    if gmail_pending:
        _pending()
        status=_get_status("mail-thread")
        _set_status("mail-thread","awaiting_approval",**{
            **{k:v for k,v in status.items() if k!="stage"},
            "pending_gmail_action":{"name":"email_archive","args":{"message_ids":["abc123"]}},
            "pending_gmail_run_id":proposal.id})
    else:
        _set_status("mail-thread","awaiting_approval",pending_reply="Separate unseen reply",pending_sender="+1555")
    if gmail_key and gmail_pending:
        before=_get_status("mail-thread")
        if recover is True:
            with monkeypatch.context() as failure:
                def unavailable(*a,**k):
                    raise OSError("synthetic status write failure")
                failure.setattr(threads,"_set_status",unavailable)
                with pytest.raises(OSError):
                    threads.gmail_decision_core("mail-thread","approve","exact-token")
            run,replayed=threads.gmail_decision_core("mail-thread","approve","exact-token")
            assert replayed
        else:
            run,_=threads.gmail_decision_core("mail-thread","approve","exact-token")
        if recover=="restart":
            from manage.web import state
            monkeypatch.setattr(web.MANAGER,"list",lambda:["mail-thread"])
            state._recover_interrupted_threads()
            assert _get_status("mail-thread")["stage"]=="paused"
            assert (run.id,"mail-thread") in scheduled
        if recover=="changed-interrupt":
            chat.interrupt_id="successor-interrupt"
        if stale:
            chat.ids=["def456"]
            _set_status("mail-thread","awaiting_approval",**{
                **{k:v for k,v in before.items() if k!="stage"},
                "pending_gmail_token":"successor-token",
                "pending_gmail_action":{"name":"email_archive","args":{"message_ids":chat.ids}}})
    else:
        run=threads._create_run("mail-thread",None,resume_decision={"type":"approve"},
                            dispatch_key="gmail-approval:email_archive:exact-token" if gmail_key else None,
                            work_id=proposal.work_id if gmail_key and gmail_pending else None)
    before=_get_status("mail-thread")
    if wait:
        import threading
        queued=threading.Event()
        acquire=threads.THREAD_QUEUE.acquire
        def waiting_acquire(tid,**kwargs):
            callback=kwargs.get("on_state_change")
            if callback:
                def observed(stage):
                    callback(stage)
                    if stage=="queued":
                        queued.set()
                kwargs["on_state_change"]=observed
            return acquire(tid,**kwargs)
        monkeypatch.setattr(threads.THREAD_QUEUE,"acquire",waiting_acquire)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with acquire("another-thread"):
                future=pool.submit(threads._execute_run,run.id,"mail-thread")
                assert queued.wait(2)
                assert _get_status("mail-thread")==before
            future.result(2)
    else:
        threads._execute_run(run.id,"mail-thread")
    if gmail_key and gmail_pending and not stale and recover!="changed-interrupt":
        assert chat.decisions[0]=={"type":"approve"}
        if other_reply:
            assert chat.decisions[1]["type"]=="reject"
        else:
            assert len(chat.decisions)==1
    else:
        assert chat.decisions==[] and chat.pending
        assert _get_status("mail-thread")==before
        assert threads._runs().get("mail-thread",run.id).status=="cancelled"
        assert threads._runs().get("mail-thread",proposal.id).status=="awaiting_approval"



@pytest.mark.parametrize("sender,key",[("",None),(None,"email-approval:exact-token")])
def test_reply_run_cannot_fall_through_to_email_approval(client,monkeypatch,sender,key):
    class Chat:
        agent=None
        pending=True
        decisions=[]
        def pending_reply(self):
            return None
        def pending_action(self,name):
            return None
        def pending_email(self):
            return {"to":"reader@example.test","subject":"Receipt","body":"Complete body"} if self.pending else None
        def resume_action(self,decision):
            self.decisions.append(decision)
            self.pending=False
            return "Sent"
        def get_messages(self):
            return []
        def get_raw_messages(self):
            return []
    chat=Chat()
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:checkpoint_chat(chat, k.get("configurable")))
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    monkeypatch.setattr(threads._RESUME_SCHEDULER,"submit",lambda *a,**k:None)
    _set_status("mail-thread","awaiting_approval",pending_email_token="exact-token",
                pending_email_to="reader@example.test",pending_email_subject="Receipt",pending_email_body="Complete body")
    before=_get_status("mail-thread")
    # Empty legacy senders normalize to None in persisted Runs; only the key binds kind.
    run=threads._create_run("mail-thread",None,sender=sender,dispatch_key=key,
                            resume_decision={"type":"approve"})
    threads._execute_run(run.id,"mail-thread")
    # A dispatch prefix without a stored exact binding is not approval authority.
    assert chat.decisions == []
    assert _get_status("mail-thread") == before
    assert threads._runs().get("mail-thread",run.id).status == "cancelled"


@pytest.mark.parametrize("decision",["approve","reject"])
def test_gmail_core_binds_requested_kind_before_admission_and_on_replay(client,decision):
    proposal = _pending(proposal=True)
    before=_get_status("mail-thread")
    with pytest.raises(HTTPException) as mismatch:
        threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="email_archive")
    assert mismatch.value.status_code==409
    assert _get_status("mail-thread")==before and threads._runs().list("mail-thread")==[proposal]
    run,replayed=threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="email_delete")
    assert not replayed
    assert threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="email_delete")== (run,True)
    with pytest.raises(HTTPException) as replay_mismatch:
        threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="email_archive")
    assert replay_mismatch.value.status_code==409
    assert [item.id for item in threads._runs().list("mail-thread")] == [proposal.id, run.id]
    assert threads._runs().get("mail-thread", proposal.id).approvals[0].decision == run.resume_decision


@pytest.mark.parametrize("proposal_state",["running","awaiting_approval"])
def test_startup_requeues_interrupted_gmail_preview_without_resuming_agent(client,monkeypatch,proposal_state):
    from manage.web import state
    proposal=threads._create_run("mail-thread","Archive this")
    threads._runs().claim("mail-thread",proposal.id)
    if proposal_state=="awaiting_approval":
        threads._runs().transition("mail-thread",proposal.id,"awaiting_approval")
    action={"name":"email_archive","args":{"message_ids":["abc123"]}}
    _set_status("mail-thread","awaiting_approval",pending_gmail_run_id=proposal.id,
                pending_gmail_action=action,pending_gmail_token="exact-token",
                pending_gmail_messages=[],pending_gmail_preview_pending=True,
                pending_gmail_error="Preparing complete Gmail preview.")
    scheduled=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,"submit",lambda *a,**k:scheduled.append(a))
    monkeypatch.setattr(web.MANAGER,"list",lambda:["mail-thread"])
    def forbidden(*a,**k):
        pytest.fail("Preview recovery must not invoke the agent or sandbox")
    monkeypatch.setattr(threads,"_process_message",forbidden)
    monkeypatch.setattr(threads,"_get_sandbox_backend",forbidden)
    reads=[]
    def preview(shown):
        reads.append(shown)
        return [{"id":"abc123","body":"Recovered complete preview"}]
    monkeypatch.setattr(threads,"gmail_action_preview",preview)
    state._recover_interrupted_threads()
    assert (proposal.id,"mail-thread") in scheduled
    with threads.THREAD_QUEUE.acquire("another-thread"):
        threads._execute_run(proposal.id,"mail-thread")
    status=_get_status("mail-thread")
    assert status["stage"]=="awaiting_approval" and status["pending_gmail_token"]=="exact-token"
    assert status["pending_gmail_messages"][0]["body"]=="Recovered complete preview"
    assert not status["pending_gmail_preview_pending"] and not status["pending_gmail_error"]
    assert threads._runs().get("mail-thread",proposal.id).status=="awaiting_approval"
    threads._execute_run(proposal.id,"mail-thread")
    assert reads==[action]


def test_gmail_decision_requires_the_shown_checkpoint_identity(client):
    _pending()
    status=_get_status("mail-thread")
    status.pop("pending_gmail_interrupt_id")
    _set_status("mail-thread","awaiting_approval",**{k:v for k,v in status.items() if k!="stage"})
    before=_get_status("mail-thread")
    with pytest.raises(HTTPException) as missing:
        threads.gmail_decision_core("mail-thread","approve","exact-token")
    assert missing.value.status_code==409
    assert _get_status("mail-thread")==before and threads._runs().list("mail-thread")==[]


@pytest.mark.parametrize("kind", ["email_archive", "email_delete"])
@pytest.mark.parametrize("new_gate", [False, True])
@pytest.mark.parametrize("preview_error", [False, True])
def test_upgrade_repreviews_legacy_card_at_actual_checkpoint(client, tmp_path, monkeypatch,
                                                           kind, new_gate, preview_error):
    import sqlite3
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import MessagesState, StateGraph, START, END
    from langgraph.types import Command, interrupt
    from assist.thread import Thread
    from manage.web import state
    from assist.gmail import GmailError

    tid = "mail-thread"
    action = {"name": kind, "args": {"message_ids": ["abc123"]}}
    decisions = []
    def gate(snapshot):
        decisions.append(interrupt({"action_requests": [action]}))
        return {"messages": [AIMessage("Synthetic gate completed")]}
    def later(snapshot):
        if new_gate:
            decisions.append(interrupt({"action_requests": [action]}))
        return {"messages": [AIMessage("Synthetic work completed")]}
    def build(connection):
        graph = StateGraph(MessagesState)
        graph.add_node("gate", gate); graph.add_node("later", later)
        graph.add_edge(START, "gate"); graph.add_edge("gate", "later"); graph.add_edge("later", END)
        return graph.compile(checkpointer=SqliteSaver(connection))
    config = {"configurable": {"thread_id": tid}}
    connection = sqlite3.connect(tmp_path / "legacy.db", check_same_thread=False)
    graph = build(connection)
    graph.invoke({"messages": [HumanMessage("Please organize this message")]}, config, durability="sync")
    old_id = graph.get_state(config).interrupts[0].id
    if new_gate:
        graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config, durability="sync")
        assert graph.get_state(config).interrupts[0].id != old_id
    connection.close()
    connection = sqlite3.connect(tmp_path / "legacy.db", check_same_thread=False)
    graph = build(connection)
    def get_chat(*args, **kwargs):
        chat = Thread.__new__(Thread)
        chat.thread_id = tid; chat.agent = graph; chat.runconfig = config
        # The companion-owned helper reads this actual checkpoint identity.
        def identity(name, args=None):
            assert name == kind and args == action["args"]
            return graph.get_state(config).interrupts[0].id
        chat.pending_action_interrupt_id = identity
        return chat
    monkeypatch.setattr(web.MANAGER, "get", get_chat)
    monkeypatch.setattr(web.MANAGER, "list", lambda: [tid])
    jobs = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *a, **k: jobs.append(a))
    def forbidden(*args, **kwargs):
        pytest.fail("Legacy card migration must not invoke the agent or sandbox")
    monkeypatch.setattr(threads, "_process_message", forbidden)
    monkeypatch.setattr(threads, "_get_sandbox_backend", forbidden)
    reads = []
    def preview(current):
        reads.append(current)
        if preview_error:
            raise GmailError("Synthetic preview unavailable")
        return [{"id": "abc123", "body": "Fresh complete preview"}]
    monkeypatch.setattr(threads, "gmail_action_preview", preview)
    proposal = threads._create_run(tid, "Please organize this message")
    threads._runs().claim(tid, proposal.id)
    threads._runs().transition(tid, proposal.id, "awaiting_approval")
    # Exact deployed494 card shape: no interrupt identity or preview-pending flag.
    _set_status(tid, "awaiting_approval", pending_gmail_run_id=proposal.id,
                pending_gmail_action=action, pending_gmail_token="old-token",
                pending_gmail_messages=[{"id": "abc123", "body": "Old complete preview"}],
                pending_gmail_error="")
    try:
        for decision in ("approve", "reject"):
            with pytest.raises(HTTPException) as stale:
                threads.gmail_decision_core(tid, decision, "old-token")
            assert stale.value.status_code == 409
        state._recover_interrupted_threads()
        status = _get_status(tid)
        assert status["pending_gmail_token"] != "old-token"
        assert status["pending_gmail_interrupt_id"] == graph.get_state(config).interrupts[0].id
        assert status["pending_gmail_preview_pending"] and status["pending_gmail_messages"] == []
        assert jobs.count((proposal.id, tid)) == 1
        with threads.THREAD_QUEUE.acquire("other-thread"):
            threads._execute_run(proposal.id, tid)
        status = _get_status(tid)
        assert reads == [action] and not status["pending_gmail_preview_pending"]
        for decision in ("approve", "reject"):
            with pytest.raises(HTTPException) as stale:
                threads.gmail_decision_core(tid, decision, "old-token")
            assert stale.value.status_code == 409
        if preview_error:
            with pytest.raises(HTTPException):
                threads.gmail_decision_core(tid, "approve", status["pending_gmail_token"])
        admitted, replayed = threads.gmail_decision_core(
            tid, "reject" if preview_error else "approve", status["pending_gmail_token"])
        assert not replayed and admitted.work_id == proposal.work_id
        assert admitted.resume_decision["approval_interrupt_id"] == graph.get_state(config).interrupts[0].id
        assert len(decisions) == int(new_gate)  # No checkpoint consumption during migration.
    finally:
        connection.close()


@pytest.mark.parametrize("proposal_state", ["missing", "unknown", "pending", "running", "success"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_gmail_admission_requires_durable_awaiting_proposal(client, proposal_state, decision):
    _pending()
    if proposal_state != "missing":
        if proposal_state == "unknown":
            proposal_id = "missing-run"
        else:
            proposal = threads._create_run("mail-thread", "Original request")
            if proposal_state in {"running", "success"}:
                threads._runs().claim("mail-thread", proposal.id)
            if proposal_state == "success":
                threads._runs().transition("mail-thread", proposal.id, "success")
            proposal_id = proposal.id
        _set_status("mail-thread", "awaiting_approval", **{
            **{k:v for k,v in _get_status("mail-thread").items() if k != "stage"},
            "pending_gmail_run_id":proposal_id})
    before_status = _get_status("mail-thread")
    before_runs = threads._runs().list("mail-thread")
    with pytest.raises(HTTPException) as refused:
        threads.gmail_decision_core("mail-thread", decision, "exact-token")
    assert refused.value.status_code == 409
    assert _get_status("mail-thread") == before_status
    assert threads._runs().list("mail-thread") == before_runs
