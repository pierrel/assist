"""Real web approval admission/worker checks, isolated from mail and local models."""
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
    monkeypatch.setattr(web.MANAGER,"root_dir",str(tmp_path))
    monkeypatch.setattr(web.MANAGER,"thread_dir",lambda tid:str(tmp_path/tid))
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:SimpleNamespace(
        get_messages=lambda:[],pending_reply=lambda:None))
    monkeypatch.setattr(threads,"get_cached_description",lambda _:"Mail")
    return TestClient(web.app)


def _pending(error=""):
    _set_status("mail-thread","awaiting_approval",pending_gmail_token="exact-token",
                pending_gmail_action={"name":"gmail_delete","args":{"message_ids":["abc123"]}},
                pending_gmail_messages=[{"id":"abc123","from":"Sender <sender@example.test>",
                                         "to":"reader@example.test","subject":"Receipt", "date":"2026-10-01",
                                         "body":"Full <script>alert(1)</script>\nbody",
                                         "url":"https://mail.google.com/mail/u/0/#all/abc123"}],
                pending_gmail_error=error)


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
    _pending()
    def decide(_):
        return threads.gmail_decision_core("mail-thread","approve","exact-token")
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies=list(pool.map(decide,range(2)))
    assert replies[0][0].id == replies[1][0].id
    assert sorted(replayed for _,replayed in replies)==[False,True]
    runs=threads._runs().list("mail-thread")
    assert len(runs)==1 and runs[0].resume_decision=={"type":"approve"}
    assert runs[0].dispatch_key=="gmail-approval:gmail_delete:exact-token"
    assert not _get_status("mail-thread").get("pending_gmail_token")
    with pytest.raises(HTTPException) as failure:
        threads.gmail_decision_core("mail-thread","reject","exact-token")
    assert failure.value.status_code==409
    # An old approval replay cannot consume a new proposal.
    _pending()
    _set_status("mail-thread","awaiting_approval",**{
        **{k:v for k,v in _get_status("mail-thread").items() if k!="stage"},
        "pending_gmail_token":"successor-token"})
    assert threads.gmail_decision_core("mail-thread","approve","exact-token")[1]
    assert _get_status("mail-thread")["pending_gmail_token"]=="successor-token"


def test_incomplete_preview_can_only_reject(client,monkeypatch):
    _pending(error="Complete preview unavailable")
    executed=[]
    monkeypatch.setattr(threads,"_execute_run",lambda *a:executed.append(a))
    page=client.get("/thread/mail-thread").text
    assert "Complete preview unavailable" in page and "Approve move" not in page
    assert client.post("/thread/mail-thread/gmail/approve",data={"token":"exact-token"}).status_code==409
    result=client.post("/thread/mail-thread/gmail/reject",data={"token":"exact-token"},follow_redirects=False)
    assert result.status_code==303 and len(executed)==1
    assert threads._runs().list("mail-thread")[0].resume_decision=={"type":"reject"}


def test_new_message_refused_before_durable_admission_and_stale_decisions(client):
    _pending()
    assert client.post("/thread/mail-thread/message",data={"text":"Another request"}).status_code==409
    assert threads._runs().list("mail-thread")==[]
    for decision in ("approve","reject"):
        assert client.post(f"/thread/mail-thread/gmail/{decision}",data={"token":"old-token"}).status_code==409
    assert _get_status("mail-thread")["pending_gmail_token"]=="exact-token"


def test_worker_gmail_proposal_and_approval_resume(client,monkeypatch):
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
            if self.pending and name=="gmail_archive":
                return {"message_ids":["abc123"]}
            return None
        def pending_actions(self):
            args=self.pending_action("gmail_archive")
            return [{"name":"gmail_archive","args":args}] if args else []
        def resume_actions(self,decisions):
            self.decisions.extend(decisions)
            self.pending=False
            return "Archived"
        def get_messages(self):
            return [{"role":"user","content":"Archive this"},{"role":"assistant","content":"Approval required"}]
    chat=Chat()
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:chat)
    def touch(*args):
        if chat.pending:
            raise OSError("synthetic timestamp write failure")
    monkeypatch.setattr(web.MANAGER,"touch",touch)
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    scheduled=[]
    monkeypatch.setattr(threads._RESUME_SCHEDULER,"submit",lambda *a,**k:scheduled.append(a))
    import threading
    preview_started=threading.Event()
    release_preview=threading.Event()
    def preview(action):
        preview_started.set()
        assert release_preview.wait(2)
        return [{"id":"abc123","body":"Full booking", "subject":"Receipt"}]
    monkeypatch.setattr(threads,"gmail_action_preview",preview)
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
            with pytest.raises(HTTPException):
                threads.gmail_decision_core("mail-thread","approve",loading["pending_gmail_token"])
        finally:
            release_preview.set()
        assert request.result(2).status_code==303
        continuation.result(2)
    status=_get_status("mail-thread")
    assert status["stage"]=="awaiting_approval"
    assert status["pending_gmail_action"]["name"]=="gmail_archive"
    # A system continuation stays pending rather than entering the interrupted graph.
    assert threads._runs().get("mail-thread",queued.id).status=="pending"
    # Avoid executing that artificial continuation after the tested approval resumes.
    threads._runs().transition("mail-thread",queued.id,"cancelled")
    successor,replayed=threads.gmail_decision_core("mail-thread","approve",status["pending_gmail_token"])
    assert not replayed
    # A previously admitted user Run cannot enter the paused graph after token consumption.
    threads._execute_run(chat.follower.id,"mail-thread")
    assert threads._runs().get("mail-thread",chat.follower.id).status=="pending"
    assert chat.messages==["Archive this"]
    threads._dispatch_pending_after("mail-thread")
    assert scheduled[-1]==(successor.id,"mail-thread")
    threads._execute_run(successor.id,"mail-thread")
    assert chat.decisions==[{"type":"approve"}]
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
    _pending()
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
    _pending()
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
    assert len(threads._runs().list("mail-thread"))==1


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
            return {"message_ids":["abc123"]} if self.pending and name=="gmail_delete" else None
        def pending_actions(self):
            return [{"name":"gmail_delete","args":{"message_ids":["abc123"]}}] if self.pending else []
        def resume_actions(self,decisions):
            assert decisions==[{"type":"reject"}]
            self.pending=False
            return "Rejected"
        def get_messages(self):
            return [{"role":"user","content":"Trash this"},{"role":"assistant","content":"Proposal"}]
    chat=Chat()
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:chat)
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
])
def test_worker_binds_gmail_decision_to_its_dispatch_kind(client,monkeypatch,gmail_key,other_reply,gmail_pending,stale,wait,recover):
    class Chat:
        agent=None
        pending=True
        ids=["abc123"]
        decisions=[]
        def pending_reply(self):
            return {"text":"Separate unseen reply"} if self.pending and other_reply else None
        def pending_action(self,name):
            return {"message_ids":self.ids} if self.pending and gmail_pending and name=="gmail_archive" else None
        def pending_actions(self):
            return ([{"name":"gmail_archive","args":{"message_ids":self.ids}}]
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
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:chat)
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    monkeypatch.setattr(threads,"gmail_action_preview",lambda _: [{"id":"abc123","body":"Complete preview"}])
    monkeypatch.setattr(threads._RESUME_SCHEDULER,"submit",lambda *a,**k:None)
    proposal=threads._create_run("mail-thread","Original proposal")
    threads._runs().claim("mail-thread",proposal.id)
    threads._runs().transition("mail-thread",proposal.id,"awaiting_approval")
    if gmail_pending:
        _pending()
        status=_get_status("mail-thread")
        _set_status("mail-thread","awaiting_approval",**{
            **{k:v for k,v in status.items() if k!="stage"},
            "pending_gmail_action":{"name":"gmail_archive","args":{"message_ids":["abc123"]}},
            "pending_gmail_run_id":proposal.id})
    else:
        _set_status("mail-thread","awaiting_approval",pending_reply="Separate unseen reply",pending_sender="+1555")
    if gmail_key and gmail_pending:
        before=_get_status("mail-thread")
        if recover:
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
        if stale:
            chat.ids=["def456"]
            _set_status("mail-thread","awaiting_approval",**{
                **{k:v for k,v in before.items() if k!="stage"},
                "pending_gmail_token":"successor-token",
                "pending_gmail_action":{"name":"gmail_archive","args":{"message_ids":chat.ids}}})
    else:
        run=threads._create_run("mail-thread",None,resume_decision={"type":"approve"},
                            dispatch_key="gmail-approval:gmail_archive:exact-token" if gmail_key else None,
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
    if gmail_key and gmail_pending and not stale:
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
    monkeypatch.setattr(web.MANAGER,"get",lambda *a,**k:chat)
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
    assert chat.decisions==([] if key is None else [{"type":"approve"}])
    if key is None:
        assert _get_status("mail-thread")==before
        assert threads._runs().get("mail-thread",run.id).status=="cancelled"


@pytest.mark.parametrize("decision",["approve","reject"])
def test_gmail_core_binds_requested_kind_before_admission_and_on_replay(client,decision):
    _pending()
    before=_get_status("mail-thread")
    with pytest.raises(HTTPException) as mismatch:
        threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="gmail_archive")
    assert mismatch.value.status_code==409
    assert _get_status("mail-thread")==before and threads._runs().list("mail-thread")==[]
    run,replayed=threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="gmail_delete")
    assert not replayed
    assert threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="gmail_delete")== (run,True)
    with pytest.raises(HTTPException) as replay_mismatch:
        threads.gmail_decision_core("mail-thread",decision,"exact-token",expected_kind="gmail_archive")
    assert replay_mismatch.value.status_code==409
    assert threads._runs().list("mail-thread")==[run]
