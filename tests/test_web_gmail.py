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
    assert runs[0].dispatch_key=="gmail-approval:exact-token"
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
        def message(self,text):
            self.pending=True
            return "Approval required"
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
    monkeypatch.setattr(web.MANAGER,"touch",lambda *a:None)
    monkeypatch.setattr(threads,"_get_sandbox_backend",lambda *a,**k:None)
    monkeypatch.setattr(threads,"_get_domain_manager",lambda *a,**k:None)
    monkeypatch.setattr(threads,"gmail_action_preview",lambda action:[{
        "id":"abc123","body":"Full booking", "subject":"Receipt"}])
    response=client.post("/thread/mail-thread/message",data={"text":"Archive this"},follow_redirects=False)
    assert response.status_code==303
    status=_get_status("mail-thread")
    assert status["stage"]=="awaiting_approval"
    assert status["pending_gmail_action"]["name"]=="gmail_archive"
    # A system continuation stays pending rather than entering the interrupted graph.
    queued=threads._create_run("mail-thread","Followup",origin="continuation")
    threads._execute_run(queued.id,"mail-thread")
    assert threads._runs().get("mail-thread",queued.id).status=="pending"
    # Avoid executing that artificial continuation after the tested approval resumes.
    threads._runs().transition("mail-thread",queued.id,"cancelled")
    response=client.post("/thread/mail-thread/gmail/approve",data={"token":status["pending_gmail_token"]},follow_redirects=False)
    assert response.status_code==303
    assert chat.decisions==[{"type":"approve"}]
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
