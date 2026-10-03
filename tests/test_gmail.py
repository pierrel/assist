"""Synthetic Gmail boundary tests. No real mailbox credentials or network."""
import base64
import json
from email.message import EmailMessage
from types import SimpleNamespace

import pytest

from assist import gmail


def _mail(body="Your booking ZX-42 is confirmed.", subject="Booking receipt", html=False):
    message = EmailMessage()
    message["From"] = "Travel Desk <desk@example.test>"
    message["To"] = "reader@example.test"
    message["Subject"] = subject
    message["Date"] = "Thu, 01 Oct 2026 10:00:00 +0000"
    message["List-Unsubscribe"] = "<mailto:leave@example.test>, <https://example.test/leave>"
    message.set_content(body, subtype="html" if html else "plain")
    return message.as_bytes()


def _decoded(body="Your booking ZX-42 is confirmed.", html=False):
    return gmail.decode_message("abc123", _mail(body, html=html),
                                {"internalDate": "1790848800000", "labelIds": ["INBOX"]})


class _Response:
    def __init__(self, value=None, status=200, chunks=None):
        self.status_code = status
        self.chunks = chunks if chunks is not None else [json.dumps(value or {}).encode()]
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, size):
        return iter(self.chunks)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    path = tmp_path / "private" / "gmail.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"client_id": "synthetic-client", "client_secret": "secret",
                               "refresh_token": "private-refresh", "token_uri": gmail._TOKEN_URL,
                               "scopes": [gmail.GMAIL_SCOPE]}))
    path.chmod(0o600)
    monkeypatch.setenv("ASSIST_GMAIL_TOKEN_FILE", str(path))
    monkeypatch.setenv("ASSIST_THREADS_DIR", str(tmp_path / "threads"))
    return path


def test_html_mail_is_text_and_links_without_scripts_or_external_requests(monkeypatch):
    monkeypatch.setattr(gmail.requests, "request", lambda *a, **k: pytest.fail("mail fetched content"))
    value = _decoded('<p>Hello &amp; welcome</p><script>steal()</script>'
                     '<img src="https://tracker.example.test/pixel">'
                     '<a href="https://example.test/account">Account</a>'
                     '<a href="javascript:steal()">Unsafe</a>', html=True)
    assert "Hello & welcome" in value["body"]
    assert "steal()" not in value["body"]
    assert value["links"] == ["https://example.test/account"]
    assert value["list_unsubscribe"].startswith("<mailto:")
    assert value["url"].endswith("abc123")
    assert "Untrusted" in value["trust"]


def test_mime_alternative_prefers_text_and_omits_attachment():
    message = EmailMessage()
    message.set_content("Plain evidence")
    message.add_alternative("<p>HTML alternative</p>", subtype="html")
    message.add_attachment(b"private attachment", maintype="application", subtype="octet-stream",
                           filename="private.dat")
    value = gmail.decode_message("abc123", message.as_bytes(), {})
    assert value["body"] == "Plain evidence\n"
    assert "attachment" not in value["body"]


def test_client_fixed_endpoints_and_auth_never_enter_results(configured, monkeypatch):
    calls = []
    payload = {"mimeType":"text/plain", "headers":[{"name":"Subject","value":"Receipt"}],
               "body":{"data":base64.urlsafe_b64encode(b"Your booking ZX-42 is confirmed.").decode()}}
    responses = [_Response({"access_token": "private-access"}),
                 _Response({"payload": payload, "internalDate": "1790848800000"})]
    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return responses[len(calls)-1]
    monkeypatch.setattr(gmail.requests, "request", request)
    result = gmail.email_read("abc123")
    assert "ZX-42" in result and "private-access" not in result and "private-refresh" not in result
    assert calls[0][1] == gmail._TOKEN_URL
    assert calls[1][1] == gmail._API + "/messages/abc123"
    assert calls[1][2]["params"] == {"format":"full"}
    assert calls[1][2]["headers"] == {"Authorization": "Bearer private-access"}
    assert all(c[2]["allow_redirects"] is False and c[2]["stream"] for c in calls)
    assert all(response.closed for response in responses)


@pytest.mark.parametrize("mode", ["public", "symlink", "in_thread", "wrong_scope", "wrong_endpoint"])
def test_credentials_fail_closed(configured, monkeypatch, mode):
    if mode == "public":
        configured.chmod(0o644)
    elif mode == "symlink":
        link = configured.parent / "link.json"
        link.symlink_to(configured)
        monkeypatch.setenv("ASSIST_GMAIL_TOKEN_FILE", str(link))
    elif mode == "in_thread":
        monkeypatch.setenv("ASSIST_THREADS_DIR", str(configured.parent))
    else:
        value = json.loads(configured.read_text())
        value["scopes" if mode == "wrong_scope" else "token_uri"] = ["gmail.send"] if mode == "wrong_scope" else "https://evil.test/token"
        configured.write_text(json.dumps(value))
    monkeypatch.setattr(gmail.requests, "request", lambda *a, **k: pytest.fail("unsafe credential used"))
    assert gmail.email_read("abc123").startswith("Email read failed:")


@pytest.mark.parametrize("message_id", ["../send", "abc/modify", "abc\\modify", "a\nAuthorization:bad", ".", "..", "x" * 513, "\ud800"])
def test_ids_cannot_select_other_endpoints(configured, monkeypatch, message_id):
    calls = []
    monkeypatch.setattr(gmail.requests, "request", lambda *a, **k: calls.append(a) or _Response({"access_token":"a"}))
    assert gmail.email_read(message_id).startswith("Email read failed:")
    assert len(calls) == 1  # OAuth only; no attacker-controlled API path.


def test_oversized_response_closes_and_does_not_leak_response(monkeypatch):
    response = _Response(chunks=[b"x" * (gmail._MAX_RESPONSE + 1)])
    monkeypatch.setattr(gmail.requests, "request", lambda *a, **k: response)
    with pytest.raises(gmail.GmailError, match="size limit"):
        gmail._json_request("GET", gmail._API)
    assert response.closed
    response = _Response({"error":"PRIVATE PROVIDER DETAIL"}, status=401)
    monkeypatch.setattr(gmail.requests, "request", lambda *a, **k: response)
    with pytest.raises(gmail.GmailError) as failure:
        gmail._json_request("GET", gmail._API)
    assert "PRIVATE" not in str(failure.value)


def test_search_and_filters_date_native_narrowing_and_pagination(monkeypatch):
    first = _decoded()
    second = {**first,"id":"def456","from":"Other <other@example.test>"}
    calls = []
    class Client:
        def request(self, *args, **kwargs):
            calls.append(kwargs["params"])
            return {"messages":[{"id":"abc123"},{"id":"def456"}],"nextPageToken":"next"}
        def read(self, message_id):
            return first if message_id == "abc123" else second
    monkeypatch.setattr(gmail,"GmailClient",Client)
    result = json.loads(gmail.email_search(query="in:inbox", sender_regex=r"desk@example\.test",
                                         subject_regex="receipt",body_regex=r"ZX-\d+",
                                         date_regex="2026",after="2026-09-01",before="2026-11-01",
                                         page_token="previous",scan_limit=2))
    assert [m["id"] for m in result["messages"]] == ["abc123"]
    assert result["scanned"] == 2 and result["complete"] is False
    assert result["next_page_token"] == "next"
    assert calls == [{"q":"in:inbox after:2026-09-01 before:2026-11-01","maxResults":2,"pageToken":"previous"}]


@pytest.mark.parametrize("kwargs", [{"scan_limit":0},{"scan_limit":51},{"scan_limit":True},
                                   {"body_regex":"["},{"body_regex":"x"*513},
                                   {"after":"2026-02-30"},{"after":"foo is:sent"}])
def test_invalid_search_never_calls_provider(monkeypatch, kwargs):
    monkeypatch.setattr(gmail,"GmailClient",lambda:pytest.fail("invalid search called provider"))
    assert gmail.email_search(**kwargs).startswith("Email search failed:")


def test_pathological_regex_is_time_bounded(monkeypatch):
    message = _decoded("a"*100000 + "!")
    monkeypatch.setattr(gmail,"GmailClient",lambda:SimpleNamespace(
        request=lambda *a,**k:{"messages":[{"id":"abc123"}]}, read=lambda _:message))
    result=json.loads(gmail.email_search(body_regex="(a+)+$"))
    assert result["complete"] is False and "timed out" in result["skipped"][0]["error"]


def test_truncated_body_search_never_claims_complete(monkeypatch):
    message = _decoded("x"*(gmail._MAX_BODY+1))
    monkeypatch.setattr(gmail,"GmailClient",lambda:SimpleNamespace(
        request=lambda *a,**k:{"messages":[{"id":"abc123"}]}, read=lambda _:message))
    result=json.loads(gmail.email_search(body_regex="not-there"))
    assert result["complete"] is False and result["incomplete_bodies"] == 1
    with pytest.raises(gmail.GmailError,match="complete approval"):
        gmail.gmail_action_preview({"name":"email_delete","args":{"message_ids":["abc123"]}})


def test_archive_and_trash_only_exact_ids_with_partial_failure(monkeypatch):
    calls=[]
    def request(method,suffix,**kwargs):
        calls.append((method,suffix,kwargs))
        if suffix.endswith("def456/trash"):
            raise gmail.GmailError("HTTP 503; outcome unknown")
        return {}
    monkeypatch.setattr(gmail,"GmailClient",lambda:SimpleNamespace(request=request))
    assert json.loads(gmail.email_archive(["abc123"]))["completed"] == ["abc123"]
    result=json.loads(gmail.email_delete(["abc123","def456"]))
    assert result["completed"] == ["abc123"] and "error" in result
    assert calls[0] == ("POST","/messages/abc123/modify",{"json":{"removeLabelIds":["INBOX"]}})
    assert calls[1] == ("POST","/messages/abc123/trash",{"json":{}})
    assert all("send" not in call[1] for call in calls)


def test_real_framework_interrupt_before_mutation_and_rejection(monkeypatch):
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    class Model(FakeMessagesListChatModel):
        def bind_tools(self,tools,**kwargs):
            return self
    mutations=[]
    monkeypatch.setattr(gmail,"_mutate",lambda ids,action:mutations.append((ids,action)) or "done")
    for decision in ("approve","reject"):
        model=Model(responses=[AIMessage(content="",tool_calls=[{
            "name":"email_delete","args":{"message_ids":["abc123"]},"id":"call-1","type":"tool_call"}]),
            AIMessage(content="Done")])
        agent=create_agent(model,tools=[gmail.email_delete],
                           middleware=[HumanInTheLoopMiddleware(interrupt_on=gmail.GMAIL_INTERRUPT_ON)],
                           checkpointer=InMemorySaver())
        config={"configurable":{"thread_id":decision}}
        result=agent.invoke({"messages":[{"role":"user","content":"Remove this mail"}]},config)
        assert result["__interrupt__"] and len(mutations)==(0 if decision=="approve" else 1)
        agent.invoke(Command(resume={"decisions":[{"type":decision}]}),config)
    assert mutations==[(["abc123"],"trash")]


def test_parallel_requests_only_execute_the_one_displayed(monkeypatch,tmp_path):
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from assist.thread import Thread
    from manage.web.threads import _pending_gmail, _resume_gmail
    class Model(FakeMessagesListChatModel):
        def bind_tools(self,tools,**kwargs):
            return self
    calls=[]
    monkeypatch.setattr(gmail,"_mutate",lambda ids,action:calls.append((ids,action)) or "done")
    agent=create_agent(Model(responses=[AIMessage(content="",tool_calls=[
        {"name":"email_delete","args":{"message_ids":["abc123"]},"id":"call-1","type":"tool_call"},
        {"name":"email_archive","args":{"message_ids":["def456"]},"id":"call-2","type":"tool_call"}]),
        AIMessage(content="Done")]),tools=[gmail.email_delete,gmail.email_archive],
        middleware=[HumanInTheLoopMiddleware(interrupt_on=gmail.GMAIL_INTERRUPT_ON)],
        checkpointer=InMemorySaver())
    chat=Thread(str(tmp_path),thread_id="parallel",agent=agent,
                model=Model(responses=[AIMessage(content="unused")]))
    chat.message("Organize these messages")
    proposal=_pending_gmail(chat)
    assert calls==[]
    _resume_gmail(chat,{"type":"approve"})
    expected_action="archive" if proposal["name"]=="email_archive" else "trash"
    assert calls==[(proposal["args"]["message_ids"],expected_action)]
    assert not chat.pending_actions()


def test_large_attachment_reference_does_not_block_small_message_body(configured,monkeypatch):
    payload={"mimeType":"multipart/mixed","parts":[
        {"mimeType":"text/plain","body":{"data":base64.urlsafe_b64encode(b"Small body evidence").decode()}},
        {"mimeType":"application/octet-stream","filename":"large.bin",
         "body":{"attachmentId":"large-attachment","size":30*1024*1024}}]}
    calls=[]
    def request(method,url,**kwargs):
        calls.append(url)
        return _Response({"access_token":"a"} if url==gmail._TOKEN_URL else {"payload":payload})
    monkeypatch.setattr(gmail.requests,"request",request)
    value=json.loads(gmail.email_read("abc123"))
    assert value["body"]=="Small body evidence"
    assert len(calls)==2 and not any("attachments" in url for url in calls)


def test_unreadable_message_preserves_other_matches_and_next_page(monkeypatch):
    def read(message_id):
        if message_id=="def456":
            raise gmail.GmailError("Message exceeds size limit")
        return _decoded()
    monkeypatch.setattr(gmail,"GmailClient",lambda:SimpleNamespace(
        request=lambda *a,**k:{"messages":[{"id":"def456"},{"id":"abc123"}],"nextPageToken":"next"},read=read))
    value=json.loads(gmail.email_search())
    assert [message["id"] for message in value["messages"]]==["abc123"]
    assert value["next_page_token"]=="next" and value["complete"] is False
    assert value["skipped"]==[{"id":"def456","error":"Message exceeds size limit"}]


def test_full_payload_preserves_unicode_and_declared_legacy_charset():
    for charset,body in (("utf-8","Café — hello"),("iso-8859-1","Café")):
        payload={"mimeType":"text/plain","headers":[{"name":"Content-Type","value":f"text/plain; charset={charset}"}],
                 "body":{"data":base64.urlsafe_b64encode(body.encode(charset)).decode()}}
        value=gmail._decode_message("abc123",gmail._payload_message(payload),{})
        assert value["body"]==body


def test_one_regex_timeout_preserves_prior_matches_and_cursor(monkeypatch):
    messages={"abc123":_decoded("aaa"),"def456":_decoded("a"*100000+"!")}
    monkeypatch.setattr(gmail,"GmailClient",lambda:SimpleNamespace(
        request=lambda *a,**k:{"messages":[{"id":"abc123"},{"id":"def456"}],"nextPageToken":"next"},
        read=lambda key:messages[key]))
    result=json.loads(gmail.email_search(body_regex="(a+)+$"))
    assert [message["id"] for message in result["messages"]]==["abc123"]
    assert result["next_page_token"]=="next" and result["complete"] is False
    assert result["skipped"][0]["id"]=="def456"


def test_null_provider_body_is_a_gmail_error(monkeypatch):
    client=object.__new__(gmail.GmailClient)
    monkeypatch.setattr(client,"request",lambda *a,**k:{"payload":{"mimeType":"text/plain","body":None}})
    with pytest.raises(gmail.GmailError,match="structure/encoding"):
        client.read("abc123")


def test_opaque_message_id_read_mutations_and_link_are_encoded(monkeypatch):
    from urllib.parse import quote
    message_id = "opaque:Q_?#%" + "Z" * 40 + "é"
    calls = []
    client = object.__new__(gmail.GmailClient)
    def request(method, suffix, **kwargs):
        calls.append((method, suffix, kwargs))
        return {"payload":{"mimeType":"text/plain","body":{"data":""}}}
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(gmail, "GmailClient", lambda: client)
    value = client.read(message_id)
    assert value["id"] == message_id
    assert value["url"] == "https://mail.google.com/mail/u/0/#all/" + quote(message_id, safe="")
    assert json.loads(gmail.email_archive([message_id]))["completed"] == [message_id]
    assert json.loads(gmail.email_delete([message_id]))["completed"] == [message_id]
    segment = "/messages/" + quote(message_id, safe="")
    assert calls == [("GET", segment, {"params":{"format":"full"}}),
                     ("POST", segment + "/modify", {"json":{"removeLabelIds":["INBOX"]}}),
                     ("POST", segment + "/trash", {"json":{}})]


@pytest.mark.parametrize("disposition", [None, "inline"])
def test_filename_text_attachment_cannot_replace_real_body(monkeypatch, disposition):
    attachment = {"mimeType": "text/plain", "filename": "notes.txt", "body": {"data": base64.urlsafe_b64encode(b"Attachment content").decode()}}
    if disposition:
        attachment["headers"] = [{"name": "Content-Disposition", "value": disposition}]
    payload = {"mimeType": "multipart/alternative", "parts": [attachment, {
        "mimeType": "text/html", "body": {"data": base64.urlsafe_b64encode(b"<p>Actual booking body</p>").decode()}}]}
    client = gmail.GmailClient.__new__(gmail.GmailClient)
    monkeypatch.setattr(client, "request", lambda *a, **k: {"payload": payload})
    result = client.read("abc123")
    assert result["body"].strip() == "Actual booking body"
    assert not result["body_truncated"]
