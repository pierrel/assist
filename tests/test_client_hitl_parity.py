"""Shared-client approval contracts over real temporary service stores."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from assist.egress.store import EgressRequest, EgressStore
from assist.geo.catalog import CATALOG_FILE, Catalog
from assist.geo.proposals import Proposal, ProposalStore
from assist.geo.provisioner import Provisioner
from assist.geo.registry import RegionRegistry
from assist.run_service import RunService
from assist.thread import Thread
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from manage.web import egress, geo, phone_api, state, threads
from manage.web.app import app


HEADERS = {"Authorization": "Bearer parity-test-token"}
BASE = "/api/v1/phone/threads/parity-thread"


@pytest.fixture
def parity(tmp_path, monkeypatch):
    (tmp_path / "parity-thread").mkdir()
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "parity-test-token")
    monkeypatch.setattr(state.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(threads, "_require_deep_thread", lambda tid: None)
    runs = RunService(str(tmp_path))
    monkeypatch.setattr(threads, "_runs", lambda: runs)
    store = EgressStore(str(tmp_path / "egress"))
    monkeypatch.setattr(egress, "EGRESS_STORE", store)
    monkeypatch.setattr(threads, "EGRESS_STORE", store)
    monkeypatch.setattr(geo, "GEO_PROPOSALS", None)
    monkeypatch.setattr(geo, "GEO_REGISTRY", None)
    callbacks = []
    monkeypatch.setattr(egress, "_dispatch_egress_resolution", lambda tid: callbacks.append(("main", tid)))
    monkeypatch.setattr(egress, "_resume_egress_waiters", lambda tid: callbacks.append(("child", tid)))
    state._set_status("parity-thread", "ready")
    return TestClient(app), store, runs, callbacks


def request(store, host="docs.example.test"):
    value = EgressRequest(host, 443, "Read synthetic documentation", "parity-thread",
                          created_at=datetime.now(timezone.utc).isoformat())
    assert store.add_pending(value) is None
    return value


def proposals(client):
    response = client.get(BASE + "/approvals", headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()["proposals"]


def test_ready_thread_exposes_egress_without_a_run(parity):
    client, store, runs, _ = parity
    request(store)
    assert client.get(BASE + "/approvals").status_code == 401
    proposal, = proposals(client)
    assert proposal["kind"] == "egress"
    assert proposal["host"] == "docs.example.test" and proposal["port"] == 443
    assert proposal["decisions"] == ["hour", "always", "decline"]
    assert proposal["state"] == "pending" and runs.list("parity-thread") == []


@pytest.mark.parametrize("decision", ["hour", "always", "decline"])
def test_egress_uses_exact_existing_service_and_replays_without_dispatch(parity, decision):
    client, store, runs, callbacks = parity
    original = request(store)
    proposal, = proposals(client)
    payload = {"kind": "egress", "token": proposal["token"], "decision": decision}
    first = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert first.status_code == 200, first.text
    saved = store.get(original.key)
    assert saved.state == ("declined" if decision == "decline" else "approved")
    assert callbacks == [("main", "parity-thread"), ("child", "parity-thread")]
    replay = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert replay.status_code == 200 and replay.json()["replayed"] is True
    assert len(callbacks) == 2 and runs.list("parity-thread") == []


def test_stale_egress_card_cannot_approve_recreated_same_destination(parity):
    client, store, _, callbacks = parity
    original = request(store)
    old, = proposals(client)
    store.remove_thread("parity-thread")
    replacement = replace(original, created_at="2099-01-01T00:00:00+00:00")
    store.add_pending(replacement)
    response = client.post(BASE + "/approval", headers=HEADERS, json={
        "kind": "egress", "token": old["token"], "decision": "always"})
    assert response.status_code == 409
    assert store.get(original.key).state == "pending" and callbacks == []


def test_independent_pending_requests_dispatch_only_after_last_decision(parity):
    client, store, _, callbacks = parity
    request(store, "one.example.test")
    request(store, "two.example.test")
    first, second = proposals(client)
    for index, proposal in enumerate((first, second)):
        response = client.post(BASE + "/approval", headers=HEADERS, json={
            "kind": "egress", "token": proposal["token"], "decision": "decline"})
        assert response.status_code == 200
        assert len(callbacks) == (0 if index == 0 else 2)


def test_geo_proposal_is_discoverable_on_ready_thread(parity, tmp_path, monkeypatch):
    client, _, _, _ = parity
    props, registry = ProposalStore(str(tmp_path)), RegionRegistry(str(tmp_path))
    props.put(Proposal("test/region", "Synthetic region", (0, 0, 1, 1), 1200,
                       "parity-thread", "Plan a synthetic route", "2026-10-20T00:00:00+00:00"))
    monkeypatch.setattr(geo, "GEO_PROPOSALS", props)
    monkeypatch.setattr(geo, "GEO_REGISTRY", registry)
    proposal, = proposals(client)
    assert proposal["kind"] == "geo_download" and proposal["state"] == "pending"
    assert proposal["size_bytes"] == 1200 and "rebuild" in proposal["warning"]


@pytest.fixture
def sqlite_reply(parity, tmp_path, monkeypatch, request):
    client, _, runs, _ = parity
    effects = []
    action = {"name": "send_reply", "args": {"text": "Complete synthetic reply"}}
    unseen = {"name": "send_reply", "args": {"text": "Unseen reply"}}

    def gate(value):
        decisions = interrupt({"action_requests": [action, unseen]})["decisions"]
        if decisions[0]["type"] != "reject":
            effects.append(decisions[0].get("edited_action", action)["args"]["text"])
        assert decisions[1]["type"] == "reject"
        return {"messages": [AIMessage("Synthetic reply decision complete")]}

    connection = sqlite3.connect(tmp_path / "reply.db", check_same_thread=False)
    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    graph = builder.compile(checkpointer=SqliteSaver(connection))
    owner = runs.create("parity-thread", "general-agent", "Synthetic inbound message", sender="+15555550100")
    runs.claim("parity-thread", owner.id)
    config = {"configurable": {"thread_id": "parity-thread", "work_id": owner.work_id}}
    graph.invoke({"messages": [HumanMessage(owner.text)]}, config, durability="sync")
    if getattr(request, "param", None) != "running":
        runs.transition("parity-thread", owner.id, "awaiting_approval")

    def get(tid, configurable=None, **kwargs):
        chat = Thread.__new__(Thread)
        chat.thread_id, chat.agent = tid, graph
        chat.runconfig = {"configurable": {"thread_id": tid, **(configurable or {})}}
        chat._run = lambda value: graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
        return chat

    monkeypatch.setattr(state.MANAGER, "get", get)
    monkeypatch.setattr(state.MANAGER, "thread_default_working_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "touch", lambda tid: None)
    monkeypatch.setattr(threads, "get_cached_description", lambda tid: "Synthetic thread")
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda *args, **kwargs: None)
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *args, **kwargs: None)
    state._set_status("parity-thread", "awaiting_approval", pending_reply=action["args"]["text"],
                      pending_sender=owner.sender)
    try:
        yield client, runs, owner, effects, graph, config
    finally:
        connection.close()


@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
def test_sqlite_reply_runs_only_reviewed_action_and_keeps_receipt(sqlite_reply, decision):
    client, runs, owner, effects, _, _ = sqlite_reply
    proposal, = proposals(client)
    payload = {"kind": "send_reply", "token": proposal["token"], "decision": decision}
    if decision == "edit":
        payload["text"] = "Edited\r\nsynthetic reply"
    response = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert response.status_code == 200, response.text
    accepted = runs.get("parity-thread", response.json()["run_id"])
    assert accepted.work_id == owner.work_id and effects == []
    threads._execute_run(accepted.id, "parity-thread")
    assert runs.get("parity-thread", accepted.id).status == "success"
    assert effects == ([] if decision == "reject" else
                       ["Edited\nsynthetic reply" if decision == "edit" else "Complete synthetic reply"])
    replay = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert replay.status_code == 200 and replay.json()["run_id"] == accepted.id
    assert replay.json()["replayed"] is True
    assert len(effects) == (0 if decision == "reject" else 1)


def test_reply_conflicting_decision_and_sender_change_are_refused(sqlite_reply):
    client, runs, _, effects, _, _ = sqlite_reply
    proposal, = proposals(client)
    payload = {"kind": "send_reply", "token": proposal["token"], "decision": "approve"}
    first = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert first.status_code == 200
    conflicting = client.post(BASE + "/approval", headers=HEADERS, json={**payload, "decision": "reject"})
    assert conflicting.status_code == 409
    state._set_status("parity-thread", "awaiting_approval", pending_reply="Complete synthetic reply",
                      pending_sender="+15555550101")
    threads._execute_run(first.json()["run_id"], "parity-thread")
    assert runs.get("parity-thread", first.json()["run_id"]).status == "cancelled"
    assert effects == []


@pytest.mark.parametrize("sqlite_reply", ["running"], indirect=True)
def test_reply_publication_window_retains_checkpoint_work_and_original_receipt(sqlite_reply):
    client, runs, owner, effects, _, _ = sqlite_reply
    assert runs.get("parity-thread", owner.id).status == "running"
    proposal, = proposals(client)
    response = client.post(BASE + "/approval", headers=HEADERS, json={
        "kind": "send_reply", "token": proposal["token"], "decision": "approve"})
    assert response.status_code == 200
    accepted = runs.get("parity-thread", response.json()["run_id"])
    assert accepted.work_id == owner.work_id
    runs.transition("parity-thread", owner.id, "awaiting_approval")
    threads._execute_run(accepted.id, "parity-thread")
    original = client.get(BASE + "/runs/" + owner.id, headers=HEADERS)
    assert original.status_code == 200 and original.json()["status"] == "success"
    assert effects == ["Complete synthetic reply"]


def test_consumed_reply_gate_does_not_hide_independent_approvals(sqlite_reply, monkeypatch):
    client, _, _, effects, graph, config = sqlite_reply
    independent = request(egress.EGRESS_STORE)
    chat = state.MANAGER.get("parity-thread")
    pending = chat.pending_reply

    def consume_after_read():
        args = pending()
        if args:
            graph.invoke(Command(resume={"decisions": [{"type": "reject"}, {"type": "reject"}]}),
                         config, durability="sync")
        return args

    chat.pending_reply = consume_after_read
    monkeypatch.setattr(state.MANAGER, "get", lambda *args, **kwargs: chat)
    value, = proposals(client)
    assert value["kind"] == "egress" and value["host"] == independent.host
    assert threads.reply_approval_preview("parity-thread") is None
    threads.render_thread("parity-thread", chat)
    assert effects == []


@pytest.fixture
def download(parity, tmp_path, monkeypatch):
    client, _, _, _ = parity
    props, registry = ProposalStore(str(tmp_path)), RegionRegistry(str(tmp_path))
    proposal = Proposal("test/region", "Synthetic region", (0, 0, 1, 1), 1200,
                        "parity-thread", "Plan a synthetic route", "2026-10-20T00:00:00+00:00")
    props.put(proposal)
    release, started = threading.Event(), threading.Event()
    jobs = []

    def job(op, slug):
        jobs.append((op, slug))
        started.set()
        assert release.wait(10), "test did not release the fake download"
        return False

    provisioner = Provisioner(registry, props, job, lambda *args: None, lambda: False)
    monkeypatch.setattr(geo, "GEO_PROPOSALS", props)
    monkeypatch.setattr(geo, "GEO_REGISTRY", registry)
    monkeypatch.setattr(geo, "_PROVISIONER", provisioner)
    try:
        yield client, props, registry, proposal, provisioner, release, started, jobs
    finally:
        release.set()
        provisioner._executor.shutdown(wait=True)


def test_geo_queued_failed_and_replayed_decisions_never_repeat_job(download):
    client, props, registry, original, provisioner, release, started, jobs = download
    proposal, = proposals(client)
    payload = {"kind": "geo_download", "token": proposal["token"], "decision": "approve"}
    first = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert first.status_code == 200 and first.json()["status"] == "importing"
    assert started.wait(1)
    assert proposals(client)[0]["state"] == "importing"
    assert client.post(BASE + "/approval", headers=HEADERS, json=payload).json()["replayed"] is True
    decline = client.post(BASE + "/approval", headers=HEADERS, json={**payload, "decision": "reject"})
    assert decline.status_code == 409 and props.get(original.slug) == original
    release.set()
    provisioner._executor.submit(lambda: None).result(timeout=2)
    assert proposals(client)[0]["state"] == "failed"
    replay = client.post(BASE + "/approval", headers=HEADERS, json=payload)
    assert replay.status_code == 200 and replay.json()["status"] == "failed"
    assert replay.json()["replayed"] is True and len(jobs) == 1
    assert registry.get(original.slug).approval_id == original.identity
    # A genuinely fresh proposal has its own pending review despite the old failure.
    props.put(replace(original, created_at="2026-10-20T01:00:00+00:00"))
    assert proposals(client)[0]["state"] == "pending"
    assert client.post(BASE + "/approval", headers=HEADERS, json=payload).status_code == 409


def test_geo_replacement_and_decline_preserve_exact_origin(download):
    client, props, _, original, _, _, _, jobs = download
    old, = proposals(client)
    props.put(replace(original, origin_tid="another-thread"))
    payload = {"kind": "geo_download", "token": old["token"], "decision": "approve"}
    assert client.post(BASE + "/approval", headers=HEADERS, json=payload).status_code == 409
    assert jobs == [] and proposals(client) == []
    props.put(original)
    payload["decision"] = "reject"
    assert client.post(BASE + "/approval", headers=HEADERS, json=payload).json()["status"] == "declined"
    assert props.get(original.slug) is None and jobs == []


def test_delayed_egress_batch_cannot_skip_a_new_pending_card(parity):
    _, store, _, _ = parity
    first = request(store)
    store.resolve(first.key, "hour")
    second = request(store, "next.example.test")
    assert store.take_undispatched("parity-thread") == []
    assert not store.get(first.key).dispatched
    store.resolve(second.key, "decline")
    assert len(store.take_undispatched("parity-thread")) == 2
    assert store.take_undispatched("parity-thread") == []


def test_concurrent_egress_decisions_share_one_result(parity):
    client, store, _, callbacks = parity
    request(store)
    proposal, = proposals(client)
    payload = {"kind": "egress", "token": proposal["token"], "decision": "hour"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: client.post(BASE + "/approval", headers=HEADERS, json=payload), range(2)))
    assert [value.status_code for value in results] == [200, 200]
    assert sorted(value.json()["replayed"] for value in results) == [False, True]
    assert len(callbacks) == 2


def test_oversize_mail_does_not_hide_independent_network_request(parity):
    client, store, _, _ = parity
    request(store)
    state._set_status("parity-thread", "awaiting_approval", pending_email_token="synthetic-email-token",
                      pending_email_to="reader@example.test", pending_email_subject="Subject",
                      pending_email_body="x" * phone_api.MAX_APPROVAL_BYTES)
    values = proposals(client)
    assert values[0]["state"] == "unavailable" and values[0]["decisions"] == []
    assert values[1]["kind"] == "egress" and values[1]["state"] == "pending"


@pytest.mark.parametrize("replacement_time", ["before_delivery", "during_callback"])
def test_geo_completion_never_consumes_a_new_proposal(download, replacement_time):
    client, props, _, original, provisioner, release, _, _ = download
    proposal, = proposals(client)
    assert client.post(BASE + "/approval", headers=HEADERS, json={
        "kind": "geo_download", "token": proposal["token"], "decision": "approve"}).status_code == 200
    release.set()
    provisioner._executor.submit(lambda: None).result(timeout=2)
    replacement = replace(original, created_at="2026-10-20T02:00:00+00:00")
    sent = []

    def complete(*args):
        sent.append(args)
        if replacement_time == "during_callback":
            props.put(replacement)

    provisioner._on_complete = complete
    provisioner._health_check = lambda: True
    if replacement_time == "before_delivery":
        props.put(replacement)
    provisioner.deliver_pending()
    assert props.get(original.slug) == replacement
    assert len(sent) == (0 if replacement_time == "before_delivery" else 1)
    provisioner.deliver_pending()
    assert props.get(original.slug) == replacement and len(sent) <= 1


def test_geo_delayed_failure_delivery_cannot_describe_a_running_generation(download):
    _, props, _, original, provisioner, _, started, _ = download
    provisioner.decide(original, "approve")
    assert started.wait(1)
    sent = []
    provisioner._on_complete = lambda *args: sent.append(args)
    provisioner._health_check = lambda: True
    provisioner._deliver(original.slug, failed=True)
    assert sent == [] and props.get(original.slug) == original


def test_web_reply_content_and_token_come_from_same_preview(sqlite_reply, monkeypatch):
    _, _, _, _, _, _ = sqlite_reply
    chat = state.MANAGER.get("parity-thread")
    old = threads.approval_status

    def initial_status(tid):
        value = old(tid)
        state._set_status(tid, "awaiting_approval", pending_reply="Complete synthetic reply",
                          pending_sender="+15555550101")
        return value

    monkeypatch.setattr(threads, "approval_status", initial_status)
    page = threads.render_thread("parity-thread", chat)
    assert "to +15555550101:" in page
    assert "to +15555550100:" not in page
    assert threads.reply_approval_preview("parity-thread")["token"] in page


def test_reply_reopened_store_and_rotated_token_keep_original_receipt(sqlite_reply, monkeypatch):
    client, runs, _, effects, _, _ = sqlite_reply
    proposal, = proposals(client)
    first, replayed = threads.reply_decision_core("parity-thread", "approve", proposal["token"])
    assert not replayed
    reopened = RunService(runs.root_dir)
    monkeypatch.setattr(threads, "_runs", lambda: reopened)
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "rotated-parity-token")
    refreshed = threads.reply_approval_preview("parity-thread")
    assert refreshed["token"] != proposal["token"] and refreshed["state"] == "accepted"
    again, replayed = threads.reply_decision_core("parity-thread", "approve", refreshed["token"])
    assert replayed and again.id == first.id
    with pytest.raises(HTTPException) as rejected:
        threads.reply_decision_core("parity-thread", "reject", refreshed["token"])
    assert rejected.value.status_code == 409
    assert len(reopened.list("parity-thread")) == 2 and effects == []


def test_geo_late_size_probe_cannot_replace_an_admitted_proposal(download, tmp_path, monkeypatch):
    from assist.geo import tools
    _, props, registry, original, provisioner, _, started, _ = download
    (tmp_path / CATALOG_FILE).write_text(json.dumps([{
        "slug": original.slug, "display_name": original.display_name,
        "bbox": original.bbox, "url": "https://download.geofabrik.de/synthetic.osm.pbf"}]))
    _, _, propose = tools.geo_tools(registry, Catalog(str(tmp_path)), props)
    probing, finish_probe = threading.Event(), threading.Event()

    def size(_url):
        probing.set()
        assert finish_probe.wait(3)
        return 9999

    monkeypatch.setattr(tools, "_head_size", size)
    monkeypatch.setattr(tools, "_thread_id", lambda: "replacement-thread")
    with ThreadPoolExecutor(max_workers=1) as pool:
        publication = pool.submit(propose, original.slug, "A different synthetic request")
        try:
            assert probing.wait(1)
            assert provisioner.decide(original, "approve") == ("importing", False)
            assert started.wait(1)
        finally:
            finish_probe.set()
        assert "already loaded or downloading" in publication.result(timeout=2)
    assert props.get(original.slug) == original
    assert registry.get(original.slug).approval_id == original.identity
