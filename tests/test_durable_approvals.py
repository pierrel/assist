"""Run-owned approvals at real storage and SQLite checkpoint crash boundaries."""
from dataclasses import replace
from contextlib import nullcontext
import json
import os
import sqlite3
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

from assist.run_service import InvalidRunTransition, RunService, RunStoreUnavailable
from assist.git_sync import GitSyncError
from assist.thread import Thread
from manage.web import phone_api, state, threads


ACTION = {"name":"send_email", "args":{"to":"reader@example.test", "subject":"Subject", "body":"Body"}}
IDENTITY = ["Assistant <assistant@example.test>", "oversight@example.test"]
PROPOSAL = {"kind":"send_email", "action":ACTION, "from":IDENTITY[0], "cc":IDENTITY[1]}
DECISION = {"type":"approve", "approval_interrupt_id":"gate", "email_review_identity":IDENTITY}


@pytest.fixture
def records(tmp_path):
    (tmp_path / "approval-test").mkdir()
    service = RunService(str(tmp_path))
    owner = service.create("approval-test", "general-agent", "Email this person")
    service.claim(owner.thread_id, owner.id)
    approval = service.publish_approval(owner.thread_id, owner.id, action=ACTION,
        requests=[ACTION], proposal=PROPOSAL, interrupt_id="gate", checkpoint_id="proposal")
    return service, owner, approval


def accept(service, approval, decision=DECISION):
    return service.accept_approval(approval.thread_id, approval.id, decision, [{"type":decision["type"]}])


def test_atomic_acceptance_failure_before_replace(records, monkeypatch):
    service, owner, approval = records
    original = os.replace
    def fail(source, destination):
        if str(destination).endswith("runs.json"):
            raise OSError("crash before atomic rename")
        return original(source, destination)
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        accept(service, approval)
    restarted = RunService(service.root_dir)
    assert restarted.list(owner.thread_id) == service.list(owner.thread_id)
    assert len(restarted.list(owner.thread_id)) == 1
    assert restarted.approval(owner.thread_id, approval.id).state == "pending"
    assert restarted.approval(owner.thread_id, approval.id).decision is None


def test_atomic_acceptance_survives_lost_response_and_recovery(records, monkeypatch):
    service, owner, approval = records
    write = service._write
    def commit_then_die(*args):
        write(*args)
        raise OSError("crash after atomic rename")
    monkeypatch.setattr(service, "_write", commit_then_die)
    with pytest.raises(OSError):
        accept(service, approval)
    restarted = RunService(service.root_dir)
    accepted, replay = accept(restarted, approval)
    assert replay and len(restarted.list(owner.thread_id)) == 2
    restarted.claim(owner.thread_id, accepted.id)
    successor = restarted.continue_approval(owner.thread_id, accepted.id)
    original, replay = accept(RunService(service.root_dir), approval)
    assert replay and original.id == accepted.id
    current = restarted.approval(owner.thread_id, approval.id)
    assert current.accepted_run_id == accepted.id and current.execution_run_id == successor.id
    assert successor.approval_id == approval.id and successor.work_id == owner.work_id


def test_rejection_is_unfinished_until_checkpoint_completion(records):
    service, owner, approval = records
    run, _ = accept(service, approval, {"type":"reject"})
    record = service.unfinished_approval_continuation(owner.thread_id)
    assert record.state == "rejected" and record.execution_run_id == run.id
    service.claim(owner.thread_id, run.id)
    service.reconcile_approval(owner.thread_id, approval.id, checkpoint_id="after", consumed=True)
    assert service.approval(owner.thread_id, approval.id).state == "consumed"
    service.reconcile_approval(owner.thread_id, approval.id, checkpoint_id="end", completed=True, outcome="checkpoint_end")
    assert service.unfinished_approval_continuation(owner.thread_id) is None
    assert service.get(owner.thread_id, run.id).status == "running"
    service.transition(owner.thread_id, run.id, "success")


def test_review_revision_keeps_proposal_identity_and_forbids_accepted_refresh(records):
    service, owner, original = records
    revised = service.publish_approval(owner.thread_id, owner.id, action=ACTION,
        requests=[ACTION], proposal={**PROPOSAL, "cc":"different@example.test"},
        interrupt_id="gate", checkpoint_id="proposal")
    assert revised.id != original.id and revised.proposal_id == original.proposal_id
    assert service.approval(owner.thread_id, original.id).outcome == "new_proposal"
    with pytest.raises(InvalidRunTransition):
        accept(service, original)
    accept(service, revised)
    before = service.list(owner.thread_id)
    with pytest.raises(InvalidRunTransition):
        service.publish_approval(owner.thread_id, owner.id, action=ACTION, requests=[ACTION],
            proposal=PROPOSAL, interrupt_id="gate", checkpoint_id="proposal")
    with pytest.raises(InvalidRunTransition):
        service.publish_approval(owner.thread_id, owner.id, action=ACTION, requests=[ACTION],
            proposal=PROPOSAL, interrupt_id="gate", checkpoint_id="proposal", approval_id=revised.id)
    assert service.list(owner.thread_id) == before


def test_blocked_continuation_is_durable_and_excluded_from_dispatch(records, monkeypatch):
    service, owner, approval = records
    accepted, _ = accept(service, approval)
    follower = service.create(owner.thread_id, "general-agent", "Next question")
    service.reconcile_approval(owner.thread_id, approval.id, checkpoint_id=None, blocked_reason="unreadable checkpoint")
    restarted = RunService(service.root_dir)
    monkeypatch.setattr(threads, "_runs", lambda:restarted)
    queued = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda *args, **kwargs:queued.append(args))
    threads._dispatch_pending_after(owner.thread_id)
    assert queued == [] and restarted.get(owner.thread_id, follower.id).status == "pending"
    assert restarted.get(owner.thread_id, accepted.id).status == "cancelled"
    assert restarted.unfinished_approval_continuation(owner.thread_id).blocked_reason == "unreadable checkpoint"


@pytest.fixture
def sqlite_approval(tmp_path, monkeypatch):
    tid = "approval-test"
    (tmp_path / tid).mkdir()
    service = RunService(str(tmp_path))
    monkeypatch.setattr(threads, "_runs", lambda:service)
    monkeypatch.setattr(state.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda value:str(tmp_path / value))
    monkeypatch.setattr(state.MANAGER, "thread_default_working_dir", lambda value:str(tmp_path / value))
    monkeypatch.setattr(state.MANAGER, "touch", lambda value:None)
    monkeypatch.setattr(state.MANAGER, "list", lambda:[tid])
    monkeypatch.setattr(threads, "_require_deep_thread", lambda value:None)
    monkeypatch.setattr(threads, "_get_sandbox_backend", lambda *args, **kwargs:None)
    monkeypatch.setattr(threads, "get_cached_description", lambda value:"Synthetic thread")
    monkeypatch.setattr(threads.SandboxManager, "cleanup", lambda *args, **kwargs:None)
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "assistant@example.test")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Assistant")
    monkeypatch.setenv("EMAIL_ALWAYS_CC", "oversight@example.test")
    queued = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit", lambda rid, value, **kwargs:queued.append((rid, value)))
    return tid, tmp_path, service, queued


@pytest.mark.parametrize("window", ["accepted", "consumed", "marked", "ended", "new_gate"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_sqlite_checkpoint_crash_boundaries(sqlite_approval, monkeypatch, window, decision):
    tid, root, service, queued = sqlite_approval
    effects, decisions = [], []
    extra = {"name":"send_email", "args":{**ACTION["args"], "to":"unseen@example.test"}}
    def gate(value):
        chosen = interrupt({"action_requests":[ACTION, extra]})
        decisions.append(chosen)
        if chosen["decisions"][0]["type"] != "reject":
            effects.append(ACTION["args"])
        assert chosen["decisions"][1]["type"] == "reject"
        return {"messages":[AIMessage("Decision consumed")]}
    def finish(value):
        if window == "new_gate":
            interrupt({"action_requests":[ACTION]})
        return {"messages":[AIMessage("Done")]}
    def build(connection):
        graph = StateGraph(MessagesState)
        graph.add_node("gate", gate); graph.add_node("finish", finish)
        graph.add_edge(START, "gate"); graph.add_edge("gate", "finish"); graph.add_edge("finish", END)
        return graph.compile(checkpointer=SqliteSaver(connection),
                             interrupt_before=["finish"] if window in {"consumed", "marked"} else [])
    connection = sqlite3.connect(root / "checkpoint.db", check_same_thread=False)
    graph = build(connection)
    owner = service.create(tid, "general-agent", "Email this person")
    service.claim(tid, owner.id)
    config = {"configurable":{"thread_id":tid, "work_id":owner.work_id}}
    graph.invoke({"messages":[HumanMessage(owner.text)]}, config, durability="sync")
    snapshot = graph.get_state(config)
    approval = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION, extra],
        proposal=PROPOSAL, interrupt_id=snapshot.interrupts[0].id,
        checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
    # No status.json card is required for browser/phone acceptance.
    preview = phone_api._approval_preview(tid)["proposal"]
    assert preview["token"] == approval.id
    accepted, _ = threads.email_decision_core(tid, decision, approval.id, phone_preview=True)
    follower = service.create(tid, "general-agent", "Follower")
    service.claim(tid, accepted.id)
    if window != "accepted":
        config["configurable"].update(approval_id=approval.id)
        graph.invoke(Command(resume={"decisions":list(service.approval(tid, approval.id).decisions)}), config, durability="sync")
        if window == "marked":
            service.reconcile_approval(tid, approval.id,
                checkpoint_id=graph.get_state(config).config["configurable"]["checkpoint_id"], consumed=True)
    connection.close()
    # Reopen both durable stores, using fake effects and no model/provider calls.
    restarted = RunService(service.root_dir)
    monkeypatch.setattr(threads, "_runs", lambda:restarted)
    connection = sqlite3.connect(root / "checkpoint.db", check_same_thread=False)
    graph = build(connection)
    def get(value, configurable=None, **kwargs):
        chat = Thread.__new__(Thread)
        chat.thread_id, chat.agent = tid, graph
        chat.runconfig = {"configurable":{"thread_id":tid, **(configurable or {})}}
        chat._run = lambda value:graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
        return chat
    monkeypatch.setattr(state.MANAGER, "get", get)
    try:
        threads._execute_run(accepted.id, tid)
        while queued:
            rid, value = queued.pop(0)
            if rid == follower.id:
                break
            threads._execute_run(rid, value)
        record = restarted.approval(tid, approval.id)
        assert record.state == "completed" and record.blocked_reason is None
        assert len(decisions) == 1 and len(effects) == (0 if decision == "reject" else 1)
        assert restarted.get(tid, follower.id).status == "pending"
        replay, replayed = threads.email_decision_core(tid, decision, approval.id, phone_preview=True)
        assert replayed and replay.id == accepted.id
        if window == "new_gate":
            pending = restarted.pending_approval(tid)
            assert pending is not None and pending.id != approval.id
            assert pending.interrupt_id != approval.interrupt_id
            assert pending.proposal_id != approval.proposal_id
            assert pending.action == approval.action
    finally:
        connection.close()


@pytest.mark.parametrize("failure", [False, True])
def test_sqlite_approval_end_waits_for_git_finalization(sqlite_approval, monkeypatch, failure):
    tid, root, service, queued = sqlite_approval
    effects = []
    def gate(value):
        interrupt({"action_requests":[ACTION]})
        effects.append("sent")
        return {"messages":[AIMessage("Saved reply")]}
    connection = sqlite3.connect(root / "checkpoint.db", check_same_thread=False)
    graph = StateGraph(MessagesState)
    graph.add_node("gate", gate)
    graph.add_edge(START, "gate"); graph.add_edge("gate", END)
    graph = graph.compile(checkpointer=SqliteSaver(connection))
    owner = service.create(tid, "general-agent", "Email this person")
    service.claim(tid, owner.id)
    config = {"configurable":{"thread_id":tid, "work_id":owner.work_id}}
    graph.invoke({"messages":[HumanMessage(owner.text)]}, config, durability="sync")
    snapshot = graph.get_state(config)
    approval = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION],
        proposal=PROPOSAL, interrupt_id=snapshot.interrupts[0].id,
        checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
    accepted, _ = threads.email_decision_core(tid, "approve", approval.id, phone_preview=True)
    def get(value, configurable=None, **kwargs):
        chat = Thread.__new__(Thread)
        chat.thread_id, chat.agent = tid, graph
        chat.runconfig = {"configurable":{"thread_id":tid, **(configurable or {})}}
        chat._run = lambda value:graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
        return chat
    monkeypatch.setattr(state.MANAGER, "get", get)
    def finish(*args):
        assert service.approval(tid, approval.id).state == "completed"
        assert service.get(tid, accepted.id).status == "running"
        assert service.get(tid, accepted.id).result == "Saved reply"
        if failure:
            raise GitSyncError("Git finalization failed")
    lifecycle = SimpleNamespace(bound=True, resume=lambda:None,
        require_sandbox=lambda value:None, cleanup_model=lambda value:None,
        model_starting=lambda:None, finish_visible=finish)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", lambda *args, **kwargs:nullcontext(lifecycle))
    try:
        threads._execute_run(accepted.id, tid)
        saved = RunService(service.root_dir).get(tid, accepted.id)
        assert service.approval(tid, approval.id).state == "completed"
        assert saved.status == ("error" if failure else "success")
        assert saved.result == "Saved reply"
        assert saved.error == ("Git finalization failed" if failure else None)
        assert effects == ["sent"]
    finally:
        connection.close()


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("git_bound", [False, True])
def test_sqlite_approval_end_recovery_retains_git_fence(sqlite_approval, monkeypatch, recorded, git_bound):
    tid, root, service, queued = sqlite_approval
    effects = []
    def gate(value):
        interrupt({"action_requests":[ACTION]})
        effects.append("sent")
        return {"messages":[AIMessage("Saved reply")]}
    connection = sqlite3.connect(root / "checkpoint.db", check_same_thread=False)
    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate"); builder.add_edge("gate", END)
    graph = builder.compile(checkpointer=SqliteSaver(connection))
    owner = service.create(tid, "general-agent", "Email this person")
    service.claim(tid, owner.id)
    config = {"configurable":{"thread_id":tid, "work_id":owner.work_id}}
    graph.invoke({"messages":[HumanMessage(owner.text)]}, config, durability="sync")
    snapshot = graph.get_state(config)
    approval = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION], proposal=PROPOSAL,
        interrupt_id=snapshot.interrupts[0].id, checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
    accepted, _ = accept(service, approval)
    service.claim(tid, accepted.id)
    config["configurable"]["approval_id"] = approval.id
    graph.invoke(Command(resume={"decisions":[{"type":"approve"}]}), config, durability="sync")
    service.transition(tid, accepted.id, "running", result="Saved reply")
    if recorded:
        service.reconcile_approval(tid, approval.id,
            checkpoint_id=graph.get_state(config).config["configurable"]["checkpoint_id"],
            consumed=True, completed=True, outcome="checkpoint_end")
    connection.close()
    if git_bound:
        (root / tid / ".git").mkdir()
    restarted = RunService(service.root_dir)
    monkeypatch.setattr(threads, "_runs", lambda:restarted)
    connection = sqlite3.connect(root / "checkpoint.db", check_same_thread=False)
    graph = builder.compile(checkpointer=SqliteSaver(connection))
    monkeypatch.setattr(state.MANAGER, "get", lambda *args, **kwargs:SimpleNamespace(
        agent=graph, runconfig=config))
    try:
        threads.queue_recovery_runs()
        assert queued == [(accepted.id, tid)]
        threads._execute_run(*queued.pop(0))
        saved = restarted.get(tid, accepted.id)
        expected_error = threads.git_recovery_error(str(root / tid), str(root / tid))
        assert saved.status == ("error" if git_bound else "success")
        assert saved.error == expected_error and saved.result == "Saved reply"
        assert threads._get_status(tid)["stage"] == ("error" if git_bound else "ready")
        assert restarted.approval(tid, approval.id).state == "completed"
        assert effects == ["sent"]
    finally:
        connection.close()


@pytest.mark.parametrize("phase", ["ownership", "resume", "sandbox"])
@pytest.mark.parametrize("consumed", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_precompletion_git_error_is_atomically_repairable(sqlite_approval, monkeypatch, phase, consumed, decision):
    tid, root, service, queued = sqlite_approval
    effects = []
    def gate(value):
        choice = interrupt({"action_requests":[ACTION]})
        if choice["decisions"][0]["type"] != "reject":
            effects.append("sent")
        return {"messages":[AIMessage("Decision consumed")]}
    connection = sqlite3.connect(root / "checkpoint.db", check_same_thread=False)
    builder = StateGraph(MessagesState)
    builder.add_node("gate", gate)
    builder.add_node("finish", lambda value:{"messages":[AIMessage("Saved reply")]})
    builder.add_edge(START, "gate"); builder.add_edge("gate", "finish"); builder.add_edge("finish", END)
    graph = builder.compile(checkpointer=SqliteSaver(connection), interrupt_before=["finish"] if consumed else [])
    owner = service.create(tid, "general-agent", "Email this person")
    service.claim(tid, owner.id)
    config = {"configurable":{"thread_id":tid, "work_id":owner.work_id}}
    graph.invoke({"messages":[HumanMessage(owner.text)]}, config, durability="sync")
    snapshot = graph.get_state(config)
    approval = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION], proposal=PROPOSAL,
        interrupt_id=snapshot.interrupts[0].id, checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
    accepted, _ = threads.email_decision_core(tid, decision, approval.id, phone_preview=True)
    if consumed:
        config["configurable"]["approval_id"] = approval.id
        graph.invoke(Command(resume={"decisions":list(service.approval(tid, approval.id).decisions)}), config, durability="sync")
        service.reconcile_approval(tid, approval.id,
            checkpoint_id=graph.get_state(config).config["configurable"]["checkpoint_id"], consumed=True)
    follower = service.create(tid, "general-agent", "Follower")
    queued.clear()
    failed = True
    error = "Git verification needs operator repair"
    def boundary(name):
        if failed and name == phase:
            raise GitSyncError(error)
    lifecycle = SimpleNamespace(bound=True, resume=lambda:boundary("resume"),
        require_sandbox=lambda value:boundary("sandbox"), cleanup_model=lambda value:None,
        model_starting=lambda:None, finish_visible=lambda *args:None)
    def acquire(*args, **kwargs):
        boundary("ownership")
        return nullcontext(lifecycle)
    monkeypatch.setattr(threads.GitLifecycle, "acquire", acquire)
    def get(value, configurable=None, **kwargs):
        chat = Thread.__new__(Thread)
        chat.thread_id, chat.agent = tid, graph
        chat.runconfig = {"configurable":{"thread_id":tid, **(configurable or {})}}
        chat._run = lambda value:graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
        return chat
    monkeypatch.setattr(state.MANAGER, "get", get)
    writes = []
    write = service._write
    def observe_write(*args):
        write(*args)
        reopened = RunService(service.root_dir)
        writes.append((reopened.get(tid, accepted.id).status, reopened.approval(tid, approval.id).blocked_reason))
    monkeypatch.setattr(service, "_write", observe_write)
    try:
        threads._execute_run(accepted.id, tid)
        assert all(reason == error for status, reason in writes if status == "error")
        restarted = RunService(service.root_dir)
        record = restarted.approval(tid, approval.id)
        assert record.blocked_reason == error and record.state == ("consumed" if consumed else "rejected" if decision == "reject" else "accepted")
        assert restarted.get(tid, accepted.id).status == "error"
        assert restarted.get(tid, accepted.id).error == error
        assert effects == (["sent"] if consumed and decision == "approve" else [])
        monkeypatch.setattr(threads, "_runs", lambda:restarted)
        threads.queue_recovery_runs()
        assert queued == [] and restarted.get(tid, follower.id).status == "pending"
        assert threads.approval_status(tid)["stage"] == "error"
        failed = False
        repaired = threads.repair_approval_core(tid, approval.id)
        assert repaired.id != accepted.id and repaired.approval_id == approval.id
        assert restarted.approval(tid, approval.id).accepted_run_id == accepted.id
        threads._execute_run(repaired.id, tid)
        assert restarted.approval(tid, approval.id).state == "completed"
        assert restarted.get(tid, repaired.id).status == "success"
        assert effects == (["sent"] if decision == "approve" else [])
        assert restarted.get(tid, follower.id).status == "pending"
    finally:
        connection.close()


@pytest.mark.parametrize("corruption", ["missing", "unreadable", "foreign"])
def test_checkpoint_authority_failure_fences_followers(records, monkeypatch, corruption):
    service, owner, approval = records
    run, _ = accept(service, approval)
    follower = service.create(owner.thread_id, "general-agent", "Follower")
    def snapshot(config):
        if corruption == "unreadable":
            raise OSError("cannot read checkpoint")
        return SimpleNamespace(config=None if corruption == "missing" else {"configurable":{"checkpoint_id":"foreign"}},
            metadata={"work_id":"foreign", "approval_id":"foreign"}, interrupts=(), next=(), parent_config=None)
    chat = SimpleNamespace(agent=SimpleNamespace(get_state=snapshot), runconfig={})
    with pytest.raises((ValueError, OSError)) as error:
        threads._approval_checkpoint(chat, approval)
    service.reconcile_approval(owner.thread_id, approval.id, checkpoint_id=None, blocked_reason=str(error.value))
    restarted = RunService(service.root_dir)
    assert restarted.unfinished_approval_continuation(owner.thread_id).blocked_reason
    assert restarted.get(owner.thread_id, follower.id).status == "pending"
    assert restarted.get(owner.thread_id, run.id).status == "cancelled"


@pytest.mark.parametrize("mutation", ["namespace", "work", "anchor"])
def test_original_interrupt_does_not_bypass_modern_checkpoint_authority(records, mutation):
    service, owner, approval = records
    config = {"thread_id":owner.thread_id, "checkpoint_ns":"", "checkpoint_id":"proposal"}
    metadata = {"work_id":owner.work_id}
    if mutation == "namespace":
        config["checkpoint_ns"] = "foreign"
    elif mutation == "work":
        metadata["work_id"] = "foreign"
    else:
        config["checkpoint_id"] = "different"
    snapshot = SimpleNamespace(config={"configurable":config}, metadata=metadata, parent_config=None,
        interrupts=(SimpleNamespace(id="gate", value={"action_requests":[ACTION]}),), next=("gate",))
    with pytest.raises(ValueError):
        threads._approval_checkpoint(SimpleNamespace(agent=SimpleNamespace(get_state=lambda cfg:snapshot), runconfig={}), approval)


@pytest.mark.parametrize("phase", ["pending", "accepted", "blocked"])
def test_generic_resume_is_fenced_at_claim(sqlite_approval, phase, monkeypatch):
    tid, root, service, queued = sqlite_approval
    owner = service.create(tid, "general-agent", "Email")
    service.claim(tid, owner.id)
    record = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION],
        proposal=PROPOSAL, interrupt_id="gate", checkpoint_id="proposal")
    if phase != "pending":
        accepted, _ = accept(service, record)
        if phase == "blocked":
            service.reconcile_approval(tid, record.id, checkpoint_id=None, blocked_reason="checkpoint fault")
    follower = service.create(tid, "general-agent", None, resume=True)
    monkeypatch.setattr(state.MANAGER, "get", lambda *args, **kwargs:pytest.fail("Follower must not read or resume the approval graph"))
    threads._execute_run(follower.id, tid)
    assert service.get(tid, follower.id).status == "pending"


def test_explicit_repair_rechecks_checkpoint_and_preserves_accepted_receipt(records, monkeypatch):
    service, owner, record = records
    accepted, _ = accept(service, record)
    service.reconcile_approval(owner.thread_id, record.id, checkpoint_id=None, blocked_reason="temporary checkpoint read error")
    config = {"configurable":{"thread_id":owner.thread_id, "checkpoint_id":"proposal", "checkpoint_ns":""}}
    snapshot = SimpleNamespace(config=config, metadata={"work_id":owner.work_id}, parent_config=None,
        interrupts=(SimpleNamespace(id="gate", value={"action_requests":[ACTION]}),), next=("gate",))
    monkeypatch.setattr(threads, "_runs", lambda:service)
    monkeypatch.setattr(state.MANAGER, "get", lambda *args, **kwargs:SimpleNamespace(
        agent=SimpleNamespace(get_state=lambda cfg:snapshot), runconfig={}))
    repaired = threads.repair_approval_core(owner.thread_id, record.id)
    assert repaired.id != accepted.id and repaired.approval_id == record.id and repaired.status == "pending"
    assert service.approval(owner.thread_id, record.id).blocked_reason is None
    assert accept(service, record)[0].id == accepted.id
    assert service.get(owner.thread_id, accepted.id).status == "cancelled"


def test_verified_nonmail_gate_completes_without_deciding_new_reply(sqlite_approval, monkeypatch):
    tid, root, service, queued = sqlite_approval
    decisions = []
    def gate(value):
        decisions.append(interrupt({"action_requests":[ACTION]}))
        return {"messages":[AIMessage("Email approved")]}
    def reply(value):
        interrupt({"action_requests":[{"name":"send_reply", "args":{"text":"Separate reply"}}]})
        pytest.fail("Old email decision must not approve this reply")
    with sqlite3.connect(root / "new-reply.db", check_same_thread=False) as connection:
        builder = StateGraph(MessagesState)
        builder.add_node("gate", gate); builder.add_node("reply", reply)
        builder.add_edge(START, "gate"); builder.add_edge("gate", "reply"); builder.add_edge("reply", END)
        graph = builder.compile(checkpointer=SqliteSaver(connection))
        owner = service.create(tid, "general-agent", "Email")
        service.claim(tid, owner.id)
        config = {"configurable":{"thread_id":tid, "work_id":owner.work_id}}
        graph.invoke({"messages":[HumanMessage("Email")]}, config, durability="sync")
        snapshot = graph.get_state(config)
        record = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION], proposal=PROPOSAL,
            interrupt_id=snapshot.interrupts[0].id, checkpoint_id=snapshot.config["configurable"]["checkpoint_id"])
        accepted, _ = threads.email_decision_core(tid, "approve", record.id, phone_preview=True)
        def get(value, configurable=None, **kwargs):
            chat = Thread.__new__(Thread)
            chat.thread_id, chat.agent = tid, graph
            chat.runconfig = {"configurable":{"thread_id":tid, **(configurable or {})}}
            chat._run = lambda value:graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
            return chat
        monkeypatch.setattr(state.MANAGER, "get", get)
        threads._execute_run(accepted.id, tid)
        assert len(decisions) == 1
        assert service.approval(tid, record.id).state == "completed"
        assert service.approval(tid, record.id).outcome == "new_gate"
        assert service.unfinished_approval_continuation(tid) is None
        assert state._get_status(tid)["pending_reply"] == "Separate reply"
        assert service.get(tid, accepted.id).status == "awaiting_approval"


def test_legacy_upgrade_does_not_import_consumed_approval_before_current_gate(sqlite_approval, monkeypatch):
    tid, root, service, queued = sqlite_approval
    choices = []
    def gate(value):
        choices.append(interrupt({"action_requests":[ACTION]}))
        return {"messages":[AIMessage("Consumed")]}
    with sqlite3.connect(root / "legacy-two-gates.db", check_same_thread=False) as connection:
        builder = StateGraph(MessagesState)
        builder.add_node("first", gate); builder.add_node("second", gate)
        builder.add_edge(START, "first"); builder.add_edge("first", "second"); builder.add_edge("second", END)
        graph = builder.compile(checkpointer=SqliteSaver(connection))
        original = service.create(tid, "general-agent", "Email")
        service.claim(tid, original.id); service.transition(tid, original.id, "awaiting_approval")
        config = {"configurable":{"thread_id":tid}}
        graph.invoke({"messages":[HumanMessage("Email")]}, config, durability="sync")
        first_interrupt = graph.get_state(config).interrupts[0].id
        old = service.create(tid, "general-agent", None, work_id=original.work_id,
            dispatch_key="email-approval:first", resume_decision={**DECISION, "approval_interrupt_id":first_interrupt})
        service.claim(tid, old.id); service.transition(tid, old.id, "interrupted")
        graph.invoke(Command(resume={"decisions":[{"type":"approve"}]}), config, durability="sync")
        second_interrupt = graph.get_state(config).interrupts[0].id
        proposal = service.create(tid, "general-agent", None, work_id=original.work_id, resume=True)
        service.claim(tid, proposal.id); service.transition(tid, proposal.id, "awaiting_approval")
        current = service.create(tid, "general-agent", None, work_id=original.work_id,
            dispatch_key="email-approval:second", resume_decision={**DECISION, "approval_interrupt_id":second_interrupt})
        def get(value, configurable=None, **kwargs):
            chat = Thread.__new__(Thread)
            chat.thread_id, chat.agent = tid, graph
            chat.runconfig = {"configurable":{"thread_id":tid, **(configurable or {})}}
            chat._run = lambda value:graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
            return chat
        monkeypatch.setattr(state.MANAGER, "get", get)
        threads.queue_recovery_runs()
        record = service.unfinished_approval_continuation(tid)
        assert record.accepted_run_id == current.id and record.blocked_reason is None
        assert service.get(tid, old.id).approval_id is None
        assert queued == [(current.id, tid)]
        threads._execute_run(current.id, tid)
        assert len(choices) == 2
        assert service.approval(tid, record.id).state == "completed"


@pytest.mark.parametrize("phase", ["pending", "running", "interrupted"])
def test_early_runtime_error_blocks_approval_and_projects_error(sqlite_approval, monkeypatch, phase):
    from assist.thread_engine import ThreadEngineError
    tid, root, service, queued = sqlite_approval
    owner = service.create(tid, "general-agent", "Email")
    service.claim(tid, owner.id)
    record = service.publish_approval(tid, owner.id, action=ACTION, requests=[ACTION],
        proposal=PROPOSAL, interrupt_id="gate", checkpoint_id="proposal")
    accepted, _ = accept(service, record)
    if phase != "pending":
        service.claim(tid, accepted.id)
        if phase == "interrupted":
            service.transition(tid, accepted.id, "interrupted")
    monkeypatch.setattr(threads, "_is_pi_thread", lambda value:(_ for _ in ()).throw(ThreadEngineError("invalid engine")))
    threads._execute_run(accepted.id, tid)
    assert service.approval(tid, record.id).blocked_reason == "invalid engine"
    assert service.get(tid, accepted.id).status == ("interrupted" if phase == "interrupted" else "error")
    assert threads.approval_status(tid)["stage"] == "error"
    assert queued == []


@pytest.mark.parametrize("streaming", [False, True])
def test_approval_bad_request_never_walks_rollback_history(streaming):
    import httpx
    from openai import BadRequestError
    attempts = []
    error = BadRequestError("synthetic refusal", response=httpx.Response(400,
        request=httpx.Request("POST", "https://synthetic.invalid")), body={})
    class Agent:
        def invoke(self, value, config, **kwargs):
            attempts.append(value)
            raise error
        def stream(self, value, config, **kwargs):
            attempts.append(value)
            raise error
        def get_state_history(self, config):
            pytest.fail("Approval execution must not rewind behind a consumed decision")
    chat = Thread.__new__(Thread)
    chat.agent, chat.thread_id = Agent(), "approval-rollback-test"
    chat.runconfig = {"configurable":{"thread_id":chat.thread_id, "approval_id":"approved"}}
    chat.on_queue_state = None
    with pytest.raises(BadRequestError):
        if streaming:
            chat._observe(None, lambda value:None)
        else:
            chat._run(None)
    assert attempts == [None]


def test_record_bounds_apply_to_acceptance_as_well_as_publication(records):
    service, owner, record = records
    decision = {"type":"edit", "edited_action":{"name":"send_email", "args":{
        **ACTION["args"], "body":"x" * (2 * 1024 * 1024)}}}
    with pytest.raises(ValueError, match="size limit"):
        service.accept_approval(owner.thread_id, record.id, decision, [decision])
    assert len(service.list(owner.thread_id)) == 1
    assert service.approval(owner.thread_id, record.id).state == "pending"


def test_legacy_migration_read_failure_requires_checked_fresh_review(sqlite_approval, monkeypatch):
    tid, root, service, queued = sqlite_approval
    chosen = []
    def gate(value):
        chosen.append(interrupt({"action_requests":[ACTION]}))
        return {"messages":[AIMessage("Done")]}
    with sqlite3.connect(root / "legacy-read-error.db", check_same_thread=False) as connection:
        builder = StateGraph(MessagesState)
        builder.add_node("gate", gate); builder.add_edge(START, "gate"); builder.add_edge("gate", END)
        graph = builder.compile(checkpointer=SqliteSaver(connection))
        config = {"configurable":{"thread_id":tid}}
        graph.invoke({"messages":[HumanMessage("Email")]}, config, durability="sync")
        original = graph.get_state(config).interrupts[0].id
        owner = service.create(tid, "general-agent", "Email")
        service.claim(tid, owner.id); service.transition(tid, owner.id, "awaiting_approval")
        accepted = service.create(tid, "general-agent", None, work_id=owner.work_id,
            dispatch_key="email-approval:legacy", resume_decision={**DECISION, "approval_interrupt_id":original})
        monkeypatch.setattr(state.MANAGER, "get", lambda *args, **kwargs:SimpleNamespace(
            agent=SimpleNamespace(get_state=lambda cfg:(_ for _ in ()).throw(OSError("temporary read failure"))), runconfig={}))
        threads._import_legacy_approval_runs(tid)
        blocked = service.unfinished_approval_continuation(tid)
        assert blocked.blocked_reason and blocked.action["args"] == {}
        def get(value, configurable=None, **kwargs):
            chat = Thread.__new__(Thread)
            chat.thread_id, chat.agent = tid, graph
            chat.runconfig = {"configurable":{"thread_id":tid, **(configurable or {})}}
            chat._run = lambda value:graph.invoke(value, chat.runconfig, durability="sync")["messages"][-1].content
            return chat
        monkeypatch.setattr(state.MANAGER, "get", get)
        fresh = threads.fresh_review_approval_core(tid, blocked.id)
        assert chosen == [] and service.approval(tid, blocked.id).outcome == "fresh_review"
        assert fresh.state == "pending" and fresh.action == ACTION and fresh.decision is None
        assert fresh.interrupt_id == original and fresh.proposal_id == blocked.proposal_id
        run, _ = threads.email_decision_core(tid, "reject", fresh.id, phone_preview=True)
        threads._execute_run(run.id, tid)
        assert chosen == [{"decisions":[{"type":"reject", "message":"The user declined to send this email."}]}]
