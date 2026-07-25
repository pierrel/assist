from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

from assist.async_subagents import (
    AsyncTaskContext,
    async_task_context,
    task_tool,
)


def _runtime(tool_call_id="call-1"):
    return SimpleNamespace(tool_call_id=tool_call_id)


def test_requires_registered_web_execution_context():
    with pytest.raises(RuntimeError, match="outside a configured web run"):
        task_tool.func("find it", "context-agent", _runtime())


def test_rejects_unknown_type_before_dispatch():
    with async_task_context(AsyncTaskContext("p", "r", "w", lambda *_: {})):
        with pytest.raises(ValueError, match="unknown subagent type"):
            task_tool.func("find it", "invented-agent", _runtime())


def test_sync_dispatch_key_and_interrupt_payload(monkeypatch):
    dispatched = []
    payloads = []
    monkeypatch.setattr(
        "assist.async_subagents.interrupt",
        lambda payload: payloads.append(payload) or "child result")

    def dispatch(key, agent, description):
        dispatched.append((key, agent, description))
        return {"thread_id": "child-thread", "run_id": "child-run"}

    context = AsyncTaskContext("parent-thread", "parent-run", "parent-work", dispatch)
    with async_task_context(context):
        command = task_tool.func("inspect files", "context-agent", _runtime("tc-7"))

    assert dispatched == [("parent-work:tc-7", "context-agent", "inspect files")]
    assert payloads == [{
        "parent_thread_id": "parent-thread", "parent_run_id": "parent-run",
        "parent_work_id": "parent-work", "child_thread_id": "child-thread",
        "child_run_id": "child-run", "tool_call_id": "tc-7",
    }]
    assert list(command.update) == ["messages"]
    assert len(command.update["messages"]) == 1
    assert command.update["messages"][0].tool_call_id == "tc-7"
    assert command.update["messages"][0].content == "child result"


def test_async_tool_dispatches_off_loop(monkeypatch):
    calls = []
    monkeypatch.setattr("assist.async_subagents.interrupt", lambda _: "done")

    def dispatch(key, agent, description):
        calls.append((key, agent, description))
        return {"thread_id": "ct", "run_id": "cr"}

    context = AsyncTaskContext("pt", "pr", "pw", dispatch)
    with async_task_context(context):
        command = asyncio.run(task_tool.coroutine(
            "research it", "research-agent", _runtime("async-call")))

    assert calls == [("pw:async-call", "research-agent", "research it")]
    assert command.update["messages"][0].tool_call_id == "async-call"


def test_dispatch_must_return_complete_child_identity(monkeypatch):
    monkeypatch.setattr("assist.async_subagents.interrupt", lambda _: "done")
    context = AsyncTaskContext("pt", "pr", "pw", lambda *_: {"thread_id": "ct"})
    with async_task_context(context):
        with pytest.raises(ValueError, match="thread_id and run_id"):
            task_tool.func("inspect", "context-agent", _runtime())


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _dispatch_from(db: Path):
    def dispatch(key, agent, description):
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS children "
                "(dispatch_key TEXT PRIMARY KEY, thread_id TEXT, run_id TEXT, "
                "agent TEXT, description TEXT)")
            conn.execute(
                "INSERT OR IGNORE INTO children VALUES (?, ?, ?, ?, ?)",
                (key, "child-thread", "child-run", agent, description))
            row = conn.execute(
                "SELECT thread_id, run_id FROM children WHERE dispatch_key = ?",
                (key,),
            ).fetchone()
        return {"thread_id": row[0], "run_id": row[1]}
    return dispatch


def _compile(checkpoint_db: Path, children_db: Path):
    def delegate(_state):
        context = AsyncTaskContext(
            "parent-thread", "parent-run", "parent-work",
            _dispatch_from(children_db))
        with async_task_context(context):
            return task_tool.func(
                "inspect the workspace", "context-agent", _runtime("task-1"))

    graph = StateGraph(_State)
    graph.add_node("delegate", delegate)
    graph.add_edge(START, "delegate")
    graph.add_edge("delegate", END)
    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, graph.compile(checkpointer=saver)


def test_replay_after_process_recreation_dispatches_once_and_pairs_result(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    children_db = tmp_path / "children.sqlite"
    config = {"configurable": {"thread_id": "parent-thread"}}

    conn, graph = _compile(checkpoint_db, children_db)
    first = graph.invoke({"messages": []}, config, durability="sync")
    conn.close()

    assert first["__interrupt__"][0].value["child_run_id"] == "child-run"

    conn, graph = _compile(checkpoint_db, children_db)
    final = graph.invoke(Command(resume="durable child result"), config,
                         durability="sync")
    conn.close()

    with sqlite3.connect(children_db) as db:
        children = db.execute(
            "SELECT dispatch_key, thread_id, run_id FROM children").fetchall()
    assert children == [("parent-work:task-1", "child-thread", "child-run")]
    tool_messages = [message for message in final["messages"]
                     if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "task-1"
    assert tool_messages[0].content == "durable child result"
