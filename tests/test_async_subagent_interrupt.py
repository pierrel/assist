"""Durable proof for serialized async-child delegation.

The production design depends on a parent graph releasing ``THREAD_QUEUE`` while a
required child runs, then continuing the SAME pending task call after the child
finishes.  This test pins the LangGraph mechanism before assist builds on it:
``interrupt()`` checkpoints the pending node task and ``Command(resume=...)`` replays
that node with the resume value, including after recreating the graph and SqliteSaver.

It also runs a sibling tool node in the interrupted superstep.  The sibling must not
execute twice, and both tool results must remain paired to the original assistant tool
calls.  Direct checkpoint/message mutation is deliberately absent: only LangGraph's
supported interrupt/resume API completes the pending task.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class _State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _record(db: Path, event: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS proof_events (event TEXT NOT NULL)")
        conn.execute("INSERT INTO proof_events VALUES (?)", (event,))


def _child_for(db: Path, dispatch_key: str) -> str:
    """Durably create-or-find the child before interrupting the parent node."""
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS proof_children "
            "(dispatch_key TEXT PRIMARY KEY, child_id TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO proof_children VALUES (?, ?)",
            (dispatch_key, "child-1"),
        )
        return conn.execute(
            "SELECT child_id FROM proof_children WHERE dispatch_key = ?",
            (dispatch_key,),
        ).fetchone()[0]


def _graph(proof_db: Path):
    def required_child(_state):
        _record(proof_db, "required-child-node")
        child_id = _child_for(proof_db, "parent-work-1:task-required")
        result = interrupt({"child_run_id": child_id})
        return {
            "messages": [
                ToolMessage(str(result), tool_call_id="task-required")
            ]
        }

    def sibling(_state):
        _record(proof_db, "sibling-node")
        return {
            "messages": [ToolMessage("sibling result", tool_call_id="task-sibling")]
        }

    def finish(_state):
        _record(proof_db, "finish-node")
        return {"messages": [AIMessage("one final answer")]}

    graph = StateGraph(_State)
    graph.add_node("required_child", required_child)
    graph.add_node("sibling", sibling)
    graph.add_node("finish", finish)
    graph.add_edge(START, "required_child")
    graph.add_edge(START, "sibling")
    graph.add_edge("required_child", "finish")
    graph.add_edge("sibling", "finish")
    graph.add_edge("finish", END)
    return graph


def _compile(checkpoint_db: Path, proof_db: Path):
    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, _graph(proof_db).compile(checkpointer=saver)


def test_required_child_interrupt_resumes_once_after_process_recreation(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.sqlite"
    proof_db = tmp_path / "proof.sqlite"
    config = {"configurable": {"thread_id": "parent-thread"}}
    first_input = {
        "messages": [
            AIMessage(
                "",
                tool_calls=[
                    {"name": "required_child", "args": {}, "id": "task-required"},
                    {"name": "sibling", "args": {}, "id": "task-sibling"},
                ],
            )
        ]
    }

    conn, parent = _compile(checkpoint_db, proof_db)
    first = parent.invoke(first_input, config, durability="sync")
    conn.close()  # process boundary: no in-memory graph/checkpointer state survives.

    assert first["__interrupt__"][0].value == {"child_run_id": "child-1"}
    assert [m.tool_call_id for m in first["messages"] if isinstance(m, ToolMessage)] \
        == ["task-sibling"]

    conn, parent = _compile(checkpoint_db, proof_db)
    final = parent.invoke(Command(resume="child result"), config,
                          durability="sync")
    conn.close()

    with sqlite3.connect(proof_db) as proof:
        events = [row[0] for row in proof.execute("SELECT event FROM proof_events")]
        children = proof.execute("SELECT dispatch_key, child_id FROM proof_children").fetchall()

    # The interrupted node replays by contract, but its create-or-find side effect is
    # idempotent. The completed sibling and final synthesis each execute exactly once.
    assert events.count("required-child-node") == 2
    assert events.count("sibling-node") == 1
    assert events.count("finish-node") == 1
    assert children == [("parent-work-1:task-required", "child-1")]

    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert sorted(m.tool_call_id for m in tool_messages) == [
        "task-required", "task-sibling"
    ]
    assert [m.content for m in final["messages"] if isinstance(m, AIMessage)][-1] \
        == "one final answer"
