"""A nested sub-agent, paused mid-run, RESUMES from its durable checkpoint on a
parent replay — WITHOUT re-running its completed steps.

This pins the load-bearing property the fair-scheduling *scheduler* (the pause /
requeue / resume design, docs/2026-07-08-fair-scheduling.org) depends on: when a
turn is paused at a superstep boundary while a sub-agent is mid-flight, resuming the
turn must continue the sub-agent from where it left off, not restart it and lose its
work. That property is provided by langgraph itself, NOT by any assist code:

  - a nested sub-agent inherits the parent's (durable) checkpointer via langgraph's
    checkpointer precedence (config CONFIG_KEY_CHECKPOINTER wins over the sub-agent's
    own saver), so its per-superstep checkpoints are written to the trunk store;
  - the nested checkpoint namespace is a deterministic hash of the parent checkpoint
    + task position, so a parent replay reproduces it and finds the sub-agent's
    checkpoints again.

So this is a GENERAL mechanism — research is not special-cased, and no persistence
change is needed to make a paused research sub-agent resumable. This test guards that
assumption so a future change (e.g. giving sub-agents their own non-durable saver, or
minting a fresh per-dispatch thread_id) can't silently break lossless resume — which
would reappear only as re-run work under load, never as a failed test elsewhere.

Uses a real SqliteSaver over a temp file (no mocks — the checkpointer is exactly the
risk, per CLAUDE.md "don't mock away the risk").
"""
from __future__ import annotations

import operator
import os
import sqlite3
import tempfile
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class _PauseSignal(Exception):
    """Stands in for the scheduler's superstep-boundary pause (Phase 2 raises an
    equivalent from ThreadQueueMiddleware.after_model)."""


class _State(TypedDict):
    log: Annotated[list, operator.add]


def _build(calls: dict, paused_once: dict):
    """Parent graph: p_pre -> [sub-agent as a nested node] -> p_post. The sub-agent
    is s1 -> s2 -> s3 -> s4; s3 raises a pause the FIRST time it runs."""

    def sub_node(name: str, pause_here: bool = False):
        def run(_state):
            calls[name] = calls.get(name, 0) + 1
            if pause_here and not paused_once["done"]:
                paused_once["done"] = True
                raise _PauseSignal(f"paused before {name}")
            return {"log": [name]}
        return run

    sg = StateGraph(_State)
    for name, pause in [("s1", False), ("s2", False), ("s3", True), ("s4", False)]:
        sg.add_node(name, sub_node(name, pause))
    sg.add_edge(START, "s1")
    sg.add_edge("s1", "s2")
    sg.add_edge("s2", "s3")
    sg.add_edge("s3", "s4")
    sg.add_edge("s4", END)
    sub = sg.compile()  # nested: the parent injects its checkpointer + namespace

    def p_pre(_state):
        calls["p_pre"] = calls.get("p_pre", 0) + 1
        return {"log": ["pre"]}

    def p_post(_state):
        calls["p_post"] = calls.get("p_post", 0) + 1
        return {"log": ["post"]}

    pg = StateGraph(_State)
    pg.add_node("p_pre", p_pre)
    pg.add_node("sub", sub)  # compiled subgraph AS a node -> automatic nesting
    pg.add_node("p_post", p_post)
    pg.add_edge(START, "p_pre")
    pg.add_edge("p_pre", "sub")
    pg.add_edge("sub", "p_post")
    pg.add_edge("p_post", END)
    return pg


def test_paused_subagent_resumes_without_rerunning_completed_steps():
    calls: dict[str, int] = {}
    paused_once = {"done": False}
    graph = _build(calls, paused_once)

    _fd, tmp = tempfile.mkstemp(suffix=".sqlite"); os.close(_fd)
    conn = sqlite3.connect(tmp, check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        agent = graph.compile(checkpointer=saver)
        cfg = {"configurable": {"thread_id": "T"}}

        # Run 1: the sub-agent pauses mid-flight (before s3). s1, s2 have completed
        # and been checkpointed (durability="sync" writes each superstep).
        try:
            agent.invoke({"log": ["start"]}, cfg, durability="sync")
            raise AssertionError("expected a pause mid sub-agent")
        except _PauseSignal:
            pass
        assert calls == {"p_pre": 1, "s1": 1, "s2": 1, "s3": 1}

        # Run 2: resume the parent (input=None). The sub-agent must continue from its
        # durable checkpoint — s1/s2 are NOT re-run; s3 runs again (past the pause),
        # then s4, then the parent's p_post.
        agent.invoke(None, cfg, durability="sync")
    finally:
        conn.close()
        if os.path.exists(tmp):
            os.remove(tmp)

    # The no-lost-work contract, asserted on execution counts (the symptom), not a
    # status flag: completed steps ran exactly once across the pause/resume.
    assert calls["s1"] == 1, f"s1 re-ran on resume (lost work): {calls}"
    assert calls["s2"] == 1, f"s2 re-ran on resume (lost work): {calls}"
    assert calls["p_pre"] == 1, f"parent step re-ran on resume: {calls}"
    assert calls["s3"] == 2, f"s3 should run twice (paused, then resumed): {calls}"
    assert calls["s4"] == 1, f"s4 should complete once after resume: {calls}"
    assert calls["p_post"] == 1, f"parent should finish once after resume: {calls}"
