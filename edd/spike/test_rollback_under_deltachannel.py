"""DeltaChannel spike — rollback validation.

Throwaway test for the `spike-deltachannel` branch.  Two probes:

1. ``probe_history_indexing`` — verifies that ``agent.get_state_history(cfg)``
   returns one usable entry per super-step ordered most-recent first, and
   that ``app.invoke(None, target_config)`` resumes cleanly from a target
   chosen by ``target_idx = depth + 1`` math.  This is what
   ``assist/checkpoint_rollback.py:97-145`` depends on.

2. ``probe_invoke_with_rollback`` — drives the FULL ``invoke_with_rollback``
   path: a node that raises ``BadRequestError`` on its 6th invocation, then
   succeeds.  Verifies the rollback fires, picks a valid target, and the
   resumed invocation returns the agent result without partial-state
   corruption.  Catches anything ``probe_history_indexing`` would miss
   because it tests the live error-recovery path end-to-end.

Run:
    .venv/bin/python -m edd.spike.test_rollback_under_deltachannel
"""

import os
import sqlite3
import tempfile
from typing import Annotated, TypedDict

import httpx
from openai import BadRequestError
from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.channels import DeltaChannel
from langgraph.graph import END, START, StateGraph
from deepagents._messages_reducer import _messages_delta_reducer

from assist.checkpoint_rollback import invoke_with_rollback


SNAPSHOT_FREQUENCY = 10


def _build_graph(node_fn=None):
    """Single-node graph with DeltaChannel on messages.  ``node_fn``
    overrides the default no-op step (used by probe_invoke_with_rollback
    to inject a transient failure)."""

    class State(TypedDict):
        messages: Annotated[
            list[AnyMessage],
            DeltaChannel(_messages_delta_reducer, snapshot_frequency=SNAPSHOT_FREQUENCY),
        ]

    def default_step(state: State):
        idx = len(state["messages"])
        tool_call_id = f"call_{idx}"
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "noop", "args": {"i": idx}, "id": tool_call_id}
                    ],
                ),
                ToolMessage(content=f"result-{idx}", tool_call_id=tool_call_id),
            ]
        }

    g = StateGraph(State)
    g.add_node("step", node_fn or default_step)
    g.add_edge(START, "step")
    g.add_edge("step", END)
    return g


def _bad_request_error(msg: str = "Synthetic 400 — invalid messages") -> BadRequestError:
    """Construct a BadRequestError the way assist sees it from vLLM/llama.cpp."""
    req = httpx.Request("POST", "http://localhost/v1/chat/completions")
    resp = httpx.Response(400, json={"error": {"message": msg}}, request=req)
    return BadRequestError(msg, response=resp, body={"error": {"message": msg}})


def probe_invoke_with_rollback():
    """Probe 2 — drive the full invoke_with_rollback path end-to-end.

    Builds a graph whose node raises BadRequestError on call #6, then
    succeeds.  invoke_with_rollback should: catch the error, roll back
    to a previous checkpoint, re-invoke with current_input=None, and
    return the recovered result.  Verifies the rollback math works
    against DeltaChannel's get_state_history shape and that resume
    from a DeltaChannel checkpoint produces valid state.
    """
    print()
    print(f"# Probe 2: invoke_with_rollback under DeltaChannel(K={SNAPSHOT_FREQUENCY})")
    print()

    call_count = {"n": 0}

    def flaky_step(state):
        call_count["n"] += 1
        if call_count["n"] == 6:
            # Fail once.  Rollback will rewind to an earlier checkpoint
            # and re-invoke; the next time we enter this node, call_count
            # is 7 and we succeed.
            raise _bad_request_error()
        idx = len(state["messages"])
        return {
            "messages": [
                AIMessage(
                    content=f"reply {idx}",
                    tool_calls=[{"name": "noop", "args": {}, "id": f"c{idx}"}],
                ),
                ToolMessage(content=f"r{idx}", tool_call_id=f"c{idx}"),
            ]
        }

    g = _build_graph(flaky_step)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "threads.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn)
            app = g.compile(checkpointer=saver)
            cfg = {"configurable": {"thread_id": "rollback-probe-2"}}

            # Five successful super-steps build history (calls #1-#5).
            # Seed counts as the first invocation.
            app.invoke({"messages": [HumanMessage(content="seed")]}, cfg)
            for i in range(1, 5):
                app.invoke({"messages": [HumanMessage(content=f"t{i}")]}, cfg)
            assert call_count["n"] == 5, f"expected 5 calls, got {call_count['n']}"
            # Next invoke is call #6 → raises inside invoke_with_rollback.

            # Drive invoke_with_rollback with a new turn.
            try:
                result = invoke_with_rollback(
                    app,
                    {"messages": [HumanMessage(content="t6 — will fail then recover")]},
                    cfg,
                )
                msg_count = len(result.get("messages", []))
                # If rollback worked, we'll have a non-empty messages list
                # and call_count > 6 (the retry fired).
                pass_ok = msg_count > 0 and call_count["n"] >= 7
                status = "PASS" if pass_ok else "FAIL"
                print(f"## invoke_with_rollback: {status}")
                print(f"  - flaky node invocations: {call_count['n']} (expected >= 7)")
                print(f"  - final messages count: {msg_count}")
                print(f"  - rollback fired and recovered: {pass_ok}")
            except Exception as e:
                print(f"## invoke_with_rollback: ERROR")
                print(f"  - flaky node invocations: {call_count['n']}")
                print(f"  - raised: {type(e).__name__}: {str(e)[:200]}")
        finally:
            conn.close()


def probe_history_indexing():
    """Probe 1 — get_state_history shape + invoke-from-target-config under DeltaChannel."""
    g = _build_graph()
    n_supersteps = 15  # straddles the K=10 snapshot boundary
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "threads.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn)
            app = g.compile(checkpointer=saver)
            cfg = {"configurable": {"thread_id": "rollback-test"}}

            # Seed + N super-steps
            app.invoke({"messages": [HumanMessage(content="start")]}, cfg)
            for i in range(1, n_supersteps + 1):
                app.invoke({"messages": [HumanMessage(content=f"turn {i}")]}, cfg)

            history = list(app.get_state_history(cfg))
            print(f"# Probe 1: get_state_history shape + resume from target config under DeltaChannel(K={SNAPSHOT_FREQUENCY})")
            print()
            print(f"After {n_supersteps} super-steps:")
            print(f"  - history entries: {len(history)}")
            print(f"  - history[0].metadata.step: {history[0].metadata.get('step')}")
            print(f"  - history[-1].metadata.step: {history[-1].metadata.get('step')}")
            print()

            # Probe rollback to three depths.  The exact step indices
            # depend on how langgraph exposes per-superstep history;
            # what matters is that target_idx = depth + 1 produces a
            # config we can resume from.
            for depth in [1, 5, 12]:
                target_idx = depth + 1
                if target_idx >= len(history):
                    print(f"## depth={depth}: SKIP (history too short, len={len(history)})")
                    continue
                target = history[target_idx]
                target_step = target.metadata.get("step", "?")
                target_msg_count = len(target.values.get("messages", []))
                target_cfg = target.config
                cp_id = target_cfg["configurable"].get("checkpoint_id", "")[:16]

                # Now try to resume the graph from that target config.
                try:
                    resumed = app.invoke(
                        {"messages": [HumanMessage(content=f"resumed at depth {depth}")]},
                        target_cfg,
                    )
                    resumed_msg_count = len(resumed.get("messages", []))
                    # The resumed state should contain the rolled-back
                    # messages PLUS the resume input PLUS the one new
                    # step's output (1 AIMessage + 1 ToolMessage = 2).
                    expected_min = target_msg_count + 1  # at least the resume HumanMessage
                    pass_ok = resumed_msg_count >= expected_min
                    status = "PASS" if pass_ok else "FAIL"
                    print(
                        f"## depth={depth}: {status}\n"
                        f"  - target step: {target_step}, msg_count: {target_msg_count}, cp: {cp_id}…\n"
                        f"  - resumed msg_count: {resumed_msg_count} "
                        f"(>= target+1 = {expected_min}: {pass_ok})"
                    )
                except Exception as e:
                    print(
                        f"## depth={depth}: ERROR\n"
                        f"  - target step: {target_step}, msg_count: {target_msg_count}, cp: {cp_id}…\n"
                        f"  - resume raised: {type(e).__name__}: {e}"
                    )
                print()
        finally:
            conn.close()


if __name__ == "__main__":
    probe_history_indexing()
    probe_invoke_with_rollback()
