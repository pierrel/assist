"""DeltaChannel spike — rollback validation.

Throwaway test for the `spike-deltachannel` branch.  Verifies that
`RollbackRunnable`'s assumption — that `agent.get_state_history(cfg)`
returns one usable entry per super-step, ordered most-recent first —
still holds under `DeltaChannel`.

If a delta-only row is exposed as its own history entry, the
`target_idx = depth + 1` math in `assist/checkpoint_rollback.py:131`
can land on a partial state, breaking rollback.  Probes three
straddling cases: depth 1 (same snapshot window), depth 5 (interior
of a snapshot window), depth 12 (one step past the K=10 boundary —
the most likely failure site).

Run:
    .venv/bin/python -m edd.spike.test_rollback_under_deltachannel
"""

import os
import sqlite3
import tempfile
from typing import Annotated, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.channels import DeltaChannel
from langgraph.graph import END, START, StateGraph
from deepagents._messages_reducer import _messages_delta_reducer


SNAPSHOT_FREQUENCY = 10


def _build_graph():
    """Single-node graph with DeltaChannel on messages."""
    class State(TypedDict):
        messages: Annotated[
            list[AnyMessage],
            DeltaChannel(_messages_delta_reducer, snapshot_frequency=SNAPSHOT_FREQUENCY),
        ]

    def step(state: State):
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
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_edge("step", END)
    return g


def main():
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
            print(f"# Rollback validation under DeltaChannel(K={SNAPSHOT_FREQUENCY})")
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
    main()
