"""DeltaChannel spike — measure threads.db bytes per super-step.

Throwaway harness for the `spike-deltachannel` branch.  See
docs/2026-05-16-deltachannel-spike.org for context.

Lives under edd/spike/ (NOT edd/eval/) so scripts/run-evals.sh does
not pick it up via its `edd/eval/test_*.py` glob.

What it does:
  - Builds a minimal StateGraph with a `messages` channel and a
    single node that appends a synthetic ~50 KB ToolMessage per
    super-step.
  - Compiles with langgraph's SqliteSaver against a temp threads.db.
  - Drives it N super-steps (N in {20, 50, 100}).
  - Records bytes-on-disk + checkpoint row count + average blob
    size every 5 steps.
  - Prints a markdown row that can be pasted into the spike doc.

This harness builds its own minimal StateGraph (`_build_graph`) and
does NOT use `create_deep_agent` — so deepagents' `_DeepAgentState`
default-on DeltaChannel is NOT in play here.  To measure DeltaChannel
via this harness you MUST pass `--delta` regardless of which
deepagents version is installed.

Use it for the apples-to-apples comparison:
  - Without `--delta`: baseline (plain `add_messages` reducer,
    full-list snapshot per super-step).  Runs on any langgraph
    version; does NOT require deepagents 0.6.1.
  - With `--delta`: explicit `DeltaChannel(_messages_delta_reducer,
    snapshot_frequency=K)` wrap on the `messages` channel — the same
    reducer + channel shape that `deepagents._DeepAgentState` uses
    in production.  Requires deepagents 0.6.1 + langgraph 1.2 for the
    `_messages_delta_reducer` and `DeltaChannel` imports.

The `_messages_delta_reducer` import lives inside the `use_delta`
branch of `_build_graph`, so the baseline run does not need it
installed — useful for measuring the pre-upgrade environment.

Usage:
    .venv/bin/python -m edd.spike.measure_deltachannel --n 20 50 100
    .venv/bin/python -m edd.spike.measure_deltachannel --n 100 --delta
    .venv/bin/python -m edd.spike.measure_deltachannel --n 100 --delta --snapshot-frequency 50
"""

import argparse
import os
import sqlite3
import tempfile
from typing import Annotated, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


PAYLOAD_BYTES = 50_000  # ~ representative read_url result (50 KB)


def _payload() -> str:
    """Deterministic 50 KB string — pseudo-HTML to mimic read_url output."""
    chunk = (
        "<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco "
        "laboris nisi ut aliquip ex ea commodo consequat.</p>\n"
    )
    repeats = (PAYLOAD_BYTES // len(chunk)) + 1
    return (chunk * repeats)[:PAYLOAD_BYTES]


def _build_graph(use_delta: bool, snapshot_frequency: int = 10):
    """Build a single-node StateGraph that appends one ToolMessage per step.

    use_delta=True: wrap the `messages` channel in DeltaChannel.  Only
        valid on langgraph >= 1.2 with DeltaChannel importable.
    use_delta=False: classic `add_messages` reducer (langgraph 1.x default).
    """
    if use_delta:
        try:
            from langgraph.channels import DeltaChannel  # type: ignore
            # Use deepagents' batch-aware reducer — the same one
            # `_DeepAgentState` uses in production (deepagents 0.6.1
            # graph.py:66).  Plain `add_messages` is NOT batch-aware and
            # raises on DeltaChannel's `reducer(state, [w1, w2, ...])`
            # call shape.
            from deepagents._messages_reducer import _messages_delta_reducer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "DeltaChannel not importable — need langgraph >= 1.2 "
                f"and deepagents >= 0.6.1.  Underlying: {e}"
            ) from e

        class State(TypedDict):
            messages: Annotated[
                list[AnyMessage],
                DeltaChannel(
                    _messages_delta_reducer,
                    snapshot_frequency=snapshot_frequency,
                ),
            ]
    else:
        class State(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

    payload = _payload()

    def step(state: State):
        # Mimic a tool_call → tool_result pattern: emit one AIMessage
        # with a tool_calls list, then a ToolMessage carrying the
        # 50 KB payload.  This is the per-turn growth shape we see
        # in prod (read_url result accumulation).
        tool_call_id = f"call_{len(state['messages'])}"
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_url",
                            "args": {"url": "https://example.com"},
                            "id": tool_call_id,
                        }
                    ],
                ),
                ToolMessage(
                    content=payload,
                    tool_call_id=tool_call_id,
                ),
            ]
        }

    g = StateGraph(State)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_edge("step", END)
    return g


def _db_stats(db_path: str) -> dict:
    """Return per-table sizing.

    Under DeltaChannel, the per-step delta payload is added to the
    ``writes`` table (column ``value``), while the ``checkpoints``
    table (column ``checkpoint``) holds a much smaller snapshot blob
    that says "look at writes for the actual content."  Under the
    baseline ``add_messages`` reducer, virtually all data sits in
    the ``checkpoints`` table.  Reporting only one table can mislead
    readers about where the win comes from — measure both.

    SQLite is in WAL mode by default; uncheckpointed rows live in
    the ``-wal`` sidecar file and are absent from the main DB's
    ``os.path.getsize`` reading.  We force a TRUNCATE checkpoint
    before measuring so ``file_bytes`` reflects the actual on-disk
    footprint of every committed row.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Drain the WAL into the main DB file so file_bytes is
        # accurate.  Without this, low-N runs show suspiciously tiny
        # file_bytes (4096 = single empty page) even though the
        # per-table queries report many rows.
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        ckpt_rows, ckpt_bytes = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(checkpoint)), 0) FROM checkpoints"
        ).fetchone()
        writes_rows, writes_bytes = cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(LENGTH(value)), 0) FROM writes"
        ).fetchone()
    finally:
        conn.close()
    file_bytes = os.path.getsize(db_path)
    return {
        "file_bytes": file_bytes,
        "ckpt_rows": ckpt_rows,
        "ckpt_blob_bytes": ckpt_bytes,
        "writes_rows": writes_rows,
        "writes_value_bytes": writes_bytes,
    }


def _checkpoint_meta(db_path: str) -> dict:
    """Surface schema info for the doc — detect any new tables Layer 2/3
    wrappers don't expect (deltas table, etc)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        tables = sorted(
            r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )
    finally:
        conn.close()
    return {"tables": tables}


def measure(n: int, use_delta: bool, snapshot_frequency: int = 10) -> dict:
    """Drive the graph for n super-steps, snapshot stats every 5."""
    g = _build_graph(use_delta=use_delta, snapshot_frequency=snapshot_frequency)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "threads.db")
        # SqliteSaver wants a connection.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            saver = SqliteSaver(conn)
            app = g.compile(checkpointer=saver)
            thread_id = "spike"
            cfg = {"configurable": {"thread_id": thread_id}}

            trajectory = []
            # n total super-steps — no extra seed invoke, so the
            # advertised N matches the workload exactly.  Each invoke
            # produces one super-step with the synthetic 50 KB
            # ToolMessage payload.
            for i in range(1, n + 1):
                app.invoke({"messages": [HumanMessage(content=f"turn {i}")]}, cfg)
                if i % 5 == 0 or i == n:
                    s = _db_stats(db_path)
                    avg_ckpt = s["ckpt_blob_bytes"] // s["ckpt_rows"] if s["ckpt_rows"] else 0
                    avg_writes = s["writes_value_bytes"] // s["writes_rows"] if s["writes_rows"] else 0
                    trajectory.append({
                        "step": i,
                        "file_bytes": s["file_bytes"],
                        "ckpt_rows": s["ckpt_rows"],
                        "ckpt_blob_bytes": s["ckpt_blob_bytes"],
                        "avg_ckpt_blob_bytes": avg_ckpt,
                        "writes_rows": s["writes_rows"],
                        "writes_value_bytes": s["writes_value_bytes"],
                        "avg_writes_value_bytes": avg_writes,
                    })

            meta = _checkpoint_meta(db_path)
        finally:
            conn.close()

    return {
        "n": n,
        "use_delta": use_delta,
        "snapshot_frequency": snapshot_frequency,
        "payload_bytes_per_step": PAYLOAD_BYTES,
        "trajectory": trajectory,
        "schema": meta,
    }


def _versions() -> dict:
    import importlib.metadata as md
    out = {}
    for pkg in [
        "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite",
        "langgraph-prebuilt", "langchain", "langchain-core", "deepagents",
    ]:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "n/a"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20, 50, 100],
                    help="super-step counts to measure (default: 20 50 100)")
    ap.add_argument("--delta", action="store_true",
                    help="wrap messages channel in DeltaChannel (needs langgraph >= 1.2)")
    ap.add_argument("--snapshot-frequency", type=int, default=10,
                    help="DeltaChannel snapshot_frequency K (default: 10)")
    args = ap.parse_args()

    versions = _versions()
    print("# DeltaChannel spike — measurement run")
    print()
    print("## Versions")
    for k, v in versions.items():
        print(f"- {k} == {v}")
    print()
    print(f"## Mode: {'DeltaChannel(K=' + str(args.snapshot_frequency) + ')' if args.delta else 'baseline (add_messages)'}")
    print()

    for n in args.n:
        result = measure(n=n, use_delta=args.delta,
                         snapshot_frequency=args.snapshot_frequency)
        final = result["trajectory"][-1]
        total_blob = final["ckpt_blob_bytes"] + final["writes_value_bytes"]
        print(f"### N={n}")
        print(f"- final file bytes: {final['file_bytes']:,}")
        print(f"- checkpoints table: {final['ckpt_rows']} rows, "
              f"{final['ckpt_blob_bytes']:,} blob bytes "
              f"(avg {final['avg_ckpt_blob_bytes']:,})")
        print(f"- writes table:      {final['writes_rows']} rows, "
              f"{final['writes_value_bytes']:,} value bytes "
              f"(avg {final['avg_writes_value_bytes']:,})")
        print(f"- sum across tables: {total_blob:,} bytes "
              f"(vs file_bytes {final['file_bytes']:,} — delta is SQLite page overhead)")
        print(f"- schema tables: {result['schema']['tables']}")
        print()
        print("| step | file_bytes | ckpt rows | ckpt avg blob | writes rows | writes avg value |")
        print("|------|------------|-----------|---------------|-------------|------------------|")
        for s in result["trajectory"]:
            print(f"| {s['step']} | {s['file_bytes']:,} | "
                  f"{s['ckpt_rows']} | {s['avg_ckpt_blob_bytes']:,} | "
                  f"{s['writes_rows']} | {s['avg_writes_value_bytes']:,} |")
        print()


if __name__ == "__main__":
    main()
