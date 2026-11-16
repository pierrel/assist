"""Read selected private eval threads' latest reconstructed checkpoint messages."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from deepagents._messages_reducer import _messages_delta_reducer
from langchain_core.messages import message_to_dict
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.pregel._checkpoint import channels_from_checkpoint


def reconstruct_messages(checkpoint_path: Path, thread_ids: Sequence[str]) -> list[dict]:
    """Return selected threads' latest state messages, including delta-only rows."""
    if not thread_ids or any(not isinstance(tid, str) or not tid for tid in thread_ids):
        raise ValueError("explicit nonempty thread IDs are required")
    conn = sqlite3.connect(checkpoint_path.resolve().as_uri() + "?mode=ro", uri=True,
                           check_same_thread=False)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoints'").fetchone():
            raise ValueError("checkpoint store has no schema")
        saver = SqliteSaver(conn)
        names = {(item.config["configurable"]["thread_id"],
                  item.config["configurable"].get("checkpoint_ns", ""))
                 for tid in set(thread_ids)
                 for item in saver.list({"configurable": {"thread_id": tid}})}
        if set(thread_ids) - {tid for tid, _ in names}:
            raise ValueError("selected thread has no checkpoint")
        result = []
        for thread_id, namespace in sorted(names):
            latest = saver.get_tuple({"configurable": {
                "thread_id": thread_id, "checkpoint_ns": namespace}})
            if latest is None:
                raise RuntimeError("checkpoint namespace disappeared during export")
            channels, _ = channels_from_checkpoint(
                {"messages": DeltaChannel(_messages_delta_reducer, snapshot_frequency=50)},
                latest.checkpoint, saver=saver, config=latest.config)
            messages = channels["messages"].get()
            result.append({"thread_id": thread_id, "checkpoint_ns": namespace,
                           "checkpoint_id": latest.config["configurable"]["checkpoint_id"],
                           "messages": [message_to_dict(message) for message in messages]})
        return result
    finally:
        conn.close()


def export_trace(checkpoint_path: Path, output_path: Path,
                 thread_ids: Sequence[str]) -> None:
    """Export latest state; callback logs retain earlier model-visible history."""
    payload = {"scope": "latest_reconstructed_state", "historical_messages_covered": False,
               "threads": reconstruct_messages(checkpoint_path, thread_ids)}
    fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(payload, output, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--thread-id", action="append", required=True)
    args = parser.parse_args()
    export_trace(args.checkpoint, args.output, args.thread_id)
