"""Read native committed steering receipts before dispatching journal fallback."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3


def frame(run) -> str:
    """Keep equal text distinct when Hermes combines a batch of steers."""
    return f"[Assist follow-up id={run.id}]\n{run.text}"


def committed_receipts(agent_dir: str) -> dict[str, str]:
    """Read user-row receipt metadata across this thread's compression lineage.

    Compression can rotate sessions or archive messages. This thread has its
    own SessionDB, and every receipt carries its exact logical host work ID;
    inactive rows and ancestors must therefore remain in the scan.
    """
    directory = Path(agent_dir) / "hermes"
    database = directory / "session.db"
    if directory.is_symlink() or database.is_symlink():
        raise RuntimeError("Native steering history needs reconciliation")
    if not database.exists():
        return {}
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
        rows = connection.execute(
            "SELECT display_metadata FROM messages WHERE role = 'user' "
            "AND display_kind = 'steer' AND display_metadata IS NOT NULL")
        receipts = {}
        for (raw,) in rows:
            receipt = json.loads(raw).get("assist_steering")
            if receipt is None:
                continue
            work_id, ids = receipt["work_id"], receipt["run_ids"]
            if (not isinstance(work_id, str) or not work_id or not isinstance(ids, list)
                    or any(not isinstance(rid, str) or not rid for rid in ids)):
                raise RuntimeError("Native steering receipt needs reconciliation")
            receipts.update((rid, work_id) for rid in ids)
        return receipts
