"""Translate native steering tickets and receipts to the existing Run journal.

The web adapter owns admission fencing, owner/reset validation and fallback
dispatch. Mutation helpers here require that caller's admission fence.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
import json
from pathlib import Path
import sqlite3

from assist.run_service import Run, RunService


def frame(run: Run) -> str:
    """Keep equal text distinct when Hermes combines a batch of steers."""
    return f"[Assist follow-up id={run.id}]\n{run.text}"


def offer_followups(service: RunService, owner: Run, candidates: Sequence[Run],
                   consumed: Collection[str] = ()) -> list[dict]:
    """Offer/claim native tickets under the caller's admission fence.

    Candidates must be the owner-validated, reset-bounded journal snapshot.
    Only exact eligible IDs acknowledged after native persistence are claimed.
    """
    eligible = [item for item in candidates
                if item.status == "pending" and item.interjection_work_id == owner.work_id
                and item.sender is None and item.origin is None and item.user_event_id == item.id
                and item.mode == "turn" and item.assistant_id == "general-agent"
                and item.text and item.review_snapshot is None and item.approval_id is None]
    for item in eligible:
        if item.id in consumed:
            service.transition(owner.thread_id, item.id, "success", consumed_by=owner.id)
    return [{"id": item.id, "text": frame(item)} for item in eligible
            if item.id not in consumed]


def reconcile_receipts(service: RunService, thread_id: str, agent_dir: str) -> None:
    """Claim committed native IDs before fallback, under the admission fence.

    Missing receipts leave tickets pending; unreadable receipts propagate to
    the web dispatcher, which reports uncertain delivery without replay.
    """
    runs = service.list(thread_id)
    pending = [run for run in runs if run.status == "pending" and run.interjection_work_id]
    if not pending:
        return
    receipts = committed_receipts(agent_dir)
    for run in pending:
        if receipts.get(run.id) == run.interjection_work_id:
            owner = next(item for item in runs if item.work_id == run.interjection_work_id)
            service.transition(thread_id, run.id, "success", consumed_by=owner.id)


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
