"""Checkpoint retention wrapper.

Layer 2 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-2-checkpoint-pruning.org).
Wraps langgraph's SqliteSaver to bound per-thread checkpoint count.

On every ``put()`` the wrapper retains only the most-recent
``ASSIST_RETAIN_LAST`` checkpoints per ``(thread_id, checkpoint_ns)``
and deletes the older rows from ``checkpoints`` and ``writes`` in the
same transaction as the ``INSERT``.

Pinned against ``langgraph-checkpoint-sqlite==3.0.3``.  See
``EXPECTED_TABLES`` for the schema assertion that flags upstream drift.
"""
from __future__ import annotations

import json
import logging
import os

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

DEFAULT_RETAIN_LAST = 10

# Tables the wrapper expects to find.  A schema drift (extra or
# missing tables) is logged at first put() and disables pruning for
# the saver's lifetime — the prune is a best-effort optimization,
# never let it block a put.
EXPECTED_TABLES = frozenset({"checkpoints", "writes"})


def _resolve_retain_last() -> int:
    """Read ASSIST_RETAIN_LAST.

    - unset → DEFAULT_RETAIN_LAST
    - "0" / "" / "disabled" / "off" / "false" → 0 (no-op pass-through)
    - positive int → retain that many
    - anything else → DEFAULT_RETAIN_LAST with a warning (typo → loud
      log, never silently disabled)
    """
    raw = os.environ.get("ASSIST_RETAIN_LAST")
    if raw is None:
        return DEFAULT_RETAIN_LAST
    norm = raw.strip().lower()
    if norm in ("", "disabled", "off", "false"):
        return 0
    try:
        n = int(norm)
    except ValueError:
        logger.warning(
            "ASSIST_RETAIN_LAST=%r is not an integer; defaulting to %d",
            raw, DEFAULT_RETAIN_LAST,
        )
        return DEFAULT_RETAIN_LAST
    if n < 0:
        logger.warning(
            "ASSIST_RETAIN_LAST=%d is negative; defaulting to %d",
            n, DEFAULT_RETAIN_LAST,
        )
        return DEFAULT_RETAIN_LAST
    return n


class CheckpointRetentionSaver(SqliteSaver):
    """SqliteSaver that prunes old checkpoints inline with each put().

    Retention count is read from ``ASSIST_RETAIN_LAST`` at construction
    time (unless overridden via ``retain_last=`` kwarg for tests).
    ``retain_last == 0`` makes the wrapper a no-op pass-through —
    operational kill switch.

    The async API (``aput`` etc.) is not overridden because upstream's
    sync ``SqliteSaver`` raises ``NotImplementedError`` from all async
    methods (see ``_AIO_ERROR_MSG`` in the upstream module).
    """

    def __init__(self, *args, retain_last: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retain_last = (
            retain_last if retain_last is not None else _resolve_retain_last()
        )
        # None = not yet checked; True/False = result cached.
        self._schema_ok: bool | None = None
        if self.retain_last == 0:
            logger.info("CheckpointRetentionSaver: pruning disabled")
        else:
            logger.info(
                "CheckpointRetentionSaver: retaining last %d checkpoints "
                "per (thread_id, checkpoint_ns)",
                self.retain_last,
            )

    def _verify_schema(self, cur) -> bool:
        """Return True if the live schema matches EXPECTED_TABLES.

        Cached after the first call.  A mismatch logs once and disables
        pruning for the saver's lifetime — operator must bump the pin
        and update EXPECTED_TABLES.
        """
        if self._schema_ok is not None:
            return self._schema_ok
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        actual = frozenset(row[0] for row in cur.fetchall())
        if actual != EXPECTED_TABLES:
            logger.warning(
                "CheckpointRetentionSaver: schema drift detected — "
                "expected %s, got %s (extras=%s, missing=%s); "
                "pruning disabled.  Bump langgraph-checkpoint-sqlite "
                "and update EXPECTED_TABLES in assist/checkpointer.py.",
                sorted(EXPECTED_TABLES), sorted(actual),
                sorted(actual - EXPECTED_TABLES),
                sorted(EXPECTED_TABLES - actual),
            )
            self._schema_ok = False
        else:
            self._schema_ok = True
        return self._schema_ok

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint and prune older ones in one transaction.

        Mirrors upstream SqliteSaver.put (langgraph-checkpoint-sqlite==3.0.3,
        lines 380-436) so the INSERT and the prune share a single
        cursor + lock acquisition + commit.  Re-validate this body on
        langgraph upgrades.

        Prune failures are caught and logged — the INSERT is the
        load-bearing operation; the prune self-heals on the next put.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")

        with self.cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, "
                "parent_checkpoint_id, type, checkpoint, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(thread_id),
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    type_,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )

            if self.retain_last > 0 and self._verify_schema(cur):
                try:
                    self._prune(cur, str(thread_id), checkpoint_ns)
                except Exception:
                    logger.exception(
                        "Prune failed for thread_id=%r checkpoint_ns=%r; "
                        "checkpoint inserted, prune self-heals on next put.",
                        thread_id, checkpoint_ns,
                    )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def _prune(self, cur, thread_id: str, checkpoint_ns: str) -> None:
        """Delete checkpoints beyond the retention window for one (tid, ns).

        Scoped by the full ``(thread_id, checkpoint_ns)`` pair so two
        namespaces under the same ``thread_id`` retain N each.
        Deletes from ``writes`` before ``checkpoints`` because writes
        references checkpoints semantically, even though no FK
        enforces it — keeps the invariant clean if a future schema
        adds one.
        """
        cur.execute(
            "SELECT checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "ORDER BY checkpoint_id DESC LIMIT -1 OFFSET ?",
            (thread_id, checkpoint_ns, self.retain_last),
        )
        old_ids = [row[0] for row in cur.fetchall()]
        if not old_ids:
            return
        placeholders = ",".join("?" * len(old_ids))
        cur.execute(
            f"DELETE FROM writes "
            f"WHERE thread_id = ? AND checkpoint_ns = ? "
            f"AND checkpoint_id IN ({placeholders})",
            (thread_id, checkpoint_ns, *old_ids),
        )
        cur.execute(
            f"DELETE FROM checkpoints "
            f"WHERE thread_id = ? AND checkpoint_ns = ? "
            f"AND checkpoint_id IN ({placeholders})",
            (thread_id, checkpoint_ns, *old_ids),
        )
