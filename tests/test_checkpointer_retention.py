"""Tests for ``assist.checkpointer.CheckpointRetentionSaver``.

Layer 2 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-2-checkpoint-pruning.org).

Real sqlite + real upstream SqliteSaver behind the wrapper.  Mocks
would hide the schema-drift case the wrapper is meant to detect, and
the row-count assertions need a real DB anyway.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from unittest import TestCase

from langgraph.checkpoint.base import empty_checkpoint

from assist.checkpointer import (
    DEFAULT_RETAIN_LAST,
    CheckpointRetentionSaver,
    _resolve_retain_last,
)


def _config(thread_id: str, ns: str = "") -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}


def _ckpt(seq: int) -> dict:
    """Return a Checkpoint with a deterministic, lex-sortable id.

    Real langgraph uses uuid6 (monotonic by time); zero-padded
    integers reproduce the DESC-ordering invariant the wrapper
    relies on without test flakiness from clock resolution.
    """
    c = empty_checkpoint()
    c["id"] = f"{seq:08d}"
    return c


def _put_n(saver: CheckpointRetentionSaver, n: int, *, tid: str, ns: str = "") -> list[str]:
    """Write ``n`` checkpoints; return the ids in insertion order."""
    ids = []
    for i in range(n):
        c = _ckpt(i)
        saver.put(_config(tid, ns), c, {}, {})
        ids.append(c["id"])
    return ids


def _ids_in_db(saver: CheckpointRetentionSaver, tid: str, ns: str = "") -> list[str]:
    cur = saver.conn.cursor()
    cur.execute(
        "SELECT checkpoint_id FROM checkpoints "
        "WHERE thread_id = ? AND checkpoint_ns = ? "
        "ORDER BY checkpoint_id DESC",
        (tid, ns),
    )
    return [row[0] for row in cur.fetchall()]


def _writes_count(saver: CheckpointRetentionSaver, tid: str, ns: str = "") -> int:
    cur = saver.conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM writes WHERE thread_id = ? AND checkpoint_ns = ?",
        (tid, ns),
    )
    return cur.fetchone()[0]


def _new_saver(retain_last: int = 10) -> CheckpointRetentionSaver:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    return CheckpointRetentionSaver(conn, retain_last=retain_last)


class TestBasicPrune(TestCase):
    def test_15_puts_retains_only_last_10(self):
        saver = _new_saver(retain_last=10)
        ids = _put_n(saver, 15, tid="t1")
        survivors = _ids_in_db(saver, "t1")
        # DESC order; 5 oldest dropped.
        self.assertEqual(survivors, list(reversed(ids[5:])))
        self.assertEqual(len(survivors), 10)


class TestNamespaceIsolation(TestCase):
    """Two namespaces under the same thread retain N each independently."""

    def test_each_ns_retains_independently(self):
        saver = _new_saver(retain_last=10)
        _put_n(saver, 15, tid="t1", ns="")
        _put_n(saver, 15, tid="t1", ns="sub")
        self.assertEqual(len(_ids_in_db(saver, "t1", "")), 10)
        self.assertEqual(len(_ids_in_db(saver, "t1", "sub")), 10)


class TestThreadIsolation(TestCase):
    """Two threads retain N each — DELETE must include thread_id predicate."""

    def test_each_thread_retains_independently(self):
        saver = _new_saver(retain_last=10)
        _put_n(saver, 15, tid="ta")
        _put_n(saver, 15, tid="tb")
        self.assertEqual(len(_ids_in_db(saver, "ta")), 10)
        self.assertEqual(len(_ids_in_db(saver, "tb")), 10)


class TestWritesPrunedInLockstep(TestCase):
    """``writes`` rows for pruned checkpoints are also deleted."""

    def test_writes_for_pruned_checkpoints_are_deleted(self):
        saver = _new_saver(retain_last=10)
        # Seed 15 checkpoints; for each, drop one write row.  Use
        # put_writes so the ids match.
        for i in range(15):
            c = _ckpt(i)
            saver.put(_config("t1"), c, {}, {})
            saver.put_writes(
                {"configurable": {
                    "thread_id": "t1",
                    "checkpoint_ns": "",
                    "checkpoint_id": c["id"],
                }},
                [("channel", "value")],
                task_id=f"task-{i}",
            )
        # Only 10 checkpoints survive → only 10 writes rows survive.
        self.assertEqual(_writes_count(saver, "t1"), 10)
        survivors = set(_ids_in_db(saver, "t1"))
        cur = saver.conn.cursor()
        cur.execute(
            "SELECT DISTINCT checkpoint_id FROM writes WHERE thread_id = ?",
            ("t1",),
        )
        write_ids = set(row[0] for row in cur.fetchall())
        self.assertEqual(write_ids, survivors)


class TestDeleteThreadStillWipes(TestCase):
    """Layer 0 hard_delete path: ``delete_thread`` must still drop everything."""

    def test_delete_thread_after_prune_wipes_all(self):
        saver = _new_saver(retain_last=10)
        _put_n(saver, 15, tid="t1")
        self.assertEqual(len(_ids_in_db(saver, "t1")), 10)
        saver.delete_thread("t1")
        self.assertEqual(_ids_in_db(saver, "t1"), [])
        self.assertEqual(_writes_count(saver, "t1"), 0)


class TestRollbackHasEnoughDepth(TestCase):
    """N=10 covers RollbackRunnable.max_rollback_depth=3 with margin."""

    def test_can_walk_back_3_checkpoints_after_prune(self):
        saver = _new_saver(retain_last=10)
        _put_n(saver, 15, tid="t1")
        survivors_desc = _ids_in_db(saver, "t1")
        # 3rd-most-recent must still exist.
        self.assertGreaterEqual(len(survivors_desc), 4)
        third_back = survivors_desc[3]
        cur = saver.conn.cursor()
        cur.execute(
            "SELECT 1 FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
            ("t1", third_back),
        )
        self.assertIsNotNone(cur.fetchone())


class TestRetentionDisabled(TestCase):
    """retain_last=0 → wrapper is a no-op pass-through."""

    def test_zero_skips_prune(self):
        saver = _new_saver(retain_last=0)
        _put_n(saver, 25, tid="t1")
        self.assertEqual(len(_ids_in_db(saver, "t1")), 25)


class TestPruneFailsOpen(TestCase):
    """A failure inside ``_prune`` must not lose the put."""

    def test_delete_failure_is_swallowed_insert_survives(self):
        saver = _new_saver(retain_last=10)
        _put_n(saver, 10, tid="t1")  # seed the prune to actually have work

        original_prune = saver._prune
        calls = {"n": 0}

        def boom(cur, thread_id, checkpoint_ns):
            calls["n"] += 1
            raise sqlite3.OperationalError("synthetic failure")

        saver._prune = boom  # type: ignore[method-assign]
        try:
            # The 11th put should succeed even though _prune raises.
            saver.put(_config("t1"), _ckpt(100), {}, {})
        finally:
            saver._prune = original_prune  # type: ignore[method-assign]

        self.assertEqual(calls["n"], 1)
        # 10 originals + the new one = 11 (no prune happened).
        self.assertEqual(len(_ids_in_db(saver, "t1")), 11)


class TestConcurrentPuts(TestCase):
    """Two Python threads writing to different ``thread_id``s serialize via the upstream lock."""

    def test_no_deadlock_and_correct_counts(self):
        saver = _new_saver(retain_last=10)

        errors: list[BaseException] = []

        def writer(tid: str):
            try:
                _put_n(saver, 15, tid=tid)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t_a = threading.Thread(target=writer, args=("ta",))
        t_b = threading.Thread(target=writer, args=("tb",))
        t_a.start()
        t_b.start()
        t_a.join(timeout=10)
        t_b.join(timeout=10)
        self.assertFalse(t_a.is_alive())
        self.assertFalse(t_b.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(_ids_in_db(saver, "ta")), 10)
        self.assertEqual(len(_ids_in_db(saver, "tb")), 10)


class TestAputStillRaises(TestCase):
    """Sanity: we don't accidentally enable the async path."""

    def test_aput_raises_not_implemented(self):
        saver = _new_saver(retain_last=10)
        with self.assertRaises(NotImplementedError):
            asyncio.run(saver.aput(_config("t1"), _ckpt(0), {}, {}))


class TestExplainQueryPlanUsesPkIndex(TestCase):
    """The prune SELECT must hit the PK index, not a table scan."""

    def test_prune_select_uses_index(self):
        saver = _new_saver(retain_last=10)
        # Realize the schema; setup() runs lazily via cursor(), so
        # do one put first.
        saver.put(_config("t1"), _ckpt(0), {}, {})
        cur = saver.conn.cursor()
        cur.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT checkpoint_id FROM checkpoints "
            "WHERE thread_id = ? AND checkpoint_ns = ? "
            "ORDER BY checkpoint_id DESC LIMIT -1 OFFSET ?",
            ("t1", "", 10),
        )
        plan = " ".join(str(row) for row in cur.fetchall()).upper()
        # SQLite's planner reports either "USING INDEX" or "USING
        # COVERING INDEX" or "USING PRIMARY KEY".  Any of those is
        # fine; a "SCAN TABLE" without an index is not.
        self.assertIn("INDEX", plan + "PRIMARY KEY")
        self.assertNotIn("SCAN CHECKPOINTS", plan)


class TestSchemaDriftDisablesPrune(TestCase):
    """An unexpected upstream schema (extra/missing table) disables pruning."""

    def test_extra_table_disables_prune(self):
        saver = _new_saver(retain_last=10)
        # Realize the schema (lazy via cursor()), then add a sentinel
        # table that doesn't belong to the pin.  Use a non-colliding
        # seed id so the subsequent _put_n's INSERT OR REPLACE doesn't
        # overwrite it.
        saver.put(_config("t1"), _ckpt(999), {}, {})
        saver.conn.execute("CREATE TABLE sentinel (x INTEGER)")
        # Reset the cached check so the next put re-evaluates.
        saver._schema_ok = None
        with self.assertLogs("assist.checkpointer", level="WARNING") as cm:
            _put_n(saver, 15, tid="t1")
        self.assertTrue(
            any("schema drift" in m for m in cm.output),
            f"expected schema-drift warning, got: {cm.output!r}",
        )
        # Drift disables pruning → all 16 checkpoints survive (the
        # seed at id=999 + 15 more at ids 0-14).
        self.assertEqual(len(_ids_in_db(saver, "t1")), 16)


class TestEnvVarResolution(TestCase):
    def test_unset_returns_default(self, ):
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=False):
            import os as _os
            _os.environ.pop("ASSIST_RETAIN_LAST", None)
            self.assertEqual(_resolve_retain_last(), DEFAULT_RETAIN_LAST)

    def test_explicit_integer(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"ASSIST_RETAIN_LAST": "42"}):
            self.assertEqual(_resolve_retain_last(), 42)

    def test_disabled_keywords_zero_out(self):
        from unittest.mock import patch
        for raw in ("0", "disabled", "off", "false", "DISABLED", ""):
            with patch.dict("os.environ", {"ASSIST_RETAIN_LAST": raw}):
                self.assertEqual(
                    _resolve_retain_last(), 0,
                    f"expected 0 for ASSIST_RETAIN_LAST={raw!r}",
                )

    def test_invalid_falls_back_with_warning(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"ASSIST_RETAIN_LAST": "not-a-number"}):
            with self.assertLogs("assist.checkpointer", level="WARNING"):
                self.assertEqual(_resolve_retain_last(), DEFAULT_RETAIN_LAST)

    def test_negative_falls_back_with_warning(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"ASSIST_RETAIN_LAST": "-5"}):
            with self.assertLogs("assist.checkpointer", level="WARNING"):
                self.assertEqual(_resolve_retain_last(), DEFAULT_RETAIN_LAST)
