"""Integration tests for ``ThreadManager.hard_delete``.

Layer 0 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-0-thread-retention.org).

Uses a real temp ``ThreadManager`` + real sqlite + mocked
``SandboxManager``.  Mocks for sqlite would hide any future
upstream-schema mismatch; a real DB catches that.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest import TestCase
from unittest.mock import patch

from assist.thread import ThreadManager


def _seed_thread(manager: ThreadManager, tid: str) -> str:
    """Create a thread dir and a couple of fake checkpointer rows.

    Returns the thread directory path.  We poke the SqliteSaver's
    schema directly with INSERTs because the public API requires a
    full agent run; we only care that ``delete_thread`` removes the
    rows we put in.
    """
    tdir = os.path.join(manager.root_dir, tid)
    os.makedirs(os.path.join(tdir, "domain"), exist_ok=True)
    # Drop a sentinel file so we can verify rmtree happened.
    with open(os.path.join(tdir, "marker.txt"), "w") as f:
        f.write("seed")

    # Realize the SqliteSaver schema.  ``setup()`` is the upstream
    # idempotent migration helper.
    manager.checkpointer.setup()
    cur = manager.conn.cursor()
    # Mirror the schema columns we know about (checkpoints, writes).
    # Use thread_id as the only meaningful field; everything else is
    # whatever the migration created.
    cur.execute("PRAGMA table_info(checkpoints)")
    ck_cols = [row[1] for row in cur.fetchall()]
    placeholders = ", ".join("?" for _ in ck_cols)
    values = [tid if c == "thread_id" else "" for c in ck_cols]
    cur.execute(
        f"INSERT INTO checkpoints ({', '.join(ck_cols)}) VALUES ({placeholders})",
        values,
    )

    cur.execute("PRAGMA table_info(writes)")
    w_cols = [row[1] for row in cur.fetchall()]
    if w_cols:
        # ``idx`` is an int column; everything else gets an empty
        # string.  Good enough to verify deletion.
        placeholders = ", ".join("?" for _ in w_cols)
        values = [
            tid if c == "thread_id" else (0 if c == "idx" else "")
            for c in w_cols
        ]
        cur.execute(
            f"INSERT INTO writes ({', '.join(w_cols)}) VALUES ({placeholders})",
            values,
        )
    manager.conn.commit()
    return tdir


def _count_rows(manager: ThreadManager, table: str, tid: str) -> int:
    cur = manager.conn.cursor()
    cur.execute(
        f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?", (tid,)
    )
    return cur.fetchone()[0]


class TestHardDeleteRemovesAllThree(TestCase):
    """The three removal targets all land: dir, DB rows, sandbox cleanup."""

    def test_removes_dir_db_and_calls_sandbox_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000000-aaaaaaaa"
                tdir = _seed_thread(mgr, tid)
                expected_workdir = mgr.thread_default_working_dir(tid)

                self.assertTrue(os.path.isdir(tdir))
                self.assertEqual(_count_rows(mgr, "checkpoints", tid), 1)

                with patch(
                    "assist.thread.SandboxManager.cleanup"
                ) as mock_cleanup:
                    mgr.hard_delete(tid)

                mock_cleanup.assert_called_once_with(expected_workdir)
                self.assertFalse(os.path.exists(tdir))
                self.assertEqual(_count_rows(mgr, "checkpoints", tid), 0)
                self.assertEqual(_count_rows(mgr, "writes", tid), 0)
            finally:
                mgr.close()


class TestHardDeleteOnDeleteCallback(TestCase):
    """on_delete callbacks fire with the tid argument."""

    def test_single_callback_invoked_with_tid(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000001-bbbbbbbb"
                _seed_thread(mgr, tid)

                received: list[str] = []

                def cb(t: str) -> None:
                    received.append(t)

                with patch("assist.thread.SandboxManager.cleanup"):
                    mgr.hard_delete(tid, on_delete=[cb])

                self.assertEqual(received, [tid])
            finally:
                mgr.close()

    def test_multiple_callbacks_all_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000002-cccccccc"
                _seed_thread(mgr, tid)

                hits: list[str] = []
                cbs = [
                    lambda t: hits.append(f"a:{t}"),
                    lambda t: hits.append(f"b:{t}"),
                    lambda t: hits.append(f"c:{t}"),
                ]
                with patch("assist.thread.SandboxManager.cleanup"):
                    mgr.hard_delete(tid, on_delete=cbs)

                self.assertEqual(
                    hits, [f"a:{tid}", f"b:{tid}", f"c:{tid}"]
                )
            finally:
                mgr.close()


class TestHardDeleteIdempotent(TestCase):
    """Re-running on a half-deleted thread succeeds."""

    def test_dir_already_gone_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000003-dddddddd"
                tdir = _seed_thread(mgr, tid)

                # Simulate a prior crashed sweep: dir wiped, DB rows
                # still present.
                import shutil as _shutil
                _shutil.rmtree(tdir)
                self.assertFalse(os.path.exists(tdir))
                self.assertEqual(_count_rows(mgr, "checkpoints", tid), 1)

                with patch("assist.thread.SandboxManager.cleanup"):
                    # Should complete without raising.
                    mgr.hard_delete(tid)

                # Second-half cleanup should still finish.
                self.assertEqual(_count_rows(mgr, "checkpoints", tid), 0)
                self.assertFalse(os.path.exists(tdir))
            finally:
                mgr.close()

    def test_double_call_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000004-eeeeeeee"
                _seed_thread(mgr, tid)

                with patch("assist.thread.SandboxManager.cleanup"):
                    mgr.hard_delete(tid)
                    # Second call: pure no-op path, must not raise.
                    mgr.hard_delete(tid)
            finally:
                mgr.close()


class TestHardDeleteCallbackIsolation(TestCase):
    """A misbehaving callback can't break sibling callbacks or cleanup."""

    def test_raising_callback_does_not_block_subsequent(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000005-ffffffff"
                tdir = _seed_thread(mgr, tid)

                hits: list[str] = []

                def boom(t: str) -> None:
                    raise RuntimeError("intentional test failure")

                def good(t: str) -> None:
                    hits.append(t)

                with patch("assist.thread.SandboxManager.cleanup"):
                    # boom raises but good must still run.
                    mgr.hard_delete(tid, on_delete=[boom, good])

                self.assertEqual(hits, [tid])
                # And the cleanup itself completed.
                self.assertFalse(os.path.exists(tdir))
                self.assertEqual(_count_rows(mgr, "checkpoints", tid), 0)
            finally:
                mgr.close()

    def test_raising_callback_does_not_undo_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                tid = "20260504000006-gggggggg"
                tdir = _seed_thread(mgr, tid)

                def boom(t: str) -> None:
                    raise RuntimeError("nope")

                with patch("assist.thread.SandboxManager.cleanup"):
                    mgr.hard_delete(tid, on_delete=[boom])

                # Cleanup completed before callbacks ran.
                self.assertFalse(os.path.exists(tdir))
                self.assertEqual(_count_rows(mgr, "checkpoints", tid), 0)
            finally:
                mgr.close()
