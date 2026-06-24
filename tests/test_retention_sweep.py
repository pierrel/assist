"""Tests for ``assist.retention.prune_to_n_threads`` and the CLI guard.

Layer 0 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-0-thread-retention.org).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sqlite3
import tempfile
from unittest import TestCase
from unittest.mock import patch, MagicMock

from assist import retention
from assist.thread_manager import ThreadManager


def _make_thread_dir(root: str, tid: str, mtime: float) -> str:
    tdir = os.path.join(root, tid)
    os.makedirs(os.path.join(tdir, "domain"), exist_ok=True)
    os.utime(tdir, (mtime, mtime))
    return tdir


class TestPruneKeepsMostRecent(TestCase):
    """Walks the dir by mtime and hard-deletes all but the most recent N."""

    def test_keeps_n_most_recent_by_mtime(self):
        n = 10
        extras = 5
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                # Create N+5 threads with mtimes spaced 1 minute apart.
                # Thread "00" is the OLDEST; "14" is the NEWEST.
                base = 1_700_000_000.0
                tids = []
                for i in range(n + extras):
                    tid = f"thread-{i:02d}"
                    _make_thread_dir(mgr.root_dir, tid, base + i * 60)
                    tids.append(tid)

                with patch("assist.thread_manager.SandboxManager.cleanup"):
                    deleted = retention.prune_to_n_threads(
                        mgr.root_dir, n, mgr
                    )

                # Five oldest should have been deleted.
                self.assertEqual(set(deleted), {f"thread-{i:02d}" for i in range(extras)})
                self.assertEqual(len(deleted), extras)

                # Most-recent N should survive.
                survivors = sorted(
                    name for name in os.listdir(mgr.root_dir)
                    if os.path.isdir(os.path.join(mgr.root_dir, name))
                    and name != "__pycache__"
                )
                expected_survivors = sorted(
                    f"thread-{i:02d}" for i in range(extras, n + extras)
                )
                self.assertEqual(survivors, expected_survivors)
            finally:
                mgr.close()

    def test_below_floor_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                base = 1_700_000_000.0
                for i in range(3):
                    _make_thread_dir(mgr.root_dir, f"t-{i}", base + i * 60)

                with patch("assist.thread_manager.SandboxManager.cleanup") as mc:
                    deleted = retention.prune_to_n_threads(
                        mgr.root_dir, 100, mgr
                    )

                self.assertEqual(deleted, [])
                mc.assert_not_called()
                self.assertEqual(
                    sorted(
                        n for n in os.listdir(mgr.root_dir)
                        if os.path.isdir(os.path.join(mgr.root_dir, n))
                    ),
                    ["t-0", "t-1", "t-2"],
                )
            finally:
                mgr.close()

    def test_missing_root_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                # Pass a path that does not exist.
                ghost = os.path.join(tmp, "does-not-exist")
                deleted = retention.prune_to_n_threads(ghost, 5, mgr)
                self.assertEqual(deleted, [])
            finally:
                mgr.close()

    def test_one_bad_thread_does_not_block_sweep(self):
        """Per design doc: best-effort, log and continue on per-thread errors."""
        n = 2
        extras = 3
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                base = 1_700_000_000.0
                for i in range(n + extras):
                    _make_thread_dir(
                        mgr.root_dir, f"thread-{i:02d}", base + i * 60
                    )

                # Make hard_delete blow up for one of the candidates
                # (the second-oldest) and verify the rest are still
                # processed.
                original = mgr.hard_delete
                bad_tid = "thread-01"

                def flaky(tid, on_delete=None):
                    if tid == bad_tid:
                        raise RuntimeError("boom")
                    return original(tid, on_delete=on_delete)

                with patch("assist.thread_manager.SandboxManager.cleanup"), \
                     patch.object(mgr, "hard_delete", side_effect=flaky):
                    deleted = retention.prune_to_n_threads(
                        mgr.root_dir, n, mgr
                    )

                # The bad tid should NOT appear in the returned list,
                # but the others (0 and 2) should.
                self.assertNotIn(bad_tid, deleted)
                self.assertIn("thread-00", deleted)
                self.assertIn("thread-02", deleted)
            finally:
                mgr.close()


class TestForeignWriterGuard(TestCase):
    """The CLI ``__main__`` aborts if lsof reports foreign holders."""

    def _run_cli(self, env_overrides: dict, lsof_stdout: str, lsof_rc: int):
        """Invoke ``python -m assist.retention`` with patched subprocess.run.

        Returns the (returncode, stderr) of ``main()``.  We invoke
        ``main()`` in-process rather than as a subprocess so the
        ``mock.patch`` actually applies to the lsof call.
        """
        from io import StringIO

        captured_err = StringIO()

        def fake_run(cmd, *args, **kwargs):
            class R:
                returncode = lsof_rc
                stdout = lsof_stdout
                stderr = ""
            return R()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "threads.db")
            # Touch the DB so the guard runs (it skips if missing).
            open(db_path, "a").close()

            env = {
                "ASSIST_THREADS_DIR": tmp,
                "MIN_THREADS": "5",
            }
            env.update(env_overrides)
            with patch.dict(os.environ, env, clear=False), \
                 patch("assist.retention.subprocess.run", side_effect=fake_run), \
                 patch("sys.stderr", captured_err):
                try:
                    rc = retention.main()
                except SystemExit as se:
                    rc = se.code
            return rc, captured_err.getvalue()

    def test_aborts_when_foreign_pid_holds_db(self):
        # Simulate one foreign process and one row that's our own PID.
        own = str(os.getpid())
        stdout = (
            "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
            f"python  9999 deploy 10u REG  259,1 4096    1 /tmp/threads.db\n"
            f"python  {own} deploy 11u REG  259,1 4096    1 /tmp/threads.db\n"
        )
        rc, err = self._run_cli({}, stdout, lsof_rc=0)
        self.assertEqual(rc, 3)
        self.assertIn("refusing to run", err)

    def test_passes_when_only_self_holds_db(self):
        own = str(os.getpid())
        stdout = (
            "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
            f"python  {own} deploy 11u REG  259,1 4096    1 /tmp/threads.db\n"
        )
        rc, _ = self._run_cli({}, stdout, lsof_rc=0)
        # rc 0 means main() ran the (empty) prune and returned cleanly.
        self.assertEqual(rc, 0)

    def test_passes_when_lsof_reports_no_holders(self):
        # lsof returns 1 when no process holds the file.
        rc, _ = self._run_cli({}, "", lsof_rc=1)
        self.assertEqual(rc, 0)


def _insert_checkpoint(conn, tid, ckpt_id="c1"):
    """Insert one checkpoints row, introspecting columns via PRAGMA so the seed
    survives upstream SqliteSaver schema changes (repo convention — see
    test_thread_manager_hard_delete._seed_thread)."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(checkpoints)")]
    vals = [tid if c == "thread_id" else (ckpt_id if c == "checkpoint_id" else "")
            for c in cols]
    conn.execute(
        f"INSERT INTO checkpoints ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})", vals)


def _insert_write(conn, tid, ckpt_id="c1"):
    """Insert one writes row (schema-introspected; see _insert_checkpoint)."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(writes)")]
    vals = [tid if c == "thread_id"
            else (ckpt_id if c == "checkpoint_id" else (0 if c == "idx" else ""))
            for c in cols]
    conn.execute(
        f"INSERT INTO writes ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})", vals)


class TestPurgeOrphanedCheckpoints(TestCase):
    """purge_orphaned_checkpoints deletes checkpoint-threads that have no
    on-disk dir (deleted threads' leftovers + sub-agent UUID checkpoints),
    and leaves live (dir-having) threads untouched."""

    def _seed(self, mgr, tid):
        _insert_checkpoint(mgr.conn, tid)
        _insert_write(mgr.conn, tid)

    def test_purges_threads_without_a_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "live-thread", "domain"), exist_ok=True)
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                for tid in ("live-thread", "orphan-subagent-uuid", "deleted-thread"):
                    self._seed(mgr, tid)
                mgr.conn.commit()
                purged = retention.purge_orphaned_checkpoints(tmp, mgr)
                self.assertEqual(set(purged), {"orphan-subagent-uuid", "deleted-thread"})
                ck = {r[0] for r in mgr.conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
                wr = {r[0] for r in mgr.conn.execute("SELECT DISTINCT thread_id FROM writes")}
                self.assertEqual(ck, {"live-thread"})
                self.assertEqual(wr, {"live-thread"})  # batched delete removes writes rows too
            finally:
                mgr.close()

    def test_no_orphans_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "only-live", "domain"), exist_ok=True)
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                self._seed(mgr, "only-live")
                mgr.conn.commit()
                self.assertEqual(retention.purge_orphaned_checkpoints(tmp, mgr), [])
                ck = {r[0] for r in mgr.conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
                self.assertEqual(ck, {"only-live"})
            finally:
                mgr.close()

    def test_keeps_all_live_threads_with_many_orphans(self):
        # Load-bearing safety property: NO live thread is purged, regardless of
        # how many orphans surround it.
        with tempfile.TemporaryDirectory() as tmp:
            live = [f"live-{i}" for i in range(5)]
            orphans = [f"orphan-{i}" for i in range(5)]
            for t in live:
                os.makedirs(os.path.join(tmp, t, "domain"), exist_ok=True)
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                for t in live + orphans:
                    self._seed(mgr, t)
                mgr.conn.commit()
                purged = retention.purge_orphaned_checkpoints(tmp, mgr)
                self.assertEqual(set(purged), set(orphans))
                ck = {r[0] for r in mgr.conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints")}
                self.assertEqual(ck, set(live))
            finally:
                mgr.close()

    def test_returns_empty_when_no_checkpoints_table(self):
        # Fresh db (checkpointer never set up) has no checkpoints table; the
        # sweep must no-op, not raise.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "live", "domain"), exist_ok=True)
            mgr = ThreadManager(root_dir=tmp)
            try:
                self.assertEqual(retention.purge_orphaned_checkpoints(tmp, mgr), [])
            finally:
                mgr.close()

    def test_ignores_non_directory_entries(self):
        # A plain file in threads_root (the db itself, a lock) must not be
        # treated as a live thread or break the sweep.
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "live", "domain"), exist_ok=True)
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                open(os.path.join(tmp, "stray.txt"), "w").close()
                for t in ("live", "orphan"):
                    self._seed(mgr, t)
                mgr.conn.commit()
                self.assertEqual(
                    retention.purge_orphaned_checkpoints(tmp, mgr), ["orphan"])
                ck = {r[0] for r in mgr.conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints")}
                self.assertEqual(ck, {"live"})
            finally:
                mgr.close()

    def test_real_db_error_is_not_swallowed(self):
        # A non-"no such table" OperationalError (locked/corrupt) must surface,
        # not be reported as a clean no-op sweep (Copilot PR #142 review).
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "live", "domain"), exist_ok=True)
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                fake = MagicMock()
                fake.execute.side_effect = sqlite3.OperationalError("database is locked")
                with patch.object(mgr, "conn", fake):
                    with self.assertRaises(sqlite3.OperationalError):
                        retention.purge_orphaned_checkpoints(tmp, mgr)
            finally:
                mgr.close()

    def test_db_error_during_orphan_delete_propagates(self):
        # The per-orphan loop must NOT swallow a DB fault (locked/corrupt) — it
        # would report a clean sweep while skipping orphans (Copilot #142 rd2).
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                self._seed(mgr, "orphan")
                mgr.conn.commit()
                with patch.object(retention, "_delete_thread_in_batches",
                                  side_effect=sqlite3.OperationalError("database is locked")):
                    with self.assertRaises(sqlite3.OperationalError):
                        retention.purge_orphaned_checkpoints(tmp, mgr)
            finally:
                mgr.close()

    def test_batched_delete_removes_all_rows_of_a_large_orphan(self):
        # The batch loop must terminate and delete every row of an orphan that
        # exceeds _DELETE_BATCH (the 101GB-incident shape, in miniature).
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ThreadManager(root_dir=tmp)
            try:
                mgr.checkpointer.setup()
                n = retention._DELETE_BATCH + 37
                for i in range(n):
                    _insert_checkpoint(mgr.conn, "big", ckpt_id=f"c{i}")
                mgr.conn.commit()
                self.assertEqual(
                    retention.purge_orphaned_checkpoints(tmp, mgr), ["big"])
                left = mgr.conn.execute(
                    "SELECT count(*) FROM checkpoints WHERE thread_id='big'"
                ).fetchone()[0]
                self.assertEqual(left, 0)
            finally:
                mgr.close()
