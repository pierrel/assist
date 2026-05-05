"""Tests for ``assist.retention.prune_to_n_threads`` and the CLI guard.

Layer 0 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-0-thread-retention.org).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

from assist import retention
from assist.thread import ThreadManager


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

                with patch("assist.thread.SandboxManager.cleanup"):
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

                with patch("assist.thread.SandboxManager.cleanup") as mc:
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

                with patch("assist.thread.SandboxManager.cleanup"), \
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
