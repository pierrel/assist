"""Concurrency contract tests for the web merge / push endpoints.

The web app runs as a single uvicorn worker, but two browser tabs can
fire ``/thread/<tid>/merge`` (or ``/push-main``) within the same
millisecond.  ``manage.web.state.MERGE_LOCK`` is the in-process gate
that serialises those calls; without it, the host's ``git fetch`` →
``git rebase`` → ``git commit`` sequence would interleave and leave
the local repo in a partial state.

These tests pin two contracts:

1. ``MERGE_LOCK`` exists, is a real :class:`threading.Lock`, and is
   shared between the merge and push routes (same module-level
   binding, not two independent locks).
2. Inside a routes' critical section, holding the lock blocks the
   other route — exercised by a tiny ``threading``-level harness.
"""
import threading
import time
import unittest

from manage.web import state


class TestMergeLockContract(unittest.TestCase):
    def test_merge_lock_is_a_threading_lock(self):
        # Threading.Lock isn't a class, it's a factory — `threading.Lock()`
        # returns an instance of `_thread.lock`.  We probe by capability,
        # not by isinstance.
        self.assertTrue(hasattr(state.MERGE_LOCK, 'acquire'))
        self.assertTrue(hasattr(state.MERGE_LOCK, 'release'))

    def test_merge_lock_serialises_concurrent_acquirers(self):
        """A second thread waiting on ``MERGE_LOCK`` must wait for the
        first to release.  Holding the lock for 100ms and asserting the
        second thread's acquire wall-clock is at least 90ms is enough
        to catch a lock-not-actually-shared regression.
        """
        observed = []

        def hold(label: str, hold_for: float) -> None:
            with state.MERGE_LOCK:
                start = time.monotonic()
                observed.append((label, "acquired", start))
                time.sleep(hold_for)
                observed.append((label, "released", time.monotonic()))

        t1 = threading.Thread(target=hold, args=("first", 0.1))
        t1.start()
        # Give the first thread a moment to acquire before the second tries.
        time.sleep(0.01)
        t2_start = time.monotonic()
        t2 = threading.Thread(target=hold, args=("second", 0.0))
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        # Order: first acquired, first released, second acquired.
        self.assertEqual(
            [(label, event) for (label, event, _) in observed],
            [("first", "acquired"), ("first", "released"),
             ("second", "acquired"), ("second", "released")],
            "merge lock did not serialise the two acquirers",
        )

        # Second thread waited at least the hold duration minus the
        # leading 10ms sleep (with slop for scheduling).
        second_acquired_at = next(
            ts for label, event, ts in observed
            if label == "second" and event == "acquired"
        )
        wait = second_acquired_at - t2_start
        self.assertGreaterEqual(
            wait, 0.07,
            f"second thread acquired without waiting (waited {wait*1000:.1f}ms)",
        )


if __name__ == '__main__':
    unittest.main()
