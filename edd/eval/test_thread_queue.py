"""End-to-end eval for ThreadAffinityQueue.

Spawns two real Threads concurrently, each sending one short message,
and verifies that the queue serialized them — i.e., one thread observed
the "queued" state for nonzero duration before acquiring the lock.
"""
import os
import shutil
import tempfile
import threading
import time

from unittest import TestCase

from assist.thread import ThreadManager
from assist.thread_queue import THREAD_QUEUE


class _StateRecorder:
    """Capture (timestamp, state) tuples emitted by the queue."""

    def __init__(self):
        self.events: list[tuple[float, str]] = []
        self._lock = threading.Lock()

    def __call__(self, state: str) -> None:
        with self._lock:
            self.events.append((time.time(), state))

    def states(self) -> list[str]:
        with self._lock:
            return [s for _, s in self.events]

    def queued_duration(self) -> float:
        """Seconds spent in 'queued' before observing 'running'.

        0.0 if the thread never queued.
        """
        with self._lock:
            queued_at = None
            for ts, state in self.events:
                if state == "queued":
                    queued_at = ts
                elif state == "running" and queued_at is not None:
                    return ts - queued_at
        return 0.0


class TestThreadQueueE2E(TestCase):
    """Concurrent Thread.message() calls must serialize through the queue."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.tm = ThreadManager(self.root)

    def tearDown(self):
        try:
            self.tm.close()
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)

    def test_two_concurrent_threads_serialize_via_queue(self):
        """Both threads finish; one observes a 'queued' state for >0 seconds."""
        # Sanity: queue is empty at start.
        self.assertIsNone(THREAD_QUEUE.current_handle())
        self.assertEqual(THREAD_QUEUE.waiter_count(), 0)

        # Two recorders, one per thread.
        rec_a = _StateRecorder()
        rec_b = _StateRecorder()

        chat_a = self.tm.new(on_queue_state=rec_a)
        chat_b = self.tm.new(on_queue_state=rec_b)

        # Use the same prompt so cache-pressure conditions are equivalent.
        # Short prompt keeps the eval bounded.
        prompt = "What is 2+2? Answer in one short sentence."

        results: dict[str, str] = {}
        errors: dict[str, BaseException] = {}

        def run(name: str, chat):
            try:
                results[name] = chat.message(prompt)
            except BaseException as e:
                errors[name] = e

        ta = threading.Thread(target=run, args=("A", chat_a))
        tb = threading.Thread(target=run, args=("B", chat_b))

        # Start A first so its acquire wins the unloaded queue, then B
        # right after so B has to wait on A.
        start = time.time()
        ta.start()
        # Tiny delay to make A's first-acquire deterministic.
        time.sleep(0.01)
        tb.start()
        ta.join(timeout=180)
        tb.join(timeout=180)
        wall = time.time() - start

        # Both threads completed (no hangs, no exceptions).
        self.assertFalse(ta.is_alive(), "thread A did not finish in time")
        self.assertFalse(tb.is_alive(), "thread B did not finish in time")
        self.assertEqual(errors, {}, f"thread errors: {errors}")
        self.assertIn("A", results)
        self.assertIn("B", results)
        self.assertGreater(len(results["A"]), 0, "thread A returned empty")
        self.assertGreater(len(results["B"]), 0, "thread B returned empty")

        # The queue actually serialized something: at least one of the
        # threads spent measurable time in 'queued' before 'running'.
        # This proves the lock contended; if both immediately acquired,
        # the test environment isn't exercising what it claims to.
        a_wait = rec_a.queued_duration()
        b_wait = rec_b.queued_duration()
        max_wait = max(a_wait, b_wait)
        self.assertGreater(
            max_wait,
            0.0,
            f"neither thread observed a queued state; "
            f"events A={rec_a.events!r} B={rec_b.events!r}",
        )

        # The first thread (A) saw only "running" (no queued).
        self.assertEqual(rec_a.states(), ["running"])
        # The second thread (B) saw "queued" then "running".
        self.assertEqual(rec_b.states(), ["queued", "running"])

        # Queue is cleaned up after both threads finish.
        self.assertIsNone(THREAD_QUEUE.current_handle())
        self.assertEqual(THREAD_QUEUE.waiter_count(), 0)

        # Sanity ceiling: 2 short prompts on a local LLM should land in
        # well under 3 minutes total even with full prefill cost on
        # both. This is a generous upper bound — its job is to catch
        # hangs, not measure throughput.
        self.assertLess(wall, 180.0, f"wall-clock {wall:.1f}s exceeded sanity cap")
