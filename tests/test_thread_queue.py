"""Tests for ThreadAffinityQueue (assist/thread_queue.py)."""
import os
import threading
import time
from unittest.mock import patch

import pytest

from assist.thread_queue import (
    DEFAULT_HOLD_TIMEOUT_S,
    DEFAULT_WAIT_TIMEOUT_S,
    QueueWaitTimeout,
    ThreadAffinityQueue,
    active_handle,
)


def test_no_contention_runs_immediately_with_no_queued_callback():
    q = ThreadAffinityQueue()
    states = []
    with q.acquire("A", on_state_change=states.append):
        pass
    # Single holder never observed "queued"; only "running" was emitted.
    assert states == ["running"]
    assert q.current_handle() is None


def test_second_thread_observes_queued_then_running():
    q = ThreadAffinityQueue()
    holder_can_release = threading.Event()
    waiter_states: list[str] = []
    waiter_done = threading.Event()

    def hold_a():
        with q.acquire("A"):
            holder_can_release.wait(timeout=5)

    def waiter_b():
        with q.acquire("B", on_state_change=waiter_states.append):
            pass
        waiter_done.set()

    ta = threading.Thread(target=hold_a)
    ta.start()
    # Give A a head start so B definitely arrives second.
    time.sleep(0.05)
    tb = threading.Thread(target=waiter_b)
    tb.start()

    # B has not acquired yet; should be queued.
    time.sleep(0.05)
    assert waiter_states == ["queued"]
    assert q.waiter_count() == 1

    holder_can_release.set()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert waiter_done.is_set()
    assert waiter_states == ["queued", "running"]
    assert q.current_handle() is None
    assert q.waiter_count() == 0


def test_fifo_among_waiters():
    q = ThreadAffinityQueue()
    holder_can_release = threading.Event()
    completion_order: list[str] = []
    completion_lock = threading.Lock()

    def hold_a():
        with q.acquire("A"):
            holder_can_release.wait(timeout=5)
        with completion_lock:
            completion_order.append("A")

    def waiter(tid: str, started: threading.Event):
        started.set()
        with q.acquire(tid):
            # Tiny bit of work to force ordering to be observable.
            time.sleep(0.01)
        with completion_lock:
            completion_order.append(tid)

    ta = threading.Thread(target=hold_a)
    ta.start()
    time.sleep(0.05)

    # Stagger waiter starts so FIFO order is unambiguous.
    starts = [threading.Event() for _ in range(3)]
    waiters = []
    for tid, ev in zip(["B", "C", "D"], starts):
        t = threading.Thread(target=waiter, args=(tid, ev))
        waiters.append(t)
        t.start()
        ev.wait(timeout=1)
        # Allow this waiter's acquire() to enqueue before the next one.
        time.sleep(0.02)

    holder_can_release.set()
    ta.join(timeout=5)
    for w in waiters:
        w.join(timeout=5)

    assert completion_order == ["A", "B", "C", "D"]


def test_holder_exception_releases_lock():
    q = ThreadAffinityQueue()
    waiter_acquired = threading.Event()

    def hold_a():
        try:
            with q.acquire("A"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    def waiter_b():
        with q.acquire("B"):
            waiter_acquired.set()

    ta = threading.Thread(target=hold_a)
    tb = threading.Thread(target=waiter_b)
    ta.start()
    tb.start()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert waiter_acquired.is_set()
    assert q.current_handle() is None


def test_wait_timeout_raises():
    q = ThreadAffinityQueue()
    holder_can_release = threading.Event()

    def hold_a():
        with q.acquire("A"):
            holder_can_release.wait(timeout=2)

    ta = threading.Thread(target=hold_a)
    ta.start()
    time.sleep(0.05)

    with pytest.raises(QueueWaitTimeout):
        with q.acquire("B", wait_timeout_s=0.1):
            pytest.fail("should not have acquired")

    # Holder still owns the queue; we did not steal it on timeout.
    assert q.current_handle() is not None
    assert q.current_handle().thread_id == "A"

    holder_can_release.set()
    ta.join(timeout=5)
    assert q.current_handle() is None
    # Waiter cleaned itself out of the queue.
    assert q.waiter_count() == 0


def test_hold_timeout_marks_expired():
    q = ThreadAffinityQueue()
    with q.acquire("A", hold_timeout_s=0.1) as handle:
        assert handle.expired is False
        time.sleep(0.25)
        assert handle.expired is True


def test_reentrant_same_thread_id_is_noop():
    q = ThreadAffinityQueue()
    states_outer: list[str] = []
    states_inner: list[str] = []
    with q.acquire("A", on_state_change=states_outer.append) as outer:
        with q.acquire("A", on_state_change=states_inner.append) as inner:
            assert inner is outer
        # Inner exit must not release the holder.
        assert q.current_handle() is outer
    assert q.current_handle() is None
    # Outer fired "running"; inner fired nothing (no "queued", no "running").
    assert states_outer == ["running"]
    assert states_inner == []


def test_same_thread_id_from_different_os_thread_does_not_free_ride():
    # Reentrancy is gated on "same call stack as the holder", not just
    # matching thread_id.  A web double-click produces two background
    # tasks with the same tid; the second must queue, not run alongside.
    q = ThreadAffinityQueue()
    holder_can_release = threading.Event()
    waiter_states: list[str] = []
    waiter_done = threading.Event()

    def hold_a():
        with q.acquire("A"):
            holder_can_release.wait(timeout=5)

    def waiter_same_tid():
        with q.acquire("A", on_state_change=waiter_states.append):
            pass
        waiter_done.set()

    ta = threading.Thread(target=hold_a)
    ta.start()
    time.sleep(0.05)
    tb = threading.Thread(target=waiter_same_tid)
    tb.start()
    time.sleep(0.05)

    # Second caller with same tid is on a different OS thread, so it
    # must queue — NOT take the reentrant fast path.
    assert waiter_states == ["queued"]
    assert q.waiter_count() == 1

    holder_can_release.set()
    ta.join(timeout=5)
    tb.join(timeout=5)
    assert waiter_done.is_set()
    assert waiter_states == ["queued", "running"]


def test_active_handle_visible_inside_acquire():
    q = ThreadAffinityQueue()
    assert active_handle() is None
    with q.acquire("A") as handle:
        assert active_handle() is handle
    assert active_handle() is None


def test_active_handle_not_set_in_unrelated_thread():
    q = ThreadAffinityQueue()
    other_thread_handle: list = []

    def peek():
        other_thread_handle.append(active_handle())

    with q.acquire("A"):
        t = threading.Thread(target=peek)
        t.start()
        t.join(timeout=2)
    # Spawned thread did not inherit the contextvar via threading.
    assert other_thread_handle == [None]


def test_env_float_parses_valid_value(monkeypatch):
    # Test the parser directly, not via importlib.reload — reloading the
    # module recreates the contextvar singleton and breaks any other
    # test that imported it.
    from assist.thread_queue import _env_float

    monkeypatch.setenv("ASSIST_TESTING_FLOAT", "1.5")
    assert _env_float("ASSIST_TESTING_FLOAT", 99.0) == 1.5


def test_env_float_falls_back_on_unset(monkeypatch):
    from assist.thread_queue import _env_float

    monkeypatch.delenv("ASSIST_TESTING_FLOAT", raising=False)
    assert _env_float("ASSIST_TESTING_FLOAT", 99.0) == 99.0


def test_env_float_falls_back_on_invalid(monkeypatch):
    from assist.thread_queue import _env_float

    monkeypatch.setenv("ASSIST_TESTING_FLOAT", "notanumber")
    assert _env_float("ASSIST_TESTING_FLOAT", 99.0) == 99.0


def test_defaults_match_module_constants():
    # Sanity check: the values applied to a freshly-constructed queue
    # match the module-level defaults documented in the docstring.
    q = ThreadAffinityQueue()
    assert q._default_hold_timeout == DEFAULT_HOLD_TIMEOUT_S
    assert q._default_wait_timeout == DEFAULT_WAIT_TIMEOUT_S
