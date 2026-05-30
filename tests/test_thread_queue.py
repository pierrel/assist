"""Tests for ThreadAffinityQueue (assist/thread_queue.py)."""
import contextvars
import threading
import time

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


def test_hold_timeout_marks_expired_and_force_releases():
    # After the hold_timeout_s watchdog fires, both the cooperative-cancel
    # flag (`expired`) AND the queue-slot release happen.  The force release
    # is what bounds the leak window when the holder's `finally` is skipped
    # (the 2026-05-28 prod incident — see thread_queue.py docstring).
    # Poll for the watchdog's effects rather than fixed-sleep: avoids
    # flakiness on slow/loaded CI where the Timer thread may take longer
    # than a tight `time.sleep` margin to execute its callback.
    q = ThreadAffinityQueue()
    with q.acquire("A", hold_timeout_s=0.1) as handle:
        assert handle.expired is False
        assert q.current_handle() is handle
        deadline = time.time() + 2.0
        while q.current_handle() is not None and time.time() < deadline:
            time.sleep(0.01)
        assert handle.expired is True
        # The slot is vacated mid-`with`: a waiter could acquire RIGHT NOW.
        assert q.current_handle() is None
    # Holder's `finally` ran after the watchdog already released; the
    # `is handle` guard in `_release_if_holder` made it a no-op (no
    # exception, no double-notify).
    assert q.current_handle() is None


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


# --- Leak-fix regression tests ----------------------------------------
# These tests pin the contract that the queue's `_holder` slot is
# released even when `acquire`'s `finally` is interrupted partway, AND
# that the watchdog timer forcibly releases the slot when `finally` is
# bypassed entirely.  Background: docs/2026-05-29-queue-holder-leak-fix.org.


def test_cross_context_exit_leaks_holder_until_watchdog_recovers():
    # Pins the failure-mode-and-recovery for the 2026-05-28 prod incident:
    # the with-block is entered in one `contextvars.Context` and exited in
    # another, so `_active_handle.reset(token)` raises `ValueError` at the
    # top of `finally`, aborting the rest of cleanup.  The queue's policy
    # is NOT to swallow the ValueError (that hides the bug in the caller);
    # instead, the watchdog bounds the resulting `_holder` leak to
    # `hold_timeout_s`.  The known caller (`Thread.stream_message`) binds
    # its generator to a captured Context via `_ContextBoundIterator`, so
    # this path is no longer reachable from in-tree code; the test still
    # pins the queue-level recovery contract for any future caller that
    # might violate the same-Context-exit contract.
    q = ThreadAffinityQueue()
    cm = q.acquire("A", hold_timeout_s=0.1)
    ctx = contextvars.copy_context()
    # `__enter__` binds the contextvar token to `ctx`; the subsequent
    # `__exit__` runs in the test's outer context, so `reset(token)`
    # sees the context mismatch and raises ValueError (real CPython
    # behaviour — no mocking needed).
    ctx.run(cm.__enter__)
    assert q.current_handle() is not None

    # The ValueError propagates out of `finally` — the holder leaks
    # momentarily.  This is the documented failure mode.
    with pytest.raises(ValueError):
        cm.__exit__(None, None, None)
    assert q.current_handle() is not None, "holder leaked, as expected"

    # The watchdog catches the leak: `_holder` is force-released after
    # `hold_timeout_s` so the queue stays serving other threads.
    deadline = time.time() + 2.0
    while q.current_handle() is not None and time.time() < deadline:
        time.sleep(0.01)
    assert q.current_handle() is None, (
        "watchdog should have force-released the leaked slot"
    )

    # A fresh acquire goes straight to "running" with no "queued" state.
    states: list[str] = []
    with q.acquire("B", on_state_change=states.append):
        pass
    assert states == ["running"]


def test_watchdog_force_releases_when_holder_is_wedged():
    # The real leak-bound: even if the holder's `with` never exits
    # (wedged in an infinite loop, never reaches `finally`), the
    # watchdog timer must release the slot at `hold_timeout_s` so
    # waiters can proceed.
    q = ThreadAffinityQueue()
    holder_can_release = threading.Event()
    holder_done = threading.Event()
    holder_acquired = threading.Event()
    waiter_states: list[str] = []
    waiter_done = threading.Event()

    def wedged_holder():
        with q.acquire(
            "A", hold_timeout_s=0.1,
            on_state_change=lambda s: holder_acquired.set() if s == "running" else None,
        ):
            holder_can_release.wait(timeout=5.0)
        holder_done.set()

    def waiter():
        # Wait_timeout generous; the watchdog should free the slot
        # at ~0.1s, well before wait_timeout would matter.
        with q.acquire("B", wait_timeout_s=2.0,
                       on_state_change=waiter_states.append):
            pass
        waiter_done.set()

    th = threading.Thread(target=wedged_holder)
    th.start()
    # Deterministic wait for A to become holder (avoids time.sleep flakiness
    # on slow/loaded CI — the on_state_change callback above flips the Event
    # the moment A's acquire transitions to "running").
    assert holder_acquired.wait(timeout=2.0), "holder A never acquired"
    assert q.current_handle() is not None
    tw = threading.Thread(target=waiter)
    tw.start()

    # B is queued behind A.  Wait long enough for the watchdog to fire.
    waiter_done.wait(timeout=2.0)
    assert waiter_done.is_set(), "waiter never acquired — watchdog didn't release"
    # B observed both states: queued then running.
    assert waiter_states == ["queued", "running"]

    # Let the wedged holder unblock so the test thread can finish.
    holder_can_release.set()
    holder_done.wait(timeout=2.0)
    th.join(timeout=2.0)
    tw.join(timeout=2.0)

    # Assert threads actually terminated — a flaky join-on-timeout would
    # leave them alive and silently corrupt state for the next test.
    assert not th.is_alive(), "wedged_holder thread did not finish"
    assert not tw.is_alive(), "waiter thread did not finish"
    assert q.current_handle() is None
    assert q.waiter_count() == 0


def test_watchdog_does_not_clobber_newly_promoted_holder():
    # Watchdog fires on A, releasing the slot.  B is promoted and
    # becomes the new holder.  Then A's `with` finally exits — its
    # cleanup must NOT clobber B's slot.  The `is handle` guard in
    # `_release_if_holder` is what protects this; this test pins it.
    q = ThreadAffinityQueue()
    a_can_release = threading.Event()
    a_acquired = threading.Event()
    b_acquired = threading.Event()
    b_can_release = threading.Event()

    def hold_a():
        with q.acquire(
            "A", hold_timeout_s=0.1,
            on_state_change=lambda s: a_acquired.set() if s == "running" else None,
        ):
            a_can_release.wait(timeout=5.0)

    def hold_b():
        with q.acquire("B", wait_timeout_s=2.0):
            # Mark that B is now the holder (post-watchdog promotion).
            b_acquired.set()
            b_can_release.wait(timeout=5.0)

    ta = threading.Thread(target=hold_a)
    ta.start()
    # Deterministic wait for A to enter `with` and become holder.
    assert a_acquired.wait(timeout=2.0), "holder A never acquired"
    tb = threading.Thread(target=hold_b)
    tb.start()

    # Wait for B to be promoted (watchdog fires at ~0.1s).
    assert b_acquired.wait(timeout=2.0), "B never promoted — watchdog didn't release A"
    b_handle = q.current_handle()
    assert b_handle is not None
    assert b_handle.thread_id == "B"

    # Now let A exit its `with` (its cleanup must be a no-op for `_holder`).
    a_can_release.set()
    ta.join(timeout=2.0)
    assert not ta.is_alive(), "hold_a thread did not finish"

    # B is still the holder — A's `finally` didn't clobber it.
    assert q.current_handle() is b_handle

    b_can_release.set()
    tb.join(timeout=2.0)
    assert not tb.is_alive(), "hold_b thread did not finish"
    assert q.current_handle() is None
