"""Fair-scheduling queue behavior (Phase 2 of docs/2026-07-08-fair-scheduling.org):
the 10-min quantum pauses a holder ONLY under contention, the 2h cap measures
CUMULATIVE ACTIVE hold (excludes paused/queued time, can't be dodged by pausing),
and the loop-side path stays lock-free.

Un-mocked ThreadAffinityQueue + real Condition/Timers (CLAUDE.md: don't mock the
risk).  Small quanta so the tick fires in test time.
"""
import threading
import time

from assist.thread_queue import ThreadAffinityQueue, active_handle


def _spin(pred, timeout=2.0, step=0.005):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def test_no_pause_when_uncontended():
    # A holder past its quantum with NO waiter is never paused — the tick re-arms.
    q = ThreadAffinityQueue(quantum_s=0.03, hold_timeout_s=100)
    with q.acquire("A") as h:
        time.sleep(0.2)  # many quanta, nobody waiting
        assert h.pause_requested is False


def test_pause_only_under_contention():
    # With a waiter present, the holder is asked to pause at its quantum, and the
    # slot hands to the waiter once the holder yields (unwinds acquire).
    q = ThreadAffinityQueue(quantum_s=0.03, hold_timeout_s=100, wait_timeout_s=5)
    holder_started = threading.Event()
    waiter_ran = threading.Event()

    def holder():
        with q.acquire("A") as h:
            holder_started.set()
            # Simulate the agent yielding at after_model when pause_requested flips.
            _spin(lambda: h.pause_requested, timeout=3.0)
            # (leaving the `with` = the pause unwind that frees the slot)

    def waiter():
        # arrives AFTER the holder, so it's the contention that triggers the pause
        with q.acquire("B", wait_timeout_s=5):
            waiter_ran.set()

    th = threading.Thread(target=holder); th.start()
    assert holder_started.wait(2.0)
    time.sleep(0.05)  # let a quantum pass with no waiter first (proves it's contention)
    tb = threading.Thread(target=waiter); tb.start()
    th.join(timeout=5); tb.join(timeout=5)
    assert waiter_ran.is_set(), "waiter never ran after the holder paused"


def test_cap_measures_cumulative_active_and_ignores_paused_time():
    # A turn burns s1 active, "pauses" (we simulate by exiting + carrying the hold),
    # waits (paused wall-time that must NOT count), then resumes seeded with s1's
    # active ms.  The 2h cap fires only when s1 + s2 active >= cap, never on the
    # paused wall-time in between.
    cap = 0.30
    q = ThreadAffinityQueue(quantum_s=0.05, hold_timeout_s=cap, wait_timeout_s=5)

    # slice 1: ~0.15s active, exit cleanly (not a cap kill)
    with q.acquire("A") as h1:
        time.sleep(0.15)
        assert h1.expired is False, "cap fired too early on slice 1"
    carried = q.pop_hold("A")
    assert 0.12 < carried / 1000.0 < 0.25, f"carried active ms off: {carried}"

    # paused wall-time that must be excluded from the cap
    time.sleep(0.4)

    # slice 2: resume seeded with slice-1's active hold; ~0.2s more active pushes
    # cumulative (0.15 + 0.2) past the 0.30 cap -> the tick kills it.
    with q.acquire("A", accumulated_active_ms=carried) as h2:
        assert _spin(lambda: h2.expired, timeout=2.0), "cap never fired on cumulative active"


def test_peek_holder_stays_lock_free_while_slot_held_and_contended():
    # The loop-side read must never block on the Condition, even while the queue is
    # held AND a waiter is parked in cond.wait (the resume-scheduler / queued-turn
    # shape).  peek_holder is an atomic ref read; assert it returns promptly.
    q = ThreadAffinityQueue(quantum_s=100, hold_timeout_s=100, wait_timeout_s=5)
    held = threading.Event()
    release = threading.Event()

    def holder():
        with q.acquire("A"):
            held.set()
            release.wait(timeout=5)

    def waiter():
        try:
            with q.acquire("B", wait_timeout_s=5):
                pass
        except Exception:
            pass

    th = threading.Thread(target=holder); th.start()
    assert held.wait(2.0)
    tb = threading.Thread(target=waiter); tb.start()
    time.sleep(0.05)  # B now parked in cond.wait
    # peek_holder on "the loop": must be fast (no lock).
    t0 = time.perf_counter()
    for _ in range(1000):
        q.peek_holder()
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.1, f"peek_holder blocked ({elapsed:.3f}s for 1000 reads)"
    assert q.peek_holder() == "A"
    release.set(); th.join(timeout=5); tb.join(timeout=5)


def test_pop_hold_drains_so_a_fresh_turn_is_not_charged_prior_active():
    # A terminal (non-resume) turn's hold is drained; the NEXT fresh turn seeds 0.
    q = ThreadAffinityQueue(quantum_s=100, hold_timeout_s=100)
    with q.acquire("A"):
        time.sleep(0.05)
    assert q.pop_hold("A") > 0          # the finally persisted this slice
    assert q.pop_hold("A") == 0.0       # ...and it's drained (no double-charge)


def test_tick_fires_while_a_reentrant_acquire_holds_the_turn():
    # THE regression guard for the 2026-07-08 preemption bug: _process_message holds
    # the slot, then Thread._run re-acquires it (reentrant no-op) for the whole turn.
    # If that reentrant acquire holds self._cond across its yield, _on_tick (which needs
    # the lock) is blocked for the ENTIRE turn and the quantum pause never fires. The
    # unit tests missed it because they acquire directly, never through the reentrant
    # path. This drives the reentrant path un-mocked and asserts the tick still runs.
    q = ThreadAffinityQueue(quantum_s=0.05, hold_timeout_s=100, wait_timeout_s=5)
    got_pause = threading.Event()

    def holder():
        with q.acquire("A"):                 # outer hold (as _process_message does)
            with q.acquire("A") as h:        # reentrant (as Thread._run does)
                # a waiter must be present for the tick to request a pause
                threading.Thread(target=_park_waiter, args=(q,), daemon=True).start()
                if _spin(lambda: h.pause_requested, timeout=3.0):
                    got_pause.set()

    threading.Thread(target=holder, daemon=True).start()
    assert got_pause.wait(4.0), ("tick never set pause_requested during a reentrant "
                                 "acquire — it is blocked on self._cond (the bug)")


def _park_waiter(q):
    try:
        with q.acquire("B", wait_timeout_s=3):
            pass
    except Exception:
        pass
