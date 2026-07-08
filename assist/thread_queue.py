"""Per-thread LLM-affinity queue.

Serializes access at the ``Thread.message()`` boundary so concurrent
threads don't thrash llama.cpp's ``--parallel 1`` KV cache.  When two
agents alternate turns through the slot, every turn pays a full
prefill from scratch — prefill cost grows from O(T) to O(T²) per
agent.  Holding the queue for one full ``Thread.message()`` keeps the
slot's cached prefix matched, so prefill stays a per-turn delta.

Affinity, then fairness under contention:

- Affinity: holding the slot for one full turn keeps the KV prefix matched.
- Fairness: a holder that has held the slot for ``quantum_s`` AND has a
  waiter behind it is asked to PAUSE at its next superstep (cooperative,
  via ``pause_requested`` -> :class:`ThreadPauseRequested`), so a quick turn
  isn't blocked for the full length of a long research turn.  The paused
  turn is resumed later (round-robin, back of the queue) from its durable
  checkpoint — no lost work.  UNCONTENDED, the holder is never paused (the
  tick just re-arms), so a long turn with nobody waiting runs uninterrupted.
- Waiters are FIFO among themselves.  A resumed turn re-acquires like any
  other waiter, so it lands at the back — natural round-robin, no priority.
- Same ``thread_id`` re-acquiring is a no-op (re-entrant by id).

Failure-fast bounds (the tick timer, :meth:`_on_tick`, enforces both):

- ``hold_timeout_s`` — a cap on CUMULATIVE ACTIVE hold (summed across a turn's
  slices; paused/queued wall-time is excluded, and it can't be dodged by
  pausing).  On breach the holder is flagged ``expired`` so the cooperative
  cancel point in :class:`ThreadQueueMiddleware` raises :class:`ThreadHoldExpired`
  between LLM calls (terminal — a runaway backstop, not resumable), AND the slot
  is vacated immediately.  Honors the project rule that threads die on
  infrastructure failure rather than heal-and-retry.  The cap bounds the
  *detection latency* of the cooperative cancel, not the exact wall-clock
  release time — a ``wrap_model_call`` retry inside
  :class:`EmptyResponseRecoveryMiddleware` or
  :class:`BadRequestRetryMiddleware` can run one or two more LLM calls
  past the boundary before ``after_model`` next fires.  Forcible slot
  release happens regardless, logged at WARNING.
- ``wait_timeout_s`` — a waiter that can't acquire raises
  :class:`QueueWaitTimeout` and the thread errors.

After a force-release (cap breach) the original holder thread may still be
unwinding its ``with`` block while a waiter is already running; the identity
guard in :meth:`_release_if_holder` prevents the unwinding holder's late
cleanup from clobbering the new holder.

Single-process scope:

The queue is a module-level singleton.  It coordinates background
tasks within one ``manage.web`` process.  Eval CLIs that import
``assist.thread`` in a separate process get their own (empty) queue
— intentional: they're not sharing the prod LLM slot anyway.
"""

import contextvars
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_HOLD_TIMEOUT_S = _env_float("ASSIST_THREAD_HOLD_TIMEOUT_S", 7200.0)
DEFAULT_WAIT_TIMEOUT_S = _env_float("ASSIST_THREAD_QUEUE_WAIT_S", 14400.0)
# Fair-scheduling quantum: a holder that has held the slot for this long AND has a
# waiter behind it yields at its next superstep (cooperative pause), so a quick turn
# isn't blocked for the full length of a long research turn.  Uncontended, it just
# re-arms — no pause when nobody is waiting.  The 2h HOLD_TIMEOUT is a separate,
# terminal cap on *cumulative active* hold (see ``_on_tick``).
DEFAULT_QUANTUM_S = _env_float("ASSIST_THREAD_QUANTUM_S", 600.0)


class QueueWaitTimeout(Exception):
    """A waiter exceeded ``wait_timeout_s`` before reaching the head."""


class ThreadHoldExpired(Exception):
    """The holder exceeded ``hold_timeout_s`` of cumulative active hold; the turn is
    terminated (a runaway backstop — NOT resumable)."""


class ThreadPauseRequested(Exception):
    """The holder's quantum expired with a waiter present.  NON-terminal: the run
    path catches it OUTSIDE the acquire block, does not finalize the turn, marks it
    paused, and hands it to the resume scheduler.  Distinct from ThreadHoldExpired
    (terminal) so the two never share a catch clause."""


class _Handle:
    __slots__ = (
        "thread_id", "expired", "acquired_at", "pause_requested",
        "accumulated_active_ms", "quantum_s", "hold_timeout_s", "timer",
    )

    def __init__(
        self, thread_id: str, quantum_s: float, hold_timeout_s: float,
        accumulated_active_ms: float = 0.0,
    ) -> None:
        self.thread_id = thread_id
        self.expired = False
        self.acquired_at = time.time()            # start of THIS active slice
        self.pause_requested = False
        # Active hold from PRIOR slices (0 for a fresh turn; a resumed turn is
        # seeded with what it already burned, so the 2h cap can't be dodged by
        # pausing).  Never counts paused/queued wall-time.
        self.accumulated_active_ms = accumulated_active_ms
        self.quantum_s = quantum_s
        self.hold_timeout_s = hold_timeout_s
        self.timer: threading.Timer | None = None


_active_handle: contextvars.ContextVar = contextvars.ContextVar(
    "thread_queue_active_handle", default=None
)


class ThreadAffinityQueue:
    def __init__(
        self,
        hold_timeout_s: float = DEFAULT_HOLD_TIMEOUT_S,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
        quantum_s: float = DEFAULT_QUANTUM_S,
    ) -> None:
        self._cond = threading.Condition()
        self._holder: _Handle | None = None
        self._waiters: deque[str] = deque()
        self._default_hold_timeout = hold_timeout_s
        self._default_wait_timeout = wait_timeout_s
        self._default_quantum = quantum_s
        # Cumulative ACTIVE hold per thread_id, persisted across a pause so a
        # resumed turn carries what it already burned toward the 2h cap.  Written
        # in ``acquire``'s finally; drained by ``pop_hold`` on every exit (a pause
        # carries the value to the resume; a terminal exit discards it).  Touched
        # only under ``self._cond``, only off the event loop.
        self._active_hold_ms: dict[str, float] = {}

    @contextmanager
    def acquire(
        self,
        thread_id: str,
        on_state_change: Callable[[str], None] | None = None,
        wait_timeout_s: float | None = None,
        hold_timeout_s: float | None = None,
        quantum_s: float | None = None,
        accumulated_active_ms: float = 0.0,
    ) -> Iterator[_Handle]:
        """Acquire this thread's single-flight slot for the ``with`` block.

        Context-affine: the context manager sets a ``ContextVar`` token on entry
        and resets it on exit, so it must be entered, resumed across ``yield``,
        and exited in the SAME execution context.  If the wrapped generator is
        advanced/closed in a different context — e.g. driven across threads via
        ``run_in_executor`` on the default pool, since each thread has its own
        context — the token reset raises ``ValueError: <token> was created in a
        different Context``.  Drive it from a single thread/context.  (The
        ``threading.Condition`` is always used under its own lock, so it is safe
        across threads; the contextvar token is the hard constraint.)
        """
        cb = on_state_change or (lambda _: None)
        wait_timeout = (
            self._default_wait_timeout if wait_timeout_s is None else wait_timeout_s
        )
        hold_timeout = (
            self._default_hold_timeout if hold_timeout_s is None else hold_timeout_s
        )
        quantum = self._default_quantum if quantum_s is None else quantum_s

        with self._cond:
            # Reentrant only if this caller is on the holder's own call
            # stack — i.e. the active contextvar is the holder's handle.
            # A second OS thread that happens to share `thread_id` (e.g.
            # a user double-clicking Send: two background tasks for one
            # tid) does NOT free-ride; the contextvar is unset across
            # threads, so it falls through to the wait path.
            if (
                self._holder is not None
                and self._holder.thread_id == thread_id
                and _active_handle.get() is self._holder
            ):
                # No new state callback, no new watchdog.
                yield self._holder
                return

            if self._holder is not None:
                cb("queued")
                self._waiters.append(thread_id)
                deadline = time.time() + wait_timeout
                try:
                    while self._holder is not None or (
                        self._waiters and self._waiters[0] != thread_id
                    ):
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise QueueWaitTimeout(
                                f"thread {thread_id} waited {wait_timeout}s for queue"
                            )
                        self._cond.wait(timeout=remaining)
                    self._waiters.popleft()
                except BaseException:
                    try:
                        self._waiters.remove(thread_id)
                    except ValueError:
                        pass
                    self._cond.notify_all()
                    raise

            handle = _Handle(
                thread_id, quantum, hold_timeout,
                accumulated_active_ms=accumulated_active_ms,
            )
            self._holder = handle
            cb("running")

            tick = threading.Timer(quantum, self._on_tick, args=(handle,))
            tick.daemon = True
            handle.timer = tick

        # Start the tick outside the lock so a near-zero quantum (used in tests)
        # doesn't fire while we're still in __enter__.
        tick.start()
        token = _active_handle.set(handle)
        try:
            yield handle
        finally:
            # `_active_handle.reset(token)` requires same-Context exit
            # (the docstring above defines the contract).  Callers that
            # iterate across thread boundaries must bind via
            # ``contextvars.Context.run`` — see `Thread.stream_message`'s
            # `_ContextBoundIterator` for the in-tree pattern.  If the
            # contract is violated, the tick bounds the resulting
            # `_holder` leak to ``hold_timeout_s``.
            _active_handle.reset(token)
            with self._cond:
                # Persist this slice's active time so a resume carries it toward
                # the 2h cap; cancel the pending tick; release the slot — all under
                # one lock so an in-flight tick can't interleave (it would see
                # `_holder is not handle` and no-op).  ``pop_hold`` drains it: a
                # pause reads it for the resume seed, a terminal exit discards it.
                slice_ms = (time.time() - handle.acquired_at) * 1000.0
                self._active_hold_ms[thread_id] = handle.accumulated_active_ms + slice_ms
                if handle.timer is not None:
                    handle.timer.cancel()
                self._release_if_holder(handle)

    def _release_if_holder(self, handle: _Handle) -> bool:
        """Vacate the slot iff ``handle`` is still the current holder.

        Idempotent across the two release paths (``acquire``'s ``finally``
        and the tick's :meth:`_on_tick`); whichever wins the
        lock first releases, the other sees ``_holder is not handle`` and
        returns False without clobbering a newly-promoted holder.
        """
        with self._cond:
            if self._holder is handle:
                self._holder = None
                self._cond.notify_all()
                return True
            return False

    def _on_tick(self, handle: _Handle) -> None:
        """Re-arming timer callback: enforce the 2h cumulative-active cap and the
        fair-scheduling quantum.  Runs on a :class:`threading.Timer` daemon thread,
        entirely under ``self._cond`` (off the event loop, so taking the lock is
        safe — only :meth:`peek_holder` must stay lock-free).

        Order (the cap wins over the quantum, so both are never set at once):
          - cumulative active hold >= ``hold_timeout_s`` -> ``expired`` (terminal
            kill) + force-release the slot.  ``expired`` is read by
            :class:`ThreadQueueMiddleware`, which raises :class:`ThreadHoldExpired`.
          - else this slice >= ``quantum_s`` AND a waiter is present -> request a
            pause: ``pause_requested`` (read by the middleware -> non-terminal
            :class:`ThreadPauseRequested`).  Uncontended, we do NOT pause.
          - else re-arm for another quantum.
        """
        with self._cond:
            if self._holder is not handle:
                return  # already released; a stale tick.
            slice_s = time.time() - handle.acquired_at
            cumulative_s = handle.accumulated_active_ms / 1000.0 + slice_s
            if cumulative_s >= handle.hold_timeout_s:
                handle.expired = True
                self._release_if_holder(handle)
                logger.warning(
                    "thread %s hit the %.0fs cumulative-active cap — terminating",
                    handle.thread_id, handle.hold_timeout_s,
                )
                return  # terminal — do NOT re-arm.
            if slice_s >= handle.quantum_s and len(self._waiters) > 0:
                handle.pause_requested = True
                # Fall through and KEEP TICKING: the holder yields at its next
                # after_model (which cancels the timer via acquire's finally), but
                # if it doesn't yield promptly we must still enforce the 2h cap —
                # so re-arm rather than stop here.
            # Keep holding; check the cap (and re-assert the pause) next quantum.
            tick = threading.Timer(handle.quantum_s, self._on_tick, args=(handle,))
            tick.daemon = True
            handle.timer = tick
            tick.start()

    def pop_hold(self, thread_id: str) -> float:
        """Remove and return the thread's persisted cumulative-active-hold (ms).

        Called on EVERY exit of a turn's ``acquire`` block by the run path: a pause
        passes the value to the resume's ``accumulated_active_ms`` seed (so the 2h
        cap can't be dodged by pausing); a terminal exit discards it (a fresh turn
        starts at 0).  Draining on every exit keeps ``_active_hold_ms`` from growing.
        """
        with self._cond:
            return self._active_hold_ms.pop(thread_id, 0.0)

    def current_handle(self) -> _Handle | None:
        with self._cond:
            return self._holder

    def peek_holder(self) -> str | None:
        """Lock-free, best-effort read of the current holder's thread id.

        Unlike :meth:`current_handle`, this does **not** take ``self._cond``.
        It exists so callers on the asyncio event-loop thread can check who
        holds the slot WITHOUT risking a blocking lock acquire there — a
        single contended/held queue lock on the event loop freezes the whole
        web server (observed 2026-06-10: a synchronous ``current_handle()``
        in the message-POST path wedged the event loop under a long research
        turn).  The reference read is atomic under the GIL, and ``_Handle``'s
        ``thread_id`` is immutable, so the worst case is a momentarily stale
        value — fine for its only use: picking an initial UI status label
        that the background task then refines.
        """
        holder = self._holder  # atomic ref read; deliberately no lock
        return holder.thread_id if holder is not None else None

    def waiter_count(self) -> int:
        with self._cond:
            return len(self._waiters)


def active_handle() -> _Handle | None:
    """Return the holder handle visible to the current execution context.

    Set by :meth:`ThreadAffinityQueue.acquire` for the duration of the
    ``with`` block, scoped via :mod:`contextvars` so sub-agent calls in
    the same call stack see it but unrelated background work does not.
    """
    return _active_handle.get()


THREAD_QUEUE = ThreadAffinityQueue()
