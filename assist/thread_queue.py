"""Per-thread LLM-affinity queue.

Serializes access at the ``Thread.message()`` boundary so concurrent
threads don't thrash llama.cpp's ``--parallel 1`` KV cache.  When two
agents alternate turns through the slot, every turn pays a full
prefill from scratch — prefill cost grows from O(T) to O(T²) per
agent.  Holding the queue for one full ``Thread.message()`` keeps the
slot's cached prefix matched, so prefill stays a per-turn delta.

Affinity, not fairness:

- Waiters wait until the slot is vacated — either by the holder's
  clean release at ``__exit__`` or by the watchdog force-releasing it
  at ``hold_timeout_s``.  After a force-release the original holder
  thread may still be unwinding its ``with`` block while a waiter is
  already running; the identity guard in :meth:`_release_if_holder`
  prevents the unwinding holder's late cleanup from clobbering the
  new holder.
- Waiters are FIFO among themselves.
- Same ``thread_id`` re-acquiring is a no-op (re-entrant by id).

Failure-fast bounds:

- ``hold_timeout_s`` — a runaway holder is flagged ``expired`` so the
  cooperative cancel point in :class:`ThreadQueueMiddleware` raises
  :class:`ThreadHoldExpired` between LLM calls, AND the slot is vacated
  immediately so the next waiter can claim it without waiting on the
  runaway's ``finally``.  Honors the project rule that threads die on
  infrastructure failure rather than heal-and-retry.  The cap bounds the
  *detection latency* of the cooperative cancel, not the exact wall-clock
  release time — a ``wrap_model_call`` retry inside
  :class:`EmptyResponseRecoveryMiddleware` or
  :class:`BadRequestRetryMiddleware` can run one or two more LLM calls
  past expiration before ``after_model`` next fires.  Forcible slot
  release happens at the timeout regardless, logged at WARNING.
- ``wait_timeout_s`` — a waiter that can't acquire raises
  :class:`QueueWaitTimeout` and the thread errors.

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


class QueueWaitTimeout(Exception):
    """A waiter exceeded ``wait_timeout_s`` before reaching the head."""


class ThreadHoldExpired(Exception):
    """The holder exceeded ``hold_timeout_s``; cancellation is in flight."""


class _Handle:
    __slots__ = ("thread_id", "expired", "acquired_at")

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.expired = False
        self.acquired_at = time.time()


_active_handle: contextvars.ContextVar = contextvars.ContextVar(
    "thread_queue_active_handle", default=None
)


class ThreadAffinityQueue:
    def __init__(
        self,
        hold_timeout_s: float = DEFAULT_HOLD_TIMEOUT_S,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    ) -> None:
        self._cond = threading.Condition()
        self._holder: _Handle | None = None
        self._waiters: deque[str] = deque()
        self._default_hold_timeout = hold_timeout_s
        self._default_wait_timeout = wait_timeout_s

    @contextmanager
    def acquire(
        self,
        thread_id: str,
        on_state_change: Callable[[str], None] | None = None,
        wait_timeout_s: float | None = None,
        hold_timeout_s: float | None = None,
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

            handle = _Handle(thread_id)
            self._holder = handle
            cb("running")

            watchdog = threading.Timer(
                hold_timeout, self._on_hold_timeout, args=(handle,)
            )
            watchdog.daemon = True

        # Start the watchdog outside the lock so a near-zero timeout
        # (used in tests) doesn't fire while we're still in __enter__.
        watchdog.start()
        token = _active_handle.set(handle)
        try:
            yield handle
        finally:
            # `_active_handle.reset(token)` requires same-Context exit
            # (the docstring above defines the contract).  Callers that
            # iterate across thread boundaries must bind via
            # ``contextvars.Context.run`` — see `Thread.stream_message`'s
            # `_ContextBoundIterator` for the in-tree pattern.  If the
            # contract is violated, the watchdog bounds the resulting
            # `_holder` leak to ``hold_timeout_s``.
            _active_handle.reset(token)
            watchdog.cancel()
            self._release_if_holder(handle)

    def _release_if_holder(self, handle: _Handle) -> bool:
        """Vacate the slot iff ``handle`` is still the current holder.

        Idempotent across the two release paths (``acquire``'s ``finally``
        and the watchdog's :meth:`_on_hold_timeout`); whichever wins the
        lock first releases, the other sees ``_holder is not handle`` and
        returns False without clobbering a newly-promoted holder.
        """
        with self._cond:
            if self._holder is handle:
                self._holder = None
                self._cond.notify_all()
                return True
            return False

    def _on_hold_timeout(self, handle: _Handle) -> None:
        """Watchdog callback: flag holder ``expired`` and force-release the slot.

        ``expired = True`` is read by :class:`ThreadQueueMiddleware`'s
        cooperative cancel (via :func:`active_handle` — per-call-stack).
        The force-release through :meth:`_release_if_holder` is
        defense-in-depth: it bounds the leak window for any cleanup
        failure that leaves ``_holder`` set, regardless of cause.
        Logs WARNING when the release actually fired — should be cold
        in steady state.
        """
        handle.expired = True
        if self._release_if_holder(handle):
            logger.warning(
                "force-released wedged holder %s after %.1fs hold",
                handle.thread_id,
                time.time() - handle.acquired_at,
            )

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
