"""Per-thread LLM-affinity queue.

Serializes access at the ``Thread.message()`` boundary so concurrent
threads don't thrash llama.cpp's ``--parallel 1`` KV cache.  When two
agents alternate turns through the slot, every turn pays a full
prefill from scratch — prefill cost grows from O(T) to O(T²) per
agent.  Holding the queue for one full ``Thread.message()`` keeps the
slot's cached prefix matched, so prefill stays a per-turn delta.

Affinity, not fairness:

- Holder runs to completion before any waiter runs.
- Waiters are FIFO among themselves.
- Same ``thread_id`` re-acquiring is a no-op (re-entrant by id).

Failure-fast bounds:

- ``hold_timeout_s`` — a runaway holder is flagged ``expired`` so the
  cooperative cancel point in :class:`ThreadQueueMiddleware` raises
  :class:`ThreadHoldExpired` between LLM calls.  Honors the project
  rule that threads die on infrastructure failure rather than
  heal-and-retry.  The cap bounds the *detection latency*, not the
  exact wall-clock release time — a ``wrap_model_call`` retry inside
  :class:`EmptyResponseRecoveryMiddleware` or
  :class:`BadRequestRetryMiddleware` can run one or two more LLM calls
  past expiration before ``after_model`` next fires.
- ``wait_timeout_s`` — a waiter that can't acquire raises
  :class:`QueueWaitTimeout` and the thread errors.

Single-process scope:

The queue is a module-level singleton.  It coordinates background
tasks within one ``manage.web`` process.  Eval CLIs that import
``assist.thread`` in a separate process get their own (empty) queue
— intentional: they're not sharing the prod LLM slot anyway.
"""

import contextvars
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEFAULT_HOLD_TIMEOUT_S = _env_float("ASSIST_THREAD_HOLD_TIMEOUT_S", 7200.0)
DEFAULT_WAIT_TIMEOUT_S = _env_float("ASSIST_THREAD_QUEUE_WAIT_S", 900.0)


class QueueWaitTimeout(Exception):
    """A waiter exceeded ``wait_timeout_s`` before reaching the head."""


class ThreadHoldExpired(Exception):
    """The holder exceeded ``hold_timeout_s``; cancellation is in flight."""


class _Handle:
    __slots__ = ("thread_id", "expired", "acquired_at", "_watchdog")

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        self.expired = False
        self.acquired_at = time.time()
        self._watchdog: threading.Timer | None = None


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

            watchdog = threading.Timer(hold_timeout, _mark_expired, args=(handle,))
            watchdog.daemon = True
            handle._watchdog = watchdog

        # Start the watchdog outside the lock so a near-zero timeout
        # (used in tests) doesn't fire while we're still in __enter__.
        watchdog.start()
        token = _active_handle.set(handle)
        try:
            yield handle
        finally:
            _active_handle.reset(token)
            try:
                watchdog.cancel()
            except Exception:
                pass
            with self._cond:
                if self._holder is handle:
                    self._holder = None
                    self._cond.notify_all()

    def current_handle(self) -> _Handle | None:
        with self._cond:
            return self._holder

    def waiter_count(self) -> int:
        with self._cond:
            return len(self._waiters)


def _mark_expired(handle: _Handle) -> None:
    handle.expired = True


def active_handle() -> _Handle | None:
    """Return the holder handle visible to the current execution context.

    Set by :meth:`ThreadAffinityQueue.acquire` for the duration of the
    ``with`` block, scoped via :mod:`contextvars` so sub-agent calls in
    the same call stack see it but unrelated background work does not.
    """
    return _active_handle.get()


THREAD_QUEUE = ThreadAffinityQueue()
