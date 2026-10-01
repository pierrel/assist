"""Process-local boundary between durable Run work and intentional shutdown."""

from contextlib import contextmanager
import threading


class DrainClosed(RuntimeError):
    """A new Run cannot be accepted by this stopping process."""


class RunGate:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._closed = False

    @contextmanager
    def active(self):
        with self._condition:
            accepted = not self._closed
            if accepted:
                self._active += 1
        try:
            yield accepted
        finally:
            if accepted:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()

    def close_when_idle(self) -> None:
        """Wait off-loop, then exclude new work in the same zero-count step."""
        with self._condition:
            while self._active:
                self._condition.wait()
            self._closed = True

    def reopen(self) -> None:
        """Reset only after a completed in-process lifespan (mainly tests)."""
        with self._condition:
            if self._active:
                raise RuntimeError("cannot reopen while Run work is active")
            self._closed = False


RUN_GATE = RunGate()
