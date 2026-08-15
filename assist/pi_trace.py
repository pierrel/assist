"""Host-owned, redacted Pi activity records for visible web turns."""
from __future__ import annotations

import json
import os
import stat
import threading
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


_FILENAME = "pi-trace.jsonl"
_VERSION = 1
_MAX_BYTES = 2 * 1024 * 1024
_MAX_EVENTS = 512
_MAX_RUN_ID_BYTES = 256
_KINDS = {"model", "tool"}
_NAMES = {"model request", "read", "write", "bash", "load skill", "map data"}
_OUTCOMES = {"started", "completed", "did not complete"}


class PiTraceError(ValueError):
    """A Pi activity trace is malformed, unsafe, or unavailable."""


@dataclass(frozen=True)
class PiTraceEvent:
    """One redacted lifecycle event for a host-authorized Pi operation."""

    run_id: str
    sequence: int
    operation: int
    kind: Literal["model", "tool"]
    name: Literal["model request", "read", "write", "bash", "load skill", "map data"]
    outcome: Literal["started", "completed", "did not complete"]


class PiTraceStore:
    """Append-only Pi trace resource, independent from visible conversation state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _path(thread_dir: str | Path) -> Path:
        return Path(thread_dir) / _FILENAME

    @staticmethod
    def _validate(value: object) -> PiTraceEvent:
        expected = {"version", "ts", "run_id", "sequence", "operation", "kind", "name", "outcome"}
        if not isinstance(value, dict) or set(value) != expected:
            raise PiTraceError("Pi activity event has an invalid shape")
        if (value["version"] != _VERSION or not isinstance(value["ts"], str)
                or not isinstance(value["run_id"], str) or not value["run_id"]
                or not isinstance(value["sequence"], int) or isinstance(value["sequence"], bool)
                or not isinstance(value["operation"], int) or isinstance(value["operation"], bool)
                or value["sequence"] < 1 or value["operation"] < 1
                or value["kind"] not in _KINDS or value["name"] not in _NAMES
                or value["outcome"] not in _OUTCOMES):
            raise PiTraceError("Pi activity event has invalid fields")
        try:
            if (len(value["ts"].encode("utf-8")) > 128
                    or len(value["run_id"].encode("utf-8")) > _MAX_RUN_ID_BYTES):
                raise PiTraceError("Pi activity event exceeds its bound")
        except UnicodeEncodeError as error:
            raise PiTraceError("Pi activity event is not valid UTF-8") from error
        return PiTraceEvent(value["run_id"], value["sequence"], value["operation"],
                            value["kind"], value["name"], value["outcome"])

    @classmethod
    def _events(cls, raw: bytes) -> list[PiTraceEvent]:
        """Validate the complete immutable history before exposing or extending it."""
        if raw and not raw.endswith(b"\n"):
            raise PiTraceError("Pi activity trace is malformed")
        events: list[PiTraceEvent] = []
        last_sequence: dict[str, int] = {}
        started: dict[tuple[str, int], PiTraceEvent] = {}
        settled: set[tuple[str, int]] = set()
        for line in raw.splitlines():
            try:
                event = cls._validate(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError, PiTraceError) as error:
                raise PiTraceError("Pi activity trace is malformed") from error
            if event.sequence != last_sequence.get(event.run_id, 0) + 1:
                raise PiTraceError("Pi activity trace has an invalid sequence")
            last_sequence[event.run_id] = event.sequence
            key = (event.run_id, event.operation)
            if event.outcome == "started":
                if key in started:
                    raise PiTraceError("Pi activity trace repeats an operation")
                started[key] = event
            elif key not in started or key in settled:
                raise PiTraceError("Pi activity trace has an invalid operation outcome")
            else:
                prior = started[key]
                if (event.kind, event.name) != (prior.kind, prior.name):
                    raise PiTraceError("Pi activity trace changes an operation")
                settled.add(key)
            events.append(event)
            if len(events) > _MAX_EVENTS:
                raise PiTraceError("Pi activity trace exceeds its bound")
        return events

    @classmethod
    def _read(cls, path: Path) -> tuple[bytes, list[PiTraceEvent]]:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return b"", []
        except OSError as error:
            raise PiTraceError("Pi activity trace is unreadable") from error
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o022 or metadata.st_size > _MAX_BYTES):
                raise PiTraceError("Pi activity trace is unsafe")
            chunks: list[bytes] = []
            remaining = _MAX_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > _MAX_BYTES:
                raise PiTraceError("Pi activity trace exceeds its bound")
        except OSError as error:
            raise PiTraceError("Pi activity trace is unreadable") from error
        finally:
            os.close(descriptor)
        return raw, cls._events(raw)

    def get_events(self, thread_dir: str | Path) -> list[PiTraceEvent]:
        """Return complete validated activity history without affecting Pi execution."""
        with self._lock:
            _, events = self._read(self._path(thread_dir))
            return events

    def append(self, thread_dir: str | Path, event: PiTraceEvent) -> PiTraceEvent:
        """Persist one validated event atomically, rejecting conflicting replay."""
        path = self._path(thread_dir)
        event = self._validate({
            "version": _VERSION, "ts": "", "run_id": event.run_id,
            "sequence": event.sequence, "operation": event.operation,
            "kind": event.kind, "name": event.name, "outcome": event.outcome,
        })
        with self._lock:
            raw, prior = self._read(path)
            expected_sequence = 1 + max(
                (item.sequence for item in prior if item.run_id == event.run_id), default=0)
            if event.sequence != expected_sequence:
                raise PiTraceError("Pi activity trace has a conflicting sequence")
            payload = json.dumps({
                "version": _VERSION, "ts": datetime.now(UTC).isoformat(),
                "run_id": event.run_id, "sequence": event.sequence,
                "operation": event.operation, "kind": event.kind,
                "name": event.name, "outcome": event.outcome,
            }, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(raw) + len(payload) > _MAX_BYTES:
                raise PiTraceError("Pi activity trace is full")
            self._events(raw + payload)
            descriptor: int | None = None
            temporary: str | None = None
            try:
                descriptor, temporary = tempfile.mkstemp(prefix=".pi-trace-", dir=path.parent)
                os.fchmod(descriptor, 0o600)
                view = memoryview(raw + payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Pi activity trace write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                os.replace(temporary, path)
                temporary = None
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError as error:
                raise PiTraceError("Pi activity trace write failed") from error
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if temporary:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass
            return event


class PiTraceRecorder:
    """One turn's opaque operation handles over a thread-owned Pi trace resource."""

    def __init__(self, store: PiTraceStore, thread_dir: str | Path, run_id: str) -> None:
        self._store = store
        self._thread_dir = thread_dir
        self._run_id = run_id
        self._lock = threading.Lock()
        self._sequence = 0
        self._operation = 0
        self._open: dict[int, tuple[str, str]] = {}
        self._unavailable = False

    def start(self, kind: Literal["model", "tool"],
              name: Literal["model request", "read", "write", "bash", "load skill", "map data"]) -> int | None:
        """Persist an admitted operation and return an opaque handle for its terminal state."""
        with self._lock:
            if self._unavailable:
                return None
            self._sequence += 1
            self._operation += 1
            operation = self._operation
            try:
                self._store.append(self._thread_dir, PiTraceEvent(
                    self._run_id, self._sequence, operation, kind, name, "started"))
            except PiTraceError:
                self._unavailable = True
                return None
            self._open[operation] = (kind, name)
            return operation

    def settle(self, operation: int | None, completed: bool) -> None:
        """Persist one generic terminal outcome for an opaque operation handle."""
        if operation is None:
            return
        with self._lock:
            detail = self._open.pop(operation, None)
            if detail is None or self._unavailable:
                return
            self._sequence += 1
            try:
                self._store.append(self._thread_dir, PiTraceEvent(
                    self._run_id, self._sequence, operation, detail[0], detail[1],
                    "completed" if completed else "did not complete"))
            except PiTraceError:
                self._unavailable = True

    def finish_unsettled(self) -> None:
        """Close surviving operations only after every trace producer has drained."""
        with self._lock:
            operations = tuple(self._open)
        for operation in operations:
            self.settle(operation, False)
