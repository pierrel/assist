"""Durable Agent Protocol-shaped runs.

A persisted ``pending`` run is the acceptance commit.  Dispatch queues and web
status are projections which may be rebuilt from this store after a restart.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from assist.backlog import PendingMessage
from assist.record_store import PerThreadJsonStore, RecordNotFound


RunStatus = Literal[
    "pending", "running", "success", "error", "timeout", "interrupted",
    "cancelled",
]
RunMode = Literal["turn", "child"]

RUNS_FILE = "runs.json"
# ``interrupted`` is terminal for that protocol invocation. Logical work continues in
# a new run sharing work_id; recovery still reconciles interrupted parents explicitly.
NONTERMINAL_STATUSES = frozenset({"pending", "running"})
TERMINAL_STATUSES = frozenset(
    {"success", "error", "timeout", "interrupted", "cancelled"})
_STATUSES = NONTERMINAL_STATUSES | TERMINAL_STATUSES
_MODES = frozenset({"turn", "child"})


class RunNotFound(RecordNotFound):
    """No run with the requested id exists on the thread."""


class InvalidRunTransition(ValueError):
    """A requested status change is not valid for the run's current state."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Run:
    """One durable unit of agent work.

    Execution context is explicit rather than an open-ended metadata mapping so
    callers cannot select an undeclared tool profile or host resource.
    """

    thread_id: str
    assistant_id: str
    text: str | None
    id: str
    work_id: str
    status: RunStatus
    mode: RunMode
    parent_thread_id: str | None
    parent_run_id: str | None
    dispatch_key: str | None
    sender: str | None
    rider: dict | None
    origin: str | None
    resume: bool
    resume_decision: dict | None
    pending_text: str | None
    active_ms: float
    consumed_by: str | None
    error: str | None
    result: str | None
    multitask_strategy: str
    created_at: str
    updated_at: str
    # For delegate slices only: brief-form URLs canonically matched to owner Runs
    # already accepted at admission, without added or changed userinfo.
    delegate_user_urls: tuple[str, ...] = ()
    # Private host-side location snapshot for this visible run. It is excluded
    # from protocol responses and is only reconstructed into tool configuration.
    location: dict | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        if not self.delegate_user_urls:
            value.pop("delegate_user_urls")
        if self.location is None:
            value.pop("location")
        return value

    @staticmethod
    def from_dict(value: dict) -> "Run":
        if value.get("status") not in _STATUSES:
            raise ValueError(f"invalid run status: {value.get('status')!r}")
        if value.get("mode", "turn") not in _MODES:
            raise ValueError(f"invalid run mode: {value.get('mode')!r}")
        return Run(
            thread_id=str(value["thread_id"]),
            assistant_id=str(value["assistant_id"]),
            text=(str(value["text"]) if value.get("text") is not None else None),
            id=str(value["id"]),
            work_id=str(value.get("work_id") or value["id"]),
            status=value["status"],
            mode=value.get("mode", "turn"),
            parent_thread_id=value.get("parent_thread_id") or None,
            parent_run_id=value.get("parent_run_id") or None,
            dispatch_key=value.get("dispatch_key") or None,
            sender=value.get("sender") or None,
            rider=value.get("rider") or None,
            origin=value.get("origin") or None,
            resume=bool(value.get("resume", False)),
            resume_decision=value.get("resume_decision") or None,
            pending_text=(str(value["pending_text"])
                          if value.get("pending_text") is not None else None),
            active_ms=float(value.get("active_ms", 0.0)),
            consumed_by=value.get("consumed_by") or None,
            error=value.get("error") or None,
            result=value.get("result") or None,
            multitask_strategy=value.get("multitask_strategy", "enqueue"),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            delegate_user_urls=tuple(value.get("delegate_user_urls") or ()),
            location=dict(value["location"]) if value.get("location") else None,
        )


_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    # ``pending -> success`` is the interjection handoff: the accepted follower is
    # checkpointed into the active run, then terminalized with ``consumed_by`` so a
    # restart can never dispatch it as a second answer.
    "pending": frozenset({"running", "success", "timeout", "cancelled"}),
    "running": frozenset({"pending", "success", "error", "timeout",
                           "interrupted", "cancelled"}),
    # A protocol run that interrupts never becomes running again. Resumption creates a
    # NEW run on the same thread with the same logical work_id.
    "interrupted": frozenset({"cancelled"}),
    "success": frozenset(),
    "error": frozenset(),
    "timeout": frozenset(),
    "cancelled": frozenset(),
}


class RunService(PerThreadJsonStore[Run]):
    """Atomic per-thread run persistence and lifecycle operations."""

    FILENAME = RUNS_FILE
    NOTFOUND_EXC = RunNotFound

    @property
    def root_dir(self) -> str:
        return self._root

    @staticmethod
    def _from_dict(value: dict) -> Run:
        return Run.from_dict(value)

    def _read(self, thread_id: str) -> list[Run]:
        try:
            with open(self._path(thread_id)) as stream:
                values = json.load(stream)
        except FileNotFoundError:
            return []
        return [Run.from_dict(value) for value in values]

    def create(
        self,
        thread_id: str,
        assistant_id: str,
        text: str | None,
        *,
        work_id: str | None = None,
        mode: RunMode = "turn",
        parent_thread_id: str | None = None,
        parent_run_id: str | None = None,
        dispatch_key: str | None = None,
        sender: str | None = None,
        rider: dict | None = None,
        origin: str | None = None,
        resume: bool = False,
        resume_decision: dict | None = None,
        pending_text: str | None = None,
        active_ms: float = 0.0,
        run_id: str | None = None,
        cancel_pending: bool = False,
        max_runs: int | None = None,
        max_pending: int | None = None,
        multitask_strategy: str = "enqueue",
        delegate_user_urls: tuple[str, ...] = (),
        location: dict | None = None,
    ) -> Run:
        """Persist and return a pending run, the work-acceptance commit."""
        if not assistant_id:
            raise ValueError("assistant_id is required")
        if active_ms < 0:
            raise ValueError("active_ms cannot be negative")
        if mode == "child" and (not parent_thread_id or not parent_run_id
                                or not dispatch_key):
            raise ValueError(
                "a child run requires parent_thread_id, parent_run_id, and dispatch_key")
        if mode == "child" and not thread_id.startswith("sub-"):
            raise ValueError("a child run requires a sub- thread id")
        if mode == "turn" and (parent_thread_id or parent_run_id):
            raise ValueError("a turn run cannot have parent fields")
        rid = run_id or uuid.uuid4().hex
        now = _now()
        run = Run(
            thread_id=thread_id, assistant_id=assistant_id, text=text, id=rid,
            work_id=work_id or rid, status="pending", mode=mode,
            parent_thread_id=parent_thread_id, parent_run_id=parent_run_id,
            dispatch_key=dispatch_key, sender=sender,
            rider=dict(rider) if rider else None, origin=origin, resume=resume,
            resume_decision=(dict(resume_decision) if resume_decision else None),
            pending_text=pending_text,
            active_ms=float(active_ms), consumed_by=None, error=None,
            result=None,
            multitask_strategy=multitask_strategy,
            created_at=now, updated_at=now,
            delegate_user_urls=tuple(delegate_user_urls),
            location=dict(location) if location else None,
        )
        with self._lock:
            if mode == "child":
                directory = os.path.dirname(self._path(thread_id))
                marker = os.path.join(directory, ".subagent")
                new_child_directory = not os.path.exists(directory)
                if os.path.exists(directory) and not os.path.isfile(marker):
                    if os.listdir(directory):
                        raise ValueError("a child run cannot use a visible thread")
            runs = self._read(thread_id)
            if dispatch_key:
                existing = next((candidate for candidate in runs
                                 if candidate.dispatch_key == dispatch_key), None)
                if existing is not None:
                    if (existing.assistant_id != assistant_id
                            or existing.text != text or existing.mode != mode
                            or existing.parent_thread_id != parent_thread_id
                            or existing.parent_run_id != parent_run_id):
                        raise ValueError(
                            f"dispatch key conflicts with persisted run: {dispatch_key}")
                    return existing
            if cancel_pending:
                if any(candidate.status in {"running", "interrupted"}
                       for candidate in runs):
                    raise InvalidRunTransition(
                        "running or interrupted work cannot be replaced")
                runs = [replace(candidate, status="cancelled", updated_at=now)
                        if candidate.status == "pending" else candidate
                        for candidate in runs]
            if max_runs is not None and len(runs) >= max_runs:
                raise InvalidRunTransition("run history limit reached")
            if (max_pending is not None
                    and sum(candidate.status == "pending" for candidate in runs)
                    >= max_pending):
                raise InvalidRunTransition("pending run limit reached")
            if any(existing.id == rid for existing in runs):
                raise ValueError(f"run already exists: {rid}")
            if mode == "child":
                os.makedirs(directory, exist_ok=True)
                with open(marker, "a"):
                    pass
            runs.append(run)
            try:
                self._write(thread_id, runs)
            except Exception:
                if mode == "child" and new_child_directory:
                    shutil.rmtree(directory, ignore_errors=True)
                raise
        return run

    def get(self, thread_id: str, run_id: str) -> Run:
        with self._lock:
            return self._find(self._read(thread_id), run_id)

    def list(self, thread_id: str) -> list[Run]:
        return self.for_thread(thread_id)

    def peek(self, thread_id: str) -> list[Run]:
        """Lock-free read for event-loop render paths.

        Atomic replacement means readers see the whole old or whole new file.
        A concurrent disappearance or unreadable projection renders no runs.
        """
        try:
            return self._read(thread_id)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return []

    @staticmethod
    def _find(runs: list[Run], run_id: str) -> Run:
        for run in runs:
            if run.id == run_id:
                return run
        raise RunNotFound(run_id)

    def transition(
        self,
        thread_id: str,
        run_id: str,
        status: RunStatus,
        *,
        active_ms: float | None = None,
        consumed_by: str | None = None,
        error: str | None = None,
        rider: dict | None = None,
        result: str | None = None,
    ) -> Run:
        """Move a run to ``status``; repeating the same transition is a no-op."""
        if active_ms is not None and active_ms < 0:
            raise ValueError("active_ms cannot be negative")
        with self._lock:
            runs = self._read(thread_id)
            current = self._find(runs, run_id)
            if status == current.status:
                changed = replace(
                    current,
                    active_ms=(current.active_ms if active_ms is None
                               else float(active_ms)),
                    consumed_by=(current.consumed_by if consumed_by is None
                                 else consumed_by),
                    error=current.error if error is None else error,
                    rider=current.rider if rider is None else dict(rider),
                    result=current.result if result is None else result,
                )
                if changed == current:
                    return current
                changed = replace(changed, updated_at=_now())
                runs[runs.index(current)] = changed
                self._write(thread_id, runs)
                return changed
            if status not in _TRANSITIONS[current.status]:
                raise InvalidRunTransition(
                    f"cannot transition run {run_id} from {current.status} to {status}")
            changed = replace(
                current,
                status=status,
                active_ms=current.active_ms if active_ms is None else float(active_ms),
                consumed_by=consumed_by if consumed_by is not None else current.consumed_by,
                error=error,
                rider=current.rider if rider is None else dict(rider),
                result=current.result if result is None else result,
                updated_at=_now(),
            )
            runs[runs.index(current)] = changed
            self._write(thread_id, runs)
            return changed

    def claim(self, thread_id: str, run_id: str) -> Run:
        """Atomically claim pending work for execution; repeated claims are safe."""
        return self.transition(thread_id, run_id, "running")

    def cancel(self, thread_id: str, run_id: str) -> Run:
        """Move any transition-eligible run to cancelled; repeats are idempotent."""
        return self.transition(thread_id, run_id, "cancelled")

    def cancel_pending(self, thread_id: str, run_id: str) -> Run:
        """Cancel ``run_id`` only while it is still pending."""
        with self._lock:
            runs = self._read(thread_id)
            current = self._find(runs, run_id)
            if current.status != "pending":
                raise InvalidRunTransition(
                    f"cannot cancel non-pending run {run_id}: {current.status}")
            changed = replace(current, status="cancelled", updated_at=_now())
            runs[runs.index(current)] = changed
            self._write(thread_id, runs)
            return changed

    def scan_all(self) -> list[Run]:
        """Return runs across visible and hidden thread directories."""
        return self.all()

    def _read_children(self) -> list[Run]:
        try:
            thread_ids = os.listdir(self._root)
        except FileNotFoundError:
            return []
        found = []
        for thread_id in thread_ids:
            directory = os.path.join(self._root, thread_id)
            if os.path.isfile(os.path.join(directory, ".subagent")):
                try:
                    found.extend(self._read(thread_id))
                except (FileNotFoundError, NotADirectoryError):
                    continue
        return found

    def scan_children(self) -> list[Run]:
        """Return child runs without parsing visible thread histories."""
        with self._lock:
            return self._read_children()

    def import_legacy(
        self,
        thread_id: str,
        records: list[PendingMessage],
        *,
        assistant_id: str = "general-agent",
    ) -> list[Run]:
        """Import legacy pending messages once, preserving their ticket ids.

        Existing ids are returned unchanged, making restart retries idempotent.
        This method never writes the legacy journal; removing it is the caller's
        responsibility after every imported run is durable.
        """
        with self._lock:
            runs = self._read(thread_id)
            by_id = {run.id: run for run in runs}
            imported = []
            changed = False
            for record in records:
                if record.thread_id != thread_id:
                    raise ValueError("legacy record belongs to another thread")
                run = by_id.get(record.id)
                if run is None:
                    now = _now()
                    run = Run(
                        thread_id=thread_id, assistant_id=assistant_id,
                        text=record.text, id=record.id, work_id=record.id,
                        status="pending", mode="turn", parent_thread_id=None,
                        parent_run_id=None, dispatch_key=None,
                        sender=record.sender, rider=dict(record.rider) if record.rider else None,
                        origin=record.origin, resume=False, resume_decision=None,
                        pending_text=None, active_ms=0.0,
                        consumed_by=None, error=None, result=None,
                        multitask_strategy="enqueue",
                        created_at=record.enqueued_at or now, updated_at=now,
                    )
                    runs.append(run)
                    by_id[run.id] = run
                    changed = True
                elif (run.text != record.text or run.sender != record.sender
                      or run.rider != record.rider or run.origin != record.origin):
                    raise ValueError(
                        f"legacy record {record.id} conflicts with persisted run")
                imported.append(run)
            if changed:
                self._write(thread_id, runs)
            return imported
