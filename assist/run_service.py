"""Durable Agent Protocol-shaped runs.

A persisted ``pending`` run is the dispatch acceptance commit. A direct owner
event may first be ``revocation_pending`` while browser safety reset completes.
Dispatch queues and web status are projections of this store.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from assist.backlog import PendingMessage
from assist.record_store import PerThreadJsonStore, RecordNotFound


RunStatus = Literal[
    "revocation_pending", "pending", "running", "success", "error", "timeout", "interrupted",
    "cancelled", "awaiting_approval",
]
RunMode = Literal["turn", "child"]
CancelCleanup = Literal["pending", "complete"]
ObservationToken = tuple[int, int, int, int, int]

RUNS_FILE = "runs.json"
# ``interrupted`` is terminal for that protocol invocation. Logical work continues in
# a new run sharing work_id; recovery still reconciles interrupted parents explicitly.
NONTERMINAL_STATUSES = frozenset({"revocation_pending", "pending", "running"})
AWAITING_APPROVAL_STATUSES = frozenset({"awaiting_approval"})
TERMINAL_STATUSES = frozenset(
    {"success", "error", "timeout", "interrupted", "cancelled"})
_STATUSES = NONTERMINAL_STATUSES | AWAITING_APPROVAL_STATUSES | TERMINAL_STATUSES
_MODES = frozenset({"turn", "child"})


class RunNotFound(RecordNotFound):
    """No run with the requested id exists on the thread."""


class InvalidRunTransition(ValueError):
    """A requested status change is not valid for the run's current state."""


class RunStoreUnavailable(RuntimeError):
    """The durable Run store could not provide a readable or writable snapshot."""


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
    # A cancellation receipt belongs to the immutable accepted handle, not a
    # successor slice.  Its absence keeps historical records unchanged.
    cancel_cleanup: CancelCleanup | None = None
    # Only a directly admitted human turn mints its own immutable Run ID here.
    # A same-work successor may carry that ID; synthetic follow-ups do not.
    user_event_id: str | None = None
    # Direct-owner events have a durable per-thread ordering sequence. Historical
    # Runs without this field retain zero and cannot outrank a new event.
    admission_sequence: int = 0
    revocation_retry_at: str | None = None
    revocation_generation: str | None = None
    browser_reset_notice: bool = False
    browser_reset_run_id: str | None = None
    browser_boot_id: str | None = None
    browser_deadline_ns: int | None = None

    _MAX_OPAQUE_ID_CHARS = 256

    @staticmethod
    def _required_opaque_id(value: object, name: str) -> str:
        """Return one bounded durable identifier without coercing corrupt values."""
        if (not isinstance(value, str) or not value
                or len(value) > Run._MAX_OPAQUE_ID_CHARS):
            raise ValueError(f"invalid {name}")
        return value

    @staticmethod
    def _optional_opaque_id(value: object, name: str) -> str | None:
        """Return an absent or bounded durable identifier without coercion."""
        if value is None or value == "":
            return None
        return Run._required_opaque_id(value, name)

    @staticmethod
    def _optional_text(value: object, name: str) -> str | None:
        """Accept only persisted text, never a lossy string coercion."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"invalid {name}")
        return value

    @staticmethod
    def _optional_mapping(value: object, name: str) -> dict | None:
        """Accept only the map shape dispatch later dereferences."""
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"invalid {name}")
        return dict(value)

    def to_dict(self) -> dict:
        value = asdict(self)
        if not self.delegate_user_urls:
            value.pop("delegate_user_urls")
        if self.location is None:
            value.pop("location")
        if self.cancel_cleanup is None:
            value.pop("cancel_cleanup")
        if self.user_event_id is None:
            value.pop("user_event_id")
        if self.admission_sequence == 0:
            value.pop("admission_sequence")
        if self.revocation_retry_at is None:
            value.pop("revocation_retry_at")
        if self.revocation_generation is None:
            value.pop("revocation_generation")
        if not self.browser_reset_notice:
            value.pop("browser_reset_notice")
        if self.browser_reset_run_id is None:
            value.pop("browser_reset_run_id")
        if self.browser_boot_id is None:
            value.pop("browser_boot_id")
        if self.browser_deadline_ns is None:
            value.pop("browser_deadline_ns")
        return value

    @staticmethod
    def from_dict(value: dict) -> "Run":
        if not isinstance(value, dict):
            raise TypeError("run record must be an object")
        if value.get("status") not in _STATUSES:
            raise ValueError(f"invalid run status: {value.get('status')!r}")
        if value.get("mode", "turn") not in _MODES:
            raise ValueError(f"invalid run mode: {value.get('mode')!r}")
        cancel_cleanup = value.get("cancel_cleanup")
        if cancel_cleanup not in {None, "pending", "complete"}:
            raise ValueError(f"invalid cancellation cleanup: {cancel_cleanup!r}")
        delegate_user_urls = value.get("delegate_user_urls") or ()
        if (not isinstance(delegate_user_urls, list | tuple)
                or any(not isinstance(url, str) or len(url) > 4096
                       for url in delegate_user_urls)):
            raise ValueError("invalid delegate user URLs")
        active_ms = value.get("active_ms", 0.0)
        if (isinstance(active_ms, bool) or not isinstance(active_ms, int | float)
                or not math.isfinite(active_ms) or active_ms < 0):
            raise ValueError("invalid active milliseconds")
        resume = value.get("resume", False)
        if not isinstance(resume, bool):
            raise ValueError("invalid resume flag")
        sequence = value.get("admission_sequence", 0)
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("invalid admission sequence")
        browser_reset_notice = value.get("browser_reset_notice", False)
        if not isinstance(browser_reset_notice, bool):
            raise ValueError("invalid browser reset notice")
        deadline_ns = value.get("browser_deadline_ns")
        if (deadline_ns is not None and
                (isinstance(deadline_ns, bool) or not isinstance(deadline_ns, int)
                 or deadline_ns <= 0)):
            raise ValueError("invalid browser deadline")
        boot_id = Run._optional_opaque_id(value.get("browser_boot_id"),
                                          "browser boot id")
        if (boot_id is None) != (deadline_ns is None):
            raise ValueError("incomplete browser deadline")
        return Run(
            thread_id=Run._required_opaque_id(value["thread_id"], "thread id"),
            assistant_id=Run._required_opaque_id(value["assistant_id"], "assistant id"),
            text=Run._optional_text(value.get("text"), "text"),
            id=Run._required_opaque_id(value["id"], "run id"),
            work_id=Run._required_opaque_id(
                value.get("work_id") or value["id"], "work id"),
            status=value["status"],
            mode=value.get("mode", "turn"),
            parent_thread_id=Run._optional_opaque_id(
                value.get("parent_thread_id"), "parent thread id"),
            parent_run_id=Run._optional_opaque_id(value.get("parent_run_id"), "parent run id"),
            dispatch_key=Run._optional_opaque_id(value.get("dispatch_key"), "dispatch key"),
            sender=Run._optional_opaque_id(value.get("sender"), "sender"),
            rider=Run._optional_mapping(value.get("rider"), "rider"),
            origin=Run._optional_opaque_id(value.get("origin"), "origin"),
            resume=resume,
            resume_decision=Run._optional_mapping(
                value.get("resume_decision"), "resume decision"),
            pending_text=Run._optional_text(value.get("pending_text"), "pending text"),
            active_ms=float(active_ms),
            consumed_by=Run._optional_opaque_id(value.get("consumed_by"), "consumed by"),
            error=Run._optional_text(value.get("error"), "error"),
            result=Run._optional_text(value.get("result"), "result"),
            multitask_strategy=Run._required_opaque_id(
                value.get("multitask_strategy", "enqueue"), "multitask strategy"),
            created_at=Run._required_opaque_id(value["created_at"], "created at"),
            updated_at=Run._required_opaque_id(value["updated_at"], "updated at"),
            delegate_user_urls=tuple(delegate_user_urls),
            location=Run._optional_mapping(value.get("location"), "location"),
            cancel_cleanup=cancel_cleanup,
            user_event_id=Run._optional_opaque_id(
                value.get("user_event_id"), "user event id"),
            admission_sequence=sequence,
            revocation_retry_at=Run._optional_text(
                value.get("revocation_retry_at"), "revocation retry at"),
            revocation_generation=Run._optional_opaque_id(
                value.get("revocation_generation"), "revocation generation"),
            browser_reset_notice=browser_reset_notice,
            browser_reset_run_id=Run._optional_opaque_id(
                value.get("browser_reset_run_id"), "browser reset run id"),
            browser_boot_id=boot_id,
            browser_deadline_ns=deadline_ns,
        )


_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    "revocation_pending": frozenset({"pending", "cancelled", "error"}),
    # ``pending -> success`` is the interjection handoff: the accepted follower is
    # checkpointed into the active run, then terminalized with ``consumed_by`` so a
    # restart can never dispatch it as a second answer.
    "pending": frozenset({"running", "success", "timeout", "cancelled"}),
    "running": frozenset({"pending", "success", "error", "timeout",
                           "interrupted", "cancelled", "awaiting_approval"}),
    # A protocol run that interrupts never becomes running again. Resumption creates a
    # NEW run on the same thread with the same logical work_id.
    "interrupted": frozenset({"cancelled"}),
    # A user resolution starts a NEW child slice, preserving the interrupted
    # graph checkpoint and leaving this parked invocation immutable.
    "awaiting_approval": frozenset({"cancelled"}),
    "success": frozenset(),
    "error": frozenset(),
    "timeout": frozenset(),
    "cancelled": frozenset(),
}


class RunService(PerThreadJsonStore[Run]):
    """Atomic per-thread run persistence and lifecycle operations."""

    FILENAME = RUNS_FILE
    NOTFOUND_EXC = RunNotFound

    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self._revisions: dict[str, int] = {}
        self._revision_lock = threading.Lock()

    def _write(self, thread_id: str, runs: list[Run]) -> None:
        """Persist RUNS before incrementing their process-local invalidation revision."""
        super()._write(thread_id, runs)

        with self._revision_lock:
            self._revisions[thread_id] = self._revisions.get(thread_id, 0) + 1

    def revision(self, thread_id: str) -> int:
        """Return THREAD-ID's process-local durable-write invalidation revision."""
        with self._revision_lock:
            return self._revisions.get(thread_id, 0)

    def list_with_revision(self, thread_id: str) -> tuple[list[Run], int]:
        """Read one durable snapshot and its invalidation revision atomically."""
        runs, revision, _token = self.list_with_revision_and_token(thread_id)
        return runs, revision

    def list_with_revision_and_token(
            self, thread_id: str
    ) -> tuple[list[Run], int, ObservationToken | None]:
        """Read one coherent durable snapshot, revision, and file observation token."""
        with self._lock:
            runs, token = self._read_with_token(thread_id)
            return runs, self.revision(thread_id), token

    @property
    def root_dir(self) -> str:
        return self._root

    @staticmethod
    def _from_dict(value: dict) -> Run:
        return Run.from_dict(value)

    @staticmethod
    def _token(metadata: os.stat_result) -> ObservationToken:
        return (metadata.st_dev, metadata.st_ino, metadata.st_size,
                metadata.st_mtime_ns, metadata.st_ctime_ns)

    @classmethod
    def _regular_token(cls, metadata: os.stat_result) -> ObservationToken:
        if not stat.S_ISREG(metadata.st_mode):
            raise RunStoreUnavailable("Run store is unavailable")
        return cls._token(metadata)

    def _open_observation_fd(self, thread_id: str) -> int:
        """Open one private durable file without following a crafted final symlink."""
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        return os.open(self._path(thread_id), flags)

    def observation_token(self, thread_id: str) -> ObservationToken | None:
        """Probe one durable file in O(1), without parsing its Run history."""
        try:
            fd = self._open_observation_fd(thread_id)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise RunStoreUnavailable("Run store is unavailable") from error
        try:
            return self._regular_token(os.fstat(fd))
        except OSError as error:
            raise RunStoreUnavailable("Run store is unavailable") from error
        finally:
            os.close(fd)

    def _read_with_token(self, thread_id: str) -> tuple[list[Run], ObservationToken | None]:
        """Parse one coherent no-follow snapshot, retrying one in-place mutation."""
        for _attempt in range(2):
            try:
                fd = self._open_observation_fd(thread_id)
            except FileNotFoundError:
                return [], None
            except OSError as error:
                raise RunStoreUnavailable("Run store is unavailable") from error
            try:
                before = self._regular_token(os.fstat(fd))
                with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
                    values = json.load(stream)
                after = self._regular_token(os.fstat(fd))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RunStoreUnavailable("Run store is unavailable") from error
            finally:
                os.close(fd)
            if before != after:
                continue
            try:
                if not isinstance(values, list):
                    raise TypeError("run store must be a list")
                return [Run.from_dict(value) for value in values], before
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise RunStoreUnavailable("Run store is unavailable") from error
        raise RunStoreUnavailable("Run store is unavailable")

    def _read(self, thread_id: str) -> list[Run]:
        try:
            runs, _token = self._read_with_token(thread_id)
            return runs
        except RunStoreUnavailable:
            raise

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
        user_origin: bool = False,
        user_event_id: str | None = None,
        revocation_pending: bool = False,
        browser_reset_run_id: str | None = None,
    ) -> Run:
        """Persist a direct held event or a dispatchable pending Run."""
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
        if user_origin:
            if (user_event_id is not None or mode != "turn" or origin is not None
                    or sender is not None or text is None or resume):
                raise ValueError("only a fresh direct user turn can create user provenance")
            user_event_id = rid
        if revocation_pending and not user_origin:
            raise ValueError("only a direct owner event can await browser revocation")
        now = _now()
        run = Run(
            thread_id=thread_id, assistant_id=assistant_id, text=text, id=rid,
            work_id=work_id or rid,
            status="revocation_pending" if revocation_pending else "pending", mode=mode,
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
            user_event_id=user_event_id,
            browser_reset_run_id=browser_reset_run_id,
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
                        if candidate.status in {"pending", "revocation_pending"} else candidate
                        for candidate in runs]
            if max_runs is not None and len(runs) >= max_runs:
                raise InvalidRunTransition("run history limit reached")
            if (max_pending is not None
                    and sum(candidate.status in {"pending", "revocation_pending"}
                            for candidate in runs)
                    >= max_pending):
                raise InvalidRunTransition("pending run limit reached")
            if any(existing.id == rid for existing in runs):
                raise ValueError(f"run already exists: {rid}")
            if mode == "child":
                os.makedirs(directory, exist_ok=True)
                with open(marker, "a"):
                    pass
            if user_origin:
                run = replace(run, admission_sequence=max(
                    (candidate.admission_sequence for candidate in runs), default=0) + 1)
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

    def bind_browser_deadline(self, thread_id: str, run_id: str) -> tuple[str, int]:
        """Persist one boot-relative first-open deadline for the logical work."""
        with self._lock:
            runs = self._read(thread_id)
            current = self._find(runs, run_id)
            existing = [run for run in runs if run.work_id == current.work_id
                        and run.browser_deadline_ns is not None]
            if existing:
                boot_id = existing[0].browser_boot_id
                deadline_ns = min(run.browser_deadline_ns for run in existing)
                if any(run.browser_boot_id != boot_id for run in existing):
                    raise RuntimeError("browser work crossed a host reboot")
            else:
                with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
                    boot_id = stream.read().strip()
                deadline_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME) + 240_000_000_000
            if current.browser_deadline_ns is None:
                runs[runs.index(current)] = replace(
                    current, browser_boot_id=boot_id,
                    browser_deadline_ns=deadline_ns, updated_at=_now())
                self._write(thread_id, runs)
            return boot_id, deadline_ns

    def list(self, thread_id: str) -> list[Run]:
        return self.for_thread(thread_id)

    def peek(self, thread_id: str) -> list[Run]:
        """Lock-free read for event-loop render paths.

        Atomic replacement means readers see the whole old or whole new file.
        A concurrent disappearance or unreadable projection renders no runs.
        """
        try:
            return self._read(thread_id)
        except RunStoreUnavailable:
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
        revocation_retry_at: str | None = None,
        revocation_generation: str | None = None,
        browser_reset_notice: bool = False,
        browser_reset_run_id: str | None = None,
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
                    revocation_retry_at=(current.revocation_retry_at
                                         if revocation_retry_at is None else revocation_retry_at),
                    revocation_generation=(current.revocation_generation
                                           if revocation_generation is None
                                           else revocation_generation),
                    browser_reset_notice=current.browser_reset_notice or browser_reset_notice,
                    browser_reset_run_id=(current.browser_reset_run_id
                                          if browser_reset_run_id is None
                                          else browser_reset_run_id),
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
                revocation_retry_at=revocation_retry_at,
                revocation_generation=(current.revocation_generation
                                       if revocation_generation is None
                                       else revocation_generation),
                browser_reset_notice=current.browser_reset_notice or browser_reset_notice,
                browser_reset_run_id=(current.browser_reset_run_id
                                      if browser_reset_run_id is None
                                      else browser_reset_run_id),
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

    def cancel_logical(self, thread_id: str, accepted_id: str) -> list[Run]:
        """Atomically cancel a pending logical slice and receipt its cleanup.

        The accepted handle owns the receipt so a retry can distinguish its
        incomplete cleanup from an unrelated terminal cancellation.  The
        returned records are the durable post-write snapshot.
        """
        with self._lock:
            runs = self._read(thread_id)
            accepted = self._find(runs, accepted_id)
            work = [run for run in runs if run.work_id == accepted.work_id]
            selected = work[-1]
            if selected.status != "pending":
                raise InvalidRunTransition(
                    f"cannot cancel non-pending logical run {selected.id}: {selected.status}")
            now = _now()
            updated = []
            for run in runs:
                changes = {}
                if run.id == selected.id or (
                        run.work_id == accepted.work_id and run.status == "interrupted"):
                    changes.update(status="cancelled", updated_at=now)
                if run.id == accepted_id:
                    changes["cancel_cleanup"] = "pending"
                    changes.setdefault("updated_at", now)
                updated.append(replace(run, **changes) if changes else run)
            self._write(thread_id, updated)
            return updated

    def complete_cancel_cleanup(self, thread_id: str, accepted_id: str) -> Run:
        """Durably record that ACCEPTED-ID's idempotent cancellation cleanup ended."""
        with self._lock:
            runs = self._read(thread_id)
            accepted = self._find(runs, accepted_id)
            if accepted.cancel_cleanup == "complete":
                return accepted
            if accepted.cancel_cleanup != "pending":
                raise InvalidRunTransition(
                    f"run {accepted_id} has no pending cancellation cleanup")
            completed = replace(accepted, cancel_cleanup="complete", updated_at=_now())
            runs[runs.index(accepted)] = completed
            self._write(thread_id, runs)
            return completed

    def fail_initialization(
            self, thread_id: str, run_id: str, error: str
    ) -> list[Run]:
        """Atomically mark the failed initializer and its pending followers erroneous."""
        with self._lock:
            runs = self._read(thread_id)
            self._find(runs, run_id)
            now = _now()
            changed = False
            failed = []
            updated_runs = []
            for run in runs:
                should_fail = run.id == run_id or run.status in {
                    "pending", "revocation_pending"}
                if should_fail and run.status in {"pending", "revocation_pending", "running"}:
                    run = replace(run, status="error", error=error, updated_at=now)
                    changed = True
                if should_fail and run.status == "error":
                    failed.append(run)
                updated_runs.append(run)
            if changed:
                self._write(thread_id, updated_runs)
            return failed

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
