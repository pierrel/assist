"""Private, bounded live-conversation capture storage and judging.

This module deliberately uses fixed-schema model calls rather than an Assist
thread: recorded evidence must not acquire tools, a sandbox, a checkpointer, or
a workspace while it is being judged.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import secrets
import stat
import threading
import fcntl
from time import monotonic
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from assist.thread_queue import THREAD_QUEUE, QueueWaitTimeout
from assist.visible_conversation import VisibleRecord
from edd.outcome_judge import Evidence, OutcomeJudge, OutcomeObservation, OutcomeRequirement


_ID_RE = re.compile(r"[A-Za-z0-9_-]{20,64}\Z")
_PROMPT_PATH = Path(__file__).with_name("live_capture_interpreter_prompt.md")
CAPTURE_PROJECTION_VERSION = 1
CAPTURE_OBSERVATION_VERSION = 1
MAX_REASON_BYTES = 8_000
MAX_RECORDS = 240
MAX_RECORD_BYTES = 32_000
MAX_SNAPSHOT_BYTES = 512_000
MAX_STORE_BYTES = 64 * 1024 * 1024
MAX_RECOVERY = 128
CAPTURE_QUEUE_HOLD_TIMEOUT_S = 120.0
CAPTURE_QUEUE_POLL_TIMEOUT_S = 1.0
CAPTURE_MODEL_WALL_TIMEOUT_S = 110.0
CAPTURE_OUTPUT_TOKENS = 1_024
_REJECTED_CAPTURE_RESERVATION_BYTES = MAX_REASON_BYTES + 4_096
# A judged result retains the canonical observation as well as the immutable
# transcript.  Reserve for that duplicate evidence, criterion ID lists, and
# bounded model output before admitting the snapshot.
_RESULT_RESERVATION_BYTES = 2 * MAX_SNAPSHOT_BYTES + 256 * 1024


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CaptureCriterion(_ClosedModel):
    description: str = Field(min_length=1, max_length=600)


class CaptureStorageFull(ValueError):
    """The private store cannot safely persist another capture card."""


class CaptureCriteria(_ClosedModel):
    status: Literal["criteria", "needs_clarification"]
    requested: tuple[CaptureCriterion, ...] = ()
    forbidden: tuple[CaptureCriterion, ...] = ()
    clarification: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def validate_criteria(self) -> "CaptureCriteria":
        if self.status == "criteria" and not self.requested:
            raise ValueError("criteria need a requested outcome")
        if self.status == "needs_clarification" and self.requested:
            raise ValueError("clarification cannot include requested outcomes")
        if len(self.requested) > 4 or len(self.forbidden) > 4:
            raise ValueError("criteria are bounded")
        return self


@dataclass(frozen=True)
class InterpretedCriteria:
    criteria: CaptureCriteria
    model: str
    prompt_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _record_json(record: VisibleRecord) -> dict[str, object]:
    return {
        "id": record.id, "order": record.order, "role": record.role,
        "text": record.text, "source_kind": record.source_kind,
        "capture_eligible": record.capture_eligible,
    }


class CaptureStore:
    """Owner-only immutable snapshots and mutable worker result records."""

    def __init__(self, root: str | os.PathLike[str], *, threads_root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        thread_root = Path(threads_root).expanduser().resolve()
        if self.root == thread_root or thread_root in self.root.parents or self.root in thread_root.parents:
            raise ValueError("capture root must be outside the thread root")
        self._lock = threading.RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._check_directory(self.root)
        self._index = self.root / "index.json"
        if not self._index.exists():
            self._write_json(self._index, {"version": 1, "threads": {}})

    @staticmethod
    def _check_directory(path: Path) -> None:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise ValueError("capture path must be a real directory")
        if st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise ValueError("capture directory must be owner-only")

    @staticmethod
    def _check_id(capture_id: str) -> None:
        if not _ID_RE.fullmatch(capture_id):
            raise ValueError("invalid capture identifier")

    def _path(self, capture_id: str, filename: str) -> Path:
        self._check_id(capture_id)
        if filename not in {"request.json", "transcript.json", "result.json"}:
            raise ValueError("invalid capture filename")
        self._check_directory(self.root / capture_id)
        path = self.root / capture_id / filename
        if path.parent.parent != self.root:
            raise ValueError("invalid capture path")
        return path

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode) \
                or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise ValueError("capture file must be an owner-only regular file")
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("capture JSON must be an object")
        return value

    def _write_json(self, path: Path, value: object) -> None:
        data = _json_bytes(value)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            os.chmod(path, 0o600)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _index_value(self) -> dict[str, Any]:
        return self._read_json(self._index)

    def _store_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            st = path.lstat()
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
        return total

    def create(
        self, *, thread_id: str, reason: str, scope: Literal["last_3", "entire"],
        records: tuple[VisibleRecord, ...], turn_range: tuple[int, int] = (1, 1),
        source_revision: str | None = None,
    ) -> dict[str, Any]:
        reason_bytes = reason.encode()
        if not reason.strip():
            raise ValueError("a capture reason is required")
        if len(reason_bytes) > MAX_REASON_BYTES:
            raise ValueError("capture reason is too large")
        if turn_range[0] < 1 or turn_range[1] < turn_range[0]:
            raise ValueError("invalid capture turn range")
        with self._lock:
            stored = self._store_bytes()
            if stored + _REJECTED_CAPTURE_RESERVATION_BYTES > MAX_STORE_BYTES:
                raise CaptureStorageFull("private capture storage is full")
            if len(records) > MAX_RECORDS:
                return self._create_rejected(thread_id, reason, scope, turn_range, "too many visible records")
            if any(len(record.text) > MAX_RECORD_BYTES for record in records):
                return self._create_rejected(thread_id, reason, scope, turn_range, "a visible record is too large")
            if any(len(record.text.encode()) > MAX_RECORD_BYTES for record in records):
                return self._create_rejected(thread_id, reason, scope, turn_range, "a visible record is too large")
            transcript = {
                "projection_version": CAPTURE_PROJECTION_VERSION,
                "observation_schema_version": CAPTURE_OBSERVATION_VERSION,
                "records": [_record_json(record) for record in records],
            }
            transcript_bytes = _json_bytes(transcript)
            if len(transcript_bytes) > MAX_SNAPSHOT_BYTES:
                return self._create_rejected(thread_id, reason, scope, turn_range, "selected conversation is too large")
            if stored + len(transcript_bytes) + _RESULT_RESERVATION_BYTES > MAX_STORE_BYTES:
                return self._create_rejected(
                    thread_id, reason, scope, turn_range, "private capture storage is full", status="failed")
            return self._create_files(thread_id, reason, scope, turn_range, transcript, transcript_bytes, source_revision, "queued", None)

    def _create_rejected(self, thread_id: str, reason: str, scope: str,
                         turn_range: tuple[int, int], error: str,
                         *, status: Literal["failed", "needs_shorter_scope"] = "needs_shorter_scope") -> dict[str, Any]:
        with self._lock:
            return self._create_files(thread_id, reason, scope, turn_range, None, None, None, status, error)

    def _create_files(
        self, thread_id: str, reason: str, scope: str, turn_range: tuple[int, int],
        transcript: dict[str, Any] | None,
        transcript_bytes: bytes | None, source_revision: str | None, status: str, error: str | None,
    ) -> dict[str, Any]:
        capture_id = secrets.token_urlsafe(20)
        self._check_id(capture_id)
        directory = self.root / capture_id
        directory.mkdir(mode=0o700)
        self._check_directory(directory)
        created_at = _utc_now()
        request = {
            "capture_id": capture_id, "thread_id": thread_id, "reason": reason,
            "scope": scope, "turn_range": list(turn_range), "created_at": created_at,
            "source_revision": source_revision,
        }
        result = {"status": status, "updated_at": created_at}
        if error:
            result["error"] = error
        self._write_json(directory / "request.json", request)
        if transcript is not None and transcript_bytes is not None:
            self._write_json(directory / "transcript.json", transcript)
            result["transcript_sha256"] = hashlib.sha256(transcript_bytes).hexdigest()
            result["transcript_bytes"] = len(transcript_bytes)
        self._write_json(directory / "result.json", result)
        index = self._index_value()
        index.setdefault("threads", {}).setdefault(thread_id, []).insert(0, {
            "capture_id": capture_id, "status": status, "updated_at": created_at,
        })
        self._write_json(self._index, index)
        return self.get_for_thread(thread_id, capture_id)

    def get_for_thread(self, thread_id: str, capture_id: str) -> dict[str, Any]:
        with self._lock:
            request = self._read_json(self._path(capture_id, "request.json"))
            if request.get("thread_id") != thread_id:
                raise FileNotFoundError("capture is not part of this thread")
            result = self._read_json(self._path(capture_id, "result.json"))
            transcript_path = self._path(capture_id, "transcript.json")
            try:
                transcript = self._read_json(transcript_path)
            except FileNotFoundError:
                transcript = None
            return {"request": request, "result": result, "transcript": transcript}

    def list_for_thread(self, thread_id: str) -> list[dict[str, Any]]:
        with self._lock:
            index = self._index_value()
            return list(index.get("threads", {}).get(thread_id, []))

    def list_for_threads(self) -> dict[str, list[dict[str, Any]]]:
        """Return copied per-thread summaries in one index read."""
        with self._lock:
            threads = self._index_value().get("threads", {})
            if not isinstance(threads, dict):
                return {}
            return {
                thread_id: [dict(entry) for entry in entries if isinstance(entry, dict)]
                for thread_id, entries in threads.items()
                if isinstance(thread_id, str) and isinstance(entries, list)
            }

    def update_result(self, thread_id: str, capture_id: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self.get_for_thread(thread_id, capture_id)
            result = {**current["result"], **result, "updated_at": _utc_now()}
            result_path = self._path(capture_id, "result.json")
            new_bytes = len(_json_bytes(result))
            if self._store_bytes() - result_path.stat().st_size + new_bytes > MAX_STORE_BYTES:
                raise CaptureStorageFull("private capture storage is full")
            self._write_json(result_path, result)
            index = self._index_value()
            entries = index.get("threads", {}).get(thread_id, [])
            for entry in entries:
                if entry.get("capture_id") == capture_id:
                    entry["status"] = result["status"]
                    entry["updated_at"] = result["updated_at"]
            self._write_json(self._index, index)
            return {**current, "result": result}

    def pending(self) -> list[tuple[str, str]]:
        with self._lock:
            pending: list[tuple[str, str]] = []
            threads = self._index_value().get("threads", {})
            if not isinstance(threads, dict):
                return pending
            for thread_id, entries in threads.items():
                if not isinstance(thread_id, str) or not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    capture_id = entry.get("capture_id")
                    if (entry.get("status") in {"queued", "interpreting", "judging"}
                            and isinstance(capture_id, str)
                            and _ID_RE.fullmatch(capture_id)):
                        pending.append((thread_id, capture_id))
            return pending[:MAX_RECOVERY]


class CaptureInterpreter:
    """One tool-free fixed-schema interpretation of a capture reason."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model
        self._prompt = _PROMPT_PATH.read_text()
        self._prompt_sha256 = hashlib.sha256(self._prompt.encode()).hexdigest()

    def interpret(self, reason: str, transcript: dict[str, Any]) -> InterpretedCriteria:
        selected = self._model
        if selected is None:
            from assist.model_manager import select_capture_model
            selected = select_capture_model()
        model = selected.bind(
            response_format={"type": "json_schema", "json_schema": {
                "name": "assist_capture_criteria", "strict": True,
                "schema": CaptureCriteria.model_json_schema(),
            }},
            max_tokens=CAPTURE_OUTPUT_TOKENS, seed=0,
        )
        payload = json.dumps({"reason": reason, "transcript": transcript}, sort_keys=True)
        response = model.invoke([
            SystemMessage(content=self._prompt),
            HumanMessage(content=("BEGIN UNTRUSTED LIVE CAPTURE\n" + payload + "\nEND UNTRUSTED LIVE CAPTURE")),
        ])
        if not isinstance(response.content, str):
            raise TypeError("capture interpreter response must be text")
        model_name = response.response_metadata.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip() or response.response_metadata.get("finish_reason") != "stop":
            raise ValueError("capture interpreter did not finish cleanly")
        return InterpretedCriteria(CaptureCriteria.model_validate_json(response.content), model_name, self._prompt_sha256)


def build_observation(criteria: CaptureCriteria, transcript: dict[str, Any]) -> OutcomeObservation:
    """Attach every canonical record to every criterion deterministically."""
    if criteria.status != "criteria":
        raise ValueError("clarification has no judgeable observation")
    records = transcript.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("capture transcript has no records")
    evidence = tuple(Evidence(
        id=str(record["id"]), kind="prompt" if record["role"] == "user" else "response",
        state="present", content=str(record["text"]),
    ) for record in records)
    ids = tuple(item.id for item in evidence)
    requested = tuple(OutcomeRequirement(id=f"requested-{n}", description=item.description, evidence_ids=ids)
                      for n, item in enumerate(criteria.requested, start=1))
    forbidden = tuple(OutcomeRequirement(id=f"forbidden-{n}", description=item.description, evidence_ids=ids)
                      for n, item in enumerate(criteria.forbidden, start=1))
    return OutcomeObservation(requested=requested, forbidden=forbidden, evidence=evidence)


def _capture_model_process(kind: str, payload: dict[str, Any], result) -> None:
    """Run one model call in an isolated child that the worker can kill."""
    try:
        from assist.model_manager import select_capture_model

        model = select_capture_model()
        if kind == "interpret":
            interpreted = CaptureInterpreter(model).interpret(payload["reason"], payload["transcript"])
            value = {
                "criteria": interpreted.criteria.model_dump(mode="json"),
                "model": interpreted.model, "prompt_sha256": interpreted.prompt_sha256,
            }
        elif kind == "judge":
            observation = OutcomeObservation.model_validate(payload["observation"])
            judged = OutcomeJudge(model, max_tokens=CAPTURE_OUTPUT_TOKENS).judge_with_provenance(observation)
            value = {
                "verdict": judged.verdict.model_dump(mode="json"),
                "model": judged.model, "prompt_sha256": judged.prompt_sha256,
            }
        else:
            raise ValueError("unknown capture model call")
        result.send({"ok": True, "value": value})
    except Exception:
        result.send({"ok": False})
    finally:
        result.close()


def _bounded_model_call(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Kill a raw model request before its queue hold can be force-released."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_capture_model_process, args=(kind, payload, child))
    process.start()
    child.close()
    try:
        process.join(CAPTURE_MODEL_WALL_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(2)
            raise TimeoutError("capture model call exceeded its wall-clock deadline")
        if not parent.poll() or not (response := parent.recv()).get("ok"):
            raise RuntimeError("capture model call failed")
        value = response.get("value")
        if not isinstance(value, dict):
            raise RuntimeError("capture model call returned invalid data")
        return value
    finally:
        if process.is_alive():
            process.kill()
            process.join(2)
        parent.close()


class CaptureWorker:
    """Exclusive bounded background owner for interpreter and judge calls."""

    def __init__(self, store: CaptureStore, *, model_factory: Callable[[], Any] | None = None) -> None:
        self.store = store
        self._model_factory = model_factory
        self._queue: Queue[tuple[str, str] | None] = Queue(maxsize=MAX_RECOVERY)
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._seen: set[tuple[str, str]] = set()
        self._seen_lock = threading.Lock()
        self._lock_file: Any | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        lock_path = self.store.root / "worker.lock"
        fd = -1
        try:
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            lock_stat = os.fstat(fd)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.getuid() \
                    or lock_stat.st_mode & 0o077:
                raise ValueError("capture worker lock must be an owner-only regular file")
            self._lock_file = os.fdopen(fd, "a+")
        except (OSError, ValueError) as error:
            if self._lock_file is None and fd >= 0:
                os.close(fd)
            raise ValueError("capture worker lock must be an owner-only regular file") from error
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError("another capture worker owns this store") from error
        try:
            for item in self.store.pending():
                self.submit(*item)
            self._stopping.clear()
            self._thread = threading.Thread(target=self._run, name="assist-capture-worker", daemon=True)
            self._thread.start()
        except BaseException:
            with self._seen_lock:
                self._seen.clear()
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            raise

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=CAPTURE_MODEL_WALL_TIMEOUT_S + 5)
            if self._thread.is_alive():
                raise RuntimeError("capture worker did not stop before its bounded deadline")
            self._thread = None
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def submit(self, thread_id: str, capture_id: str) -> None:
        item = (thread_id, capture_id)
        with self._seen_lock:
            if item in self._seen:
                return
            self._seen.add(item)
            try:
                self._queue.put_nowait(item)
            except Exception:
                self._seen.discard(item)
                raise RuntimeError("capture worker backlog is full") from None

    def _model(self) -> Any:
        if self._model_factory is not None:
            return self._model_factory()
        from assist.model_manager import select_capture_model
        return select_capture_model()

    def _interpret(self, reason: str, transcript: dict[str, Any]) -> InterpretedCriteria:
        if self._model_factory is not None:
            return CaptureInterpreter(self._model()).interpret(reason, transcript)
        response = _bounded_model_call("interpret", {"reason": reason, "transcript": transcript})
        return InterpretedCriteria(
            CaptureCriteria.model_validate(response["criteria"]), response["model"], response["prompt_sha256"],
        )

    def _judge(self, observation: OutcomeObservation):
        if self._model_factory is not None:
            return OutcomeJudge(self._model(), max_tokens=CAPTURE_OUTPUT_TOKENS).judge_with_provenance(observation)
        response = _bounded_model_call("judge", {"observation": observation.model_dump(mode="json")})
        from edd.outcome_judge import JudgedOutcome, OutcomeVerdict
        return JudgedOutcome(
            OutcomeVerdict.model_validate(response["verdict"]), response["model"], response["prompt_sha256"],
        )

    def _call(self, capture_id: str, fn: Callable[[], Any]) -> Any:
        queue_id = f"capture:{capture_id}"
        try:
            deadline = monotonic() + CAPTURE_QUEUE_HOLD_TIMEOUT_S
            while not self._stopping.is_set():
                try:
                    with THREAD_QUEUE.acquire(
                        queue_id, user_priority=False,
                        wait_timeout_s=min(CAPTURE_QUEUE_POLL_TIMEOUT_S, max(0.01, deadline - monotonic())),
                        hold_timeout_s=CAPTURE_QUEUE_HOLD_TIMEOUT_S,
                        quantum_s=CAPTURE_QUEUE_HOLD_TIMEOUT_S,
                    ):
                        if self._stopping.is_set():
                            raise RuntimeError("capture worker is stopping")
                        return fn()
                except QueueWaitTimeout:
                    if monotonic() >= deadline:
                        raise
            raise RuntimeError("capture worker is stopping")
        finally:
            THREAD_QUEUE.pop_hold(queue_id)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                return
            thread_id, capture_id = item
            with self._seen_lock:
                self._seen.discard(item)
            try:
                self._process(thread_id, capture_id)
            except Exception:
                # The page receives the safe stored status below; do not log the
                # private transcript/reason through an exception repr.
                try:
                    self.store.update_result(thread_id, capture_id, {"status": "failed", "error": "capture processing failed"})
                except Exception:
                    pass

    def _process(self, thread_id: str, capture_id: str) -> None:
        current = self.store.get_for_thread(thread_id, capture_id)
        result = current["result"]
        if result["status"] in {"pass", "partial", "fail", "failed", "needs_shorter_scope", "needs_clarification"}:
            return
        transcript = current["transcript"]
        if transcript is None:
            self.store.update_result(thread_id, capture_id, {"status": "failed", "error": "capture has no transcript"})
            return
        if result["status"] in {"queued", "interpreting"} and "criteria" not in result:
            self.store.update_result(thread_id, capture_id, {"status": "interpreting"})
            interpreted = self._call(capture_id, lambda: self._interpret(current["request"]["reason"], transcript))
            if interpreted.criteria.status == "needs_clarification":
                self.store.update_result(thread_id, capture_id, {
                    "status": "needs_clarification", "clarification": interpreted.criteria.clarification,
                    "criteria": interpreted.criteria.model_dump(mode="json"),
                    "interpreter": {"model": interpreted.model, "prompt_sha256": interpreted.prompt_sha256},
                })
                return
            observation = build_observation(interpreted.criteria, transcript)
            result = self.store.update_result(thread_id, capture_id, {
                "status": "judging", "criteria": interpreted.criteria.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
                "interpreter": {"model": interpreted.model, "prompt_sha256": interpreted.prompt_sha256},
            })["result"]
        observation = OutcomeObservation.model_validate(result["observation"])
        judged = self._call(capture_id, lambda: self._judge(observation))
        self.store.update_result(thread_id, capture_id, {
            "status": judged.verdict.overall, "criteria": result["criteria"],
            "observation": result["observation"], "interpreter": result["interpreter"],
            "judge": {"model": judged.model, "prompt_sha256": judged.prompt_sha256},
            "verdict": judged.verdict.model_dump(mode="json"),
        })
