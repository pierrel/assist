"""Host-owned, idempotent transcript for Pi preview threads.

Pi workers receive a bounded copy of this transcript and never write it.  The
web host appends the user event only after its durable Run is claimed, then an
assistant event only after the worker's current generation is still authorized.
"""
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

from assist.pi_skills import PiLoadedSkill, PiSkillError


_FILENAME = "pi-conversation.jsonl"
_VERSION = 1
_MAX_BYTES = 2 * 1024 * 1024
_MAX_EVENT_BYTES = 128 * 1024
_MAX_RUN_ID_BYTES = 256
PI_HISTORY_LIMIT = 32


class PiConversationError(ValueError):
    """The Pi transcript is malformed, unsafe, or conflicts with a Run."""


@dataclass(frozen=True)
class PiMessage:
    """One visible Pi message, attributable to the Run that produced it."""

    run_id: str
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class PiConversationContext:
    """The visible history and complete skill records for one fresh Pi worker."""

    messages: tuple[PiMessage, ...]
    loaded_skills: tuple[PiLoadedSkill, ...]
    evicted_skills: tuple[PiLoadedSkill, ...]


@dataclass(frozen=True)
class _StampedEvent:
    """One durable value with the timestamp assigned when it was committed."""

    value: PiMessage | PiLoadedSkill
    timestamp: str


class PiConversationStore:
    """Compactable host transcript with Run-keyed idempotency in one web process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _path(thread_dir: str | Path) -> Path:
        return Path(thread_dir) / _FILENAME

    @staticmethod
    def _validate(value: object) -> _StampedEvent:
        if not isinstance(value, dict):
            raise PiConversationError("Pi transcript event has an invalid shape")
        if value.get("role") == "skill":
            if set(value) != {"version", "ts", "run_id", "role", "name", "body", "body_sha256", "declared_tools"}:
                raise PiConversationError("Pi transcript event has an invalid shape")
            if value["version"] != _VERSION or not isinstance(value["ts"], str):
                raise PiConversationError("Pi transcript event has invalid fields")
            try:
                if len(value["ts"].encode("utf-8")) > 128:
                    raise PiConversationError("Pi transcript event exceeds its bound")
            except UnicodeEncodeError as error:
                raise PiConversationError("Pi transcript event is not valid UTF-8") from error
            if not isinstance(value["declared_tools"], list):
                raise PiConversationError("Pi transcript skill is invalid")
            try:
                loaded = PiLoadedSkill(
                    value["name"], value["body"], value["body_sha256"],
                    tuple(value["declared_tools"]), value["run_id"])
            except (PiSkillError, TypeError) as error:
                raise PiConversationError("Pi transcript skill is invalid") from error
            return _StampedEvent(loaded, value["ts"])
        if set(value) != {"version", "ts", "run_id", "role", "text"}:
            raise PiConversationError("Pi transcript event has an invalid shape")
        if (value["version"] != _VERSION or not isinstance(value["ts"], str)
                or not isinstance(value["run_id"], str) or not value["run_id"]
                or not isinstance(value["role"], str)
                or value["role"] not in {"user", "assistant"}
                or not isinstance(value["text"], str)):
            raise PiConversationError("Pi transcript event has invalid fields")
        try:
            if (len(value["ts"].encode("utf-8")) > 128
                    or len(value["run_id"].encode("utf-8")) > _MAX_RUN_ID_BYTES
                    or len(value["text"].encode("utf-8")) > _MAX_EVENT_BYTES):
                raise PiConversationError("Pi transcript event exceeds its bound")
        except UnicodeEncodeError as error:
            raise PiConversationError("Pi transcript event is not valid UTF-8") from error
        return _StampedEvent(PiMessage(value["run_id"], value["role"], value["text"]), value["ts"])

    @staticmethod
    def _read(path: Path) -> tuple[bytes, list[_StampedEvent]]:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return b"", []
        except OSError as error:
            raise PiConversationError("Pi transcript is unreadable") from error
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o022 or metadata.st_size > _MAX_BYTES):
                raise PiConversationError("Pi transcript is unsafe")
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
                raise PiConversationError("Pi transcript exceeds its bound")
        except OSError as error:
            raise PiConversationError("Pi transcript is unreadable") from error
        finally:
            os.close(descriptor)
        events: list[_StampedEvent] = []
        seen: set[tuple[str, str]] = set()
        user_runs: set[str] = set()
        skill_names: set[str] = set()
        previous_assistant_run: str | None = None
        for line in raw.splitlines():
            try:
                message = PiConversationStore._validate(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError, PiConversationError) as error:
                raise PiConversationError("Pi transcript is malformed") from error
            event = message.value
            if isinstance(event, PiLoadedSkill):
                if event.run_id != previous_assistant_run or event.name in skill_names:
                    raise PiConversationError("Pi transcript has an invalid Run sequence")
                skill_names.add(event.name)
                events.append(message)
                continue
            key = (event.run_id, event.role)
            if key in seen or (event.role == "assistant" and event.run_id not in user_runs):
                raise PiConversationError("Pi transcript has an invalid Run sequence")
            seen.add(key)
            if event.role == "user":
                user_runs.add(event.run_id)
                previous_assistant_run = None
            else:
                previous_assistant_run = event.run_id
            events.append(message)
        return raw, events

    def get_messages(self, thread_dir: str | Path) -> list[PiMessage]:
        """Read the visible transcript; no partial/corrupt history is accepted."""
        with self._lock:
            _, events = self._read(self._path(thread_dir))
            return [event.value for event in events if isinstance(event.value, PiMessage)]

    @staticmethod
    def _message(run_id: str, role: Literal["user", "assistant"], text: str) -> PiMessage:
        if (not isinstance(run_id, str) or not run_id
                or not isinstance(role, str) or role not in {"user", "assistant"}):
            raise PiConversationError("Pi transcript event identity is invalid")
        if not isinstance(text, str):
            raise PiConversationError("Pi transcript event exceeds its bound")
        try:
            run_id_size = len(run_id.encode("utf-8"))
            text_size = len(text.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise PiConversationError("Pi transcript event is not valid UTF-8") from error
        if run_id_size > _MAX_RUN_ID_BYTES or text_size > _MAX_EVENT_BYTES:
            raise PiConversationError("Pi transcript event exceeds its bound")
        return PiMessage(run_id, role, text)

    @staticmethod
    def _stamp(value: PiMessage | PiLoadedSkill) -> _StampedEvent:
        return _StampedEvent(value, datetime.now(UTC).isoformat())

    @staticmethod
    def _raw(events: list[_StampedEvent]) -> bytes:
        records: list[dict[str, object]] = []
        for event in events:
            value = event.value
            if isinstance(value, PiMessage):
                records.append({"version": _VERSION, "ts": event.timestamp,
                                "run_id": value.run_id, "role": value.role,
                                "text": value.text})
            else:
                records.append({"version": _VERSION, "ts": event.timestamp,
                                "run_id": value.run_id, "role": "skill",
                                "name": value.name, "body": value.body,
                                "body_sha256": value.body_sha256,
                                "declared_tools": list(value.declared_tools)})
        raw = b"".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                       for record in records)
        if len(raw) > _MAX_BYTES:
            raise PiConversationError("Pi transcript is unsafe or full")
        return raw

    @staticmethod
    def _write(path: Path, events: list[_StampedEvent]) -> None:
        raw = PiConversationStore._raw(events)
        descriptor: int | None = None
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".pi-conversation-", dir=path.parent)
            os.fchmod(descriptor, 0o600)
            output = memoryview(raw)
            while output:
                written = os.write(descriptor, output)
                if written <= 0:
                    raise OSError("Pi transcript write made no progress")
                output = output[written:]
            os.fsync(descriptor)
            os.replace(temporary, path)
            temporary = None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            raise PiConversationError("Pi transcript write failed") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def append(self, thread_dir: str | Path, run_id: str,
               role: Literal["user", "assistant"], text: str) -> PiMessage:
        """Append one visible event, or confirm the exact idempotent prior event."""
        message = self._message(run_id, role, text)
        path = self._path(thread_dir)
        with self._lock:
            _, events = self._read(path)
            messages = [event.value for event in events if isinstance(event.value, PiMessage)]
            for prior in messages:
                if prior.run_id == run_id and prior.role == role:
                    if prior == message:
                        return prior
                    raise PiConversationError("Pi Run already has a conflicting transcript event")
            if role == "assistant" and not any(
                    prior.run_id == run_id and prior.role == "user" for prior in messages):
                raise PiConversationError("Pi assistant event has no user event for its Run")
            self._write(path, [*events, self._stamp(message)])
        return message

    def context(self, thread_dir: str | Path, *, max_messages: int,
                exclude_run_id: str | None = None) -> PiConversationContext:
        """Return skill records only when their completed Run remains in history."""
        if not isinstance(max_messages, int) or isinstance(max_messages, bool) or max_messages < 1:
            raise PiConversationError("Pi history limit is invalid")
        with self._lock:
            _, events = self._read(self._path(thread_dir))
        messages = tuple(event.value for event in events
                         if isinstance(event.value, PiMessage) and event.value.run_id != exclude_run_id)[-max_messages:]
        roles_by_run: dict[str, set[str]] = {}
        for message in messages:
            roles_by_run.setdefault(message.run_id, set()).add(message.role)
        complete_run_ids = {run_id for run_id, roles in roles_by_run.items()
                            if roles == {"user", "assistant"}}
        all_skills = tuple(event.value for event in events if isinstance(event.value, PiLoadedSkill))
        skills = tuple(skill for skill in all_skills if skill.run_id in complete_run_ids)
        evicted = tuple(skill for skill in all_skills if skill.run_id not in complete_run_ids)
        return PiConversationContext(messages, skills, evicted)

    def completed_reply(self, thread_dir: str | Path, run_id: str) -> PiMessage | None:
        """Return the durable Pi completion witness for one Run, if present."""
        with self._lock:
            _, events = self._read(self._path(thread_dir))
        return next((event.value for event in events if isinstance(event.value, PiMessage)
                     and event.value.run_id == run_id and event.value.role == "assistant"), None)

    def complete(self, thread_dir: str | Path, run_id: str, reply: str,
                 promoted: tuple[PiLoadedSkill, ...],
                 evicted: tuple[PiLoadedSkill, ...]) -> PiMessage:
        """Atomically record one reply, replace promoted names, and evict exact records."""
        message = self._message(run_id, "assistant", reply)
        if len({skill.name for skill in promoted}) != len(promoted):
            raise PiConversationError("Pi promoted skills are not unique")
        if any(skill.run_id != run_id for skill in promoted):
            raise PiConversationError("Pi promoted skill Run is invalid")
        path = self._path(thread_dir)
        with self._lock:
            _, events = self._read(path)
            messages = [event.value for event in events if isinstance(event.value, PiMessage)]
            existing = next((event for event in messages
                             if event.run_id == run_id and event.role == "assistant"), None)
            if existing is not None:
                if existing == message:
                    return existing
                raise PiConversationError("Pi Run already has a conflicting transcript event")
            if not any(event.run_id == run_id and event.role == "user" for event in messages):
                raise PiConversationError("Pi assistant event has no user event for its Run")
            evicted_set = set(evicted)
            promoted_names = {skill.name for skill in promoted}
            kept = [event for event in events if not isinstance(event.value, PiLoadedSkill)
                    or (event.value not in evicted_set and event.value.name not in promoted_names)]
            additions = [self._stamp(message),
                         *(self._stamp(skill) for skill in sorted(promoted, key=lambda skill: skill.name))]
            self._write(path, [*kept, *additions])
        return message
