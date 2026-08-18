"""Host-owned durable Pi history, projection state, and crash recovery."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from assist.pi_skills import PiLoadedSkill, PiSkillError


_LEGACY = "pi-conversation.jsonl"
_DIRECTORY = "pi-history"
_MANIFEST = "manifest.json"
_STATE = "state.json"
_PENDING = "pending.json"
_VERSION = 1
_MAX_EVENT_BYTES = 128 * 1024
_MAX_RUN_ID_BYTES = 256
_MAX_SEGMENT_BYTES = 256 * 1024
_MAX_RAW_BYTES = 64 * 1024 * 1024
_MAX_SEGMENTS = _MAX_RAW_BYTES // _MAX_SEGMENT_BYTES
_MAX_STATE_BYTES = 640 * 1024
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
PI_HISTORY_LIMIT = 32


class PiConversationError(ValueError):
    """The Pi history is malformed, unsafe, or conflicts with a Run."""


@dataclass(frozen=True)
class PiMessage:
    run_id: str
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class PiHistorySummary:
    body: str
    body_sha256: str
    last_run_id: str

    def __post_init__(self) -> None:
        _text(self.body, 32 * 1024, "Pi history summary")
        _identifier(self.last_run_id, "Pi history summary")
        if hashlib.sha256(self.body.encode("utf-8")).hexdigest() != self.body_sha256:
            raise PiConversationError("Pi history summary digest is invalid")


@dataclass(frozen=True)
class PiCompactionCandidate:
    body: str
    last_run_id: str


@dataclass(frozen=True)
class PiConversationContext:
    messages: tuple[PiMessage, ...]
    loaded_skills: tuple[PiLoadedSkill, ...]
    evicted_skills: tuple[PiLoadedSkill, ...]
    summary: PiHistorySummary | None = None
    compaction_candidate: PiCompactionCandidate | None = None


@dataclass(frozen=True)
class _StampedMessage:
    value: PiMessage
    timestamp: str


def _text(value: object, limit: int, subject: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PiConversationError(f"{subject} is invalid")
    try:
        if len(value.encode("utf-8")) > limit:
            raise PiConversationError(f"{subject} exceeds its bound")
    except UnicodeEncodeError as error:
        raise PiConversationError(f"{subject} is not valid UTF-8") from error
    return value


def _identifier(value: object, subject: str) -> str:
    return _text(value, _MAX_RUN_ID_BYTES, subject)


def _skill_dict(skill: PiLoadedSkill) -> dict[str, object]:
    return {"name": skill.name, "body": skill.body, "body_sha256": skill.body_sha256,
            "declared_tools": list(skill.declared_tools), "run_id": skill.run_id}


def _summary_dict(summary: PiHistorySummary | None) -> dict[str, object] | None:
    if summary is None:
        return None
    return {"body": summary.body, "body_sha256": summary.body_sha256,
            "last_run_id": summary.last_run_id}


class PiConversationStore:
    """Append visible history while atomically replacing its small projection state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _legacy(thread_dir: str | Path) -> Path:
        return Path(thread_dir) / _LEGACY

    @staticmethod
    def _root(thread_dir: str | Path) -> Path:
        return Path(thread_dir) / _DIRECTORY

    @classmethod
    def _path(cls, thread_dir: str | Path, name: str) -> Path:
        return cls._root(thread_dir) / name

    @staticmethod
    def _safe_read(path: Path, limit: int, subject: str, *, missing: bytes | None = None) -> bytes:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            if missing is not None:
                return missing
            raise PiConversationError(f"{subject} is missing")
        except OSError as error:
            raise PiConversationError(f"{subject} is unreadable") from error
        try:
            metadata = os.fstat(descriptor)
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o022 or metadata.st_size > limit):
                raise PiConversationError(f"{subject} is unsafe")
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > limit:
                raise PiConversationError(f"{subject} exceeds its bound")
            return raw
        except OSError as error:
            raise PiConversationError(f"{subject} is unreadable") from error
        finally:
            os.close(descriptor)

    @staticmethod
    def _atomic_write(path: Path, raw: bytes, subject: str) -> None:
        descriptor: int | None = None
        temporary: str | None = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
            os.fchmod(descriptor, 0o600)
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.replace(temporary, path)
            temporary = None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            raise PiConversationError(f"{subject} write failed") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _unlink(path: Path, subject: str) -> None:
        try:
            path.unlink()
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            raise PiConversationError(f"{subject} cleanup failed") from error

    @staticmethod
    def _message(run_id: str, role: Literal["user", "assistant"], text: str) -> PiMessage:
        _identifier(run_id, "Pi history Run identity")
        if role not in {"user", "assistant"}:
            raise PiConversationError("Pi history role is invalid")
        _text(text, _MAX_EVENT_BYTES, "Pi history message")
        return PiMessage(run_id, role, text)

    @staticmethod
    def _parse_message(value: object) -> _StampedMessage:
        if not isinstance(value, dict) or set(value) != {"version", "ts", "run_id", "role", "text"}:
            raise PiConversationError("Pi history event has an invalid shape")
        if value["version"] != _VERSION:
            raise PiConversationError("Pi history event has an invalid version")
        _text(value["ts"], 128, "Pi history timestamp")
        if not isinstance(value["role"], str) or value["role"] not in {"user", "assistant"}:
            raise PiConversationError("Pi history role is invalid")
        return _StampedMessage(PiConversationStore._message(value["run_id"], value["role"], value["text"]), value["ts"])

    @staticmethod
    def _raw_message(event: _StampedMessage) -> bytes:
        value = event.value
        return json.dumps({"version": _VERSION, "ts": event.timestamp, "run_id": value.run_id,
                           "role": value.role, "text": value.text}, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8") + b"\n"

    @staticmethod
    def _stamp(message: PiMessage) -> _StampedMessage:
        return _StampedMessage(message, datetime.now(UTC).isoformat())

    @staticmethod
    def _decode_json(raw: bytes, subject: str) -> object:
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PiConversationError(f"{subject} is malformed") from error

    def _ensure_root(self, thread_dir: str | Path) -> None:
        root = self._root(thread_dir)
        if root.exists():
            metadata = os.lstat(root)
            if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077):
                raise PiConversationError("Pi history directory is unsafe")
            return
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            self._ensure_root(thread_dir)
        except OSError as error:
            raise PiConversationError("Pi history directory could not be created") from error

    def _manifest(self, thread_dir: str | Path) -> dict[str, object]:
        value = self._decode_json(self._safe_read(self._path(thread_dir, _MANIFEST), 64 * 1024,
                                                   "Pi history manifest"), "Pi history manifest")
        if (not isinstance(value, dict) or set(value) != {"version", "generation", "segments"}
                or value["version"] != _VERSION or not isinstance(value["generation"], int)
                or isinstance(value["generation"], bool) or value["generation"] < 0
                or not isinstance(value["segments"], list) or len(value["segments"]) > _MAX_SEGMENTS):
            raise PiConversationError("Pi history manifest is invalid")
        names = value["segments"]
        if (any(not isinstance(name, str) or not name.startswith("segment-") or not name.endswith(".jsonl")
                or len(name) != len("segment-") + 16 + len(".jsonl") for name in names)
                or len(set(names)) != len(names) or names != sorted(names)):
            raise PiConversationError("Pi history manifest is invalid")
        return value

    def _state(self, thread_dir: str | Path) -> tuple[bytes, PiHistorySummary | None, tuple[PiLoadedSkill, ...]]:
        raw = self._safe_read(self._path(thread_dir, _STATE), _MAX_STATE_BYTES, "Pi history state")
        summary, skills = self._parse_state(raw)
        return raw, summary, skills

    @staticmethod
    def _parse_state(raw: bytes) -> tuple[PiHistorySummary | None, tuple[PiLoadedSkill, ...]]:
        if len(raw) > _MAX_STATE_BYTES:
            raise PiConversationError("Pi history state exceeds its bound")
        value = PiConversationStore._decode_json(raw, "Pi history state")
        if not isinstance(value, dict) or set(value) != {"version", "summary", "skills"} or value["version"] != _VERSION:
            raise PiConversationError("Pi history state is invalid")
        summary_value = value["summary"]
        if summary_value is None:
            summary = None
        elif isinstance(summary_value, dict) and set(summary_value) == {"body", "body_sha256", "last_run_id"}:
            try:
                summary = PiHistorySummary(summary_value["body"], summary_value["body_sha256"], summary_value["last_run_id"])
            except (PiConversationError, TypeError) as error:
                raise PiConversationError("Pi history state is invalid") from error
        else:
            raise PiConversationError("Pi history state is invalid")
        if not isinstance(value["skills"], list) or len(value["skills"]) > 5:
            raise PiConversationError("Pi history state is invalid")
        skills: list[PiLoadedSkill] = []
        for item in value["skills"]:
            if not isinstance(item, dict) or set(item) != {"name", "body", "body_sha256", "declared_tools", "run_id"} or not isinstance(item["declared_tools"], list):
                raise PiConversationError("Pi history state is invalid")
            try:
                skills.append(PiLoadedSkill(item["name"], item["body"], item["body_sha256"], tuple(item["declared_tools"]), item["run_id"]))
            except (PiSkillError, TypeError) as error:
                raise PiConversationError("Pi history state is invalid") from error
        if len({skill.name for skill in skills}) != len(skills):
            raise PiConversationError("Pi history state is invalid")
        return summary, tuple(skills)

    @staticmethod
    def _state_raw(summary: PiHistorySummary | None, skills: tuple[PiLoadedSkill, ...]) -> bytes:
        raw = json.dumps({"version": _VERSION, "summary": _summary_dict(summary),
                          "skills": [_skill_dict(skill) for skill in skills]}, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
        if len(raw) > _MAX_STATE_BYTES:
            raise PiConversationError("Pi history state exceeds its bound")
        return raw

    def _events(self, thread_dir: str | Path, manifest: dict[str, object]) -> list[_StampedMessage]:
        events: list[_StampedMessage] = []
        total = 0
        for name in manifest["segments"]:
            assert isinstance(name, str)
            raw = self._safe_read(self._path(thread_dir, name), _MAX_SEGMENT_BYTES, "Pi history segment")
            total += len(raw)
            if total > _MAX_RAW_BYTES:
                raise PiConversationError("Pi history exceeds its bound")
            for line in raw.splitlines():
                events.append(self._parse_message(self._decode_json(line, "Pi history segment")))
        self._validate_events(events)
        return events

    @staticmethod
    def _validate_events(events: list[_StampedMessage]) -> None:
        seen: set[tuple[str, str]] = set()
        users: set[str] = set()
        for event in events:
            key = (event.value.run_id, event.value.role)
            if key in seen or (event.value.role == "assistant" and event.value.run_id not in users):
                raise PiConversationError("Pi history has an invalid Run sequence")
            seen.add(key)
            if event.value.role == "user":
                users.add(event.value.run_id)

    def _write_manifest(self, thread_dir: str | Path, manifest: dict[str, object]) -> None:
        self._atomic_write(self._path(thread_dir, _MANIFEST), json.dumps(manifest, separators=(",", ":")).encode(), "Pi history manifest")

    def _append_raw(self, thread_dir: str | Path, message: PiMessage) -> None:
        manifest = self._manifest(thread_dir)
        line = self._raw_message(self._stamp(message))
        names = list(manifest["segments"])
        target: Path | None = self._path(thread_dir, names[-1]) if names else None
        if target is None or len(self._safe_read(target, _MAX_SEGMENT_BYTES, "Pi history segment")) + len(line) > _MAX_SEGMENT_BYTES:
            if len(names) >= _MAX_SEGMENTS:
                raise PiConversationError("Pi history exceeds its bound")
            name = f"segment-{len(names) + 1:016d}.jsonl"
            self._atomic_write(self._path(thread_dir, name), line, "Pi history segment")
            names.append(name)
        else:
            self._atomic_write(target, self._safe_read(target, _MAX_SEGMENT_BYTES,
                                                        "Pi history segment") + line,
                               "Pi history segment")
        self._write_manifest(thread_dir, {"version": _VERSION, "generation": manifest["generation"] + 1, "segments": names})

    def _migrate(self, thread_dir: str | Path) -> None:
        root = self._root(thread_dir)
        legacy = self._legacy(thread_dir)
        if root.exists():
            self._ensure_root(thread_dir)
            if not legacy.exists():
                try:
                    self._manifest(thread_dir)
                    self._state(thread_dir)
                    return
                except PiConversationError:
                    # No visible append can exist without an already-published
                    # state.  This is only an interrupted empty initialization.
                    try:
                        manifest = self._manifest(thread_dir)
                    except PiConversationError:
                        manifest = None
                    if manifest is None or not manifest["segments"]:
                        self._discard_incomplete_root(thread_dir)
                    else:
                        raise
            try:
                self._manifest(thread_dir)
                self._state(thread_dir)
                return
            except PiConversationError:
                self._discard_incomplete_root(thread_dir)
        if not legacy.exists():
            self._ensure_root(thread_dir)
            self._write_manifest(thread_dir, {"version": _VERSION, "generation": 0, "segments": []})
            self._atomic_write(self._path(thread_dir, _STATE), self._state_raw(None, ()), "Pi history state")
            return
        raw = self._safe_read(legacy, 2 * 1024 * 1024, "Pi legacy transcript")
        old_events: list[_StampedMessage] = []
        old_skills: list[PiLoadedSkill] = []
        previous_assistant: str | None = None
        skill_names: set[str] = set()
        for line in raw.splitlines():
            value = self._decode_json(line, "Pi legacy transcript")
            if isinstance(value, dict) and value.get("role") == "skill":
                if (set(value) != {"version", "ts", "run_id", "role", "name", "body", "body_sha256", "declared_tools"}
                        or value.get("version") != _VERSION or not isinstance(value.get("declared_tools"), list)):
                    raise PiConversationError("Pi legacy transcript is malformed")
                try:
                    _text(value["ts"], 128, "Pi history timestamp")
                    skill = PiLoadedSkill(value["name"], value["body"], value["body_sha256"], tuple(value["declared_tools"]), value["run_id"])
                except (KeyError, PiSkillError, TypeError) as error:
                    raise PiConversationError("Pi legacy transcript is malformed") from error
                if skill.run_id != previous_assistant or skill.name in skill_names:
                    raise PiConversationError("Pi legacy transcript has an invalid Run sequence")
                old_skills.append(skill); skill_names.add(skill.name)
            else:
                event = self._parse_message(value)
                old_events.append(event)
                previous_assistant = event.value.run_id if event.value.role == "assistant" else None
        self._validate_events(old_events)
        self._ensure_root(thread_dir)
        names: list[str] = []
        segment = bytearray()
        for event in old_events:
            line = self._raw_message(event)
            if segment and len(segment) + len(line) > _MAX_SEGMENT_BYTES:
                name = f"segment-{len(names) + 1:016d}.jsonl"
                self._atomic_write(self._path(thread_dir, name), bytes(segment), "Pi history segment")
                names.append(name); segment.clear()
            segment.extend(line)
        if segment:
            name = f"segment-{len(names) + 1:016d}.jsonl"
            self._atomic_write(self._path(thread_dir, name), bytes(segment), "Pi history segment")
            names.append(name)
        messages = [event.value for event in old_events][-PI_HISTORY_LIMIT:]
        complete = {message.run_id for message in messages if message.role == "user"} & {message.run_id for message in messages if message.role == "assistant"}
        skills = tuple(skill for skill in old_skills if skill.run_id in complete)
        self._write_manifest(thread_dir, {"version": _VERSION, "generation": 0, "segments": names})
        self._atomic_write(self._path(thread_dir, _STATE), self._state_raw(None, skills), "Pi history state")
        self._unlink(legacy, "Pi legacy transcript")

    def _discard_incomplete_root(self, thread_dir: str | Path) -> None:
        """Discard only an incomplete migration while its legacy source survives."""
        root = self._root(thread_dir)
        self._ensure_root(thread_dir)
        try:
            entries = list(root.iterdir())
            for entry in entries:
                metadata = os.lstat(entry)
                allowed = (entry.name in {_MANIFEST, _STATE, _PENDING}
                           or (entry.name.startswith("segment-") and entry.name.endswith(".jsonl"))
                           or entry.name.startswith("."))
                if (not allowed or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                        or metadata.st_mode & 0o022):
                    raise PiConversationError("Pi history directory is unsafe")
                entry.unlink()
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            root.rmdir()
            parent = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except OSError as error:
            raise PiConversationError("Pi incomplete history cleanup failed") from error

    def _recover(self, thread_dir: str | Path) -> None:
        pending = self._path(thread_dir, _PENDING)
        raw = self._safe_read(pending, _MAX_RECEIPT_BYTES, "Pi history receipt", missing=b"")
        if not raw:
            return
        value = self._decode_json(raw, "Pi history receipt")
        if not isinstance(value, dict) or set(value) != {"run_id", "assistant", "old", "new"}:
            raise PiConversationError("Pi history receipt is invalid")
        try:
            old, new = (base64.b64decode(value[key], validate=True) for key in ("old", "new"))
            assistant = self._parse_message(value["assistant"])
        except (ValueError, TypeError, PiConversationError) as error:
            raise PiConversationError("Pi history receipt is invalid") from error
        if assistant.value.role != "assistant" or assistant.value.run_id != value["run_id"]:
            raise PiConversationError("Pi history receipt is invalid")
        try:
            old_projection = self._parse_state(old)
            new_projection = self._parse_state(new)
        except PiConversationError as error:
            raise PiConversationError("Pi history receipt is invalid") from error
        current, _summary, _skills = self._state(thread_dir)
        events = self._events(thread_dir, self._manifest(thread_dir))
        self._validate_projection(*old_projection, events)
        self._validate_projection(*new_projection, events)
        present = any(event.value == assistant.value for event in events)
        if present and current == new:
            pass
        elif present and current == old:
            self._atomic_write(self._path(thread_dir, _STATE), new, "Pi history state")
        elif not present and current == new:
            self._atomic_write(self._path(thread_dir, _STATE), old, "Pi history state")
        elif not present and current == old:
            pass
        else:
            raise PiConversationError("Pi history receipt conflicts with durable state")
        self._unlink(pending, "Pi history receipt")

    def _snapshot(self, thread_dir: str | Path) -> tuple[list[_StampedMessage], PiHistorySummary | None, tuple[PiLoadedSkill, ...]]:
        self._migrate(thread_dir)
        self._recover(thread_dir)
        raw, summary, skills = self._state(thread_dir)
        del raw
        events = self._events(thread_dir, self._manifest(thread_dir))
        self._validate_projection(summary, skills, events)
        return events, summary, skills

    @staticmethod
    def _validate_projection(summary: PiHistorySummary | None, skills: tuple[PiLoadedSkill, ...],
                             events: list[_StampedMessage]) -> None:
        completed = {event.value.run_id for event in events if event.value.role == "user"} & {
            event.value.run_id for event in events if event.value.role == "assistant"}
        if summary is not None:
            positions = [index for index, event in enumerate(events)
                         if event.value.run_id == summary.last_run_id and event.value.role == "assistant"]
            if len(positions) != 1:
                raise PiConversationError("Pi history summary boundary is invalid")
        if any(skill.run_id not in completed for skill in skills):
            raise PiConversationError("Pi history skill projection is invalid")

    def get_messages(self, thread_dir: str | Path) -> list[PiMessage]:
        with self._lock:
            events, _summary, _skills = self._snapshot(thread_dir)
            return [event.value for event in events]

    def append(self, thread_dir: str | Path, run_id: str, role: Literal["user", "assistant"], text: str) -> PiMessage:
        message = self._message(run_id, role, text)
        with self._lock:
            events, _summary, _skills = self._snapshot(thread_dir)
            existing = next((event.value for event in events if event.value.run_id == run_id and event.value.role == role), None)
            if existing is not None:
                if existing == message:
                    return existing
                raise PiConversationError("Pi Run already has a conflicting history event")
            if role == "assistant" and not any(event.value.run_id == run_id and event.value.role == "user" for event in events):
                raise PiConversationError("Pi assistant event has no user event for its Run")
            self._append_raw(thread_dir, message)
        return message

    def context(self, thread_dir: str | Path, *, max_messages: int, exclude_run_id: str | None = None) -> PiConversationContext:
        if not isinstance(max_messages, int) or isinstance(max_messages, bool) or max_messages < 1:
            raise PiConversationError("Pi history limit is invalid")
        with self._lock:
            events, summary, all_skills = self._snapshot(thread_dir)
        all_messages = [event.value for event in events if event.value.run_id != exclude_run_id]
        selected = all_messages[-max_messages:]
        if selected and selected[0].role == "assistant":
            selected.pop(0)
        messages = tuple(selected)
        complete = {message.run_id for message in messages if message.role == "user"} & {message.run_id for message in messages if message.role == "assistant"}
        loaded = tuple(skill for skill in all_skills if skill.run_id in complete)
        evicted = tuple(skill for skill in all_skills if skill.run_id not in complete)
        start = 0
        if summary is not None:
            start = next(index + 1 for index, event in enumerate(events)
                         if event.value.run_id == summary.last_run_id and event.value.role == "assistant")
        first_tail = next((index for index, event in enumerate(events) if messages and event.value == messages[0]), len(events))
        candidate: PiCompactionCandidate | None = None
        prefix = events[start:first_tail]
        # A checkpoint may advance only if it consumes the entire omitted prefix.
        # Any interrupted Run there remains raw context rather than becoming an
        # invisible hole between the checkpoint and tail.
        if len(prefix) % 2 == 0:
            parts = ([f"Existing checkpoint:\n{summary.body}\n\nNew raw history:"]
                     if summary is not None else [])
            total = sum(len(part.encode("utf-8")) for part in parts)
            last_run_id: str | None = None
            complete = True
            for index in range(0, len(prefix), 2):
                user, assistant = prefix[index].value, prefix[index + 1].value
                if user.role != "user" or assistant.role != "assistant" or user.run_id != assistant.run_id:
                    complete = False
                    break
                part = f"Run {user.run_id} user:\n{user.text}\n\nRun {assistant.run_id} assistant:\n{assistant.text}"
                total += len(part.encode("utf-8")) + 2
                if total > 384 * 1024:
                    complete = False
                    break
                parts.append(part)
                last_run_id = assistant.run_id
            if complete and last_run_id is not None:
                candidate = PiCompactionCandidate("\n\n".join(parts), last_run_id)
        return PiConversationContext(messages, loaded, evicted, summary, candidate)

    def completed_reply(self, thread_dir: str | Path, run_id: str) -> PiMessage | None:
        with self._lock:
            events, _summary, _skills = self._snapshot(thread_dir)
        return next((event.value for event in events if event.value.run_id == run_id and event.value.role == "assistant"), None)

    def complete(self, thread_dir: str | Path, run_id: str, reply: str, promoted: tuple[PiLoadedSkill, ...], evicted: tuple[PiLoadedSkill, ...], summary: PiHistorySummary | None = None) -> PiMessage:
        message = self._message(run_id, "assistant", reply)
        if len({skill.name for skill in promoted}) != len(promoted) or any(skill.run_id != run_id for skill in promoted):
            raise PiConversationError("Pi promoted skills are invalid")
        with self._lock:
            events, old_summary, old_skills = self._snapshot(thread_dir)
            existing = next((event.value for event in events if event.value.run_id == run_id and event.value.role == "assistant"), None)
            if existing is not None:
                if existing == message:
                    return existing
                raise PiConversationError("Pi Run already has a conflicting history event")
            if not any(event.value.run_id == run_id and event.value.role == "user" for event in events):
                raise PiConversationError("Pi assistant event has no user event for its Run")
            evicted_set = set(evicted); promoted_names = {skill.name for skill in promoted}
            skills = tuple(skill for skill in old_skills if skill not in evicted_set and skill.name not in promoted_names) + tuple(sorted(promoted, key=lambda skill: skill.name))
            next_summary = summary if summary is not None else old_summary
            old = self._state_raw(old_summary, old_skills)
            new = self._state_raw(next_summary, skills)
            receipt = {"run_id": run_id, "assistant": json.loads(self._raw_message(self._stamp(message))),
                       "old": base64.b64encode(old).decode(), "new": base64.b64encode(new).decode()}
            receipt_raw = json.dumps(receipt, separators=(",", ":")).encode()
            if len(receipt_raw) > _MAX_RECEIPT_BYTES:
                raise PiConversationError("Pi history receipt exceeds its bound")
            self._atomic_write(self._path(thread_dir, _PENDING), receipt_raw, "Pi history receipt")
            self._append_raw(thread_dir, message)
            self._atomic_write(self._path(thread_dir, _STATE), new, "Pi history state")
            self._unlink(self._path(thread_dir, _PENDING), "Pi history receipt")
        return message
