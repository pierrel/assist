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


_FILENAME = "pi-conversation.jsonl"
_VERSION = 1
_MAX_BYTES = 2 * 1024 * 1024
_MAX_EVENT_BYTES = 128 * 1024
_MAX_RUN_ID_BYTES = 256


class PiConversationError(ValueError):
    """The Pi transcript is malformed, unsafe, or conflicts with a Run."""


@dataclass(frozen=True)
class PiMessage:
    """One visible Pi message, attributable to the Run that produced it."""

    run_id: str
    role: Literal["user", "assistant"]
    text: str


class PiConversationStore:
    """Logically append-only transcript with Run-keyed idempotency in one web process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @staticmethod
    def _path(thread_dir: str | Path) -> Path:
        return Path(thread_dir) / _FILENAME

    @staticmethod
    def _validate(value: object) -> PiMessage:
        if not isinstance(value, dict) or set(value) != {"version", "ts", "run_id", "role", "text"}:
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
        return PiMessage(value["run_id"], value["role"], value["text"])

    @staticmethod
    def _read(path: Path) -> tuple[bytes, list[PiMessage]]:
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
        messages: list[PiMessage] = []
        seen: set[tuple[str, str]] = set()
        user_runs: set[str] = set()
        for line in raw.splitlines():
            try:
                message = PiConversationStore._validate(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError, PiConversationError) as error:
                raise PiConversationError("Pi transcript is malformed") from error
            key = (message.run_id, message.role)
            if key in seen or (message.role == "assistant" and message.run_id not in user_runs):
                raise PiConversationError("Pi transcript has an invalid Run sequence")
            seen.add(key)
            if message.role == "user":
                user_runs.add(message.run_id)
            messages.append(message)
        return raw, messages

    def get_messages(self, thread_dir: str | Path) -> list[PiMessage]:
        """Read the visible transcript; no partial/corrupt history is accepted."""
        with self._lock:
            _, messages = self._read(self._path(thread_dir))
            return messages

    def append(self, thread_dir: str | Path, run_id: str,
               role: Literal["user", "assistant"], text: str) -> PiMessage:
        """Append one visible event, or confirm the exact idempotent prior event."""
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
        path = self._path(thread_dir)
        message = PiMessage(run_id, role, text)
        with self._lock:
            raw, previous = self._read(path)
            for prior in previous:
                if prior.run_id == run_id and prior.role == role:
                    if prior == message:
                        return prior
                    raise PiConversationError("Pi Run already has a conflicting transcript event")
            if role == "assistant" and not any(
                    prior.run_id == run_id and prior.role == "user" for prior in previous):
                raise PiConversationError("Pi assistant event has no user event for its Run")
            payload = json.dumps({
                "version": _VERSION, "ts": datetime.now(UTC).isoformat(),
                "run_id": run_id, "role": role, "text": text,
            }, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
            if len(raw) + len(payload) > _MAX_BYTES:
                raise PiConversationError("Pi transcript is unsafe or full")
            descriptor: int | None = None
            temporary: str | None = None
            try:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=".pi-conversation-", dir=path.parent)
                os.fchmod(descriptor, 0o600)
                output = memoryview(raw + payload)
                while output:
                    written = os.write(descriptor, output)
                    if written <= 0:
                        raise OSError("Pi transcript write made no progress")
                    output = output[written:]
                os.fsync(descriptor)
                os.replace(temporary, path)
                temporary = None
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
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
        return message
