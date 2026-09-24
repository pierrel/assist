"""Authenticated, structured Assist-web API for the EmacsOS phone client.

Browser routes intentionally remain form/HTML routes.  This module is the
separate machine interface: it authenticates every request, emits only visible
conversation data, and keeps all blocking thread/worktree operations off the
single FastAPI event loop.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import inspect
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import threading
import unicodedata
import uuid
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import convert_to_messages
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assist.domain_manager import current_branch
from assist.run_service import (AWAITING_APPROVAL_STATUSES, InvalidRunTransition,
                                ObservationToken, RunStoreUnavailable, TERMINAL_STATUSES)
from assist.thread import _messages_to_dicts
from assist.thread_engine import ThreadEngineError, read_thread_engine
from assist.visible_conversation import visible_records_from_dicts
from manage.web import state
from manage.web import threads
from manage.web.run_stream import RUN_STREAMS, encode_sse


PHONE_API_PREFIX = "/api/v1/phone"
PHONE_API_TOKEN_ENV = "ASSIST_PHONE_API_TOKEN"
MAX_BODY_BYTES = 66_000
MAX_MESSAGE_CHARS = 64_000
MAX_HISTORY_MESSAGES = 80
MAX_SNAPSHOT_MESSAGE_BYTES = 32 * 1024
MAX_SNAPSHOT_BYTES = 256 * 1024
MAX_DIFF_BYTES = 256 * 1024
MAX_FILES = 1_000
MAX_WORKSPACE_NODES = 2_000
MAX_HISTORY_SCAN_WRITES = 1_024
MAX_HISTORY_WRITE_BYTES = 1 * 1024 * 1024
MAX_HISTORY_SCAN_BYTES = 4 * 1024 * 1024
MAX_THREADS = 500
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_FILE_BYTES = 4 * 1024 * 1024
MAX_DESCRIPTION_CHARS = 120
MAX_PHONE_THREADS = 200
MAX_PHONE_PENDING_RUNS = 4
MAX_PHONE_INITIALIZATIONS = 1
MAX_PHONE_SSE_RECORD_BYTES = 48 * 1024
PHONE_SSE_SEND_TIMEOUT_SECONDS = 5
PHONE_SSE_LIFETIME_SECONDS = 30 * 60
_SSE_SLOTS = threading.BoundedSemaphore(4)
_ARCHIVE_SLOTS = threading.BoundedSemaphore(1)
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MESSAGE_SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,199}\Z")
_HISTORY_CURSOR_RE = re.compile(r"(?:m-[0-9a-f]{32}|c-[A-Za-z0-9_-]{24,240})\Z")
_SEALED_ID_RE = re.compile(r"[A-Za-z0-9_-]{24,240}\Z")
_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{15,127}\Z")
_FILE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z0-9][A-Za-z0-9._/-]{0,240}"
    r"\.[A-Za-z0-9]{1,16})(?![A-Za-z0-9_./-])"
)


class _BoundedSSEStreamingResponse(StreamingResponse):
    """Bound phone observation work to 30 minutes plus one final-send grace period."""

    def __init__(self, *args, on_close=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_close = on_close

    async def stream_response(self, send) -> None:
        async def bounded_send(message: dict[str, Any]) -> None:
            with anyio.fail_after(PHONE_SSE_SEND_TIMEOUT_SECONDS):
                await send(message)

        try:
            with anyio.fail_after(PHONE_SSE_LIFETIME_SECONDS
                                  + PHONE_SSE_SEND_TIMEOUT_SECONDS):
                await bounded_send({"type": "http.response.start", "status": self.status_code,
                                    "headers": self.raw_headers})
                async for chunk in self.body_iterator:
                    if not isinstance(chunk, bytes | memoryview):
                        chunk = chunk.encode(self.charset)
                    await bounded_send({"type": "http.response.body", "body": chunk,
                                        "more_body": True})
                await bounded_send({"type": "http.response.body", "body": b"",
                                    "more_body": False})
        except TimeoutError:
            return
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
            if self._on_close is not None:
                result = self._on_close()
                if inspect.isawaitable(result):
                    await result


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _CreateThread(_StrictModel):
    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]
    repo_key: str | None = None
    harness: str = "deepagents"


class _SendMessage(_StrictModel):
    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_CHARS)]


async def _validated_body(request: Request, model: type[_StrictModel]) -> _StrictModel:
    """Read one bounded strict JSON request without trusting Content-Length."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from error
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
        body.extend(chunk)
    try:
        return model.model_validate_json(bytes(body))
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error


def _require_id(value: str, label: str = "resource id") -> str:
    if value in {".", ".."} or not _ID_RE.fullmatch(value):
        raise HTTPException(status_code=422, detail=f"Invalid {label}")
    return value


def _request_key(request: Request) -> str:
    value = request.headers.get("idempotency-key", "")
    if not _KEY_RE.fullmatch(value):
        raise HTTPException(status_code=422, detail="Invalid Idempotency-Key")
    return value


def _authenticate(request: Request) -> None:
    """Require the dedicated phone token before any thread lookup."""
    configured = os.environ.get(PHONE_API_TOKEN_ENV)
    if not configured:
        raise HTTPException(status_code=503, detail="Phone API is not configured")
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, configured):
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Bearer"})


def _repo_key(repo: str) -> str:
    return hashlib.sha256(repo.encode("utf-8")).hexdigest()[:20]


def _stored_thread_title(tid: str) -> str:
    """Return a bounded stored title without ever creating a model request."""
    cached = state.DESCRIPTION_CACHE.get(tid)
    if isinstance(cached, str):
        return cached[:MAX_DESCRIPTION_CHARS]
    try:
        with open(os.path.join(_thread_dir(tid), "description.txt"), encoding="utf-8") as source:
            title = source.read(MAX_DESCRIPTION_CHARS + 1).strip()
    except OSError:
        title = ""
    if title:
        title = title[:MAX_DESCRIPTION_CHARS]
        state.DESCRIPTION_CACHE[tid] = title
        return title
    status = state._get_status(tid)
    if status.get("stage") in threads.BUSY_STAGES:
        pending = str(status.get("pending_message") or "").strip().splitlines()
        if pending:
            return pending[0][:MAX_DESCRIPTION_CHARS]
        return "New thread"
    return tid


def _normalized_search_title(title: str) -> str:
    """Drop only leading whitespace and pictographic symbols from TITLE."""
    index = 0
    while index < len(title):
        category = unicodedata.category(title[index])
        if title[index].isspace():
            index += 1
            continue
        if category in {"So", "Sk"}:
            index += 1
            while index < len(title) and unicodedata.category(title[index]) == "Mn":
                index += 1
            continue
        break
    return title[index:]


def _thread_revision_cursor(tid: str) -> str:
    """Return an opaque list-cache invalidator from the thread directory metadata."""
    try:
        activity_ns = os.stat(_thread_dir(tid)).st_mtime_ns
    except OSError:
        activity_ns = 0
    return hashlib.sha256(f"{tid}\0{activity_ns}".encode("utf-8")).hexdigest()[:24]


def _domain_choices() -> list[dict[str, str]]:
    return [
        {"repo_key": _repo_key(domain), "label": state._domain_label(domain)}
        for domain in state.DOMAINS
    ]


def _domain_for_key(key: str | None) -> str | None:
    if key is None:
        return state.DOMAINS[0] if state.DOMAINS else None
    _require_id(key, "repository key")
    return next((domain for domain in state.DOMAINS if _repo_key(domain) == key), None)


def _thread_dir(tid: str) -> str:
    _require_id(tid, "thread id")
    try:
        directory = state.MANAGER.thread_dir(tid)
    except Exception as error:
        raise HTTPException(status_code=404, detail="Thread not found") from error
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail="Thread not found")
    if (os.path.exists(os.path.join(directory, ".subagent"))
            or os.path.exists(os.path.join(directory, ".deleted"))):
        raise HTTPException(status_code=404, detail="Thread not found")
    return directory


def _message_id(tid: str, message: dict, ordinal: int) -> str:
    """Return an opaque response identity without exposing checkpoint internals."""
    position = message.get("_phone_checkpoint_position")
    if (isinstance(position, tuple) and len(position) == 4
            and all(isinstance(value, (str, int)) for value in position)):
        return _seal_id(tid, "m", (*position, str(message.get("role", ""))))
    source = message.get("message_id")
    if isinstance(source, str) and _MESSAGE_SOURCE_ID_RE.fullmatch(source):
        payload = f"{tid}\0{message.get('role', '')}\0{source}".encode("utf-8")
    else:
        payload = (f"{tid}\0{ordinal}\0{message.get('role', '')}\0"
                   f"{message.get('content', '')}").encode("utf-8")
    return "m-" + hashlib.sha256(payload).hexdigest()[:32]


def _seal_id(tid: str, prefix: str, values: tuple[object, ...]) -> str:
    """Seal a checkpoint position into an opaque, thread-bound phone token."""
    secret = os.environ.get(PHONE_API_TOKEN_ENV)
    if not secret:
        raise HTTPException(status_code=503, detail="Phone API is not configured")
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    tag = hmac.new(secret.encode("utf-8"),
                   tid.encode("utf-8") + b"\0" + prefix.encode("ascii") + b"\0" + payload,
                   hashlib.sha256).digest()[:16]
    return prefix + "-" + base64.urlsafe_b64encode(payload + tag).rstrip(b"=").decode("ascii")


def _open_sealed_id(tid: str, value: str, prefix: str) -> tuple[object, ...] | None:
    """Return a valid sealed phone token's fields, never accepting an offset."""
    marker = prefix + "-"
    if not value.startswith(marker) or not _SEALED_ID_RE.fullmatch(value[len(marker):]):
        return None
    encoded = value[len(marker):]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, binascii.Error):
        return None
    if len(raw) <= 16:
        return None
    payload, tag = raw[:-16], raw[-16:]
    secret = os.environ.get(PHONE_API_TOKEN_ENV)
    if not secret:
        return None
    expected = hmac.new(secret.encode("utf-8"),
                        tid.encode("utf-8") + b"\0" + prefix.encode("ascii") + b"\0" + payload,
                        hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(tag, expected):
        return None
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, list):
        return None
    return tuple(decoded)


def _checkpoint_position(values: tuple[object, ...], *, with_role: bool = False) -> tuple[str, str, int, int] | None:
    """Validate one sealed checkpoint position before it reaches SQLite."""
    expected = 5 if with_role else 4
    if len(values) != expected:
        return None
    checkpoint_id, task_id, index, item_index = values[:4]
    if (not isinstance(checkpoint_id, str) or not isinstance(task_id, str)
            or len(checkpoint_id) > 128 or len(task_id) > 256
            or not isinstance(index, int) or not isinstance(item_index, int)
            or index < 0 or item_index < 0):
        return None
    if with_role and values[4] not in {"user", "assistant", "tools"}:
        return None
    return checkpoint_id, task_id, index, item_index


def _message_timestamp(message: dict) -> str | None:
    """Return an optional persisted message timestamp without inventing one."""
    for key in ("timestamp", "created_at", "ts"):
        value = message.get(key)
        if isinstance(value, str) and len(value) <= 128:
            return value
    return None


def _workspace_entries(tid: str, *, include_size: bool = True,
                       with_truncation: bool = False) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], bool]:
    """List a bounded regular-file workspace without following any symlink."""
    root = state.MANAGER.thread_default_working_dir(tid)
    entries: list[dict[str, Any]] = []
    truncated = False
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return (entries, truncated) if with_truncation else entries
    try:
        nodes = 0
        walker = os.fwalk(".", dir_fd=root_fd, follow_symlinks=False)
        try:
            for current, dirs, files, current_fd in walker:
                base = "" if current == "." else current.removeprefix("./")
                safe_dirs = []
                for name in dirs:
                    nodes += 1
                    if nodes > MAX_WORKSPACE_NODES:
                        truncated = True
                        break
                    try:
                        mode = os.stat(name, dir_fd=current_fd, follow_symlinks=False).st_mode
                    except OSError:
                        continue
                    if name != ".git" and stat.S_ISDIR(mode):
                        safe_dirs.append(name)
                dirs[:] = safe_dirs
                if truncated:
                    break
                for name in files:
                    nodes += 1
                    if nodes > MAX_WORKSPACE_NODES or len(entries) >= MAX_FILES:
                        truncated = True
                        break
                    try:
                        metadata = os.stat(name, dir_fd=current_fd, follow_symlinks=False)
                    except OSError:
                        continue
                    if name == ".git" or not stat.S_ISREG(metadata.st_mode):
                        continue
                    relative = f"{base}/{name}" if base else name
                    entries.append({"path": relative, "type": "file", "size": metadata.st_size}
                                   if include_size else {"path": relative})
                if truncated:
                    break
        finally:
            walker.close()
    finally:
        os.close(root_fd)
    entries.sort(key=lambda entry: entry["path"])
    return (entries, truncated) if with_truncation else entries


def _file_references(text: str, known_paths: set[str]) -> list[dict[str, str]]:
    """Return only response paths that exactly match a safe worktree entry."""
    matches: list[dict[str, str]] = []
    for match in _FILE_TOKEN_RE.finditer(text):
        path = match.group(1)
        if path in known_paths and not any(item["path"] == path for item in matches):
            matches.append({"path": path, "label": path})
        if len(matches) == 12:
            break
    return matches


def _thread_workspace(tid: str) -> dict[str, Any]:
    manager = state._get_domain_manager(tid)
    if manager is None or not manager.repo:
        return {"repo_key": None, "repo_label": "No repository", "branch": None,
                "revision": None, "dirty": False}
    branch = current_branch(manager.repo_path) or None
    revision = None
    try:
        result = subprocess.run(
            ["git", "-C", manager.repo_path, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
        )
        if result.returncode == 0:
            revision = result.stdout.strip() or None
    except OSError:
        pass
    try:
        dirty = manager.has_changes_vs_main()
    except Exception:
        dirty = False
    return {"repo_key": _repo_key(manager.repo), "repo_label": state._domain_label(manager.repo),
            "branch": branch, "revision": revision, "dirty": dirty}


def _thread_messages(tid: str) -> list[dict]:
    """Legacy full projection for Pi threads and browser-compatible test fixtures."""
    chat, pi_messages, _, _ = threads._thread_messages_for_fragment(tid)
    return pi_messages if pi_messages is not None else (
        [] if chat is None else chat.get_web_messages())


def _checkpoint_history(tid: str, before: tuple[str, str, int, int] | None) -> tuple[list[dict], bool]:
    """Read a bounded suffix of root message writes without hydrating graph state.

    LangGraph persists each new message as a ``writes`` row.  Reading that append
    stream is intentionally separate from ``Thread.get_web_messages()``: the
    latter restores the complete ``messages`` channel before its caller can page
    it.  The phone path reads at most the declared number and bytes of writes.
    """
    saver = state.MANAGER.checkpointer
    where = "thread_id = ? AND checkpoint_ns = '' AND channel = 'messages'"
    params: list[object] = [tid]
    if before is not None:
        checkpoint_id, task_id, index, _item_index = before
        where += (" AND (checkpoint_id < ? OR (checkpoint_id = ? AND "
                  "(task_id < ? OR (task_id = ? AND idx <= ?))))")
        params.extend([checkpoint_id, checkpoint_id, task_id, task_id, index])
    query = ("SELECT checkpoint_id, task_id, idx, type, length(value) "
             "FROM writes WHERE " + where +
             " ORDER BY checkpoint_id DESC, task_id DESC, idx DESC LIMIT ?")
    params.append(MAX_HISTORY_SCAN_WRITES + 1)
    rows: list[tuple[str, str, int, str, int]] = []
    with saver.cursor(transaction=False) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        has_more = len(rows) > MAX_HISTORY_SCAN_WRITES
        rows = rows[:MAX_HISTORY_SCAN_WRITES]
        newest_first: list[dict] = []
        loaded_bytes = 0
        for checkpoint_id, task_id, index, kind, value_size in rows:
            if not isinstance(value_size, int) or value_size < 0:
                continue
            if value_size > MAX_HISTORY_WRITE_BYTES or loaded_bytes + value_size > MAX_HISTORY_SCAN_BYTES:
                newest_first.append({
                    "role": "assistant", "content": "[Message exceeds the mobile history limit.]",
                    "message_id": f"checkpoint:{checkpoint_id}:{task_id}:{index}:oversize",
                    "_phone_checkpoint_position": (checkpoint_id, task_id, index, 0),
                })
                has_more = True
                continue
            cursor.execute(
                "SELECT value FROM writes WHERE thread_id = ? AND checkpoint_ns = '' "
                "AND checkpoint_id = ? AND task_id = ? AND idx = ? AND channel = 'messages'",
                (tid, checkpoint_id, task_id, index),
            )
            row = cursor.fetchone()
            if row is None:
                continue
            loaded_bytes += value_size
            try:
                value = saver.serde.loads_typed((kind, row[0]))
                items = value if isinstance(value, list) else [value]
                messages = convert_to_messages(items)
            except Exception:
                newest_first.append({
                    "role": "assistant", "content": "[Message is unavailable on this device.]",
                    "message_id": f"checkpoint:{checkpoint_id}:{task_id}:{index}:invalid",
                    "_phone_checkpoint_position": (checkpoint_id, task_id, index, 0),
                })
                continue
            for item_index in range(len(messages) - 1, -1, -1):
                if (before is not None and checkpoint_id == before[0] and task_id == before[1]
                        and index == before[2] and item_index >= before[3]):
                    continue
                projected = _messages_to_dicts(
                    [messages[item_index]], split_tool_call_content=True,
                    include_message_ids=False)
                for message in reversed(projected):
                    message["message_id"] = f"checkpoint:{checkpoint_id}:{task_id}:{index}:{item_index}"
                    message["_phone_checkpoint_position"] = (checkpoint_id, task_id, index, item_index)
                    newest_first.append(message)
    newest_first.reverse()
    return newest_first, has_more


def _thread_history(tid: str, before: str | None) -> tuple[list[dict], bool, bool]:
    """Return chronological phone history plus whether it is checkpoint-backed."""
    # Pi conversations do not use the LangGraph write store.  They are small,
    # local records and keep the legacy sealed-hash cursor until their own store
    # gains the same append reader.
    if threads._is_pi_thread(tid):
        raw_messages = _thread_messages(tid)
        if before is None:
            return raw_messages, False, False
        end = next((ordinal - 1 for ordinal, raw in enumerate(raw_messages, start=1)
                    if _message_id(tid, raw, ordinal) == before), -1)
        if end < 0:
            raise HTTPException(status_code=409, detail="History cursor is no longer available")
        return raw_messages[:end], False, False
    if before is not None:
        cursor = _checkpoint_position(_open_sealed_id(tid, before, "c") or ())
        if cursor is None:
            raise HTTPException(status_code=422, detail="Invalid history cursor")
    else:
        cursor = None
    messages, has_more = _checkpoint_history(tid, cursor)
    return messages, has_more, True


def _snapshot(tid: str, before: str | None = None) -> dict[str, Any]:
    _thread_dir(tid)
    if before is not None and not _HISTORY_CURSOR_RE.fullmatch(before):
        raise HTTPException(status_code=422, detail="Invalid history cursor")
    raw_messages, stored_has_more, checkpoint_backed = _thread_history(tid, before)
    end = len(raw_messages)
    start = max(0, end - MAX_HISTORY_MESSAGES)
    selected = raw_messages[start:end]
    records = visible_records_from_dicts(selected)
    known_paths = {entry["path"] for entry in _workspace_entries(tid, include_size=False)}
    status = state._get_status(tid)
    messages: list[dict[str, Any]] = []
    total_bytes = 0
    truncated = False
    covered_raw = 0
    last_consumed_raw: tuple[dict, int] | None = None
    # Consume this page newest-first, otherwise a byte cap advances the cursor
    # past messages it never returned. ``next_before`` is the oldest returned
    # opaque checkpoint cursor, so an appended message cannot shift the next page.
    pairs = list(enumerate(zip(selected, records), start=start + 1))
    for ordinal, (raw, record) in reversed(pairs):
        if record.role not in {"user", "assistant"}:
            covered_raw += 1
            last_consumed_raw = raw, ordinal
            continue
        encoded_text = record.text.encode("utf-8")
        available = min(MAX_SNAPSHOT_MESSAGE_BYTES, MAX_SNAPSHOT_BYTES - total_bytes)
        if available <= 0:
            truncated = True
            break
        if len(encoded_text) > available:
            text = encoded_text[:available].decode("utf-8", "ignore")
            truncated = True
        else:
            text = record.text
        total_bytes += len(text.encode("utf-8"))
        item = {
            "id": _message_id(tid, raw, ordinal),
            "role": record.role,
            "text": text,
            "kind": record.source_kind,
            "timestamp": _message_timestamp(raw),
            "state": ("incomplete" if (before is None
                      and record.role == "user"
                      and status.get("stage") in threads.BUSY_STAGES
                      and not any(later.role == "assistant"
                                  for later in records[(ordinal - 1 - start) + 1:]))
                      else "final"),
        }
        if record.role == "assistant" and record.source_kind == "assistant":
            item["file_refs"] = _file_references(text, known_paths)
        messages.append(item)
        covered_raw += 1
        last_consumed_raw = raw, ordinal
    messages.reverse()
    revision = hashlib.sha256(json.dumps(
        [status.get("stage", "ready"),
         *[(item["id"], item["role"], item["state"]) for item in messages]],
        separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:24]
    has_older = start > 0 or covered_raw < len(selected) or stored_has_more
    next_before = None
    if has_older and last_consumed_raw is not None:
        cursor_raw, cursor_ordinal = last_consumed_raw
        next_before = (
            _seal_id(tid, "c", tuple(cursor_raw["_phone_checkpoint_position"]))
            if checkpoint_backed and "_phone_checkpoint_position" in cursor_raw
            else _message_id(tid, cursor_raw, cursor_ordinal)
        )
    return {
        "thread": {
            "id": tid,
            "description": _stored_thread_title(tid),
            "harness": read_thread_engine(_thread_dir(tid)).name,
            "status": status.get("stage", "ready"),
            "error": ("Thread failed; inspect Assist Web for details."
                      if status.get("error") else None),
            "workspace": _thread_workspace(tid),
            "revision": revision,
        },
        "messages": messages,
        "has_older_messages": has_older,
        "next_before": next_before,
        "truncated": truncated,
    }


def _thread_repo_summary(tid: str, status: dict[str, Any]) -> tuple[str | None, str]:
    """Return chooser metadata from setup state or the thread's durable repository."""
    domain = status.get("domain")
    if not isinstance(domain, str) or not domain:
        manager = state._get_domain_manager(tid)
        domain = manager.repo if manager is not None else None
        if not isinstance(domain, str) or not domain:
            return None, "No repository"
    return _repo_key(domain), state._domain_label(domain)


def _list_threads() -> dict[str, Any]:
    values: list[tuple[int, dict[str, Any]]] = []
    for tid in state.MANAGER.list()[:MAX_THREADS]:
        try:
            status = state._get_status(tid)
            repo_key, repo_label = _thread_repo_summary(tid, status)
            title = _stored_thread_title(tid)
            activity_at = os.stat(_thread_dir(tid)).st_mtime
            values.append((threads._thread_status_rank(tid, status.get("stage", "ready")), {
                "id": tid,
                "description": title,
                "search_description": _normalized_search_title(title),
                "harness": read_thread_engine(_thread_dir(tid)).name,
                "status": status.get("stage", "ready"),
                "repo_key": repo_key,
                "repo_label": repo_label,
                "unread": state._has_unseen_response(tid),
                "activity_at": activity_at,
                "revision": _thread_revision_cursor(tid),
            }))
        except (OSError, ThreadEngineError):
            continue
    values.sort(key=lambda item: item[0])
    harnesses = [{"key": "deepagents", "label": "Deep Agents"}]
    if state.PI_PREVIEW.admits("pi"):
        harnesses.append({"key": "pi", "label": "Pi preview"})
    return {"threads": [value for _, value in values], "repositories": _domain_choices(),
            "harnesses": harnesses}


def _phone_dispatch_key(key: str) -> str:
    return "phone:" + key


def _phone_thread_id(key: str) -> str:
    return "phone-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _phone_thread_limit_reached() -> bool:
    """Keep one authenticated phone client from creating unlimited thread stores."""
    return sum(tid.startswith("phone-") for tid in state.MANAGER.list()) >= MAX_PHONE_THREADS


def _phone_initialization_limit_reached() -> bool:
    """Leave the shared first-thread worker available to ordinary web starts."""
    return sum(
        tid.startswith("phone-")
        and state._get_status(tid).get("stage") in {"initializing", "cloning"}
        for tid in state.MANAGER.list()
    ) >= MAX_PHONE_INITIALIZATIONS


def _find_dispatch(tid: str, dispatch_key: str):
    return next((run for run in threads._runs().list(tid)
                 if run.dispatch_key == dispatch_key), None)


def _submit_existing(tid: str, text: str, key: str, *, run_id: str | None = None,
                     work_id: str | None = None) -> tuple[Any, bool, bool]:
    """Durably accept one idempotent turn under the browser and Run fences."""
    _thread_dir(tid)
    dispatch_key = _phone_dispatch_key(key)
    try:
        with threads.browser_authority.fence(state.MANAGER.root_dir, tid) as browser_state:
            map_record = False
            try:
                map_dir = threads.configured_directory()
                if map_dir is not None:
                    map_record = bool(threads.browser_records(map_dir, tid))
            except (OSError, RuntimeError, ValueError):
                pass
            with threads._RUN_ADMISSION_LOCK:
                replay = _find_dispatch(tid, dispatch_key)
                if replay is not None:
                    if replay.text != text:
                        raise HTTPException(status_code=409, detail="Idempotency-Key conflicts with prior message")
                    return replay, False, True
                try:
                    if not threads._pi_message_admits(tid):
                        raise HTTPException(status_code=503, detail="Pi preview is unavailable")
                except ThreadEngineError as error:
                    raise HTTPException(status_code=409, detail="Thread harness is unavailable") from error
                try:
                    run, busy = threads._accept_message_run_locked(
                        tid, text, dispatch_key=dispatch_key,
                        max_pending=MAX_PHONE_PENDING_RUNS, run_id=run_id,
                        work_id=work_id, browser_state=browser_state,
                        browser_map_record=map_record)
                except threads._EmailApprovalPending as error:
                    raise HTTPException(
                        status_code=409, detail="Resolve the pending approval first") from error
                except InvalidRunTransition as error:
                    raise HTTPException(status_code=429, detail=str(error)) from error
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        raise HTTPException(status_code=503, detail="Browser admission is unavailable") from error
    if run.status == "revocation_pending":
        threads._queue_browser_revocation(tid)
    return run, busy, False


def _create_and_submit(body: _CreateThread, key: str, *, run_id: str | None = None,
                       work_id: str | None = None) -> tuple[str, Any, str | None, bool]:
    """Create a deterministic phone draft only when its first message arrives."""
    domain = _domain_for_key(body.repo_key)
    if body.repo_key is not None and domain is None:
        raise HTTPException(status_code=422, detail="Unknown repository")
    tid = _phone_thread_id(key)
    dispatch_key = _phone_dispatch_key(key)
    with threads._RUN_ADMISSION_LOCK:
        if os.path.isdir(state.MANAGER.thread_dir(tid)):
            replay = _find_dispatch(tid, dispatch_key)
            if replay is None:
                if threads._runs().list(tid):
                    raise HTTPException(status_code=409, detail="Phone draft conflicts with an existing thread")
                state.MANAGER.hard_delete(tid)
            else:
                expected_domain = domain or (state.DOMAINS[0] if state.DOMAINS else None)
                try:
                    existing_engine = read_thread_engine(_thread_dir(tid)).name
                except ThreadEngineError as error:
                    raise HTTPException(status_code=409, detail="Thread harness is unavailable") from error
                if (replay.text != body.message or existing_engine != body.harness
                        or state._get_status(tid).get("domain", "") != (expected_domain or "")):
                    raise HTTPException(status_code=409, detail="Idempotency-Key conflicts with prior message")
                return tid, replay, None, True
        if _phone_thread_limit_reached():
            raise HTTPException(status_code=429, detail="Phone thread limit reached")
        if _phone_initialization_limit_reached():
            raise HTTPException(status_code=429, detail="Phone initialization is busy")
        try:
            tid, run_id, selected = threads.create_thread_with_message_core(
                body.message, domain, engine=body.harness, thread_id=tid,
                dispatch_key=dispatch_key, run_id=run_id,
                work_id=work_id)
            run = threads._runs().get(tid, run_id)
        except (ValueError, ThreadEngineError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return tid, run, selected, False


def _logical_status(tid: str, run_id: str) -> dict[str, Any]:
    """Project the immutable accepted handle over one physical-state snapshot."""
    # Fair scheduling publishes interrupted predecessor, successor, and paused
    # thread status under this lock.  Read the same unit atomically so SSE never
    # mistakes that short handoff for a terminal interrupted Run.
    with threads._RUN_ADMISSION_LOCK:
        return _logical_status_locked(tid, run_id)


def _logical_status_with_revision(tid: str, run_id: str) -> tuple[dict[str, Any], int]:
    """Return one logical projection and its process-local write invalidation revision."""
    with threads._RUN_ADMISSION_LOCK:
        return _logical_status_locked(tid, run_id, with_revision=True)


def _logical_status_with_revision_and_token(
        tid: str, run_id: str
) -> tuple[dict[str, Any], int, Any]:
    """Return one coherent durable projection, write revision, and O(1) file token."""
    with threads._RUN_ADMISSION_LOCK:
        return _logical_status_locked(tid, run_id, with_revision=True, with_token=True)


def _open_observation(tid: str, work_id: str) -> bool:
    """Register one observer unless deletion won the admission-lock handoff."""
    with threads._RUN_ADMISSION_LOCK:
        try:
            _thread_dir(tid)
        except HTTPException:
            return False
        RUN_STREAMS.open_observer(tid, work_id)
        return True


def _observation_status_with_revision_and_token(
        tid: str, run_id: str, work_id: str
) -> tuple[dict[str, Any], int, Any] | None:
    """Reproject under deletion admission, or report an already-closed observer."""
    with threads._RUN_ADMISSION_LOCK:
        if RUN_STREAMS.is_gone(tid, work_id):
            return None
        try:
            return _logical_status_locked(tid, run_id, with_revision=True, with_token=True)
        except HTTPException as error:
            if error.status_code == 404:
                # A deletion that won admission has already marked the journal;
                # any other missing durable projection is an operator-visible store fault.
                if RUN_STREAMS.is_gone(tid, work_id):
                    return None
                raise RunStoreUnavailable("Run store is unavailable") from error
            raise


def _observation_invalidation(
        tid: str, work_id: str
) -> tuple[int, Any, str] | None:
    """Read cheap observation invalidators under the deletion admission boundary."""
    with threads._RUN_ADMISSION_LOCK:
        if RUN_STREAMS.is_gone(tid, work_id):
            return None
        service = threads._runs()
        revision = service.revision(tid)
        token = service.observation_token(tid)
        if token is None:
            # The deletion winner marks the journal while holding this same lock.
            # A missing file without that mark is a durable-store fault, not a close.
            raise RunStoreUnavailable("Run store is unavailable")
        return revision, token, state._get_status(tid).get("stage", "ready")


def _logical_status_locked(
        tid: str, run_id: str, *, with_revision: bool = False,
        with_runs: bool = False, include_cleanup: bool = False, with_token: bool = False,
) -> (dict[str, Any] | tuple[dict[str, Any], int]
      | tuple[dict[str, Any], int, ObservationToken] | tuple[dict[str, Any], list[Any]]):
    """Project an accepted handle while ``_RUN_ADMISSION_LOCK`` is held."""
    _thread_dir(tid)
    _require_id(run_id, "run id")
    service = threads._runs()
    if with_token:
        runs, revision, token = service.list_with_revision_and_token(tid)
        if token is None:
            raise RunStoreUnavailable("Run store is unavailable")
    else:
        runs, revision = service.list_with_revision(tid)
    accepted = next((run for run in runs if run.id == run_id), None)
    if accepted is None:
        raise HTTPException(status_code=404, detail="Run not found")
    thread_status = state._get_status(tid).get("stage", "ready")
    work = [run for run in runs if run.work_id == accepted.work_id]
    selected = work[-1]
    status = selected.status
    if status == "interrupted":
        if thread_status in threads.BUSY_STAGES and not any(
                run.work_id != accepted.work_id and run.status == "running"
                for run in runs):
            status = "transitioning"
    projection = {"id": accepted.id, "thread_id": tid, "work_id": accepted.work_id,
                  "physical_run_id": selected.id, "status": status,
                  "error": ("Run failed; inspect Assist Web for details."
                            if selected.error else None), "updated_at": selected.updated_at,
                  "thread_status": thread_status}
    if include_cleanup:
        projection["cancel_cleanup"] = accepted.cancel_cleanup
    if with_runs:
        return projection, runs
    if with_revision:
        return (projection, revision, token) if with_token else (projection, revision)
    return projection


def _public_run_projection(projection: dict[str, Any]) -> dict[str, Any]:
    """Remove internal recovery receipts from one phone Run representation."""
    return {key: value for key, value in projection.items() if key != "cancel_cleanup"}


def _reserve_existing(tid: str, text: str, key: str) -> tuple[Any, bool, bool, bool]:
    """Reserve before durable visibility, but never make streaming admission truth."""
    run_id = uuid.uuid4().hex
    work_id = uuid.uuid4().hex
    try:
        is_pi_thread = threads._is_pi_thread(tid)
    except ThreadEngineError as error:
        raise HTTPException(status_code=409,
                            detail="Thread harness is unavailable") from error
    reserved = not is_pi_thread and RUN_STREAMS.reserve(tid, work_id)
    try:
        # A busy holder can see the durable follower as soon as its locked admission
        # returns.  Activate first, so every worker that can see that Run can publish.
        if reserved:
            RUN_STREAMS.activate(tid, work_id)
        run, busy, replay = _submit_existing(tid, text, key, run_id=run_id, work_id=work_id)
    except Exception:
        if reserved:
            RUN_STREAMS.discard(tid, work_id)
        raise
    if replay or busy:
        if replay and reserved:
            RUN_STREAMS.discard(tid, work_id)
        return run, busy, replay, (RUN_STREAMS.read(tid, run.work_id) is not None
                                   if replay else reserved)
    return run, busy, False, reserved


def _reserve_create(body: _CreateThread, key: str) -> tuple[str, Any, str | None, bool, bool]:
    run_id = uuid.uuid4().hex
    work_id = uuid.uuid4().hex
    # The final deterministic phone thread id is based on the idempotency key.
    tid = _phone_thread_id(key)
    reserved = body.harness != "pi" and RUN_STREAMS.reserve(tid, work_id)
    try:
        if reserved:
            RUN_STREAMS.activate(tid, work_id)
        tid, run, domain, replay = _create_and_submit(body, key, run_id=run_id, work_id=work_id)
    except Exception:
        if reserved:
            RUN_STREAMS.discard(tid, work_id)
        raise
    if replay:
        if reserved:
            RUN_STREAMS.discard(tid, work_id)
        return tid, run, domain, True, RUN_STREAMS.read(tid, run.work_id) is not None
    # The observer starts status-only until the dedicated initializer reaches
    # its worker, then sees this already-active journal's deltas.
    return tid, run, domain, False, reserved


def _cancel_logical_run(tid: str, run_id: str) -> tuple[int, dict[str, Any]]:
    """Cancel one accepted logical Run and durably receipt its cleanup."""
    with threads._RUN_ADMISSION_LOCK:
        current_status = state._get_status(tid)
        projection, runs = _logical_status_locked(
            tid, run_id, with_runs=True, include_cleanup=True)
        if projection["status"] == "running":
            return 409, {"detail": "Run is already executing", "outcome": "running",
                         "run": _public_run_projection(projection)}
        if projection["status"] == "transitioning":
            return 409, {"detail": "Run is transitioning", "outcome": "transitioning",
                         "run": _public_run_projection(projection)}
        if projection["status"] not in {"pending", "cancelled"}:
            detail = ("Run is awaiting approval"
                      if projection["status"] in AWAITING_APPROVAL_STATUSES
                      else "Run is already terminal")
            return 409, {"detail": detail, "outcome": projection["status"],
                         "run": _public_run_projection(projection)}
        service = threads._runs()
        try:
            if projection["status"] == "pending":
                # This one write cancels the newest pending slice, retires its
                # interrupted same-work predecessors, and leaves the accepted
                # handle with a retry receipt.
                runs = service.cancel_logical(tid, run_id)
            else:
                if projection["cancel_cleanup"] is None:
                    return 409, {"detail": "Run is already terminal", "outcome": "cancelled",
                                 "run": _public_run_projection(projection)}
                if projection["cancel_cleanup"] == "complete":
                    return 200, {"outcome": "cancelled",
                                 "run": _public_run_projection(projection)}
        except InvalidRunTransition:
            return 409, {"detail": "Run is already executing", "outcome": "running",
                         "run": _logical_status_locked(tid, run_id)}
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RunStoreUnavailable("Run store is unavailable") from error
        work_id = projection["work_id"]
        # A stale interrupted record must not pin the thread paused.  Only an
        # interrupted slice with a pending/running same-work successor is resumable.
        if (state._get_status(tid).get("stage") == "paused"
                and not any(run.status == "interrupted" and any(
                    later.work_id == run.work_id
                    and later.status in {"pending", "running"}
                    for later in runs)
                            for run in runs)):
            threads._set_status(tid, "ready")
        RUN_STREAMS.finish(tid, work_id)
        # The initializer owns a partially prepared first workspace.  Its
        # bounded worker observes this receipt after setup, changes the status
        # to ready, and dispatches any follower exactly once.
        if current_status.get("stage") not in {"initializing", "cloning"}:
            threads._dispatch_pending_after(tid, projection["physical_run_id"])
        # This is deliberately the last durable write.  If any earlier cleanup
        # step fails, the pending receipt makes its retry replay that work; once
        # complete, a repeated DELETE is a no-dispatch success.
        try:
            completed = service.complete_cancel_cleanup(tid, run_id)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RunStoreUnavailable("Run store is unavailable") from error
        selected = next(run for run in runs if run.id == projection["physical_run_id"])
        if selected.id == completed.id:
            selected = completed
        projection["status"] = "cancelled"
        projection["updated_at"] = selected.updated_at
        projection["thread_status"] = state._get_status(tid).get("stage", "ready")
        return 200, {"outcome": "cancelled", "run": _public_run_projection(projection)}


def _sse(event: str, value: dict[str, Any]) -> str:
    record = encode_sse(event, value)
    if len(record.encode("utf-8")) > MAX_PHONE_SSE_RECORD_BYTES:
        raise ValueError("phone SSE record exceeds bound")
    return record


def _diff(tid: str) -> dict[str, Any]:
    _thread_dir(tid)
    manager = state._get_domain_manager(tid)
    if manager is None:
        return {"files": [], "truncated": False, "workspace": _thread_workspace(tid)}
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            ["git", "-C", manager.repo_path, "diff", "--no-ext-diff", "--no-textconv",
             "--binary", "main..."],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        assert process.stdout is not None
        data = process.stdout.read(MAX_DIFF_BYTES + 1)
        truncated = len(data) > MAX_DIFF_BYTES
        if truncated:
            data = data[:MAX_DIFF_BYTES]
            process.kill()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise HTTPException(status_code=503, detail="Thread diff is unavailable")
    return {"files": ([{"path": "workspace", "diff": data.decode("utf-8", "replace")}]
                      if data else []),
            "truncated": truncated, "workspace": _thread_workspace(tid)}


def _workspace_manifest(tid: str) -> dict[str, Any]:
    _thread_dir(tid)
    entries, truncated = _workspace_entries(tid, with_truncation=True)
    return {"workspace": _thread_workspace(tid), "files": entries,
            "truncated": truncated}


def _open_workspace_file(root_fd: int, relative: str) -> int:
    """Open one manifest path without following a symlink at any component."""
    parts = Path(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OSError("invalid workspace path")
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _workspace_archive(tid: str) -> bytes:
    """Build a bounded, regular-file-only worktree archive for a phone mirror."""
    _thread_dir(tid)
    root = state.MANAGER.thread_default_working_dir(tid)
    buffer = io.BytesIO()
    total = 0
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise HTTPException(status_code=409, detail="Workspace is unavailable") from error
    try:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for entry in _workspace_entries(tid):
                relative = entry["path"]
                remaining = MAX_ARCHIVE_BYTES - total
                if remaining <= 0:
                    break
                try:
                    fd = _open_workspace_file(root_fd, relative)
                except OSError:
                    continue
                try:
                    metadata = os.fstat(fd)
                    if (not stat.S_ISREG(metadata.st_mode)
                            or metadata.st_size > MAX_ARCHIVE_FILE_BYTES
                            or metadata.st_size > remaining):
                        continue
                    content = bytearray()
                    while len(content) < metadata.st_size:
                        chunk = os.read(fd, min(64 * 1024, metadata.st_size - len(content)))
                        if not chunk:
                            break
                        content.extend(chunk)
                    if len(content) != metadata.st_size:
                        continue
                finally:
                    os.close(fd)
                info = tarfile.TarInfo(relative)
                info.size = len(content)
                info.mode = 0o644
                info.mtime = int(metadata.st_mtime)
                archive.addfile(info, io.BytesIO(content))
                total += len(content)
    finally:
        os.close(root_fd)
    return buffer.getvalue()


router = APIRouter(prefix=PHONE_API_PREFIX, dependencies=[Depends(_authenticate)])


@router.get("/threads")
async def list_threads() -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_list_threads)


@router.get("/threads/{tid}")
async def get_thread(tid: str) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_snapshot, tid)


@router.get("/threads/{tid}/history")
async def get_thread_history(tid: str, before: str) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_snapshot, tid, before)


@router.post("/threads")
async def create_thread(request: Request) -> dict[str, Any]:
    body = await _validated_body(request, _CreateThread)
    assert isinstance(body, _CreateThread)
    key = _request_key(request)
    try:
        tid, run, domain, replay, live_text = await anyio.to_thread.run_sync(
            _reserve_create, body, key)
    except RunStoreUnavailable as error:
        raise HTTPException(status_code=503, detail="run-store-unavailable") from error
    if not replay:
        await anyio.to_thread.run_sync(
            threads._INITIALIZATION_SCHEDULER.submit, run.id, tid, domain)
    return {"thread_id": tid, "run_id": run.id, "replayed": replay,
            "live_text": live_text}


@router.post("/threads/{tid}/messages")
async def send_message(tid: str, request: Request) -> dict[str, Any]:
    body = await _validated_body(request, _SendMessage)
    assert isinstance(body, _SendMessage)
    key = _request_key(request)
    try:
        run, busy, replay, live_text = await anyio.to_thread.run_sync(
            _reserve_existing, tid, body.message, key)
    except RunStoreUnavailable as error:
        raise HTTPException(status_code=503, detail="run-store-unavailable") from error
    if not busy and not replay:
        await anyio.to_thread.run_sync(
            lambda: threads._RESUME_SCHEDULER.submit(run.id, tid, user_priority=True))
    return {"thread_id": tid, "run_id": run.id, "replayed": replay,
            "status": run.status, "live_text": live_text}


@router.get("/threads/{tid}/runs/{run_id}")
async def get_run(tid: str, run_id: str) -> dict[str, Any]:
    try:
        return await anyio.to_thread.run_sync(_logical_status, tid, run_id)
    except RunStoreUnavailable as error:
        raise HTTPException(status_code=503, detail="run-store-unavailable") from error


@router.delete("/threads/{tid}/runs/{run_id}")
async def cancel_run(tid: str, run_id: str):
    try:
        code, value = await anyio.to_thread.run_sync(_cancel_logical_run, tid, run_id)
    except RunStoreUnavailable as error:
        raise HTTPException(status_code=503, detail="run-store-unavailable") from error
    if code != 200:
        return JSONResponse(value, status_code=code)
    return value


@router.get("/threads/{tid}/runs/{run_id}/events")
async def run_events(tid: str, run_id: str, request: Request) -> StreamingResponse:
    """Offer process-local text and durable Run projection for one logical handle.

    Reset, delta, and truncation are bounded provisional journal observations;
    status and terminal are the durable logical Run projection.
    """
    try:
        status, revision, observation_token = await anyio.to_thread.run_sync(
            _logical_status_with_revision_and_token, tid, run_id)
    except RunStoreUnavailable as error:
        raise HTTPException(status_code=503,
                            detail="run-store-unavailable") from error
    if not _SSE_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many phone event streams")
    observing = await anyio.to_thread.run_sync(
        _open_observation, tid, status["work_id"])
    journal_state = await anyio.to_thread.run_sync(
        lambda: RUN_STREAMS.snapshot_if_changed(tid, status["work_id"], None))
    journal_revision, journal = journal_state if journal_state is not None else (None, None)
    released = False

    async def release_slot() -> None:
        nonlocal released
        if not released:
            released = True
            if observing:
                await anyio.to_thread.run_sync(
                    RUN_STREAMS.close_observer, tid, status["work_id"])
            _SSE_SLOTS.release()

    async def events():
        nonlocal journal, journal_revision, status
        try:
            if not observing:
                yield _sse("closed-set", {"reason": "thread-gone"})
                return
            previous = None
            attempt = None
            seen_index = 0
            truncated_attempt = None
            known_revision = revision
            known_observation_token = observation_token
            known_thread_status = status["thread_status"]
            reproject_terminal = bool(journal and journal["terminal"])
            for sequence in range(1, 1_801):
                if await request.is_disconnected():
                    return
                needs_reprojection = reproject_terminal
                try:
                    invalidation = await anyio.to_thread.run_sync(
                        _observation_invalidation, tid, status["work_id"])
                except RunStoreUnavailable:
                    yield _sse("error", {"detail": "run-store-unavailable"})
                    return
                if invalidation is None:
                    yield _sse("closed-set", {"reason": "thread-gone"})
                    return
                current_revision, current_observation_token, current_thread_status = invalidation
                if sequence != 1:
                    journal_state = await anyio.to_thread.run_sync(
                        lambda: RUN_STREAMS.snapshot_if_changed(
                            tid, status["work_id"], journal_revision))
                    if journal_state is not None:
                        journal_revision, journal = journal_state
                        reproject_terminal = bool(journal["terminal"])
                    needs_reprojection = (needs_reprojection
                                           or current_revision != known_revision
                                           or current_observation_token != known_observation_token
                                           or current_thread_status != known_thread_status
                                           or reproject_terminal)
                elif (current_revision != known_revision
                      or current_observation_token != known_observation_token
                      or current_thread_status != known_thread_status):
                    needs_reprojection = True
                if needs_reprojection:
                    try:
                        projection = await anyio.to_thread.run_sync(
                            _observation_status_with_revision_and_token,
                            tid, run_id, status["work_id"])
                    except RunStoreUnavailable:
                        yield _sse("error", {"detail": "run-store-unavailable"})
                        return
                    if projection is None:
                        yield _sse("closed-set", {"reason": "thread-gone"})
                        return
                    status, known_revision, known_observation_token = projection
                    known_thread_status = status["thread_status"]
                    reproject_terminal = False
                if journal is not None and journal["attempt"] != attempt:
                    attempt = journal["attempt"]
                    seen_index = 0
                    truncated_attempt = None
                    yield _sse("assistant-reset", {"attempt": attempt,
                                                   "reason": "replay" if sequence == 1 else "rollback"})
                if journal is not None:
                    for delta in journal["deltas"]:
                        if delta["attempt"] == attempt and delta["index"] > seen_index:
                            yield _sse("assistant-delta", delta)
                            seen_index = delta["index"]
                    if journal["truncated"] and truncated_attempt != attempt:
                        yield _sse("assistant-truncated", {"attempt": attempt})
                        truncated_attempt = attempt
                encoded = json.dumps(status, ensure_ascii=True, separators=(",", ":"))
                if encoded != previous:
                    yield _sse("status", status)
                    previous = encoded
                if status["status"] in TERMINAL_STATUSES | AWAITING_APPROVAL_STATUSES:
                    yield _sse("terminal", status)
                    return
                await asyncio.sleep(1)
            yield _sse("error", {"detail": "event stream timed out"})
        finally:
            await release_slot()

    return _BoundedSSEStreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        on_close=release_slot)


@router.get("/threads/{tid}/diff")
async def get_diff(tid: str) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_diff, tid)


@router.get("/threads/{tid}/workspace")
async def get_workspace(tid: str) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_workspace_manifest, tid)


@router.get("/threads/{tid}/workspace/archive")
async def get_workspace_archive(tid: str) -> Response:
    if not _ARCHIVE_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="A workspace archive is already in progress")
    try:
        data = await anyio.to_thread.run_sync(_workspace_archive, tid)
    finally:
        _ARCHIVE_SLOTS.release()
    return Response(data, media_type="application/gzip",
                    headers={"Content-Disposition": 'attachment; filename="workspace.tar.gz"',
                             "Cache-Control": "no-store"})
