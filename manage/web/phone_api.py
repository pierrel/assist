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
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import threading
import unicodedata
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from langchain_core.messages import convert_to_messages
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assist.domain_manager import current_branch
from assist.run_service import InvalidRunTransition, TERMINAL_STATUSES, RunNotFound
from assist.thread import _messages_to_dicts
from assist.thread_engine import ThreadEngineError, read_thread_engine
from assist.visible_conversation import visible_records_from_dicts
from manage.web import state
from manage.web import threads


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
MAX_PHONE_RUNS_PER_THREAD = 200
MAX_PHONE_PENDING_RUNS = 4
MAX_PHONE_INITIALIZATIONS = 1
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


def _authenticate(request: Request, response: Response) -> None:
    """Require the dedicated phone token before any thread lookup."""
    response.headers["Cache-Control"] = "no-store"
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
        [] if chat is None else getattr(chat, "get_web_messages", chat.get_messages)())


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


def _thread_repo_summary(status: dict[str, Any]) -> tuple[str | None, str]:
    """Return the persisted chooser label without inspecting a Git worktree."""
    domain = status.get("domain")
    if not isinstance(domain, str) or not domain:
        return None, "No repository"
    return _repo_key(domain), state._domain_label(domain)


def _list_threads() -> dict[str, Any]:
    values: list[tuple[int, dict[str, Any]]] = []
    for tid in state.MANAGER.list()[:MAX_THREADS]:
        try:
            status = state._get_status(tid)
            repo_key, repo_label = _thread_repo_summary(status)
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


def _submit_existing(tid: str, text: str, key: str) -> tuple[Any, bool, bool]:
    """Durably accept one idempotent normal web turn under the existing lock."""
    _thread_dir(tid)
    dispatch_key = _phone_dispatch_key(key)
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
        if state._get_status(tid).get("pending_email_token"):
            raise HTTPException(status_code=409, detail="Resolve the pending approval first")
        try:
            run, busy = threads._accept_message_run_locked(
                tid, text, dispatch_key=dispatch_key,
                max_runs=MAX_PHONE_RUNS_PER_THREAD,
                max_pending=MAX_PHONE_PENDING_RUNS)
        except InvalidRunTransition as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        return run, busy, False


def _create_and_submit(body: _CreateThread, key: str) -> tuple[str, Any, str | None, bool]:
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
                dispatch_key=dispatch_key, max_runs=MAX_PHONE_RUNS_PER_THREAD,
                max_pending=MAX_PHONE_PENDING_RUNS)
            run = threads._runs().get(tid, run_id)
        except (ValueError, ThreadEngineError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return tid, run, selected, False


def _run_status(tid: str, run_id: str) -> dict[str, Any]:
    _thread_dir(tid)
    _require_id(run_id, "run id")
    try:
        run = threads._runs().get(tid, run_id)
    except RunNotFound as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    return {"id": run.id, "thread_id": tid, "status": run.status,
            "error": ("Run failed; inspect Assist Web for details."
                      if run.error else None),
            "updated_at": run.updated_at,
            "thread_status": state._get_status(tid).get("stage", "ready")}


def _cancel_pending_run(tid: str, run_id: str) -> dict[str, Any]:
    """Cancel only unclaimed work; a running model turn cannot be lied about."""
    _thread_dir(tid)
    _require_id(run_id, "run id")
    status = state._get_status(tid)
    if (status.get("stage") in {"initializing", "cloning"}
            and status.get("pending_run_id") == run_id):
        raise HTTPException(status_code=409, detail="Thread setup is already in progress")
    try:
        run = threads._runs().cancel_pending(tid, run_id)
    except RunNotFound as error:
        raise HTTPException(status_code=404, detail="Run not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail="Run is already executing") from error
    return {"id": run.id, "status": run.status}


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
    tid, run, domain, replay = await anyio.to_thread.run_sync(_create_and_submit, body, key)
    if not replay:
        threads._INITIALIZATION_SCHEDULER.submit(run.id, tid, domain)
    return {"thread_id": tid, "run_id": run.id, "replayed": replay}


@router.post("/threads/{tid}/messages")
async def send_message(tid: str, request: Request) -> dict[str, Any]:
    body = await _validated_body(request, _SendMessage)
    assert isinstance(body, _SendMessage)
    key = _request_key(request)
    run, busy, replay = await anyio.to_thread.run_sync(_submit_existing, tid, body.message, key)
    if not busy and not replay:
        threads._RESUME_SCHEDULER.submit(run.id, tid, user_priority=True)
    return {"thread_id": tid, "run_id": run.id, "replayed": replay,
            "status": run.status}


@router.get("/threads/{tid}/runs/{run_id}")
async def get_run(tid: str, run_id: str) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_run_status, tid, run_id)


@router.delete("/threads/{tid}/runs/{run_id}")
async def cancel_run(tid: str, run_id: str) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_cancel_pending_run, tid, run_id)


@router.get("/threads/{tid}/runs/{run_id}/events")
async def run_events(tid: str, run_id: str, request: Request) -> StreamingResponse:
    """Offer bounded status/final events while durable state remains authoritative."""
    await anyio.to_thread.run_sync(_run_status, tid, run_id)
    if not _SSE_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many phone event streams")

    async def events():
        try:
            previous = None
            for sequence in range(1, 1_801):
                if await request.is_disconnected():
                    return
                status = await anyio.to_thread.run_sync(_run_status, tid, run_id)
                encoded = json.dumps(status, separators=(",", ":"))
                if encoded != previous:
                    yield f"id: {sequence}\nevent: status\ndata: {encoded}\n\n"
                    previous = encoded
                if status["status"] in TERMINAL_STATUSES:
                    yield "event: terminal\ndata: {}\n\n"
                    return
                await asyncio.sleep(1)
            yield "event: error\ndata: {\"detail\":\"event stream timed out\"}\n\n"
        finally:
            _SSE_SLOTS.release()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


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
