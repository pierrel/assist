"""Authenticated, structured Assist-web API for the EmacsOS phone client.

Browser routes intentionally remain form/HTML routes.  This module is the
separate machine interface: it authenticates every request, emits only visible
conversation data, and keeps all blocking thread/worktree operations off the
single FastAPI event loop.
"""
from __future__ import annotations

import asyncio
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
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assist.domain_manager import current_branch
from assist.phone_pins import MAX_PINS, create_pin, delete_pin, list_pins
from assist.run_service import NONTERMINAL_STATUSES, TERMINAL_STATUSES, RunNotFound
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
MAX_THREADS = 500
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_FILE_BYTES = 4 * 1024 * 1024
_SSE_SLOTS = threading.BoundedSemaphore(4)
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
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


class _CreatePin(_StrictModel):
    response_id: Annotated[str, Field(min_length=1, max_length=128)]


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
    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail="Thread not found")
    if (os.path.exists(os.path.join(directory, ".subagent"))
            or os.path.exists(os.path.join(directory, ".deleted"))):
        raise HTTPException(status_code=404, detail="Thread not found")
    return directory


def _message_id(tid: str, message: dict, ordinal: int) -> str:
    """Return an opaque response identity without exposing checkpoint internals."""
    source = message.get("message_id")
    if isinstance(source, str) and _ID_RE.fullmatch(source):
        payload = f"{tid}\0{source}".encode("utf-8")
    else:
        payload = (f"{tid}\0{ordinal}\0{message.get('role', '')}\0"
                   f"{message.get('content', '')}").encode("utf-8")
    return "m-" + hashlib.sha256(payload).hexdigest()[:32]


def _workspace_entries(tid: str, *, include_size: bool = True) -> list[dict[str, Any]]:
    """List regular worktree entries with no host paths or symlink traversal."""
    root = Path(state.MANAGER.thread_default_working_dir(tid)).resolve()
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name != ".git" and not (Path(current) / name).is_symlink()]
        for name in files:
            if name == ".git":
                continue
            path = Path(current) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                relative = path.relative_to(root).as_posix()
                size = path.stat().st_size
            except OSError:
                continue
            entries.append({"path": relative, "type": "file", "size": size} if include_size
                           else {"path": relative})
            if len(entries) >= MAX_FILES:
                return entries
    entries.sort(key=lambda entry: entry["path"])
    return entries


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
    chat, pi_messages, _, _ = threads._thread_messages_for_fragment(tid)
    return pi_messages if pi_messages is not None else (
        [] if chat is None else getattr(chat, "get_web_messages", chat.get_messages)())


def _snapshot(tid: str, before: int = 0) -> dict[str, Any]:
    _thread_dir(tid)
    raw_messages = _thread_messages(tid)
    if before < 0 or before > len(raw_messages):
        raise HTTPException(status_code=422, detail="Invalid history cursor")
    end = len(raw_messages) - before
    start = max(0, end - MAX_HISTORY_MESSAGES)
    selected = raw_messages[start:end]
    records = visible_records_from_dicts(selected)
    known_paths = {entry["path"] for entry in _workspace_entries(tid, include_size=False)}
    messages: list[dict[str, Any]] = []
    total_bytes = 0
    truncated = False
    covered_raw = 0
    # `before` counts raw records from the newest end.  Consume this page from
    # that same end, otherwise a byte cap advances the cursor past messages it
    # never returned.  Reverse before responding so the transcript stays
    # chronological.
    pairs = list(enumerate(zip(selected, records), start=start + 1))
    for ordinal, (raw, record) in reversed(pairs):
        if record.role not in {"user", "assistant"}:
            covered_raw += 1
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
        }
        if record.role == "assistant" and record.source_kind == "assistant":
            item["file_refs"] = _file_references(text, known_paths)
        messages.append(item)
        covered_raw += 1
    messages.reverse()
    status = state._get_status(tid)
    revision = hashlib.sha256(json.dumps(
        [(item["id"], item["role"]) for item in messages], separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:24]
    return {
        "thread": {
            "id": tid,
            "description": state._thread_title(tid),
            "harness": read_thread_engine(_thread_dir(tid)).name,
            "status": status.get("stage", "ready"),
            "error": str(status.get("error", ""))[:500] or None,
            "workspace": _thread_workspace(tid),
            "revision": revision,
        },
        "messages": messages,
        "has_older_messages": start > 0 or covered_raw < len(selected),
        "next_before": (before + covered_raw
                        if start > 0 or covered_raw < len(selected) else None),
        "truncated": truncated,
    }


def _list_threads() -> dict[str, Any]:
    values: list[tuple[int, dict[str, Any]]] = []
    for tid in state.MANAGER.list()[:MAX_THREADS]:
        try:
            status = state._get_status(tid)
            workspace = _thread_workspace(tid)
            values.append((threads._thread_status_rank(tid, status.get("stage", "ready")), {
                "id": tid,
                "description": state._thread_title(tid),
                "harness": read_thread_engine(_thread_dir(tid)).name,
                "status": status.get("stage", "ready"),
                "repo_key": workspace["repo_key"],
                "repo_label": workspace["repo_label"],
                "unread": state._has_unseen_response(tid),
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
        run, busy = threads._accept_message_run_locked(tid, text, dispatch_key=dispatch_key)
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
                raise HTTPException(status_code=409, detail="Phone draft conflicts with an existing thread")
            expected_domain = domain or (state.DOMAINS[0] if state.DOMAINS else None)
            try:
                existing_engine = read_thread_engine(_thread_dir(tid)).name
            except ThreadEngineError as error:
                raise HTTPException(status_code=409, detail="Thread harness is unavailable") from error
            if (replay.text != body.message or existing_engine != body.harness
                    or state._get_status(tid).get("domain", "") != (expected_domain or "")):
                raise HTTPException(status_code=409, detail="Idempotency-Key conflicts with prior message")
            return tid, replay, None, True
        try:
            tid, run_id, selected = threads.create_thread_with_message_core(
                body.message, domain, engine=body.harness, thread_id=tid,
                dispatch_key=dispatch_key)
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
            "error": (run.error or "")[:500] or None,
            "updated_at": run.updated_at,
            "thread_status": state._get_status(tid).get("stage", "ready")}


def _cancel_pending_run(tid: str, run_id: str) -> dict[str, Any]:
    """Cancel only unclaimed work; a running model turn cannot be lied about."""
    _thread_dir(tid)
    _require_id(run_id, "run id")
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
    entries = _workspace_entries(tid)
    return {"workspace": _thread_workspace(tid), "files": entries,
            "truncated": len(entries) >= MAX_FILES}


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
    root = Path(state.MANAGER.thread_default_working_dir(tid)).resolve()
    buffer = io.BytesIO()
    total = 0
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
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
async def get_thread_history(tid: str, before: int) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_snapshot, tid, before)


@router.post("/threads")
async def create_thread(request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    body = await _validated_body(request, _CreateThread)
    assert isinstance(body, _CreateThread)
    key = _request_key(request)
    tid, run, domain, replay = await anyio.to_thread.run_sync(_create_and_submit, body, key)
    if not replay:
        background_tasks.add_task(threads._initialize_thread, tid, run.id, domain)
    return {"thread_id": tid, "run_id": run.id, "replayed": replay}


@router.post("/threads/{tid}/messages")
async def send_message(tid: str, request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    body = await _validated_body(request, _SendMessage)
    assert isinstance(body, _SendMessage)
    key = _request_key(request)
    run, busy, replay = await anyio.to_thread.run_sync(_submit_existing, tid, body.message, key)
    if not busy and not replay:
        background_tasks.add_task(threads._execute_run, run.id, tid, user_priority=True)
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
    data = await anyio.to_thread.run_sync(_workspace_archive, tid)
    return Response(data, media_type="application/gzip",
                    headers={"Content-Disposition": 'attachment; filename="workspace.tar.gz"',
                             "Cache-Control": "no-store"})


@router.get("/threads/{tid}/pins")
async def get_pins(tid: str) -> dict[str, Any]:
    directory = await anyio.to_thread.run_sync(_thread_dir, tid)
    pins = await anyio.to_thread.run_sync(list_pins, directory)
    return {"pins": [pin.__dict__ for pin in pins], "limit": MAX_PINS}


@router.post("/threads/{tid}/pins")
async def pin_response(tid: str, request: Request) -> dict[str, Any]:
    body = await _validated_body(request, _CreatePin)
    assert isinstance(body, _CreatePin)

    def create() -> dict[str, Any]:
        directory = _thread_dir(tid)
        snapshot = _snapshot(tid)
        response = next((message for message in snapshot["messages"]
                         if message["id"] == body.response_id
                         and message["role"] == "assistant"
                         and message["kind"] == "assistant"), None)
        if response is None:
            raise HTTPException(status_code=404, detail="Assistant response not found")
        try:
            pin = create_pin(directory, body.response_id, response["text"])
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return pin.__dict__

    return await anyio.to_thread.run_sync(create)


@router.delete("/threads/{tid}/pins/{response_id}")
async def unpin_response(tid: str, response_id: str) -> Response:
    def remove() -> None:
        directory = _thread_dir(tid)
        _require_id(response_id, "response id")
        delete_pin(directory, response_id)
    await anyio.to_thread.run_sync(remove)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
