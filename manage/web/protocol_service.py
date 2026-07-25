"""Web composition root for the Agent Protocol HTTP adapter."""
from __future__ import annotations

import os
import threading
from datetime import UTC, datetime
from fastapi import HTTPException

from assist.run_service import (
    InvalidRunTransition,
    NONTERMINAL_STATUSES,
    TERMINAL_STATUSES,
)
from manage.web.state import BUSY_STAGES, MANAGER, _get_status
from manage.web.threads import (
    _RESUME_SCHEDULER,
    _RUN_ADMISSION_LOCK,
    _create_run,
    _dispatch_pending_after,
    _runs,
)


class WebAgentProtocolService:
    """Map protocol operations onto the web app's sole durable run service."""

    MAX_THREADS = 1024
    MAX_RUNS_PER_THREAD = 256
    MAX_PENDING_PER_THREAD = 32

    def __init__(self) -> None:
        self._admission_lock = threading.Lock()

    def create_thread(self) -> dict:
        with self._admission_lock:
            if len(MANAGER.list()) >= self.MAX_THREADS:
                raise HTTPException(status_code=429, detail="Thread limit reached")
            now = datetime.now(UTC).isoformat()
            return {"thread_id": MANAGER.reserve(), "created_at": now,
                    "updated_at": now, "status": "idle"}

    def get_thread(self, thread_id: str) -> dict:
        tdir = MANAGER.thread_dir(thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(thread_id)
        messages = MANAGER.get(thread_id, sandbox_backend=None).get_messages()
        runs = _runs().list(thread_id)
        return {
            "thread_id": thread_id,
            "updated_at": datetime.fromtimestamp(
                os.path.getmtime(tdir), UTC).isoformat(),
            "status": ("busy" if (_get_status(thread_id).get("stage") in BUSY_STAGES
                                   or any(run.status in NONTERMINAL_STATUSES
                                          for run in runs)) else "idle"),
            "values": {"messages": messages},
        }

    def create_run(self, thread_id: str, assistant_id: str, text: str,
                   *, multitask_strategy: str | None = None):
        with _RUN_ADMISSION_LOCK:
            if not os.path.isdir(MANAGER.thread_dir(thread_id)):
                raise FileNotFoundError(thread_id)
            if multitask_strategy == "interrupt":
                current_runs = _runs().list(thread_id)
                children = _runs().scan_children()
                unsafe = any(
                    run.status == "running"
                    or (run.status == "interrupted" and (
                        _get_status(thread_id).get("stage") == "paused"
                        or any(child.parent_thread_id == thread_id
                               and child.parent_run_id == run.id
                               and child.status in {"pending", "running"}
                               for child in children)))
                    for run in current_runs)
                if unsafe:
                    raise HTTPException(
                        status_code=409,
                        detail="A running or interrupted invocation cannot be cancelled safely")
                try:
                    run = _create_run(
                        thread_id, text, assistant_id=assistant_id,
                        cancel_pending=True, max_runs=self.MAX_RUNS_PER_THREAD,
                        max_pending=self.MAX_PENDING_PER_THREAD,
                        multitask_strategy="interrupt")
                except InvalidRunTransition as exc:
                    status = 429 if "limit reached" in str(exc) else 409
                    raise HTTPException(status_code=status, detail=str(exc)) from exc
                _RESUME_SCHEDULER.submit(run.id, thread_id)
                return run

            try:
                run = _create_run(
                    thread_id, text, assistant_id=assistant_id,
                    max_runs=self.MAX_RUNS_PER_THREAD,
                    max_pending=self.MAX_PENDING_PER_THREAD)
            except InvalidRunTransition as exc:
                status = 429 if "limit reached" in str(exc) else 409
                raise HTTPException(status_code=status, detail=str(exc)) from exc
            _dispatch_pending_after(thread_id)
            return run

    def get_run(self, thread_id: str, run_id: str):
        return _runs().get(thread_id, run_id)

    def cancel_run(self, thread_id: str, run_id: str):
        run = _runs().get(thread_id, run_id)
        if run.mode == "child" or run.status in {"running", "interrupted"}:
            raise HTTPException(
                status_code=409,
                detail="This invocation cannot be cancelled safely")
        if run.status in TERMINAL_STATUSES:
            return run
        try:
            cancelled = _runs().cancel_pending(thread_id, run_id)
        except InvalidRunTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _dispatch_pending_after(thread_id, run_id)
        return cancelled


SERVICE = WebAgentProtocolService()
