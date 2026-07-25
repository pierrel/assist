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
from assist.thread_queue import THREAD_QUEUE


class WebAgentProtocolService:
    """Map protocol operations onto the web app's sole durable run service."""

    MAX_THREADS = 1024
    MAX_HIDDEN_THREADS = 256
    MAX_ACTIVE_TASKS_PER_PARENT = 32
    MAX_RETAINED_TASKS_PER_PARENT = 128
    MAX_LISTED_TERMINAL_TASKS = 32
    MAX_RUNS_PER_THREAD = 32
    MAX_PENDING_PER_THREAD = 32

    def __init__(self) -> None:
        self._admission_lock = threading.Lock()

    @staticmethod
    def _task_snapshot(thread_id: str, runs: list) -> dict | None:
        if not runs or runs[0].mode != "child":
            return None
        latest = runs[-1]
        first = runs[0]
        description = next(
            (candidate.text for candidate in reversed(runs) if candidate.text),
            "")
        return {
            "task_id": thread_id,
            "agent_name": latest.assistant_id,
            "description": description,
            "status": "running" if latest.status == "interrupted" else latest.status,
            "run_id": latest.id,
            "work_id": latest.work_id,
            "parent_thread_id": first.parent_thread_id,
            "result": latest.result,
            "error": latest.error,
            "created_at": first.created_at,
            "updated_at": latest.updated_at,
        }

    def create_thread(self, thread_id: str | None = None,
                      metadata: dict | None = None) -> dict:
        with self._admission_lock:
            hidden = metadata is not None
            if not hidden and len(MANAGER.list()) >= self.MAX_THREADS:
                raise HTTPException(status_code=429, detail="Thread limit reached")
            if hidden:
                required = {"parent_thread_id", "parent_run_id", "dispatch_key"}
                if set(metadata or {}) != required:
                    raise HTTPException(status_code=422, detail="Invalid task metadata")
                if not os.path.isdir(MANAGER.thread_dir(metadata["parent_thread_id"])):
                    raise FileNotFoundError(metadata["parent_thread_id"])
                existing = thread_id and os.path.isdir(MANAGER.thread_dir(thread_id))
                child_runs = _runs().scan_children()
                latest_by_task = {}
                for child in child_runs:
                    if child.parent_thread_id == metadata["parent_thread_id"]:
                        latest_by_task[child.thread_id] = child
                active_count = sum(
                    child.status in NONTERMINAL_STATUSES
                    or child.status == "interrupted"
                    for child in latest_by_task.values())
                if (not existing
                        and active_count >= self.MAX_ACTIVE_TASKS_PER_PARENT):
                    raise HTTPException(
                        status_code=429, detail="Active task limit reached")
                if not existing and len(latest_by_task) >= self.MAX_RETAINED_TASKS_PER_PARENT:
                    parent_runs = _runs().list(metadata["parent_thread_id"])

                    def wake_consumed(child) -> bool:
                        key = f"task-completion:{child.id}"
                        start = next((index for index, candidate in enumerate(parent_runs)
                                      if candidate.dispatch_key == key), None)
                        if start is None:
                            return False
                        wake_work_id = parent_runs[start].work_id
                        end = next((index for index in range(start + 1, len(parent_runs))
                                    if (parent_runs[index].dispatch_key or "").startswith(
                                        "task-completion:")), len(parent_runs))
                        return any(candidate.work_id == wake_work_id
                                   and candidate.status == "success"
                                   for candidate in parent_runs[start:end])

                    terminal = sorted(
                        (child for child in latest_by_task.values()
                         if (child.status == "cancelled"
                             or (child.status in {"success", "error", "timeout"}
                                 and wake_consumed(child)))),
                        key=lambda child: child.created_at)
                    for child in terminal[:
                            len(latest_by_task) - self.MAX_RETAINED_TASKS_PER_PARENT + 1]:
                        MANAGER.hard_delete(child.thread_id)
                hidden_count = sum(
                    os.path.isfile(os.path.join(MANAGER.root_dir, name, ".subagent"))
                    for name in os.listdir(MANAGER.root_dir))
                if not existing and hidden_count >= self.MAX_HIDDEN_THREADS:
                    raise HTTPException(status_code=429, detail="Task limit reached")
            now = datetime.now(UTC).isoformat()
            return {"thread_id": MANAGER.reserve(
                        thread_id, hidden=(metadata if hidden else False)), "created_at": now,
                    "updated_at": now, "status": "idle"}

    def get_thread(self, thread_id: str) -> dict:
        tdir = MANAGER.thread_dir(thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(thread_id)
        runs = _runs().list(thread_id)
        task = self._task_snapshot(thread_id, runs)
        if task is not None:
            values = {"async_task": task}
        else:
            by_task = {}
            for child in _runs().scan_children():
                if child.parent_thread_id == thread_id:
                    by_task.setdefault(child.thread_id, []).append(child)
            tasks = [snapshot for child_tid, child_runs in by_task.items()
                     if (snapshot := self._task_snapshot(
                         child_tid, child_runs)) is not None]
            tasks.sort(key=lambda value: value["created_at"])
            active = [value for value in tasks
                      if value["status"] in NONTERMINAL_STATUSES
                      or value["status"] == "running"]
            terminal = [value for value in tasks if value not in active]
            shown = active + terminal[-self.MAX_LISTED_TERMINAL_TASKS:]
            for value in shown:
                value["description"] = str(value["description"] or "")[:500]
                value["result"] = None
                value["error"] = None
            values = {
                "async_tasks": shown,
                "async_tasks_truncated": len(shown) < len(tasks),
            }
        return {
            "thread_id": thread_id,
            "updated_at": datetime.fromtimestamp(
                os.path.getmtime(tdir), UTC).isoformat(),
            "status": ("busy" if (_get_status(thread_id).get("stage") in BUSY_STAGES
                                   or any(run.status in NONTERMINAL_STATUSES
                                          for run in runs)) else "idle"),
            "values": values,
        }

    def create_run(self, thread_id: str, assistant_id: str, text: str,
                   *, multitask_strategy: str | None = None,
                   metadata: dict | None = None):
        with _RUN_ADMISSION_LOCK:
            if not os.path.isdir(MANAGER.thread_dir(thread_id)):
                raise FileNotFoundError(thread_id)
            marker = os.path.join(MANAGER.thread_dir(thread_id), ".subagent")
            if not os.path.isfile(marker):
                raise HTTPException(status_code=409, detail="Runs are task-only")
            required = {"parent_thread_id", "parent_run_id", "dispatch_key"}
            if set(metadata or {}) != required:
                raise HTTPException(status_code=422, detail="Invalid task metadata")
            runs = _runs().list(thread_id)
            replay = next((candidate for candidate in runs
                           if candidate.dispatch_key == metadata["dispatch_key"]), None)
            if replay is not None:
                if replay.assistant_id != assistant_id or replay.text != text:
                    raise HTTPException(status_code=409, detail="Task key conflict")
                return replay
            interrupt = multitask_strategy == "interrupt"
            active = any(run.status in {"running", "interrupted"} for run in runs)
            if (interrupt and runs and not active
                    and runs[-1].status in TERMINAL_STATUSES):
                raise HTTPException(status_code=409, detail="Task already completed")
            if interrupt:
                if not active:
                    for candidate in runs:
                        if candidate.status == "pending":
                            _runs().cancel_pending(thread_id, candidate.id)
            try:
                run = _create_run(
                    thread_id, text, assistant_id=assistant_id,
                    work_id=(runs[0].work_id if runs else thread_id), mode="child",
                    parent_thread_id=metadata["parent_thread_id"],
                    parent_run_id=metadata["parent_run_id"],
                    dispatch_key=metadata["dispatch_key"],
                    max_runs=self.MAX_RUNS_PER_THREAD,
                    max_pending=self.MAX_PENDING_PER_THREAD,
                    multitask_strategy=(multitask_strategy or "enqueue"))
            except InvalidRunTransition as exc:
                status = 429 if "limit reached" in str(exc) else 409
                raise HTTPException(status_code=status, detail=str(exc)) from exc
            if not active:
                _RESUME_SCHEDULER.submit(run.id, thread_id)
            return run

    def get_run(self, thread_id: str, run_id: str):
        return _runs().get(thread_id, run_id)

    def cancel_run(self, thread_id: str, run_id: str):
        with _RUN_ADMISSION_LOCK:
            run = _runs().get(thread_id, run_id)
            logical = [candidate for candidate in _runs().list(thread_id)
                       if candidate.work_id == run.work_id]
            running = next((candidate for candidate in logical
                            if candidate.status == "running"), None)
            existing_marker = next((candidate for candidate in reversed(logical)
                                    if candidate.status == "pending"
                                    and candidate.multitask_strategy == "cancel"), None)
            if running is not None and existing_marker is not None:
                _RESUME_SCHEDULER.submit(existing_marker.id, thread_id)
                THREAD_QUEUE.request_pause(thread_id)
                return existing_marker
            for candidate in logical:
                if candidate.status == "pending":
                    _runs().cancel_pending(thread_id, candidate.id)
            if running is None:
                interrupted = [candidate for candidate in logical
                               if candidate.status == "interrupted"]
                for candidate in interrupted:
                    _runs().cancel(thread_id, candidate.id)
                result = interrupted[-1] if interrupted else run
                if interrupted:
                    result = _runs().get(thread_id, result.id)
                elif run.status == "pending":
                    result = _runs().get(thread_id, run.id)
                _dispatch_pending_after(thread_id, run_id)
                return result
            marker = _create_run(
                thread_id, None, assistant_id=running.assistant_id,
                work_id=running.work_id, mode="child",
                parent_thread_id=running.parent_thread_id,
                parent_run_id=running.parent_run_id,
                dispatch_key=f"task-cancel:{running.work_id}",
                max_runs=self.MAX_RUNS_PER_THREAD,
                max_pending=self.MAX_PENDING_PER_THREAD,
                multitask_strategy="cancel")
            _RESUME_SCHEDULER.submit(marker.id, thread_id)
            THREAD_QUEUE.request_pause(thread_id)
            return marker


SERVICE = WebAgentProtocolService()
