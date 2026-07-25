"""Agent Protocol-backed subagent tools for the web supervisor.

The tools are intentionally shaped like Deep Agents' async-subagent tools, but
use an injected in-process ASGI app.  Deep Agents 0.6.x has no public client
injection seam for an application-owned ASGI app, and its synchronous tool path
rejects ``url=None``.  Keeping the transport here avoids a public listener and
keeps durable task truth in the web Run service.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any, Iterator

import anyio
import httpx
from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool
from langgraph_sdk.client import LangGraphClient
from pydantic import Field
from starlette.types import ASGIApp


SUBAGENTS = {
    "context-agent": (
        "Discover relevant context in the user's local files. Read-only; use "
        "before answering whenever local context could inform the response."
    ),
    "research-agent": (
        "Research external topics thoroughly and return grounded findings with "
        "their sources."
    ),
    "critique-agent": (
        "Review a supplied code diff for correctness, missing tests, simplicity, "
        "and security issues."
    ),
}


@dataclass(frozen=True)
class AsyncTaskContext:
    """Identity of the ordinary parent Run executing task-management tools."""

    parent_thread_id: str
    parent_run_id: str
    parent_work_id: str


_CONTEXT: ContextVar[AsyncTaskContext | None] = ContextVar(
    "assist_async_task_context", default=None)
_APP: ASGIApp | None = None


def configure_async_subagent_app(app: ASGIApp) -> None:
    """Install the private Agent Protocol ASGI app used by all five tools."""
    global _APP
    _APP = app


@contextmanager
def async_task_context(context: AsyncTaskContext) -> Iterator[None]:
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _context(runtime: ToolRuntime) -> tuple[AsyncTaskContext, str]:
    context = _CONTEXT.get()
    if context is None:
        raise RuntimeError("subagent tool invoked outside a configured web run")
    if not runtime.tool_call_id:
        raise ValueError("tool call ID is required for subagent management")
    return context, runtime.tool_call_id


async def _with_client(operation):
    if _APP is None:
        raise RuntimeError("subagent ASGI app is not configured")
    transport = httpx.ASGITransport(app=_APP)
    async with httpx.AsyncClient(
            base_url="http://assist-agent", transport=transport) as raw:
        return await operation(LangGraphClient(raw))


def _run(operation):
    """Run one private ASGI exchange from the graph's synchronous worker."""
    return anyio.run(_with_client, operation)


def _task_id(context: AsyncTaskContext, tool_call_id: str) -> str:
    key = f"{context.parent_work_id}:{tool_call_id}"
    return "sub-" + hashlib.sha256(key.encode()).hexdigest()[:24]


def _task_value(thread: dict[str, Any]) -> dict[str, Any]:
    value = (thread.get("values") or {}).get("async_task")
    if not isinstance(value, dict):
        raise ValueError("thread is not a subagent task")
    return value


def _format_task(task: dict[str, Any], *, include_result: bool = True) -> str:
    description = str(task.get("description") or "")
    fields = {
        "task_id": task.get("task_id"),
        "agent_name": task.get("agent_name"),
        "description": description[:500],
        "status": task.get("status"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if include_result and task.get("status") in {
            "success", "error", "timeout", "cancelled"}:
        fields["result"] = task.get("result")
        fields["error"] = task.get("error")
    return json.dumps(fields, ensure_ascii=False)


_TaskId = Annotated[str, Field(
    description="Exact full task_id returned by start_async_task.")]


def _start_async_task(
    description: Annotated[str, Field(
        min_length=1, max_length=64_000,
        description="Complete, self-contained work for the subagent.")],
    subagent_type: Annotated[
        str, Field(description="One listed subagent type.")],
    runtime: ToolRuntime,
) -> str:
    context, tool_call_id = _context(runtime)
    if subagent_type not in SUBAGENTS:
        allowed = ", ".join(f"`{name}`" for name in SUBAGENTS)
        return f"Unknown subagent type `{subagent_type}`. Available: {allowed}"
    task_id = _task_id(context, tool_call_id)
    dispatch_key = f"{context.parent_work_id}:{tool_call_id}"

    async def launch(client):
        await client.threads.create(
            thread_id=task_id,
            if_exists="do_nothing",
            metadata={
                "parent_thread_id": context.parent_thread_id,
                "parent_run_id": context.parent_run_id,
                "dispatch_key": dispatch_key,
            },
        )
        return await client.runs.create(
            task_id,
            subagent_type,
            input={"messages": [{"role": "user", "content": description}]},
            metadata={
                "parent_thread_id": context.parent_thread_id,
                "parent_run_id": context.parent_run_id,
                "dispatch_key": dispatch_key,
            },
        )

    _run(launch)
    return (f"Started subagent. task_id: {task_id}. In the user reply, call it a "
            "subagent or task, never background or async. Report this full ID and "
            "return now; the result will trigger a follow-up.")


def _check_async_task(
    task_id: _TaskId,
    runtime: ToolRuntime,
) -> str:
    context, _ = _context(runtime)

    async def check(client):
        return await client.threads.get(task_id)

    try:
        task = _task_value(_run(check))
    except (httpx.HTTPStatusError, ValueError):
        return f"Task `{task_id}` was not found in this conversation."
    if task.get("parent_thread_id") != context.parent_thread_id:
        return f"Task `{task_id}` was not found in this conversation."
    return _format_task(task)


def _list_async_tasks(runtime: ToolRuntime) -> str:
    context, _ = _context(runtime)

    async def get_parent(client):
        return await client.threads.get(context.parent_thread_id)

    values = (_run(get_parent).get("values") or {})
    tasks = (values.get("async_tasks") or [])[-64:]
    if not tasks:
        return "No subagent tasks exist for this conversation."
    rendered = "\n".join(_format_task(task, include_result=False) for task in tasks)
    if values.get("async_tasks_truncated"):
        rendered += "\nOlder completed task history was pruned from this listing."
    return rendered


def _update_async_task(
    task_id: _TaskId,
    instructions: Annotated[str, Field(
        min_length=1, max_length=64_000,
        description="Complete replacement or follow-up instructions for the task.")],
    runtime: ToolRuntime,
) -> str:
    context, tool_call_id = _context(runtime)
    update_key = f"{context.parent_work_id}:{tool_call_id}"

    async def update(client):
        thread = await client.threads.get(task_id)
        task = _task_value(thread)
        if task.get("parent_thread_id") != context.parent_thread_id:
            return None
        if task.get("status") in {"success", "error", "timeout", "cancelled"}:
            return thread
        await client.runs.create(
            task_id,
            task["agent_name"],
            input={"messages": [{"role": "user", "content": instructions}]},
            multitask_strategy="interrupt",
            metadata={
                "parent_thread_id": context.parent_thread_id,
                "parent_run_id": context.parent_run_id,
                "dispatch_key": update_key,
            },
        )
        return await client.threads.get(task_id)

    thread = _run(update)
    if thread is None:
        return f"Task `{task_id}` was not found in this conversation."
    task = _task_value(thread)
    if task.get("status") in {"success", "error", "timeout", "cancelled"}:
        return f"Task `{task_id}` already completed with status {task['status']}."
    return "Task updated (queued): " + _format_task(task, include_result=False)


def _cancel_async_task(
    task_id: _TaskId,
    runtime: ToolRuntime,
) -> str:
    context, _ = _context(runtime)

    async def cancel(client):
        before = _task_value(await client.threads.get(task_id))
        if before.get("parent_thread_id") != context.parent_thread_id:
            return None, None
        if before.get("status") in {"success", "error", "timeout", "cancelled"}:
            return before, before
        await client.runs.cancel(task_id, before["run_id"])
        return before, _task_value(await client.threads.get(task_id))

    before, after = _run(cancel)
    if before is None:
        return f"Task `{task_id}` was not found in this conversation."
    if before.get("status") in {"success", "error", "timeout"}:
        return f"Task `{task_id}` already completed with status {before['status']}."
    if before.get("status") == "cancelled":
        return f"Task `{task_id}` is already cancelled."
    verb = "Cancellation requested" if after.get("status") == "pending" else "Task cancelled"
    return verb + ": " + _format_task(after, include_result=False)


_AVAILABLE = "\n".join(
    f"- {name}: {description}" for name, description in SUBAGENTS.items())

async_task_tools = (
    StructuredTool.from_function(
        name="start_async_task", func=_start_async_task,
        description=("Start a subagent and return its task ID immediately. In the "
                     "user reply, call it a subagent or task, never background or "
                     "async. Never poll in the launch turn. Available types:\n"
                     + _AVAILABLE)),
    StructuredTool.from_function(
        name="check_async_task", func=_check_async_task,
        description="Fetch one task's current status and terminal result using its full ID."),
    StructuredTool.from_function(
        name="update_async_task", func=_update_async_task,
        description=("Queue replacement instructions for a task. Active inference "
                     "or tool work finishes its current graph slice first.")),
    StructuredTool.from_function(
        name="cancel_async_task", func=_cancel_async_task,
        description=("Cancel pending work or request that active work stop at its "
                     "next model boundary; in-flight inference/tools are not preempted.")),
    StructuredTool.from_function(
        name="list_async_tasks", func=_list_async_tasks,
        description="List this conversation's durable tasks with fresh statuses and full IDs."),
)
