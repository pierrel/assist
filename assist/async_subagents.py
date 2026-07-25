"""Async-only subagent ``task`` tool for the web run executor.

The tool durably dispatches a child through an executor-provided callback, then
uses LangGraph's public interrupt/resume mechanism to release the parent run.
Replaying the interrupted node calls the same idempotent dispatch key and pairs
the child result to the original tool call.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TypedDict

import anyio
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


SUBAGENTS = {
    "context-agent": (
        "Discover relevant context in the user's local files. Read-only; use "
        "before answering whenever local context could inform the response."
    ),
    "research-agent": (
        "Research external topics thoroughly and return grounded findings with "
        "their sources."
    ),
    "background-research-agent": (
        "Delegate research to run after this turn. It returns immediately and the "
        "findings are posted to this conversation as an automatic follow-up. Use it "
        "when local context already supports a useful answer now."
    ),
    "critique-agent": (
        "Review a supplied code diff for correctness, missing tests, simplicity, "
        "and security issues."
    ),
}


class ChildIdentity(TypedDict):
    """Identity of the durably created-or-found child work."""

    thread_id: str
    run_id: str


Dispatch = Callable[[str, str, str], ChildIdentity]


@dataclass(frozen=True)
class AsyncTaskContext:
    """Parent identity and child dispatcher registered for one web execution."""

    parent_thread_id: str
    parent_run_id: str
    parent_work_id: str
    dispatch: Dispatch


_CONTEXT: ContextVar[AsyncTaskContext | None] = ContextVar(
    "assist_async_task_context", default=None)


@contextmanager
def async_task_context(context: AsyncTaskContext) -> Iterator[None]:
    """Register the parent execution context for task calls in this context."""
    token = _CONTEXT.set(context)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


class TaskSchema(BaseModel):
    description: str = Field(
        description="A complete, self-contained description of the delegated task.")
    subagent_type: str = Field(
        description="The subagent type to use; it must be one of the listed types.")


_AVAILABLE = "\n".join(f"- {name}: {description}"
                       for name, description in SUBAGENTS.items())
TASK_DESCRIPTION = f"""Delegate a task to a specialized async subagent.

The parent run pauses while the child runs and resumes with its result. Available
subagent types:
{_AVAILABLE}"""


def _prepare(subagent_type: str, runtime: ToolRuntime) -> tuple[AsyncTaskContext, str]:
    if subagent_type not in SUBAGENTS:
        allowed = ", ".join(f"`{name}`" for name in SUBAGENTS)
        raise ValueError(
            f"unknown subagent type {subagent_type!r}; available types: {allowed}")
    if not runtime.tool_call_id:
        raise ValueError("tool call ID is required for subagent invocation")
    context = _CONTEXT.get()
    if context is None:
        raise RuntimeError(
            "async subagent task invoked outside a configured web run; "
            "the web executor must register async_task_context")
    return context, f"{context.parent_work_id}:{runtime.tool_call_id}"


def _interrupt_and_pair(
    context: AsyncTaskContext,
    child: ChildIdentity,
    tool_call_id: str,
) -> Command:
    if not child.get("thread_id") or not child.get("run_id"):
        raise ValueError("async subagent dispatch must return child thread_id and run_id")
    result = interrupt({
        "parent_thread_id": context.parent_thread_id,
        "parent_run_id": context.parent_run_id,
        "parent_work_id": context.parent_work_id,
        "child_thread_id": child["thread_id"],
        "child_run_id": child["run_id"],
        "tool_call_id": tool_call_id,
    })
    content = result if isinstance(result, str) else str(result)
    return Command(update={
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
    })


def _deferred_pair(child: ChildIdentity, tool_call_id: str) -> Command:
    return Command(update={"messages": [ToolMessage(
        content=("Research was scheduled and will be posted automatically as a "
                 f"follow-up (task_id: {child['run_id']}). Answer the user now "
                 "from the context already available."),
        tool_call_id=tool_call_id,
    )]})


def _pair(context: AsyncTaskContext, child: ChildIdentity,
          subagent_type: str, tool_call_id: str) -> Command:
    if subagent_type == "background-research-agent":
        return _deferred_pair(child, tool_call_id)
    return _interrupt_and_pair(context, child, tool_call_id)


def _task(
    description: str,
    subagent_type: str,
    runtime: ToolRuntime,
) -> Command:
    context, key = _prepare(subagent_type, runtime)
    child = context.dispatch(key, subagent_type, description)
    return _pair(context, child, subagent_type, runtime.tool_call_id)


async def _atask(
    description: str,
    subagent_type: str,
    runtime: ToolRuntime,
) -> Command:
    context, key = _prepare(subagent_type, runtime)
    child = await anyio.to_thread.run_sync(
        context.dispatch, key, subagent_type, description)
    return _pair(context, child, subagent_type, runtime.tool_call_id)


task_tool = StructuredTool.from_function(
    name="task",
    func=_task,
    coroutine=_atask,
    description=TASK_DESCRIPTION,
    infer_schema=False,
    args_schema=TaskSchema,
)


def _task_no_background(description: str, subagent_type: str,
                        runtime: ToolRuntime) -> Command:
    if subagent_type == "background-research-agent":
        raise ValueError("background delegation is unavailable in this run")
    return _task(description, subagent_type, runtime)


async def _atask_no_background(description: str, subagent_type: str,
                               runtime: ToolRuntime) -> Command:
    if subagent_type == "background-research-agent":
        raise ValueError("background delegation is unavailable in this run")
    return await _atask(description, subagent_type, runtime)


task_tool_no_background = StructuredTool.from_function(
    name="task",
    func=_task_no_background,
    coroutine=_atask_no_background,
    description=("Delegate one required task to an async specialized subagent. "
                 "The parent pauses and resumes with the result. Available types:\n"
                 + "\n".join(
                     f"- {name}: {description}" for name, description in SUBAGENTS.items()
                     if name != "background-research-agent")),
    infer_schema=False,
    args_schema=TaskSchema,
)
