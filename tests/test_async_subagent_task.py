from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from assist.async_subagents import (
    AsyncTaskContext,
    async_task_context,
    async_task_tools,
    configure_async_subagent_app,
)


TOOLS = {tool.name: tool for tool in async_task_tools}


def _runtime(call_id="call-1"):
    return SimpleNamespace(tool_call_id=call_id)


@pytest.fixture
def protocol():
    app = FastAPI()
    calls = []
    tasks = {}

    @app.post("/threads")
    async def create_thread(request: Request):
        body = await request.json()
        calls.append(("thread", body))
        tid = body["thread_id"]
        tasks.setdefault(tid, {
            "task_id": tid,
            "agent_name": None,
            "description": None,
            "status": "pending",
            "run_id": None,
            "parent_thread_id": body["metadata"]["parent_thread_id"],
            "created_at": "2026-07-25T00:00:00Z",
            "updated_at": "2026-07-25T00:00:00Z",
        })
        return {"thread_id": tid, "status": "idle", "values": {}}

    @app.post("/threads/{tid}/runs")
    async def create_run(tid: str, request: Request):
        body = await request.json()
        calls.append(("run", tid, body))
        task = tasks[tid]
        task.update({
            "agent_name": body["assistant_id"],
            "description": body["input"]["messages"][0]["content"],
            "run_id": "run-" + body["metadata"]["dispatch_key"],
            "status": "pending",
        })
        return {
            "run_id": task["run_id"], "thread_id": tid,
            "assistant_id": task["agent_name"], "status": "pending",
            "multitask_strategy": body.get("multitask_strategy", "enqueue"),
        }

    @app.get("/threads/{tid}")
    def get_thread(tid: str):
        if tid == "parent":
            values = {"async_tasks": list(tasks.values())}
        elif tid in tasks:
            values = {"async_task": tasks[tid]}
        else:
            return {"thread_id": tid, "status": "idle", "values": {}}
        return {"thread_id": tid, "status": "idle", "values": values}

    @app.post("/threads/{tid}/runs/{run_id}/cancel")
    def cancel(tid: str, run_id: str):
        calls.append(("cancel", tid, run_id))
        tasks[tid]["status"] = "cancelled"
        return None

    configure_async_subagent_app(app)
    return calls, tasks


def test_five_upstream_shaped_tools_exist():
    assert set(TOOLS) == {
        "start_async_task", "check_async_task", "update_async_task",
        "cancel_async_task", "list_async_tasks",
    }


def test_requires_registered_web_execution_context(protocol):
    with pytest.raises(RuntimeError, match="outside a configured web run"):
        TOOLS["start_async_task"].func(
            "find it", "context-agent", _runtime())


def test_start_is_deterministic_and_uses_asgi(protocol):
    calls, _ = protocol
    context = AsyncTaskContext("parent", "parent-run", "parent-work")
    with async_task_context(context):
        first = TOOLS["start_async_task"].func(
            "inspect files", "context-agent", _runtime("tc-7"))
        second = TOOLS["start_async_task"].func(
            "inspect files", "context-agent", _runtime("tc-7"))

    task_id = first.split("task_id: ", 1)[1].split(".", 1)[0]
    assert second == first
    assert task_id.startswith("sub-") and len(task_id) == 28
    assert [call[1]["thread_id"] for call in calls if call[0] == "thread"] == [
        task_id, task_id]
    assert all(call[2]["metadata"]["dispatch_key"] == "parent-work:tc-7"
               for call in calls if call[0] == "run")


def test_start_admits_delegate_agent(protocol):
    calls, _ = protocol
    with async_task_context(AsyncTaskContext("parent", "parent-run", "work")):
        result = TOOLS["start_async_task"].func(
            "complete one task", "delegate-agent", _runtime("delegate"))

    assert "task_id: sub-" in result
    run = next(call for call in calls if call[0] == "run")
    assert run[2]["assistant_id"] == "delegate-agent"
    assert run[2]["input"]["messages"] == [
        {"role": "user", "content": "complete one task"}]


def test_tool_node_injects_runtime_when_model_calls_start(protocol):
    """Exercise LangGraph's real tool boundary, not ``StructuredTool.func``."""
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(async_task_tools))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)

    with async_task_context(AsyncTaskContext("parent", "parent-run", "work")):
        result = graph.compile().invoke({"messages": [AIMessage(
            content="", tool_calls=[{
                "name": "start_async_task",
                "args": {"description": "inspect", "subagent_type": "context-agent"},
                "id": "model-call-1",
                "type": "tool_call",
            }])]}, {"configurable": {"thread_id": "parent"}})

    assert "task_id: sub-" in result["messages"][-1].content


def test_check_list_update_and_cancel_are_parent_scoped(protocol):
    _, tasks = protocol
    context = AsyncTaskContext("parent", "parent-run", "parent-work")
    with async_task_context(context):
        launched = TOOLS["start_async_task"].func(
            "inspect", "context-agent", _runtime("start"))
        task_id = launched.split("task_id: ", 1)[1].split(".", 1)[0]
        assert task_id in TOOLS["list_async_tasks"].func(_runtime("list"))
        assert '"status": "pending"' in TOOLS["check_async_task"].func(
            task_id, _runtime("check"))
        updated = TOOLS["update_async_task"].func(
            task_id, "inspect only org files", _runtime("update"))
        assert "Task updated" in updated
        assert tasks[task_id]["description"] == "inspect only org files"
        cancelled = TOOLS["cancel_async_task"].func(
            task_id, _runtime("cancel"))
        assert "Task cancelled" in cancelled
        assert tasks[task_id]["status"] == "cancelled"

    with async_task_context(AsyncTaskContext("other", "r", "w")):
        assert "not found in this conversation" in TOOLS[
            "check_async_task"].func(task_id, _runtime("foreign"))


def test_unknown_agent_is_rejected_without_asgi_call(protocol):
    calls, _ = protocol
    with async_task_context(AsyncTaskContext("parent", "r", "w")):
        result = TOOLS["start_async_task"].func(
            "invent", "invented-agent", _runtime())
    assert "Unknown subagent" in result
    assert calls == []
