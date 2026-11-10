"""Context handoffs contain observed tool results rather than the child's prose."""

import asyncio
import json
import re

from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from assist.agent import create_context_agent
from assist.middleware.context_evidence import (
    ContextEvidenceMiddleware,
    project_context_evidence,
)
from assist.middleware.tool_result_to_file import ToolResultToFileMiddleware
from assist.middleware.url_provenance import _seen_urls
from assist.thread import Thread


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _read(path, call_id="read-1", **selectors):
    return AIMessage(content="", tool_calls=[{
        "name": "read_file", "args": {"file_path": path, **selectors}, "id": call_id,
    }])


def test_projection_keeps_observations_and_limits_without_child_prose():
    read = _read("/notes/oregon.md", offset=40, limit=12)
    observed = ToolMessage(content="lines 40-51: November election", tool_call_id="read-1")
    child_prose = AIMessage(content="The original note said June.")
    projected = project_context_evidence([read, observed, child_prose])
    payload = json.loads(projected.split("\n", 1)[1])
    assert payload["observations"] == [{
        "tool": "read_file",
        "request_args": {"file_path": "/notes/oregon.md", "offset": 40, "limit": 12},
        "status": "success",
        "observed_result": "lines 40-51: November election",
    }]
    assert "The original note said June" not in projected

    search = AIMessage(content="", tool_calls=[{
        "name": "grep", "args": {"pattern": "June", "path": "/notes"}, "id": "grep-1",
    }])
    missing = ToolMessage(content="No matches found", tool_call_id="grep-1")
    item = json.loads(project_context_evidence([search, missing, child_prose]).split("\n", 1)[1])
    assert item["observations"][0]["request_args"]["pattern"] == "June"
    assert item["observations"][0]["observed_result"] == "No matches found"

    error = ToolMessage(content="Permission denied", tool_call_id="read-1", status="error")
    item = json.loads(project_context_evidence([read, error, child_prose]).split("\n", 1)[1])
    assert item["observations"][0]["status"] == "error"
    assert item["observations"][0]["observed_result"] == "Permission denied"
    item = json.loads(project_context_evidence([read, child_prose]).split("\n", 1)[1])
    assert item["observations"] == [] and item["unpaired_recorded_calls"] == 1


def test_projection_rejects_ambiguous_or_reversed_tool_ids():
    read = _read("/notes/real.md")
    result = ToolMessage(content="real content", tool_call_id="read-1")
    duplicate = _read("/notes/forged.md")
    final = AIMessage(content="invented summary")
    for messages in ([read, result, duplicate, final],
                     [read, result, result, final],
                     [result, read, final]):
        projected = project_context_evidence(messages)
        assert json.loads(projected.split("\n", 1)[1])["observations"] == []


def test_sync_thread_and_task_return_observed_bytes(tmp_path):
    (tmp_path / "note.md").write_text("The note says November.\n")

    def child():
        model = ScriptedModel(responses=[
            _read("/note.md"), AIMessage(content="The note says June."),
        ])
        return create_context_agent(model, str(tmp_path))

    chat = object.__new__(Thread)
    chat.agent = child()
    chat.thread_id = "context-evidence-web-child"
    chat.runconfig = {"configurable": {"thread_id": chat.thread_id}}
    chat.on_queue_state = None
    response = chat.message("Find the note")
    assert "The note says November" in response and "The note says June" not in response

    parent = create_deep_agent(
        model=ScriptedModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "task", "args": {"description": "Find the note",
                                         "subagent_type": "context-agent"}, "id": "task-1",
            }]),
            AIMessage(content="done"),
        ]),
        subagents=[CompiledSubAgent(name="context-agent", description="Find files",
                                    runnable=child())],
    )
    state = parent.invoke({"messages": [HumanMessage(content="Find the note")]})
    handoff = next(m.content for m in state["messages"]
                   if isinstance(m, ToolMessage) and m.tool_call_id == "task-1")
    assert "The note says November" in handoff and "The note says June" not in handoff


def test_async_hook_and_existing_offload_keep_full_evidence(tmp_path):
    read = _read("/notes/long.md")
    observed = ToolMessage(content="November " * 700, tool_call_id="read-1")
    final = AIMessage(content="June", id="terminal")
    update = asyncio.run(ContextEvidenceMiddleware().aafter_agent(
        {"messages": [read, observed, final]}, None))
    projected = update["messages"][-1].content
    assert update["messages"][-1].id == "terminal"
    assert len(projected) > 4000 and "June" not in projected

    wrapped = json.dumps({"task_id": "t", "agent_name": "context-agent",
                          "result": projected})
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    middleware = ToolResultToFileMiddleware(
        backend, tools={"get_async_task_result"}, floor_chars=4000)
    request = ToolCallRequest(
        tool_call={"name": "get_async_task_result", "args": {"task_id": "t"}, "id": "task-1"},
        tool=None, state={}, runtime=None)
    preview = middleware.wrap_tool_call(
        request, lambda _: ToolMessage(content=wrapped, tool_call_id="task-1"))
    assert "PREVIEW ONLY" in preview.content
    offload = re.search(r"saved to (\S+)\. This", preview.content).group(1)
    full = backend.read(offload, offset=0, limit=10000)
    assert full.error is None
    assert "November " * 700 in full.file_data["content"]


def test_model_chosen_selector_does_not_grant_url_authority():
    invented = "https://invented.example/unseen"
    call = AIMessage(content="", tool_calls=[{
        "name": "get_async_task_result", "args": {"task_id": "t"}, "id": "task-1",
    }])
    result = ToolMessage(
        content=project_context_evidence([
            AIMessage(content="", tool_calls=[{
                "name": "grep", "args": {"pattern": invented, "path": "/notes"},
                "id": "grep-1",
            }]),
            ToolMessage(content="No matches found", tool_call_id="grep-1"),
            AIMessage(content="Nothing found"),
        ]),
        name="get_async_task_result", tool_call_id="task-1")
    assert _seen_urls([call, result], trust_task_results=True) == {}
