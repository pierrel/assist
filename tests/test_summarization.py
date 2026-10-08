"""Offline regressions for real serialized requests and durable compaction."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai import BadRequestError

# Install the same provider harness used by all Assist agent factories.
import assist.agent  # noqa: F401
from assist.middleware.bad_request_retry import BadRequestRetryMiddleware
from assist.middleware.summarization import BoundedSummarizationMiddleware

LIMIT = 131072


def _error(tokens=168576):
    body = {"error": {"code": 400, "type": "exceed_context_size_error",
                      "message": f"request ({tokens} tokens) exceeds the available context size ({LIMIT} tokens)",
                      "n_prompt_tokens": tokens, "n_ctx": LIMIT}}
    response = httpx.Response(400, json=body, request=httpx.Request("POST", "http://unit.test/v1/chat/completions"))
    return BadRequestError(body["error"]["message"], response=response, body=body["error"])


def _model(provider):
    model = ChatOpenAI(model="test", api_key="EMPTY", base_url="http://unit.test/v1",
                       http_client=httpx.Client(transport=httpx.MockTransport(provider)),
                       http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
                       max_retries=0)
    model.profile = {"max_input_tokens": LIMIT}
    return model


def _response(text):
    return httpx.Response(200, json={"id": "test", "object": "chat.completion", "model": "test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


def test_overflow_reaches_compaction_without_sanitization():
    middleware = BadRequestRetryMiddleware()
    calls = []
    def reject(request):
        calls.append(request)
        raise _error()
    with pytest.raises(ContextOverflowError):
        middleware.wrap_model_call(ModelRequest(model=_model(lambda r: _response("ok")),
                                                messages=[HumanMessage("hello")]), reject)
    assert len(calls) == 1
    assert middleware._retry_count == 0


def test_dense_history_and_oversized_summary_continue_without_losing_raw_history():
    calls = []
    def provider(request):
        payload = json.loads(request.content)
        summary = "max_completion_tokens" in payload
        # A dense tokenizer assigns one token per serialized byte. Counting the
        # actual ChatOpenAI request also includes escaping, roles and tool schema.
        tokens = len(request.content)
        calls.append((summary, tokens))
        if tokens > LIMIT:
            return httpx.Response(400, json=_error(tokens).body)
        return _response("Retain the synthetic goal." if summary else "Conversation continued.")
    model = _model(provider)
    backend = StateBackend()
    graph = create_deep_agent(model, backend=backend, subagents=[],
        middleware=[BoundedSummarizationMiddleware(model, backend), BadRequestRetryMiddleware()],
        checkpointer=InMemorySaver())
    messages = [HumanMessage("Retain the synthetic goal."),
                AIMessage(content="", tool_calls=[{"id": "search", "name": "email_search", "args": {}}]),
                ToolMessage("x" * 160000, tool_call_id="search"), HumanMessage("Continue our conversation.")]
    config = {"configurable": {"thread_id": "dense-history"}}
    result = graph.invoke({"messages": messages}, config, durability="sync")
    state = graph.get_state(config).values
    assert result["messages"][-1].content == "Conversation continued."
    assert [m.content for m in state["messages"][:4]] == [m.content for m in messages]
    assert len(state["messages"]) == 5
    event = state["_summarization_event"]
    assert event["cutoff_index"] == 3
    assert "Retain the synthetic goal" in event["summary_message"].content
    assert "Error generating summary" not in event["summary_message"].content
    assert state["files"]
    assert any(summary and size > LIMIT for summary, size in calls)
    assert calls[-1][1] <= LIMIT
    assert len(calls) == 6


def test_summary_failure_is_not_committed():
    def provider(request):
        payload = json.loads(request.content)
        if "max_completion_tokens" in payload:
            return httpx.Response(400, json={"error": {"type": "invalid_request_error", "message": "bad summary request"}})
        return httpx.Response(400, json=_error().body)
    model = _model(provider)
    backend = StateBackend()
    graph = create_deep_agent(model, backend=backend, subagents=[],
        middleware=[BoundedSummarizationMiddleware(model, backend), BadRequestRetryMiddleware()],
        checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "failed-summary"}}
    messages = [HumanMessage("old goal"), AIMessage("previous reply"), HumanMessage("continue")]
    with pytest.raises(BadRequestError, match="bad summary request"):
        graph.invoke({"messages": messages}, config)
    state = graph.get_state(config).values
    assert "_summarization_event" not in state
    assert len(state["messages"]) == 3


def test_post_compaction_overflow_advances_cutoff_and_preserves_tool_pairs():
    model = _model(lambda r: _response("brief summary"))
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    messages = [HumanMessage("old"), AIMessage("reply"), HumanMessage("middle"), AIMessage("reply"),
        AIMessage(content="", tool_calls=[{"id": "call", "name": "read_file", "args": {}}]),
        ToolMessage("content", tool_call_id="call"), HumanMessage("latest")]
    middleware._determine_cutoff_index = lambda messages: 2
    middleware._get_backend = lambda state, runtime: StateBackend()
    middleware._offload_to_backend = lambda backend, older: "/conversation_history/test.md"
    attempted = []
    def handler(request):
        attempted.append(request.messages)
        if len(attempted) < 3:
            raise ContextOverflowError("serialized request exceeds available context")
        return ModelResponse(result=[AIMessage("continued")])
    response = middleware.wrap_model_call(ModelRequest(model=model, messages=messages, state={}), handler)
    assert len(attempted) == 3
    assert [len(a) for a in attempted] == [7, 6, 4]
    assert response.command.update["_summarization_event"]["cutoff_index"] == 4
    assert attempted[-1][1:] == messages[4:]


def test_summary_reduction_must_make_progress():
    model = _model(lambda r: _response("a summary that expands the original input"))
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    count = 0
    def provider(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(400, json=_error().body)
        return _response("a summary that expands the original input")
    middleware._summary_model = _model(provider).bind(max_tokens=2048)
    with pytest.raises(ValueError, match="did not reduce"):
        middleware._summarize_text("short")
    assert count == 3


def test_overflow_after_tool_result_can_summarize_the_complete_final_group():
    model = _model(lambda r: _response("synthetic goal and complete tool result"))
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    middleware._get_backend = lambda state, runtime: StateBackend()
    middleware._offload_to_backend = lambda backend, older: "/conversation_history/test.md"
    messages = [HumanMessage("prior summary", additional_kwargs={"lc_source": "summarization"}),
        AIMessage(content="", tool_calls=[{"id": "call", "name": "read_file", "args": {}}]),
        ToolMessage("x" * 160000, tool_call_id="call")]
    attempted = []
    def handler(request):
        attempted.append(request.messages)
        if len(request.messages) > 1:
            raise ContextOverflowError("complete request exceeds context")
        return ModelResponse(result=[AIMessage("continued")])
    result = middleware.wrap_model_call(ModelRequest(model=model, messages=messages, state={}), handler)
    assert len(attempted[-1]) == 1
    assert result.command.update["_summarization_event"]["cutoff_index"] == 3
    assert len(attempted) <= 3


def test_summary_output_cap_reaches_llama_without_losing_template_options():
    payloads = []
    def provider(request):
        payloads.append(json.loads(request.content))
        return _response("brief summary")
    model = _model(provider)
    model.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    assert middleware._create_summary([HumanMessage("synthetic goal")]) == "brief summary"
    assert payloads[0]["max_tokens"] == 2048
    assert payloads[0]["max_completion_tokens"] == 2048
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_forced_compaction_uses_the_installed_bounded_middleware():
    model = _model(lambda request: _response("brief synthetic summary"))
    backend = StateBackend()
    with patch("assist.middleware.summarization.compute_summarization_defaults",
               return_value={"trigger": ("messages", 5), "keep": ("messages", 2)}):
        middleware = BoundedSummarizationMiddleware(model, backend)
    graph = create_deep_agent(model, backend=backend, subagents=[],
        middleware=[middleware, BadRequestRetryMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "forced-compaction"}}
    messages = [HumanMessage("old goal"), AIMessage("old reply"),
                HumanMessage("middle"), AIMessage("middle reply"), HumanMessage("continue")]
    graph.invoke({"messages": messages}, config)
    state = graph.get_state(config).values
    assert state["_summarization_event"]["cutoff_index"] == 3
    assert state["messages"][:5] == messages


def test_async_summary_splits_an_actual_overflow():
    middleware = BoundedSummarizationMiddleware(_model(lambda r: _response("unused")), StateBackend())
    prompt_overhead = len(middleware._summary_prompt(""))
    calls = []

    async def summarize(prompt, config):
        assert config["metadata"] == {"lc_source": "summarization"}
        calls.append(prompt)
        if len(prompt) > prompt_overhead + 8000:
            raise _error()
        return AIMessage("brief synthetic summary")

    middleware._summary_model = SimpleNamespace(ainvoke=summarize)
    result = asyncio.run(middleware._asummarize_text("x" * 12000))
    assert result == "brief synthetic summary"
    assert len(calls) == 4
    assert all(len(prompt) <= prompt_overhead + 8000 for prompt in calls[1:])


@pytest.mark.parametrize("native_error", [False, True])
def test_async_post_compaction_overflow_advances_without_changing_raw_state(native_error):
    model = _model(lambda r: _response("unused"))
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    middleware._summary_model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage("brief summary")))
    middleware._determine_cutoff_index = lambda messages: 2
    middleware._get_backend = lambda state, runtime: StateBackend()
    middleware._aoffload_to_backend = AsyncMock(return_value="/conversation_history/test.md")
    messages = [HumanMessage("old"), AIMessage("reply"), HumanMessage("middle"), AIMessage("reply"),
        AIMessage(content="", tool_calls=[{"id": "call", "name": "read_file", "args": {}}]),
        ToolMessage("content", tool_call_id="call"), HumanMessage("latest")]
    state = {"messages": messages}
    attempted = []

    async def handler(request):
        attempted.append(request.messages)
        if len(attempted) < 3:
            raise _error() if native_error else ContextOverflowError("request exceeds context")
        return ModelResponse(result=[AIMessage("continued")])

    result = asyncio.run(middleware.awrap_model_call(
        ModelRequest(model=model, messages=messages, state=state), handler))
    assert [len(a) for a in attempted] == [7, 6, 4]
    assert result.command.update["_summarization_event"]["cutoff_index"] == 4
    assert attempted[-1][1:] == messages[4:]
    assert state == {"messages": messages}
    assert middleware._aoffload_to_backend.await_count == 2


def test_async_summary_failure_leaves_the_durable_event_unset():
    model = _model(lambda r: _response("unused"))
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    response = httpx.Response(400, request=httpx.Request("POST", "http://unit.test"))
    error = BadRequestError("bad summary request", response=response,
                            body={"type": "invalid_request_error"})
    middleware._summary_model = SimpleNamespace(ainvoke=AsyncMock(side_effect=error))
    middleware._get_backend = lambda state, runtime: StateBackend()
    middleware._aoffload_to_backend = AsyncMock(return_value="/conversation_history/test.md")
    messages = [HumanMessage("old goal"), AIMessage("old reply"), HumanMessage("continue")]
    state = {"messages": messages}

    async def handler(request):
        raise _error()

    with pytest.raises(BadRequestError, match="bad summary request"):
        asyncio.run(middleware.awrap_model_call(
            ModelRequest(model=model, messages=messages, state=state), handler))
    assert state == {"messages": messages}
    assert "_summarization_event" not in state
    middleware._aoffload_to_backend.assert_awaited_once()


def test_second_overflow_can_shrink_only_the_summary_without_losing_the_user_turn():
    calls = []
    summary_calls = 0

    def provider(request):
        nonlocal summary_calls
        payload = json.loads(request.content)
        summary = "max_completion_tokens" in payload
        calls.append((summary, len(request.content)))
        if len(request.content) > LIMIT:
            return httpx.Response(400, json=_error(len(request.content)).body)
        if summary:
            summary_calls += 1
            return _response("s" * 1800 if summary_calls == 1 else "brief goal")
        return _response("continued")

    model = _model(provider)
    backend = StateBackend()
    middleware = BoundedSummarizationMiddleware(model, backend)
    middleware._determine_cutoff_index = lambda messages: 2
    graph = create_deep_agent(model, backend=backend, subagents=[],
        middleware=[middleware, BadRequestRetryMiddleware()], checkpointer=InMemorySaver())
    messages = [HumanMessage("x" * 6000), AIMessage("previous reply"), HumanMessage("u" * 115000)]
    config = {"configurable": {"thread_id": "summary-only-recovery"}}
    result = graph.invoke({"messages": messages}, config)
    state = graph.get_state(config).values
    assert result["messages"][-1].content == "continued"
    assert state["messages"][:3] == messages
    event = state["_summarization_event"]
    assert event["cutoff_index"] == 2
    assert "brief goal" in event["summary_message"].content
    assert event["file_path"] in state["files"]
    assert summary_calls == 2
    assert sum(not summary and size > LIMIT for summary, size in calls) == 2
    assert calls[-1][1] <= LIMIT


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("shrinks", [False, True])
def test_summary_only_recovery_requires_progress_and_preserves_raw_cutoff(asynchronous, shrinks):
    model = _model(lambda r: _response("unused"))
    middleware = BoundedSummarizationMiddleware(model, StateBackend())
    path = "/conversation_history/test.md"
    prior = middleware._event(None, 2, "previous summary" * 100, path)
    raw = [HumanMessage("old goal"), AIMessage("old reply"), HumanMessage("latest question")]
    state = {"messages": raw, "_summarization_event": prior}
    new_summary = "brief" if shrinks else "previous summary" * 100
    middleware._summary_model = SimpleNamespace(
        invoke=lambda *args, **kwargs: AIMessage(new_summary),
        ainvoke=AsyncMock(return_value=AIMessage(new_summary)))
    offloads = []
    middleware._get_backend = lambda state, runtime: StateBackend()
    middleware._offload_to_backend = lambda *args: offloads.append(args)
    middleware._aoffload_to_backend = AsyncMock(side_effect=AssertionError("history already offloaded"))
    attempted = []

    def handler(request):
        attempted.append(request.messages)
        if len(attempted) == 1:
            raise _error()
        return ModelResponse(result=[AIMessage("continued")])

    async def async_handler(request):
        return handler(request)

    def run():
        request = ModelRequest(model=model, messages=raw, state=state)
        return (asyncio.run(middleware.awrap_model_call(request, async_handler))
                if asynchronous else middleware.wrap_model_call(request, handler))

    if shrinks:
        result = run()
        event = result.command.update["_summarization_event"]
        assert event["cutoff_index"] == prior["cutoff_index"]
        assert event["file_path"] == path
        assert len(attempted) == 2
        assert attempted[-1][-1] == raw[-1]
    else:
        with pytest.raises(ValueError, match="did not reduce"):
            run()
        assert len(attempted) == 1
    assert state == {"messages": raw, "_summarization_event": prior}
    assert not offloads
    middleware._aoffload_to_backend.assert_not_awaited()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("write_raises", [False, True])
def test_failed_history_offload_preserves_raw_state_without_model_calls(asynchronous, write_raises):
    class FailingBackend(StateBackend):
        def write(self, path, content):
            if write_raises:
                raise OSError("history storage unavailable")
            return None

        async def awrite(self, path, content):
            return self.write(path, content)

    provider_calls = []
    def provider(request):
        provider_calls.append(request)
        return _response("brief summary")

    model = _model(provider)
    backend = FailingBackend()
    with patch("assist.middleware.summarization.compute_summarization_defaults",
               return_value={"trigger": ("messages", 3), "keep": ("messages", 1)}):
        middleware = BoundedSummarizationMiddleware(model, backend)
    graph = create_deep_agent(model, backend=backend, subagents=[],
        middleware=[middleware], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "failed-history-offload"}}
    messages = [HumanMessage("old goal"), AIMessage("old reply"), HumanMessage("latest question")]
    with pytest.raises(RuntimeError, match="Failed to offload conversation history"):
        if asynchronous:
            asyncio.run(graph.ainvoke({"messages": messages}, config, durability="sync"))
        else:
            graph.invoke({"messages": messages}, config, durability="sync")
    state = graph.get_state(config).values
    assert state["messages"] == messages
    assert "_summarization_event" not in state
    assert not state["files"]
    assert not provider_calls
