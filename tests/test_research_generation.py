"""Offline provider-boundary and research-acceptance validator regressions."""
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from assist.agent import create_research_agent


_SOURCE = '[{"title":"The Odyssey","url":"https://example.org/odyssey","content":"Gillian Cross; illustrated narrative; ages 8-12."}]'


def _response(content="Verified findings.", *, finish="stop", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return httpx.Response(200, json={"id": "test", "object": "chat.completion", "model": "Qwen3.8-27B",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}]})


def _model(provider, **kwargs):
    model = ChatOpenAI(model=kwargs.pop("model", "Qwen3.8-27B"), temperature=0.1,
        api_key="EMPTY", max_retries=0,
        timeout=httpx.Timeout(connect=10, read=600, write=60, pool=10),
        http_client=httpx.Client(transport=httpx.MockTransport(provider)), **kwargs)
    model.profile = {"max_input_tokens": 131072}
    return model


def _run(model, root, *, leaf=True):
    agent = create_research_agent(model, str(root), leaf=leaf)
    messages = [HumanMessage("Recommend an illustrated Odyssey retelling."),
        AIMessage(content="", tool_calls=[{"name": "search_internet", "args": {"query": "Odyssey retelling"}, "id": "search"}]),
        ToolMessage(_SOURCE, name="search_internet", tool_call_id="search")]
    return agent.invoke({"messages": messages}, {"configurable": {"thread_id": "research"}})["messages"]


@pytest.mark.parametrize("leaf", [True, False])
def test_budgets_and_read_allowance_reach_provider_without_changing_shared_model(tmp_path, monkeypatch, leaf):
    monkeypatch.delenv("ASSIST_LLM_READ_TIMEOUT_S", raising=False)
    requests = []
    def provider(request):
        requests.append(request)
        return _response()
    extra = {"chat_template_kwargs": {"preserve_thinking": True}}
    model = _model(provider, extra_body=extra)
    assert _run(model, tmp_path, leaf=leaf)[-1].content == "Verified findings."
    payload = json.loads(requests[0].content)
    assert payload["max_completion_tokens"] == 24576
    assert payload["reasoning_budget_tokens"] == 16384
    assert payload["chat_template_kwargs"] == extra["chat_template_kwargs"]
    assert payload["temperature"] == 0.1
    assert requests[0].extensions["timeout"] == {"connect": 10, "read": 900, "write": 60, "pool": 10}
    assert model.max_tokens is None
    assert model.extra_body == extra
    assert model.request_timeout.read == model.root_client.timeout.read == 600


def test_smaller_explicit_budgets_and_timeout_are_retained(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSIST_LLM_READ_TIMEOUT_S", "45")
    requests = []
    def provider(request):
        requests.append(request)
        return _response()
    model = _model(provider, max_tokens=1000,
                   extra_body={"max_tokens": 800, "n_predict": -1, "reasoning_budget_tokens": 200})
    _run(model, tmp_path)
    payload = json.loads(requests[0].content)
    assert payload["max_tokens"] == payload["max_completion_tokens"] == payload["n_predict"] == 800
    assert payload["reasoning_budget_tokens"] == 200
    assert requests[0].extensions["timeout"]["read"] == 45
    assert model.extra_body["n_predict"] == -1


@pytest.mark.parametrize("kind", ["empty", "prose", "valid_tool", "partial_tool"])
def test_total_cap_returns_partial_source_evidence_without_retry_or_tool_execution(tmp_path, kind):
    requests = []
    def provider(request):
        requests.append(request)
        args = '{"file_path":"/should-not-exist.md","content":"incomplete"}'
        if kind == "partial_tool":
            args = '{"file_path":"/should-not-exist.md","content":'
        calls = [{"id": "write", "type": "function", "function": {"name": "write_file", "arguments": args}}] if "tool" in kind else None
        return _response("The Odyssey is an illustrated retelling" if kind == "prose" else "", finish="length", calls=calls)
    messages = _run(_model(provider), tmp_path)
    final = messages[-1]
    assert "partial" in final.content.lower()
    assert "https://example.org/odyssey" in final.content
    assert not final.tool_calls and not final.invalid_tool_calls
    assert len(requests) == 1, "A generation cap must not repeat the same expensive request"
    assert not any(isinstance(message, ToolMessage) and message.name == "write_file" for message in messages)
    assert not (tmp_path / "references" / "should-not-exist.md").exists()


def test_other_model_profile_is_unchanged(tmp_path):
    requests = []
    def provider(request):
        requests.append(request)
        return _response()
    _run(_model(provider, model="Qwen3.6-27B"), tmp_path)
    payload = json.loads(requests[0].content)
    assert "max_completion_tokens" not in payload
    assert "reasoning_budget_tokens" not in payload
    assert requests[0].extensions["timeout"]["read"] == 600


def test_compacted_request_retains_current_raw_evidence_and_excludes_old_or_pending_results():
    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from assist.middleware.research_generation import ResearchGenerationMiddleware
    old = [HumanMessage("Previous topic"),
        AIMessage(content="", tool_calls=[{"name": "read_url", "args": {"url": "https://example.org/old"}, "id": "old"}]),
        ToolMessage("Old private material", name="read_url", tool_call_id="old")]
    current = [HumanMessage("Current research"),
        AIMessage(content="", tool_calls=[
            {"name": "read_url", "args": {"url": "https://example.org/current"}, "id": "current"},
            {"name": "read_url", "args": {"url": "https://example.org/pending"}, "id": "pending"}]),
        ToolMessage("Source preview; full page is offloaded.", name="read_url", tool_call_id="current")]
    capped = AIMessage(content="", response_metadata={"finish_reason": "length"},
        tool_calls=[{"name": "write_file", "args": {"file_path": "/x", "content": "x"}, "id": "valid"}],
        invalid_tool_calls=[{"name": "write_file", "args": "{", "id": "invalid", "error": "partial"}],
        additional_kwargs={"tool_calls": [{"id": "raw", "function": {"name": "write_file", "arguments": "{"}}]})
    calls = []
    def handler(request):
        calls.append(request)
        return ModelResponse(result=[capped])
    request = ModelRequest(model=_model(lambda r: _response()),
        messages=[HumanMessage("Effective compacted summary")], state={"messages": old + current})
    result = ResearchGenerationMiddleware().wrap_model_call(request, handler).result[0]
    assert "https://example.org/current" in result.content
    assert "Source preview; full page is offloaded." in result.content
    assert "Old private material" not in result.content
    assert "https://example.org/pending" not in result.content
    assert not result.tool_calls and not result.invalid_tool_calls
    assert "tool_calls" not in result.additional_kwargs
    assert len(calls) == 1


def test_capped_lead_keeps_completed_source_backed_worker_reply(tmp_path):
    from unittest.mock import patch
    from langchain_core.tools import tool
    requests = []
    searches = []
    @tool("search_internet")
    def search(query: str) -> str:
        """Search for retellings."""
        searches.append(query)
        return _SOURCE
    def provider(request):
        requests.append(request)
        number = len(requests)
        if number == 1:
            return _response("", calls=[{"id": "worker", "type": "function", "function": {
                "name": "task", "arguments": json.dumps({"subagent_type": "research-agent", "description": "Find illustrated Odyssey retellings"})}}])
        if number == 2:
            return _response("", calls=[{"id": "source", "type": "function", "function": {
                "name": "search_internet", "arguments": '{"query":"illustrated Odyssey retellings"}'}}])
        if number == 3:
            return _response("The Odyssey by Gillian Cross is an illustrated narrative. Source: https://example.org/odyssey")
        return _response("", finish="length")
    with patch("assist.agent.search_internet", search):
        agent = create_research_agent(_model(provider), str(tmp_path), leaf=False)
        messages = agent.invoke({"messages": [HumanMessage("Recommend an illustrated Odyssey retelling.")]},
                                {"configurable": {"thread_id": "research"}})["messages"]
    assert searches == ["illustrated Odyssey retellings"]
    assert len(requests) == 4
    assert "https://example.org/odyssey" in messages[-1].content
    assert "Gillian Cross" in messages[-1].content
    assert "untrusted worker output" in messages[-1].content
    assert not messages[-1].tool_calls


@pytest.mark.parametrize("result, expected", [
    ("Book One by Writer One, illustrated narrative.\n\nBook Two by Writer Two, a comic strip.\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 2),
    ("Book One by Writer One, chapter-book prose. Source: https://example.org/one\n\n"
     "Book Two by Writer Two, a visual comic. Source: https://example.org/two", 2),
    ("Recommend Book One and Book Two by Unknown Writer.\n\n"
     "Author directory: Writer One, Writer Two.\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 0),
    ("Book One by Writer One, narrative prose. Source: https://example.org/two\n\n"
     "Book Two by Writer Two, a comic strip. Source: https://example.org/one\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 0),
    ("Book One by Writer One, narrative prose.\n\nBook Two by Writer Two, a comic strip.\n\n"
     "Sources:\nhttps://example.org/one", 1),
    ("## Titles\nBook One\nBook Two\n## Authors\nWriter One\nWriter Two\n"
     "## Sources\nhttps://example.org/one\nhttps://example.org/two", 0),
    ("Some illustrated chapter books.\n\n"
     "Sources:\nBook One by Writer One https://example.org/one\n"
     "Book Two by Writer Two https://example.org/two", 0),
    ("The child enjoys chapter books.\n\nBook One by Writer One.\n\nBook Two by Writer Two.\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 0),
    ("Book One by Writer One.\n\nA continuous narrative in prose.\n\n"
     "Book Two by Writer Two.\n\nA visual comic-strip introduction.\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 2),
    ("Book One by Writer One.\n\nBook Two by Writer Two, narrative prose.\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 0),
    ("Book One by Writer One.\n\nNarrative prose. Source: https://example.org/two\n\n"
     "Book Two by Writer Two.\n\nVisual comic. Source: https://example.org/one\n\n"
     "Sources:\nhttps://example.org/one\nhttps://example.org/two", 0),
])
def test_recommendation_validation_pairs_identity_source_and_format(result, expected):
    from edd.eval.test_research_generation import _count_sourced_recommendations
    recommendations = [("Writer One", "Book One", "https://example.org/one", r"chapter|narrative|prose"),
                       ("Writer Two", "Book Two", "https://example.org/two", r"comic|visual")]
    assert _count_sourced_recommendations(result, recommendations) == expected


@pytest.mark.parametrize("marker", ["-", "*", "+", "1.", "1)", "  -", "  1)", "|", ""])
def test_recommendation_format_does_not_cross_markdown_items(marker):
    from edd.eval.test_research_generation import _count_sourced_recommendations
    if marker in ("|", ""):
        result = f"{marker}Book One|Writer One|https://example.org/one|\n{marker}Book Two|Writer Two|https://example.org/two|narrative prose and visual comic|"
    else:
        result = f"{marker} Book One by Writer One.\n{marker} Book Two by Writer Two, narrative prose and visual comic."
    result += "\n\nSources:\nhttps://example.org/one\nhttps://example.org/two"
    recommendations = [("Writer One", "Book One", "https://example.org/one", r"narrative|prose"),
                       ("Writer Two", "Book Two", "https://example.org/two", r"comic|visual")]
    assert _count_sourced_recommendations(result, recommendations) == 1


@pytest.mark.parametrize("following", ["- Unrelated narrative prose.", "1) Unrelated narrative prose.",
                                      "## Unrelated narrative prose", "Other|Writer|narrative prose"])
def test_recommendation_does_not_borrow_new_item_without_known_identity(following):
    from edd.eval.test_research_generation import _count_sourced_recommendations
    result = "Book One by Writer One.\n" + following + "\n\nSources:\nhttps://example.org/one"
    assert _count_sourced_recommendations(result, [
        ("Writer One", "Book One", "https://example.org/one", r"narrative|prose")]) == 0


@pytest.mark.parametrize("bound, expected", [(50000, 24576), (1234, 1234)])
def test_request_completion_alias_is_capped_in_actual_provider_payload(bound, expected):
    from langchain.agents.middleware.types import ModelRequest
    from assist.middleware.research_generation import ResearchGenerationMiddleware
    requests = []
    def provider(request):
        requests.append(request)
        return _response()
    model = _model(provider)
    request = ModelRequest(model=model, messages=[HumanMessage("Research")], state={"messages": []},
                           model_settings={"max_completion_tokens": bound})
    ResearchGenerationMiddleware().wrap_model_call(request, lambda bounded:
        bounded.model.invoke(bounded.messages, **bounded.model_settings))
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["max_completion_tokens"] == expected
    assert "max_tokens" not in payload
    assert request.model_settings == {"max_completion_tokens": bound}


@pytest.mark.parametrize("body, expected", [
    ("Book One by Writer One. Book Two by Writer Two, narrative prose and visual comic.", 1),
    ("Writer One's Book One. Writer Two's Book Two, narrative prose and visual comic.", 1),
    ("Book One by Writer One, narrative prose https://example.org/two; "
     "Book Two by Writer Two, visual comic https://example.org/one", 0),
    ("Book One by Writer One, narrative prose https://example.org/one; "
     "Book Two by Writer Two, visual comic https://example.org/two", 2),
])
def test_recommendation_evidence_stops_at_same_paragraph_identity(body, expected):
    from edd.eval.test_research_generation import _count_sourced_recommendations
    result = body + "\n\nSources:\nhttps://example.org/one\nhttps://example.org/two"
    rows = [("Writer One", "Book One", "https://example.org/one", r"narrative|prose"),
            ("Writer Two", "Book Two", "https://example.org/two", r"comic|visual")]
    assert _count_sourced_recommendations(result, rows) == expected


@pytest.mark.parametrize("second", ["Book Two by Writer One", "Book Two"])
def test_same_author_is_not_reused_across_recommendation_identities(second):
    from edd.eval.test_research_generation import _count_sourced_recommendations
    result = (f"Book One by Writer One. {second}, narrative prose and visual comic.\n\n"
              "Sources:\nhttps://example.org/one\nhttps://example.org/two")
    rows = [("Writer One", "Book One", "https://example.org/one", r"narrative|prose"),
            ("Writer One", "Book Two", "https://example.org/two", r"comic|visual")]
    assert _count_sourced_recommendations(result, rows) == (1 if "by Writer One" in second else 0)
