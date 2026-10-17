"""Offline checks for captured replay provenance, stream assembly and scoring."""
import json

import httpx
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_openai import ChatOpenAI
import pytest

from assist.middleware.research_generation import ResearchGenerationMiddleware
from edd.eval import test_research_generation_replay as replay


@pytest.fixture
def captured():
    fixture = json.loads((replay._FIXTURES / "request_748.json").read_text())
    sources, failed = replay._sources(fixture["messages"])
    return fixture, sources, failed


def test_fixture_preserves_context_and_contains_no_original_host_or_referral(captured):
    fixture, sources, failed = captured
    provenance = json.loads((replay._FIXTURES / "provenance.json").read_text())
    assert replay._context_sha(fixture) == provenance["sanitized_context_sha256"]
    assert len(fixture["messages"]) == 18 and len(fixture["tools"]) == 11
    assert len(sources) == 72
    assert len(failed) == 1
    assert "/home/" not in json.dumps(fixture)
    assert "ic_referral=" not in json.dumps(fixture)


@pytest.mark.parametrize("kind", ["source", "failed", "invented", "wrong_type", "missing_id", "length"])
def test_progress_requires_valid_source_url_after_final_response(captured, kind):
    _, sources, failed = captured
    url = next(row["url"] for row in sources if row["url"] not in failed)
    if kind == "failed":
        url = next(iter(failed))
    elif kind == "invented":
        url = "https://example.org/invented-book"
    elif kind == "wrong_type":
        url = [url]
    message = AIMessage("", tool_calls=[{"name": "read_url", "args": {"url": url},
        "id": "" if kind == "missing_id" else "call"}],
        response_metadata={"finish_reason": "length" if kind == "length" else "tool_calls"})
    assert replay._useful(message, sources, failed) == (kind == "source")


@pytest.mark.parametrize("finish", ["tool_calls", "length"])
def test_native_stream_fragments_and_reasoning_survive_but_capped_calls_do_not(captured, tmp_path, finish):
    fixture, sources, failed = captured
    url = next(row["url"] for row in sources if row["url"] not in failed)
    arguments = json.dumps({"url": url})
    deltas = [
        {"reasoning_content": "Private native reasoning preserved for diagnosis."},
        {"tool_calls": [{"index": 0, "id": "call", "type": "function",
                         "function": {"name": "read_url", "arguments": arguments[:20]}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[20:]}}]},
    ]
    sent = []
    def provider(request):
        if request.url.path.endswith("/props"):
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 131072}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": []})
        payload = json.loads(request.content)
        sent.append(payload)
        events = [{"id": "replay", "object": "chat.completion.chunk", "created": 0,
                   "model": fixture["model"], "choices": [{"index": 0, "delta": delta,
                   "finish_reason": finish if index == 2 else None}]}
                  for index, delta in enumerate(deltas)]
        events.append({"id": "replay", "object": "chat.completion.chunk", "created": 0,
                       "model": fixture["model"], "choices": [],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)

    with httpx.Client(transport=httpx.MockTransport(provider)) as client:
        model = ChatOpenAI(model=fixture["model"], temperature=0.1, api_key="EMPTY",
                           max_retries=0, http_client=client, timeout=httpx.Timeout(600))
        request = ModelRequest(model=model, messages=convert_to_messages(fixture["messages"]), state={},
                               model_settings={"max_tokens": 32768, "extra_body": {}})
        metrics = {"seed": 0}
        response = ResearchGenerationMiddleware().wrap_model_call(request,
            lambda bounded: replay._stream_response(bounded, fixture, tmp_path, metrics, model.root_client))
    assert sent[0]["messages"] == fixture["messages"] and sent[0]["tools"] == fixture["tools"]
    assert sent[0]["reasoning_budget_tokens"] == 16384
    assert sent[0]["max_completion_tokens"] == 24576
    assert metrics["reasoning_delta_events"] == 1 and metrics["usage"]["completion_tokens"] == 20
    assert "Private native reasoning" in (tmp_path / "stream.jsonl").read_text()
    assert replay._useful(response, sources, failed) == (finish == "tool_calls")
    if finish == "length":
        assert not response.tool_calls and "partial" in response.text.lower()
    else:
        assert response.tool_calls[0]["args"] == {"url": url}


def test_grounded_book_answer_or_report_is_progress_but_unlinked_claim_is_not(captured):
    _, sources, failed = captured
    url = next(row["url"] for row in sources if "Gillian Cross" in row["content"])
    answer = "The Odyssey by Gillian Cross is an illustrated narrative retelling. Source: " + url
    assert replay._useful(AIMessage(answer, response_metadata={"finish_reason": "stop"}), sources, failed)
    assert not replay._useful(AIMessage(answer.replace(url, "https://example.org/unknown"),
                                       response_metadata={"finish_reason": "stop"}), sources, failed)
    report = AIMessage("", tool_calls=[{"name": "write_file", "id": "report",
        "args": {"file_path": "/report.md", "content": answer}}],
        response_metadata={"finish_reason": "tool_calls"})
    assert replay._useful(report, sources, failed)


def test_authoritative_captured_usborne_answer_is_valid(captured):
    _, sources, failed = captured
    answer = ("The Odyssey by Russell Punter, illustrated by Fabiano Fiorin. "
              "Usborne Graphic Novels, age7+. https://usborne.com/us/the-odyssey-9781806460526")
    assert replay._useful(AIMessage(answer, response_metadata={"finish_reason": "stop"}), sources, failed)


@pytest.mark.parametrize("search_args,with_fetch,expected", [
    ({"query": "Odyssey chapter books", "max_results": 8}, True, True),
    ({"query": "Odyssey chapter books"}, False, False),
    ({"query": ["Odyssey"]}, True, False),
    ({"query": "Odyssey", "max_results": "8"}, True, False),
])
def test_valid_search_companion_does_not_discard_source_backed_fetch(captured, search_args, with_fetch, expected):
    _, sources, failed = captured
    calls = [{"name": "search_internet", "id": "search", "args": search_args}]
    if with_fetch:
        url = next(row["url"] for row in sources if row["url"] not in failed)
        calls.append({"name": "read_url", "id": "fetch", "args": {"url": url}})
    message = AIMessage("", tool_calls=calls, response_metadata={"finish_reason": "tool_calls"})
    assert replay._useful(message, sources, failed) == expected


@pytest.mark.parametrize("kind", ["answer", "report"])
def test_longer_williams_identity_survives_shorter_title_normalization(captured, kind):
    _, sources, failed = captured
    url = next(row["url"] for row in sources if "booktopia" in row["url"]
               and "Marcia Williams" in (row["title"] + " " + row["content"]))
    answer = "The Iliad and the Odyssey by Marcia Williams is a comic-strip retelling. Source: " + url
    message = (AIMessage(answer, response_metadata={"finish_reason": "stop"}) if kind == "answer" else
               AIMessage("", tool_calls=[{"name": "write_file", "id": "report",
                   "args": {"file_path": "/report.md", "content": answer}}],
                   response_metadata={"finish_reason": "tool_calls"}))
    assert replay._useful(message, sources, failed)
