"""Exact source binding from retrieval through streamed and stored replies."""

import json
import os
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from assist import thread as thread_module
from assist import tools
from assist.thread_datastore import (
    ReferenceBindingMiddleware, ReferenceStream, ThreadDataStore,
    get_thread_data, register_source, resolve_references, source_url,
    resolve_host_report,
)


URL = "https://decider.example/2026/10/12/the-10-best-horror-movies-now-streaming-on-shudder"
WRONG_URL = URL.replace("the-10", "the-11")


def test_search_result_to_stream_and_canonical_message(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    monkeypatch.setenv("ASSIST_SEARCH_URL", "http://search.example")
    monkeypatch.setattr(tools, "_query_engine", lambda *_args: (
        [{"title": "Decider", "url": URL, "content": "Film list"}], False))
    raw_result = ToolMessage(
        content=tools.search_internet("horror movies to stream"),
        tool_call_id="search-1", name="search_internet")
    request = SimpleNamespace(tool_call={"name": "search_internet", "id": "search-1"})
    bound = ReferenceBindingMiddleware()._bind_result(request, raw_result)
    ref = store.list()[0]["id"]
    assert f"ref:{ref}" in bound.content
    assert URL in bound.content  # navigation retains the exact returned URL

    raw_answer = f"Watch [Decider](ref:{ref})."
    canonical = ReferenceBindingMiddleware().after_model(
        {"messages": [AIMessage(content=raw_answer, id="answer-1")]}, None)
    saved = canonical["messages"][0]
    assert saved.id == "answer-1"
    assert saved.content == f"Watch [Decider](<{URL}>)."
    assert thread_module._messages_to_dicts([saved]) == [
        {"role": "assistant", "content": f"Watch [Decider](<{URL}>)."}]

    chat = object.__new__(thread_module.Thread)
    chat.thread_id = "test-source-binding"
    chat.on_queue_state = None
    chat.runconfig = {"configurable": {"thread_id": chat.thread_id}}
    chat.thread_datastore_path = store.path
    chat.agent = object()
    deltas = []

    def fake_stream(_agent, _input, _config, receive, *, on_reset, **_kwargs):
        for piece in ("Watch [Dec", "ider](ref:", ref[:5], ref[5:], ")."):
            receive((AIMessage(content=piece), {}))
        return {"messages": [saved]}

    monkeypatch.setattr(thread_module, "stream_with_rollback", fake_stream)
    reply = chat.observe_message("What should I watch?", deltas.append)
    assert reply == saved.content
    assert "".join(deltas) == saved.content
    assert all("ref:" not in delta for delta in deltas)


def test_unknown_refs_have_recoverable_feedback_and_ordinary_urls_are_unchanged(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL, "Decider")
    assert resolve_references(f"[Decider]({URL})", store) == f"[Decider]({URL})"
    assert resolve_references(f"[Decider]({WRONG_URL})", store) == f"[Decider]({WRONG_URL})"
    assert resolve_references(f"Source: {WRONG_URL}", store) == f"Source: {WRONG_URL}"
    assert "ask me to verify" in resolve_references("[Decider](ref:r0000000000000000)", store)
    assert "ask me to verify" in resolve_references("[Decider](ref:bad!)", store)
    for malformed_target in ("r123[bad", "r123\nbad"):
        malformed_link = f"[Decider](ref:{malformed_target})"
        assert "ref:" not in resolve_references(malformed_link, store)
        assert "ask me to verify" in ReferenceStream(store).push(malformed_link)
    malformed = ReferenceStream(store)
    assert malformed.push("[Decider](ref:bad") == ""
    assert "ask me to verify" in malformed.push("!)")
    assert "ref:" not in resolve_references("[Decider](ref:" + "a" * 300 + ")", store)
    assert "ask me to verify" in resolve_references("[Decider](ref:)", store)
    stream = ReferenceStream(store)
    assert stream.push("[Decider](https://decider.example/2026/10/12/the-11") == "[Decider](https://decider.example/2026/10/12/the-11"
    assert stream.push("-best-horror-movies-now-streaming-on-shudder)") == "-best-horror-movies-now-streaming-on-shudder)"
    stream = ReferenceStream(store)
    assert stream.push("Source ref:") == "Source "
    assert stream.push(record_id + ".") == URL + "."
    assert stream.push("", final=True) == ""
    assert resolve_references("[Source](HTTPS://wrong.example/page)", store) == "[Source](HTTPS://wrong.example/page)"
    assert "ask me to verify" in resolve_references("ref:" + "a" * 100, store)
    assert resolve_references("ref:", store) == "[source link unavailable: ask me to verify it again]"
    wikipedia = "https://en.wikipedia.org/wiki/Mercury_(element)"
    register_source(store, wikipedia)
    assert resolve_references(f"Source: {wikipedia}.", store) == f"Source: {wikipedia}."


def test_bound_url_containing_ref_text_stays_byte_exact(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    url = "https://source.example/article?q=ref:r00000000000000000000000000000000"
    record_id = register_source(store, url)
    assert resolve_references(f"[Source](ref:{record_id})", store) == f"[Source](<{url}>)"
    assert resolve_references(f"[Source]({url})", store) == f"[Source]({url})"
    assert resolve_references(f"Visit {url}", store) == f"Visit {url}"
    text = f"Visit https://owner.example/?q=ref:{record_id} now."
    for split in range(len(text) + 1):
        stream = ReferenceStream(store)
        actual = (stream.push(text[:split]) + stream.push(text[split:])
                  + stream.push("", final=True))
        assert actual == text, split


@pytest.mark.parametrize("url", [
    "https://source.example/a&copy;",
    "https://source.example/a&#42;",
    r"https://source.example/a\)b",
    "https://source.example/a`q`",
    "https://source.example/a|b",
])
def test_bound_markdown_renders_exact_href_without_changing_ordinary_urls(tmp_path, url):
    from assist.safe_markdown import render_markdown

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hrefs = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.hrefs.append(dict(attrs)["href"])

    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, url)
    assert source_url(store, record_id) == url
    raw = f"[Source](ref:{record_id})"
    resolved = resolve_references(raw, store)
    for split in range(len(raw) + 1):
        stream = ReferenceStream(store)
        assert (stream.push(raw[:split]) + stream.push(raw[split:])
                + stream.push("", final=True)) == resolved
    links = Links()
    links.feed(render_markdown(resolved))
    assert links.hrefs == [url]
    table_links = Links()
    table_links.feed(render_markdown(f"| Source |\n| --- |\n| {resolved} |",
                                     extensions=["fenced_code", "tables"]))
    assert table_links.hrefs == [url]
    assert resolve_references(f"[Owner](<{url}>)", store) == f"[Owner](<{url}>)"


def test_source_with_unicode_whitespace_is_not_registered(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    assert register_source(store, "https://source.example/a\u00a0b") is None
    assert store.list(kind="source") == []


def test_failed_long_read_url_offload_does_not_register_source(tmp_path, monkeypatch):
    from assist.middleware.tool_result_to_file import ToolResultToFileMiddleware

    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    backend = SimpleNamespace(write=lambda path, content: SimpleNamespace(error=None))
    offload = ToolResultToFileMiddleware(
        backend, tools={"read_url"}, floor_chars=4000,
        untrusted=True, page_navigation=True)
    binding = ReferenceBindingMiddleware()
    request = SimpleNamespace(tool_call={"name": "read_url", "id": "failed-fetch",
                                         "args": {"url": "https://failed.example/page"}})
    failed = ToolMessage(content="Error fetching URL: " + "X" * 5000,
                         name="read_url", tool_call_id="failed-fetch")
    result = offload.wrap_tool_call(
        request, lambda req: binding.wrap_tool_call(req, lambda _req: failed))
    assert "Source ID:" not in result.content
    assert store.list(kind="source") == []

    succeeded = ToolMessage(content="Page body " + "X" * 5000,
                            name="read_url", tool_call_id="successful-fetch")
    preview = offload.wrap_tool_call(
        request, lambda req: binding.wrap_tool_call(req, lambda _req: succeeded))
    assert "Source ID: ref:" in preview.content
    assert len(store.list(kind="source")) == 1


@pytest.mark.parametrize("template", [
    "See [source ref:{record_id}].",
    "See [source ref:{record_id}](https://owner.example/page).",
])
def test_stream_resolves_references_inside_bracket_labels_across_splits(tmp_path, template):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)
    raw = template.format(record_id=record_id)
    expected = resolve_references(raw, store)
    for split in range(len(raw) + 1):
        stream = ReferenceStream(store)
        actual = stream.push(raw[:split]) + stream.push(raw[split:]) + stream.push("", final=True)
        assert actual == expected, split


def test_stream_preserves_paragraph_after_unclosed_reference_target(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)
    raw = f"[Source](ref:{record_id}\n\nThe next paragraph is still here."
    expected = resolve_references(raw, store)
    for split in range(len(raw) + 1):
        stream = ReferenceStream(store)
        actual = stream.push(raw[:split]) + stream.push(raw[split:]) + stream.push("", final=True)
        assert actual == expected, split


def test_stream_discards_late_close_after_bounded_invalid_target(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    raw = "[S](ref:" + "a" * 2550 + ") Next paragraph."
    expected = resolve_references(raw, store)
    split = raw.index(")")
    stream = ReferenceStream(store)
    actual = stream.push(raw[:split]) + stream.push(raw[split:]) + stream.push("", final=True)
    assert actual == expected


@pytest.mark.parametrize("separator", ["", "\n"])
def test_stream_retains_valid_reference_after_bounded_invalid_target(tmp_path, separator):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)
    raw = "[S](ref:" + "a" * 2520 + separator + f"[B](ref:{record_id})"
    expected = resolve_references(raw, store)
    stream = ReferenceStream(store)
    actual = stream.push(raw[:-1]) + stream.push(raw[-1:]) + stream.push("", final=True)
    assert actual == expected


@pytest.mark.parametrize("label", [r"Guide \[PDF\]", "Guide [PDF]"])
def test_citation_label_with_brackets_resolves_in_history_and_stream(tmp_path, label):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)
    raw = f"[{label}](ref:{record_id})"
    expected = f"[{label}](<{URL}>)"
    assert resolve_references(raw, store) == expected
    for split in range(len(raw) + 1):
        stream = ReferenceStream(store)
        actual = stream.push(raw[:split]) + stream.push(raw[split:]) + stream.push("", final=True)
        assert actual == expected, split


def test_stream_discards_rest_of_overlong_bare_reference(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    raw = "ref:" + "a" * 300 + " Next paragraph."
    expected = resolve_references(raw, store)
    for split in range(260, 310):
        stream = ReferenceStream(store)
        actual = stream.push(raw[:split]) + stream.push(raw[split:]) + stream.push("", final=True)
        assert actual == expected, split


def test_nested_malformed_brackets_match_canonical_stream(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    raw = "[[Y](ref:]]"
    expected = resolve_references(raw, store)
    for split in range(len(raw) + 1):
        stream = ReferenceStream(store)
        actual = stream.push(raw[:split]) + stream.push(raw[split:]) + stream.push("", final=True)
        assert actual == expected, split


def test_fact_check_report_read_resolves_host_source_without_registering_text(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    request = SimpleNamespace(tool_call={"name": "read_file", "id": "read-1",
                                         "args": {"file_path": "final_report.md"}})
    result = ToolMessage(content=f"[Source](ref:{record_id})",
                         tool_call_id="read-1", name="read_file")

    assert ReferenceBindingMiddleware()._bind_result(request, result).content == result.content
    bound = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(request, result)
    assert bound.content == f"[Source](<{URL}>)"
    assert len(store.list(kind="source")) == 1


def test_report_binding_is_linear_on_incomplete_refs_and_bounds_expansion(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    incomplete = "[a](ref:" * 2000
    assert "ref:" not in resolve_references(
        incomplete, store, max_output_bytes=8 * 1024 * 1024)
    large_url = "https://example.test/" + "a" * 8000
    record_id = register_source(store, large_url)
    with pytest.raises(ValueError, match="exceeds 8 MiB"):
        resolve_references(f"[S](ref:{record_id})" * 1100, store,
                           max_output_bytes=8 * 1024 * 1024)


def test_local_report_binding_rejects_fifo_without_blocking(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    refs = tmp_path / "references"
    refs.mkdir()
    fifo = refs / "report.md"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        resolve_host_report(str(fifo), store)


def test_plain_brackets_do_not_stall_reply_stream(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    stream = ReferenceStream(store)
    assert stream.push("Choose [one] or [two] today.") == "Choose [one] or [two] today."


def test_malformed_search_url_does_not_fail_tool_result(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    request = SimpleNamespace(tool_call={"name": "search_internet", "id": "search-1"})
    result = ToolMessage(content=str([{"title": "Bad", "url": "http://[bad",
                                       "content": "snippet"}]),
                         tool_call_id="search-1", name="search_internet")
    bound = ReferenceBindingMiddleware()._bind_result(request, result)
    assert "http://[bad" in bound.content
    assert "source_id" not in bound.content


def test_datastore_text_get_list_search_survive_recreation_and_are_thread_scoped(tmp_path, monkeypatch):
    from assist.thread_datastore import list_thread_data, search_thread_data

    own = ThreadDataStore(str(tmp_path / "own.sqlite"))
    other = ThreadDataStore(str(tmp_path / "other.sqlite"))
    register_source(own, URL, "Decider")
    register_source(other, "https://other.example/source", "Other")
    note_id = own.put_text("A durable note about Mercury", "note", "Astronomy")
    long_id = own.put_text("x" * 3000, "note", "Long")
    own = ThreadDataStore(own.path)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: own)
    assert URL in list_thread_data()
    assert "other.example" not in list_thread_data()
    assert URL in search_thread_data("Decider")
    assert own.get(note_id) == {"id": note_id, "kind": "note", "value": "A durable note about Mercury", "title": "Astronomy"}
    assert "Mercury" in get_thread_data(note_id)
    assert "Astronomy" in list_thread_data(kind="note")
    assert "Mercury" in search_thread_data("Mercury")
    listed_long = next(row for row in json.loads(list_thread_data(kind="note"))
                       if row["id"] == long_id)
    assert listed_long["truncated"] is True
    assert len(listed_long["value"]) == 2000
    assert len(json.loads(get_thread_data(long_id))["value"]) == 3000
    assert search_thread_data("missing") == "[]"
    assert search_thread_data("' OR 1=1 --") == "[]"
    assert "Limit must" in list_thread_data(limit=100)
    assert list_thread_data(kind="arbitrary") == "[]"
    assert "not found" in get_thread_data("r" + "0" * 32)
    assert "not found" in get_thread_data(next(iter(other.list()))["id"])
    with pytest.raises(ValueError, match="trusted retrieval"):
        own.put_text("https://forged.example", "source")


def test_generic_text_store_follows_thread_manager_recreation(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from assist.thread_manager import ThreadManager

    manager = ThreadManager(str(tmp_path / "threads"))
    manager.reserve("one")
    path = manager.thread_datastore_path("one")
    record_id = ThreadDataStore(path).put_text("meeting notes", "note", "Today")
    manager.close()

    reopened = ThreadManager(str(tmp_path / "threads"))
    reopened._model = Mock()
    built = []
    monkeypatch.setattr("assist.thread_manager.Thread", lambda *args, **kwargs: built.append(kwargs) or Mock())
    try:
        reopened.get("one")
        assert built[0]["thread_datastore_path"] == path
        assert ThreadDataStore(path).get(record_id)["value"] == "meeting notes"
        assert ThreadDataStore(path).search("meeting", kind="note")[0]["id"] == record_id
    finally:
        reopened.close()


def test_explicit_reference_binding_preserves_ordinary_links_in_stream_and_history(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)
    ordinary = (
        " [Owner](https://owner.example/home)"
        " [Mail](https://mail.google.com/mail/u/0/#inbox/abc)"
        " `curl https://api.example/v1`"
    )
    raw = f"[Decider](ref:{record_id})" + ordinary
    expected = f"[Decider](<{URL}>)" + ordinary
    assert resolve_references(raw, store) == expected
    stream = ReferenceStream(store)
    chunks = (raw[:8], raw[8:32], raw[32:54], raw[54:76], raw[76:])
    assert "".join(stream.push(chunk) for chunk in chunks) + stream.push("", final=True) == expected


def test_framework_checkpoint_stores_resolved_message(tmp_path):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver

    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

    graph = create_agent(
        Model(responses=[AIMessage(content=f"[Decider](ref:{record_id})")]),
        tools=[], middleware=[ReferenceBindingMiddleware()],
        checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "stored-thread",
                               "thread_datastore_path": store.path}}
    graph.invoke({"messages": [{"role": "user", "content": "Find a film"}]}, config)
    saved = graph.get_state(config).values["messages"][-1]
    assert saved.content == f"[Decider](<{URL}>)"


def test_low_level_stream_yields_only_resolved_markdown(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)

    class Agent:
        def stream(self, *_args, **_kwargs):
            for part in ("[Dec", "ider](ref:", record_id, ")"):
                yield ("messages", (AIMessageChunk(content=part), {}))

    chat = object.__new__(thread_module.Thread)
    chat.thread_id = "low-level-stream"
    chat.on_queue_state = None
    chat.runconfig = {"configurable": {"thread_id": chat.thread_id}}
    chat.thread_datastore_path = store.path
    chat.agent = Agent()
    chunks = list(chat.stream_message("Find a film"))
    assert "".join(thread_module.extract_content_text(payload)
                   for mode, payload in chunks if mode == "messages") == f"[Decider](<{URL}>)"


def test_stream_does_not_join_citations_across_tool_messages(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL)

    class Agent:
        def stream(self, *_args, **_kwargs):
            yield ("messages", (AIMessageChunk(content="[Decider](ref:", id="first"), {}))
            yield ("messages", (ToolMessage(content="tool result", tool_call_id="t"), {}))
            yield ("messages", (AIMessageChunk(content=record_id + ")", id="second"), {}))

    chat = object.__new__(thread_module.Thread)
    chat.thread_id = "split-messages"
    chat.on_queue_state = None
    chat.runconfig = {"configurable": {"thread_id": chat.thread_id}}
    chat.thread_datastore_path = store.path
    chat.agent = Agent()
    chunks = list(chat.stream_message("Find a film"))
    text = "".join(thread_module.extract_content_text(payload)
                   for mode, payload in chunks if mode == "messages")
    assert URL not in text
    assert "ask me to verify" in text


def test_model_can_get_list_and_search_host_records_without_write_tool(tmp_path):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from assist.thread_datastore import list_thread_data, search_thread_data

    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, URL, "Decider")

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

    graph = create_agent(
        Model(responses=[AIMessage(content="", tool_calls=[
            {"name": "search_thread_data", "args": {"query": "Decider"},
             "id": "lookup-1", "type": "tool_call"}]),
            AIMessage(content="", tool_calls=[
                {"name": "get_thread_data", "args": {"record_id": record_id},
                 "id": "get-1", "type": "tool_call"}]),
            AIMessage(content="Found it")]),
        tools=[get_thread_data, list_thread_data, search_thread_data],
        middleware=[ReferenceBindingMiddleware()], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lookup-thread",
                               "thread_datastore_path": store.path}}
    result = graph.invoke({"messages": [{"role": "user", "content": "Find source"}]}, config)
    tool_results = [message for message in result["messages"]
                    if isinstance(message, ToolMessage)]
    assert len(tool_results) == 2
    assert all(record_id in item.content and URL in item.content
               for item in tool_results)

