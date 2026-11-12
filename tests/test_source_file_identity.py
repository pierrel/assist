"""Canonical source destinations invert exactly without confusing bare URI bytes."""

from html.parser import HTMLParser

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from markdown import markdown

from assist.thread_datastore import (
    BOUND_FILE_URLS_ARTIFACT, ReferenceBindingMiddleware, ThreadDataStore,
    _decode_source_destination, _file_source_records, _source_destination,
    register_source, validate_host_report,
)


PAIRS = [
    ("https://source.example/a|b", "https://source.example/a&#124;b"),
    ("https://source.example/a\\b", "https://source.example/a&#92;b"),
    ("https://source.example/a`b", "https://source.example/a&#96;b"),
    ("https://source.example/a&copy;", "https://source.example/a&amp;copy;"),
]


def request(name, **args):
    return ToolCallRequest(tool_call={"name": name, "args": args, "id": "call", "type": "tool_call"},
                           tool=None, state={}, runtime=None)


class Hrefs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.values.append(dict(attrs)["href"])


@pytest.mark.parametrize("urls", PAIRS)
def test_each_colliding_source_saves_renders_and_checks_to_its_own_record(tmp_path, monkeypatch, urls):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    ids = {url: register_source(store, url) for url in urls}
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    middleware = ReferenceBindingMiddleware()
    assert _source_destination(urls[0]) != _source_destination(urls[1])
    for index, url in enumerate(urls):
        path = f"/{index}.md"
        def write(prepared):
            args = prepared.tool_call["args"]
            result = backend.write(args["file_path"], args["content"])
            assert not result.error
            return ToolMessage(name="write_file", content="Saved", tool_call_id="call")
        middleware.wrap_tool_call(request("write_file", file_path=path,
            content=f"[Source](ref:{ids[url]})"), write)
        saved = workspace / f"{index}.md"
        before = saved.read_bytes()
        rendered = Hrefs()
        rendered.feed(markdown(before.decode(), extensions=["tables"]))
        assert rendered.values == [url]
        assert _file_source_records(before.decode(), store) == [store.get(ids[url])]
        checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
            request("read_file", file_path=path),
            ToolMessage(name="read_file", content=backend.read(path), tool_call_id="call"))
        assert checked.status == "success"
        assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == (url,)
        validate_host_report(str(saved))
        assert saved.read_bytes() == before
        assert _decode_source_destination(_source_destination(url)) == url
    assert len(store.list(kind="source")) == 2


@pytest.mark.parametrize("wrapper", ["Bare: {}", "`{}`", "`<{}>`", "    <{}>", "~~~\n<{}>\n~~~", ">     <{}>", "<{}>"])
def test_bare_and_code_literal_uri_keeps_its_own_identity(tmp_path, wrapper):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    a, b = PAIRS[0]
    register_source(store, a)
    b_id = register_source(store, b)
    text = wrapper.format(b)
    assert _file_source_records(text, store) == [store.get(b_id)]


@pytest.mark.parametrize("url", ["https://source.example/a?x=1&y=2", "https://source.example/a&copy",
                                "https://source.example/a&bogus;", "https://source.example/a&amp;b"])
def test_canonical_inverse_does_not_decode_unrelated_or_unterminated_entities(tmp_path, url):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    rid = register_source(store, url)
    encoded = _source_destination(url)
    assert _decode_source_destination(encoded) == url
    assert _file_source_records(f"[S](<{encoded}>)", store) == [store.get(rid)]
    assert _file_source_records(url, store) == [store.get(rid)]


def test_bare_literal_and_generic_uri_cannot_borrow_an_encoded_source_identity(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    a, b = PAIRS[0]
    register_source(store, a)
    store.put_text(b)
    assert _file_source_records(b, store) == []
    assert _file_source_records("`<" + b + ">`", store) == []
    assert len(store.list(kind="source")) == 1


def test_noncanonical_angle_encoding_does_not_select_a_different_source(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    register_source(store, PAIRS[0][0])
    register_source(store, "https://source.example/a&vert;b")
    with pytest.raises(ValueError, match="noncanonical"):
        _file_source_records("[S](<https://source.example/a&vert;b>)", store)


@pytest.mark.parametrize("text", [
    ">     [S](<{}>)", "> [S](<{}>)", "- [S](<{}>)",
    "<!-- [S](<{}>) -->", "<pre>\n[S](<{}>)\n</pre>",
])
def test_unsupported_encoded_link_context_fails_without_selecting_another_record(tmp_path, monkeypatch, text):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    a, b = PAIRS[0]
    register_source(store, a)
    register_source(store, b)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    original = text.format(b)
    report = tmp_path / "report.md"
    report.write_text(original)
    with pytest.raises(ValueError, match="ambiguous"):
        _file_source_records(original, store)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"),
        ToolMessage(name="read_file", content=original, tool_call_id="call"))
    assert checked.status == "error" and "ambiguous" in checked.content
    assert not checked.artifact or not checked.artifact.get(BOUND_FILE_URLS_ARTIFACT)
    assert report.read_bytes() == original.encode()
    assert len(store.list(kind="source")) == 2


@pytest.mark.parametrize("text", [
    "paragraph\n    [S](<{}>)", "- item\n    [S](<{}>)", "1. item\n    [S](<{}>)",
    "> paragraph\n    [S](<{}>)", "- item\n\n    [S](<{}>)", r"\[S](<{}>)",
    "`[S](<{}>)`", "    [S](<{}>)",
])
def test_uncertain_link_shaped_indentation_or_escaping_never_chooses_literal_collision(tmp_path, text):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    a, b = PAIRS[0]
    register_source(store, a)
    register_source(store, b)
    original = text.format(b)
    with pytest.raises(ValueError, match="ambiguous"):
        _file_source_records(original, store)


def test_escaped_producer_link_reports_verification_failure_without_byte_changes(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    a, b = PAIRS[0]
    aid = register_source(store, a)
    register_source(store, b)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    from assist.thread_datastore import resolve_references
    saved = resolve_references(r"\[S](ref:" + aid + ")", store, file_context=True)
    file = tmp_path / "report.md"
    file.write_text(saved)
    before = file.read_bytes()
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"),
        ToolMessage(name="read_file", content="     1\t" + saved, tool_call_id="call"))
    assert checked.status == "error" and "ambiguous" in checked.content
    assert file.read_bytes() == before
