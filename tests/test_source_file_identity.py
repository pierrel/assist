"""Canonical source destinations invert exactly without confusing bare URI bytes."""

from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
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


def read_file(backend, path):
    """Use the installed tool's ReadResult and line-number formatting path."""
    tool = FilesystemMiddleware(backend=backend)._create_read_file_tool()
    return tool.func(file_path=path, runtime=SimpleNamespace(tool_call_id="call"))


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
            read_file(backend, path))
        assert checked.status == "success"
        assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == (url,)
        validate_host_report(str(saved))
        assert saved.read_bytes() == before
        assert _decode_source_destination(_source_destination(url)) == url
    assert len(store.list(kind="source")) == 2


@pytest.mark.parametrize("wrapper", ["Bare: {}", "`{}`", "`<{}>`", "    <{}>", "~~~\n<{}>\n~~~",
                                    "paragraph `<{}>`", "paragraph\n\n~~~\n<{}>\n~~~", "\n    <{}>"])
def test_bare_and_code_literal_uri_keeps_its_own_identity(tmp_path, monkeypatch, wrapper):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    a, b = PAIRS[0]
    register_source(store, a)
    b_id = register_source(store, b)
    text = wrapper.format(b)
    assert _file_source_records(text, store) == [store.get(b_id)]
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    saved = tmp_path / "report.md"
    saved.write_text(text)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"), read_file(backend, "/report.md"))
    assert checked.status == "success"
    assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == (b,)
    assert saved.read_bytes() == text.encode()


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


@pytest.mark.parametrize("urls", PAIRS)
@pytest.mark.parametrize("wrapper", ["[S]({})", '[S](<{}> "Title")', "[S](<{}> )", "[S]( {} )", "[S]( <{}> )",
                                    "[S](\n{})", "[S](\n<{}>)", "[S](\r\n{})", "[S](\r\n<{}>)",
                                    '[S](\n<{}> "Title")', "[S](<{}>\n)", "<{}>", ">     <{}>",
                                    "[S][id]\n\n[id]: <{}>", "[S][id]\n\n[id]: {}"])
def test_unsupported_encoded_destinations_fail_before_raw_collision_selection(tmp_path, monkeypatch, urls, wrapper):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    for url in urls:
        register_source(store, url)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    original = wrapper.format(_source_destination(urls[0]))
    saved = tmp_path / "report.md"
    saved.write_text(original)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    with pytest.raises(ValueError, match="unsupported encoded"):
        _file_source_records(original, store)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"),
        read_file(backend, "/report.md"))
    assert checked.status == "error" and "unsupported encoded" in checked.content
    assert not checked.artifact or not checked.artifact.get(BOUND_FILE_URLS_ARTIFACT)
    assert saved.read_bytes() == original.encode()
    assert len(store.list(kind="source")) == 2


@pytest.mark.parametrize("prefix", ["2026 edition ", "123456789012345678901234567890 edition ", "2)without space ", "2.without space "])
def test_numeric_prose_reference_write_and_real_read_keep_source_identity(tmp_path, monkeypatch, prefix):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    url = PAIRS[0][0]
    rid = register_source(store, url)
    register_source(store, PAIRS[0][1])
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    def write(prepared):
        args = prepared.tool_call["args"]
        assert not backend.write(args["file_path"], args["content"]).error
        return ToolMessage(name="write_file", content="Saved", tool_call_id="call")
    ReferenceBindingMiddleware().wrap_tool_call(request("write_file", file_path="/report.md",
        content=prefix + f"[S](ref:{rid})"), write)
    before = (tmp_path / "report.md").read_bytes()
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"),
        read_file(backend, "/report.md"))
    assert checked.status == "success"
    assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == (url,)
    assert (tmp_path / "report.md").read_bytes() == before


@pytest.mark.parametrize("prefix", ["1. ", "2) ", "123456789012345678901234567890. "])
def test_actual_ordered_list_prefix_remains_unsupported_without_truncation(tmp_path, prefix):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    for url in PAIRS[0]:
        register_source(store, url)
    with pytest.raises(ValueError, match="ambiguous"):
        _file_source_records(prefix + f"[S](<{_source_destination(PAIRS[0][0])}>)", store)


def test_unsupported_named_entity_destination_cannot_select_literal_source(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    register_source(store, "https://source.example/a|b")
    register_source(store, "https://source.example/a&vert;b")
    with pytest.raises(ValueError, match="unsupported encoded"):
        _file_source_records("[S](https://source.example/a&vert;b)", store)


def test_unsupported_backslash_destination_cannot_select_literal_source(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    register_source(store, "https://source.example/a)b")
    register_source(store, "https://source.example/a\\)b")
    with pytest.raises(ValueError, match="unsupported encoded"):
        _file_source_records("[S](https://source.example/a\\)b)", store)


@pytest.mark.parametrize("wrapper", ["[S]({})", '[S](<{}> "Title")', "[S](<{}> )", "<{}>", "[S][id]\n\n[id]: <{}>"])
def test_unserialized_destinations_keep_exact_existing_source(tmp_path, wrapper):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    url = "https://source.example/exact"
    rid = register_source(store, url)
    assert _file_source_records(wrapper.format(url), store) == [store.get(rid)]


@pytest.mark.parametrize("wrapper", ["paragraph\n    <{}>", "- item\n    <{}>",
                                    "paragraph\n    [S]({})", "- item\n\n    [S]({})"])
def test_uncertain_indentation_cannot_bypass_actual_checker_with_literal_collision(tmp_path, monkeypatch, wrapper):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    for url in PAIRS[0]:
        register_source(store, url)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    original = wrapper.format(PAIRS[0][1])
    parsed = Hrefs()
    parsed.feed(markdown(original))
    assert parsed.values == [PAIRS[0][0]]
    saved = tmp_path / "report.md"
    saved.write_text(original)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"), read_file(backend, "/report.md"))
    assert checked.status == "error"
    assert not checked.artifact or not checked.artifact.get(BOUND_FILE_URLS_ARTIFACT)
    assert saved.read_bytes() == original.encode()


def test_later_indented_angle_is_conservatively_unsupported_without_container_grammar(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    for url in PAIRS[0]:
        register_source(store, url)
    original = "paragraph\n\n    <" + PAIRS[0][1] + ">"
    rendered = Hrefs()
    rendered.feed(markdown(original))
    assert rendered.values == []
    with pytest.raises(ValueError, match="unsupported encoded"):
        _file_source_records(original, store)


@pytest.mark.parametrize("wrapper", ['<a href={}>source</a>', '<a/href={}>source'])
def test_html_attribute_cannot_select_raw_entity_collision_through_actual_read(tmp_path, monkeypatch, wrapper):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    for url in PAIRS[0]:
        register_source(store, url)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    original = wrapper.format(PAIRS[0][1])
    parsed = Hrefs()
    parsed.feed(markdown(original))
    assert parsed.values == [PAIRS[0][0]]
    saved = tmp_path / "report.md"
    saved.write_text(original)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"), read_file(backend, "/report.md"))
    assert checked.status == "error"
    assert not checked.artifact or not checked.artifact.get(BOUND_FILE_URLS_ARTIFACT)
    assert saved.read_bytes() == original.encode()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_multiline_backtick_heuristic_cannot_bypass_actual_checker(tmp_path, monkeypatch, newline):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    for url in PAIRS[0]:
        register_source(store, url)
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    original = '`start' + newline * 2 + '<' + PAIRS[0][1] + '>' + newline * 2 + 'end`'
    parsed = Hrefs()
    parsed.feed(markdown(original))
    assert parsed.values == [PAIRS[0][0]]
    saved = tmp_path / "report.md"
    saved.write_bytes(original.encode())
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        request("read_file", file_path="/report.md"), read_file(backend, "/report.md"))
    assert checked.status == "error"
    assert not checked.artifact or not checked.artifact.get(BOUND_FILE_URLS_ARTIFACT)
    assert saved.read_bytes() == original.encode()
