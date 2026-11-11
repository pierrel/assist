"""Exact inline text, literal boundaries and real file-edit preimages."""

import json

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

from assist.thread_datastore import (
    BOUND_FILE_URLS_ARTIFACT, ReferenceBindingMiddleware, ReferenceStream,
    ThreadDataStore, put_thread_data, register_source, resolve_host_report,
    resolve_references, source_url,
)


def _request(name, **args):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "file-call", "type": "tool_call"},
        tool=None, state={}, runtime=None)


def test_registration_is_immutable_text_and_cannot_mint_a_source(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    value = "'); DROP TABLE records; --"
    first = json.loads(put_thread_data(value, "Exact quoted text"))
    second = json.loads(put_thread_data("https://untrusted.example/item"))
    assert first["reference"] == f"ref:{first['id']}"
    assert store.get(first["id"])["value"] == value
    assert store.get(first["id"])["kind"] == "text"
    assert source_url(store, second["id"]) is None
    assert len(store.list()) == 2
    assert ThreadDataStore(store.path).get(first["id"])["value"] == value


@pytest.mark.parametrize("value", ["literal ref:r123", "A ReF: token", "ref:", "x" * (65536 + 1)])
def test_registration_rejects_reserved_syntax_and_oversize_without_writes(tmp_path, monkeypatch, value):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    result = put_thread_data(value)
    assert "invalid" in result.lower() or "reserved" in result.lower()
    assert store.list() == []


def test_inline_values_match_stream_checkpoint_and_ordinary_text(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    name = store.put_text("Søren Quill")
    identifier = store.put_text("QL-01lI-7b92-0O5a")
    quote = store.put_text("Hold at 4 °C; do not freeze.")
    source = register_source(store, "https://source.example/exact")
    ordinary = (f"\n[Owner](https://owner.example/item?literal=ref:{name})"
                " `curl https://api.example/v1`")
    raw = (f"Recipient: ref:{name}; ID: `ref:{identifier}`.\n"
           f"| Label | ref:{quote} |\n[ref:{name}](ref:{source})" + ordinary)
    expected = ("Recipient: Søren Quill; ID: `QL-01lI-7b92-0O5a`.\n"
                "| Label | Hold at 4 °C; do not freeze. |\n"
                "[Søren Quill](<https://source.example/exact>)" + ordinary)
    assert resolve_references(raw, store) == expected
    assert resolve_references(expected, store) == expected
    saved = ReferenceBindingMiddleware().after_model(
        {"messages": [AIMessage(content=raw, id="answer")]}, None)["messages"][0]
    assert saved.content == expected
    assert saved.id == "answer"
    for split in range(len(raw) + 1):
        stream = ReferenceStream(store)
        assert stream.push(raw[:split]) == ""
        assert stream.push(raw[split:]) == ""
        assert stream.push("", final=True) == expected


@pytest.mark.parametrize("placement", ["[Label](ref:{text})", "ref:{text}:rSOURCE",
                                         "ref:{text} Label](ref:{source})"])
def test_ambiguous_or_unexpanded_generic_refs_never_write(tmp_path, monkeypatch, placement):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    source = register_source(store, "https://source.example/exact")
    value = "ref" if ":rSOURCE" in placement else "["
    text = store.put_text(value)
    raw = placement.format(text=text, source=source).replace("rSOURCE", source)
    called = []
    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("write_file", file_path="/note.md", content=raw),
        lambda request: called.append(request))
    assert result.status == "error"
    assert called == []
    with pytest.raises(ValueError):
        resolve_references(raw, store)
    canonical = ReferenceBindingMiddleware().after_model(
        {"messages": [AIMessage(content=raw)]}, None)["messages"][0].content
    assert "reference unresolved" in canonical
    stream = ReferenceStream(store)
    assert stream.push(raw[:8]) == ""
    assert stream.push(raw[8:]) == ""
    assert stream.push("", final=True) == canonical


def test_explicit_literal_escape_is_preserved_through_repeated_binding(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    literal = r"Keep \ref:r0123456789abcdef0123456789abcdef literally."
    record = json.loads(put_thread_data(literal))
    raw = f"Quote: ref:{record['id']}"
    assert resolve_references(raw, store) == "Quote: " + literal
    assert resolve_references("Quote: " + literal, store) == "Quote: " + literal
    escaped_token = "\\" + record["reference"]
    assert resolve_references(escaped_token, store) == escaped_token


def test_stream_rechecks_generic_availability_after_a_message_boundary(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    stream = ReferenceStream(store)
    assert stream.push("Before.") == "Before."
    stream.reset()
    record = store.put_text("Søren Quill")
    assert stream.push(f"Recipient: ref:{record}") == ""
    assert stream.push("", final=True) == "Recipient: Søren Quill"


def test_generic_expansion_is_bounded_without_truncating_exact_values(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    record = store.put_text("é" * 65536)
    with pytest.raises(ValueError):
        resolve_references((f"ref:{record} " * 65), store)


def test_inline_file_write_read_update_preserves_source_draft_and_final_preimages(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    name = store.put_text("Søren Quill")
    identifier = store.put_text("QL-01lI-7b92-0O5a")
    literal = r"Keep \ref:r0123456789abcdef0123456789abcdef literally."
    literal_record = store.put_text(literal)
    source = register_source(store, "https://source.example/exact")
    row = f"| ref:{name} | ref:{identifier} | Awaiting inspection | [Source](ref:{source}) |"
    original = "Header\n" + row + f"\nFooter: ref:{literal_record}\n"
    binding = ReferenceBindingMiddleware()

    def handler(request):
        args = request.tool_call["args"]
        if request.tool_call["name"] == "write_file":
            effect = backend.write(args["file_path"], args["content"])
        else:
            effect = backend.edit(args["file_path"], args["old_string"], args["new_string"])
        return ToolMessage(content=effect.error or "Updated", status="error" if effect.error else "success",
                           name=request.tool_call["name"], tool_call_id="file-call")

    assert binding.wrap_tool_call(_request("write_file", file_path="/note.md", content=original), handler).status == "success"
    note = workspace / "note.md"
    draft = note.read_text()
    assert "Søren Quill | QL-01lI-7b92-0O5a" in draft
    assert f"[Source](ref:{source})" in draft
    old = "\n" + row + "\n"
    new = old.replace("Awaiting inspection", "Passed inspection")
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md", old_string=old, new_string=new), handler).status == "success"
    assert "Passed inspection" in note.read_text()
    assert resolve_host_report(str(note), store)
    finalized = note.read_text()
    assert "https://source.example/exact" in finalized
    assert literal in finalized
    finalized_old = "\n" + next(line for line in finalized.splitlines() if "QL-" in line) + "\n"
    final_new = finalized_old.replace("Søren Quill", f"ref:{name}").replace("Passed inspection", "Archived")
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md", old_string=finalized_old, new_string=final_new), handler).status == "success"
    assert "Søren Quill | QL-01lI-7b92-0O5a | Archived" in note.read_text()
    assert literal in note.read_text()
    assert not resolve_host_report(str(note), store)


def test_reference_edit_without_preserved_newlines_has_no_effect(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("[")
    called = []
    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("edit_file", file_path="/note.md", old_string="OLD", new_string=f"ref:{text}"),
        lambda request: called.append(request))
    assert result.status == "error"
    assert "newline" in result.content.lower()
    assert called == []


def test_generic_url_never_enters_registration_or_checker_navigation_authority(tmp_path, monkeypatch):
    from assist.middleware.url_provenance import _seen_urls
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    value = "https://untrusted.example/item"
    text = store.put_text(value)
    put_result = ToolMessage(name="put_thread_data", content=json.dumps({"value": value}), tool_call_id="put")
    assert _seen_urls([put_result]) == {}
    call = {"name": "read_file", "id": "file-call", "args": {"file_path": "/report.md"}, "type": "tool_call"}
    result = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        _request("read_file", file_path="/report.md"),
        ToolMessage(name="read_file", content=f"Link: ref:{text}", tool_call_id="file-call"))
    assert value in result.content
    assert result.artifact[BOUND_FILE_URLS_ARTIFACT] == ()
    assert _seen_urls([AIMessage(content="", tool_calls=[call]), result]) == {}
