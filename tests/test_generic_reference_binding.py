"""Exact inline text, literal boundaries and real file-edit preimages."""

import json

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

from assist.thread_datastore import (
    ReferenceBindingMiddleware, ReferenceStream,
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


@pytest.mark.parametrize("value", ["literal ref:r123", "A ReF: token", "ref:", "x" * (65536 + 1),
                                  "http:// ref:unknown", "https://\nref:unknown",
                                  "[Owner](https://) ref:unknown"])
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


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quote_bearing_source_and_ordinary_urls_remain_literal(tmp_path, quote):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    url = f"https://source.example/path{quote}ref:{text}"
    source = register_source(store, url)
    raw = f"Value: ref:{text}\n[S](ref:{source})\n[Ordinary](<{url}>)"
    expected = f"Value: EXACT_ID\n[S](<{url}>)\n[Ordinary](<{url}>)"
    assert resolve_references(raw, store) == expected
    assert resolve_references(expected, store) == expected


@pytest.mark.parametrize("layout", ["[Owner](https://owner.example)[Source](ref:{source})",
                                    "`https://owner.example`[Source](ref:{source})",
                                    "`curl https://owner.example`[Source](ref:{source})",
                                    "``https://owner.example/a`b``[Source](ref:{source})",
                                    "[https://owner.example](https://owner.example)[Source](ref:{source})",
                                    "[See https://owner.example](https://owner.example)[Source](ref:{source})",
                                    "[Owner](https://owner.example/a_(b))[Source](ref:{source})",
                                    "[Owner]( https://owner.example)[Source](ref:{source})",
                                    "[Owner](https://owner.example/a\\)b)[Source](ref:{source})",
                                    "|https://owner.example|ref:{text}|",
                                    "Owner | ID\n--- | ---\nhttps://owner.example|ref:{text}\n",
                                    "Owner | ID\r\n--- | ---\r\nhttps://owner.example|ref:{text}\r\n",
                                    "https://owner.example|ref:{text}\n---|---\nOwner|ID\n",
                                    "http:// ref:{text}", "https://\nref:{text}",
                                    "[Owner](https://) ref:{text}",
                                    "| See https://owner.example/a and https://other.example|ref:{text}|",
                                    "|[Owner](https://owner.example)|ref:{text}|"])
def test_url_literals_stop_at_surrounding_link_and_table_boundaries(tmp_path, monkeypatch, layout):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("EXACT_ID")
    source = register_source(store, "https://source.example/exact")
    raw = layout.format(source=source, text=text)
    expected = raw.replace(f"ref:{text}", "EXACT_ID").replace(
        f"(ref:{source})", "(<https://source.example/exact>)")
    assert resolve_references(raw, store) == expected
    assert resolve_references(expected, store) == expected
    stream = ReferenceStream(store)
    assert stream.push(raw) == ""
    assert stream.push("", final=True) == expected
    assert ReferenceBindingMiddleware().after_model(
        {"messages": [AIMessage(content=raw)]}, None)["messages"][0].content == expected
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    calls = []

    def write(request):
        calls.append(request)
        args = request.tool_call["args"]
        effect = backend.write(args["file_path"], args["content"])
        return ToolMessage(content=effect.error or "Updated", status="error" if effect.error else "success",
                           name="write_file", tool_call_id="file-call")

    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("write_file", file_path="/note.md", content=raw),
        write)
    assert result.status == "success" and len(calls) == 1
    projected = raw.replace(f"ref:{text}", "EXACT_ID")
    assert calls[0].tool_call["args"]["content"] == projected
    assert (workspace / "note.md").read_bytes().decode() == projected
    resolve_host_report(str(workspace / "note.md"), store)
    assert (workspace / "note.md").read_bytes().decode() == expected


def test_actual_url_bytes_escaped_literals_and_known_source_drafts_stay_exact(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("EXACT_ID")
    source = register_source(store, "https://source.example/exact")
    raw = (f"https://owner.example/a|ref:{text}\n"
           f"https://owner.example|ref:{text}\n"
           f"[Owner](<https://owner.example/a'\"|ref:{text}>)\n"
           f"|https://owner.example/a\\|ref:{text}|\\ref:unknown|\n"
           f"[Source](ref:{source})")
    projected = resolve_references(raw, store, text_only=True)
    assert projected == raw
    bound = resolve_references(raw, store)
    assert bound == raw.replace(f"(ref:{source})", "(<https://source.example/exact>)")
    assert resolve_references(bound, store) == bound


@pytest.mark.parametrize("placement", ["`ref:r00000000000000000000000000000000`",
                                       "|ref:r00000000000000000000000000000000|",
                                       "Owner | ID\n--- | ---\nhttps://owner.example|ref:unknown\n",
                                       "http:// ref:unknown", "https://\nref:unknown",
                                       "[Owner](https://) ref:unknown",
                                       "|ref:{text}-suffix|", "ref:"])
def test_unknown_inline_references_fail_before_reply_or_file_effects(tmp_path, monkeypatch, placement):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("EXACT_ID")
    raw = placement.format(text=text)
    with pytest.raises(ValueError, match="unknown|malformed"):
        resolve_references(raw, store)
    calls = []
    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("write_file", file_path="/note.md", content=raw),
        lambda request: calls.append(request))
    assert result.status == "error" and calls == []
    canonical = ReferenceBindingMiddleware().after_model(
        {"messages": [AIMessage(content=raw)]}, None)["messages"][0].content
    assert "reference unresolved" in canonical
    stream = ReferenceStream(store)
    assert stream.push(raw) == ""
    assert stream.push("", final=True) == canonical


@pytest.mark.parametrize("header,table", [("`A|B` | ID", True), (r"A\|B | ID", True),
                                          ("`A|B`", False), (r"A\|B", False),
                                          ("A | B | C", False)])
def test_pipe_less_table_headers_respect_code_escapes_and_width(tmp_path, header, table):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    raw = f"{header}\n--- | ---\nhttps://owner.example|ref:{text}\n"
    expected = raw.replace(f"ref:{text}", "EXACT_ID") if table else raw
    assert resolve_references(raw, store) == expected


@pytest.mark.parametrize("opening,middle", [("```", ""), ("```python", "```notclosing\n"),
                                            ("```python", "````\n"), ("```python", "```\t\n")])
def test_table_shaped_fenced_code_keeps_literal_url_bytes(tmp_path, opening, middle):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    raw = f"{opening}\n{middle}Owner | ID\n--- | ---\nhttps://owner.example|ref:{text}\n```\n"
    assert resolve_references(raw, store) == raw


def test_full_table_write_widened_edit_and_ambiguous_fragment_feedback(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    first, second = store.put_text("FIRST_ID"), store.put_text("SECOND_ID")
    binding = ReferenceBindingMiddleware()
    calls = []

    def effect(request):
        calls.append(request)
        args = request.tool_call["args"]
        result = (backend.write(args["file_path"], args["content"])
                  if request.tool_call["name"] == "write_file" else
                  backend.edit(args["file_path"], args["old_string"], args["new_string"]))
        return ToolMessage(content=result.error or "Updated", status="error" if result.error else "success",
                           name=request.tool_call["name"], tool_call_id="file-call")

    original = f"Lead\n\nOwner | ID\n--- | ---\nhttps://owner.example|ref:{first}\n"
    assert binding.wrap_tool_call(_request("write_file", file_path="/note.md", content=original), effect).status == "success"
    note = workspace / "note.md"
    saved = original.replace(f"ref:{first}", "FIRST_ID")
    assert note.read_text() == saved
    before = len(calls)
    for reference in (f"ref:{second}", "ref:unknown"):
        result = binding.wrap_tool_call(_request("edit_file", file_path="/note.md",
            old_string="\nhttps://owner.example|FIRST_ID\n",
            new_string=f"\nhttps://owner.example|{reference}\n"), effect)
        assert result.status == "error" and "header/context" in result.content
        assert len(calls) == before and note.read_text() == saved
    old = saved[saved.index("\n"):]
    new = old.replace("FIRST_ID", f"ref:{second}")
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md", old_string=old, new_string=new), effect).status == "success"
    assert note.read_text() == saved.replace("FIRST_ID", "SECOND_ID")
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md",
        old_string="\nhttps://owner.example|SECOND_ID\n",
        new_string="\nhttps://owner.example|LITERAL_ID\n"), effect).status == "success"


@pytest.mark.parametrize("slashes", [0, 1, 2, 3, 4])
def test_url_pipe_edit_ambiguity_uses_escape_pair_parity(tmp_path, monkeypatch, slashes):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    store.put_text("EXACT_ID")
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    prefix = "https://owner.example/a" + "\\" * slashes + "|"
    request = _request("edit_file", file_path="/note.md", old_string="\n" + prefix + "OLD\n",
                       new_string="\n" + prefix + "ref:unknown\n")
    prepared = ReferenceBindingMiddleware()._bind_arguments(request)
    if slashes % 2 == 0:
        assert isinstance(prepared, ToolMessage) and prepared.status == "error"
        assert "header/context" in prepared.content
    else:
        assert isinstance(prepared, ToolCallRequest) and prepared.tool_call == request.tool_call


def test_table_shaped_paragraph_keeps_literal_url_bytes(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    raw = f"Intro\nOwner | ID\n--- | ---\nhttps://owner.example|ref:{text}\n"
    assert resolve_references(raw, store) == raw


@pytest.mark.parametrize("length", [532, 8192])
def test_long_ordinary_url_labels_bind_adjacent_sources_without_changing_url(tmp_path, length):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    source = register_source(store, "https://source.example/exact")
    prefix = f"https://owner.example/[a]/ref:{text}/"
    url = prefix + "x" * (length - len(prefix))
    raw = f"[{url}]({url})[S](ref:{source})"
    expected = raw.replace(f"(ref:{source})", "(<https://source.example/exact>)")
    assert resolve_references(raw, store) == expected
    assert resolve_references(expected, store) == expected
    assert resolve_references(raw, store, text_only=True) == raw
    for split in (0, 1, len(url), len(raw) // 2, len(raw)):
        stream = ReferenceStream(store)
        assert stream.push(raw[:split]) == ""
        assert stream.push(raw[split:]) == ""
        assert stream.push("", final=True) == expected


def test_escaped_reference_heavy_url_edit_stays_literal(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    store.put_text("EXACT_ID")
    raw = "\nhttps://owner.example/|" + r"\ref:x" * 32000 + "\n"
    assert resolve_references(raw, store, edit_context=True) == raw


def test_generic_only_file_projection_cannot_hide_an_existing_source(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("https://user.example/item")
    source = register_source(store, "https://source.example/article")
    raw = f"ref:{text}[Source](ref:{source})"
    with pytest.raises(ValueError, match="source-draft parsing"):
        resolve_references(raw, store, text_only=True)
    called = []
    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("write_file", file_path="/note.md", content=raw),
        lambda request: called.append(request))
    assert result.status == "error"
    assert called == []


@pytest.mark.parametrize("phase", ["before_text", "dry_pass"])
def test_full_and_validation_passes_stop_source_amplification_at_the_bound(tmp_path, monkeypatch, phase):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("ref" if phase == "dry_pass" else "X")
    source = register_source(store, "https://source.example/" + "x" * (8192 - 23))
    original_get = store.get
    source_reads = []

    def get(record_id):
        if record_id == source:
            source_reads.append(record_id)
        return original_get(record_id)

    monkeypatch.setattr(store, "get", get)
    raw = ((f"ref:{text}:{source} " * 2000) if phase == "dry_pass" else
           (f"ref:{source} " * 2000) + f"ref:{text}")
    with pytest.raises(ValueError):
        resolve_references(raw, store)
    assert len(source_reads) <= 1026


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
    assert result.status == "error" and "unbound generic" in result.content
    assert value not in result.content
    assert _seen_urls([AIMessage(content="", tool_calls=[call]), result]) == {}


@pytest.mark.parametrize("registered", [False, True])
def test_header_edit_preserves_untouched_url_through_finalization_and_checker_read(tmp_path, monkeypatch, registered):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("EXACT_ID")
    header = "Owner | ID\n--- | ---"
    header_id = store.put_text(header)
    source = register_source(store, "https://source.example/exact")
    literal_url = f"https://owner.example|ref:{text}"
    original = f"Lead\n\nOLD\n{literal_url}\n[S](ref:{source})\n"
    binding = ReferenceBindingMiddleware()

    def effect(request):
        args = request.tool_call["args"]
        result = (backend.write(args["file_path"], args["content"])
                  if request.tool_call["name"] == "write_file" else
                  backend.edit(args["file_path"], args["old_string"], args["new_string"]))
        return ToolMessage(content=result.error or "Updated", status="error" if result.error else "success",
                           name=request.tool_call["name"], tool_call_id="file-call")

    assert binding.wrap_tool_call(_request("write_file", file_path="/note.md", content=original), effect).status == "success"
    replacement = f"ref:{header_id}" if registered else header
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md", old_string="\nOLD\n",
        new_string=f"\n{replacement}\n"), effect).status == "success"
    note = workspace / "note.md"
    after_edit = original.replace("OLD", header)
    assert note.read_text() == after_edit
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        _request("read_file", file_path="/note.md"),
        ToolMessage(name="read_file", content=after_edit, tool_call_id="file-call"))
    assert checked.status == "success" and literal_url in checked.content
    assert "EXACT_ID" not in checked.content and "https://source.example/exact" in checked.content
    assert resolve_host_report(str(note), store)
    assert note.read_text() == after_edit.replace(f"(ref:{source})", "(<https://source.example/exact>)")


def test_raw_generic_report_draft_fails_without_rewriting_bytes(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    report = tmp_path / "report.md"
    original = f"Value: ref:{text}\n".encode()
    report.write_bytes(original)
    with pytest.raises(ValueError, match="unbound generic"):
        resolve_host_report(str(report), store)
    assert report.read_bytes() == original
