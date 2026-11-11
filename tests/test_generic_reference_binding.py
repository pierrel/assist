"""Exact inline text, literal boundaries and real file-edit preimages."""

import json

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage

from assist.thread_datastore import (
    ReferenceBindingMiddleware, ReferenceStream,
    ThreadDataStore, put_thread_data, register_source, validate_host_report,
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
    projected = expected
    assert calls[0].tool_call["args"]["content"] == projected
    assert (workspace / "note.md").read_bytes().decode() == projected
    validate_host_report(str(workspace / "note.md"))
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
    assert resolve_references(raw, store) == expected
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


def test_eager_file_binding_leaves_no_unbound_source_projection(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    text = store.put_text("https://user.example/item")
    source = register_source(store, "https://source.example/article")
    raw = f"ref:{text}[Source](ref:{source})"
    expected = f"https://user.example/item[Source](<https://source.example/article>)"
    assert resolve_references(raw, store) == expected
    called = []
    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("write_file", file_path="/note.md", content=raw),
        lambda request: called.append(request))
    assert len(called) == 1
    assert called[0].tool_call["args"]["content"] == expected


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


def test_inline_file_write_read_update_preserves_literal_source_preimages(tmp_path, monkeypatch):
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
    assert "[Source](<https://source.example/exact>)" in draft
    old = "\n" + row + "\n"
    new = old.replace("Awaiting inspection", "Passed inspection")
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md", old_string=old, new_string=new), handler).status == "success"
    assert "Passed inspection" in note.read_text()
    assert validate_host_report(str(note)) is None
    finalized = note.read_text()
    assert "https://source.example/exact" in finalized
    assert literal in finalized
    finalized_old = "\n" + next(line for line in finalized.splitlines() if "QL-" in line) + "\n"
    final_new = finalized_old.replace("Søren Quill", f"ref:{name}").replace("Passed inspection", "Archived")
    assert binding.wrap_tool_call(_request("edit_file", file_path="/note.md", old_string=finalized_old, new_string=final_new), handler).status == "success"
    assert "Søren Quill | QL-01lI-7b92-0O5a | Archived" in note.read_text()
    assert literal in note.read_text()
    assert validate_host_report(str(note)) is None


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
    assert result.status == "error" and "unbound reference" in result.content
    assert value not in result.content
    assert _seen_urls([AIMessage(content="", tool_calls=[call]), result]) == {}


@pytest.mark.parametrize("registered", [False, True])
def test_header_edit_preserves_untouched_url_through_finalization_and_checker_read(tmp_path, monkeypatch, registered):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    refs = workspace / "references"
    refs.mkdir()
    task_id = "sub-" + "a" * 24
    file_path = "/references/" + task_id + ".md"
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

    assert binding.wrap_tool_call(_request("write_file", file_path=file_path, content=original), effect).status == "success"
    replacement = f"ref:{header_id}" if registered else header
    assert binding.wrap_tool_call(_request("edit_file", file_path=file_path, old_string="\nOLD\n",
        new_string=f"\n{replacement}\n"), effect).status == "success"
    note = refs / (task_id + ".md")
    after_edit = original.replace("OLD", header).replace(f"(ref:{source})", "(<https://source.example/exact>)")
    assert note.read_text() == after_edit
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        _request("read_file", file_path=file_path),
        ToolMessage(name="read_file", content=backend.read(file_path), tool_call_id="file-call"))
    assert checked.status == "success" and literal_url in checked.content
    assert "EXACT_ID" not in checked.content and "https://source.example/exact" in checked.content
    assert validate_host_report(str(note)) is None
    assert note.read_text() == after_edit.replace(f"(ref:{source})", "(<https://source.example/exact>)")


    # All late consumers see the same literal bytes after the real write/edit.
    from types import SimpleNamespace
    from assist.research_cleanup import ReferencesCleanupRunnable
    from tests.test_research_cleanup_sandbox import LocalSandboxTransport
    from manage.web import threads
    before = note.read_bytes()
    config = {"configurable": {"thread_datastore_path": store.path}}
    for sandbox in (None, LocalSandboxTransport()):
        validator = ReferencesCleanupRunnable(None, str(refs), sandbox)
        validator._ensure_dir()
        validator._validate_report(note.name, store)
        assert note.read_bytes() == before
        if sandbox is not None:
            sandbox.work_dir = str(workspace)
        monkeypatch.setattr(threads, "MANAGER", SimpleNamespace(
            thread_datastore_path=lambda _tid: store.path))
        threads._validate_task_report("parent", f"Task ID: {task_id}\n", str(workspace), sandbox)
        assert note.read_bytes() == before
    from assist.middleware.url_provenance import BOUND_FILE_URLS_ARTIFACT, _seen_urls
    assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == ("https://source.example/exact",)
    call = {"name": "read_file", "args": {"file_path": file_path}, "id": "file-call", "type": "tool_call"}
    assert _seen_urls([AIMessage(content="", tool_calls=[call]), checked], trust_task_results=False) == {}


def test_raw_generic_report_draft_fails_without_rewriting_bytes(tmp_path):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    report = tmp_path / "report.md"
    original = f"Value: ref:{text}\n".encode()
    report.write_bytes(original)
    with pytest.raises(ValueError, match="unbound reference"):
        validate_host_report(str(report))
    assert report.read_bytes() == original


@pytest.mark.parametrize("generic_target", [False, True])
def test_saved_literal_url_reference_syntax_stays_opaque_during_source_binding(tmp_path, generic_target):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    text = store.put_text("EXACT_ID")
    source = register_source(store, "https://source.example/exact")
    target = text if generic_target else source
    literal_url = f"https://owner.example/a[S](ref:{target})"
    original = f"URL: {literal_url}\n[S](<https://source.example/exact>)\n"
    report = tmp_path / "report.md"
    report.write_bytes(original.encode())
    assert validate_host_report(str(report)) is None
    assert report.read_bytes().decode() == original.replace(
        f"\n[S](ref:{source})", "\n[S](<https://source.example/exact>)")


@pytest.mark.parametrize("url", ["https://source.example/exact", "https://source.example/a|b",
                               "https://source.example/a`b", "https://source.example/a\\b",
                               "https://source.example/a&copy;", 'https://source.example/a"b'])
def test_literal_checker_metadata_uses_only_exact_existing_host_sources(tmp_path, monkeypatch, url):
    from assist.middleware.url_provenance import _seen_urls, normalize_url
    from assist.thread_datastore import BOUND_FILE_URLS_ARTIFACT
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    source = register_source(store, url)
    untrusted = "https://untrusted.example/value"
    store.put_text(untrusted)
    body = resolve_references(f"[S](ref:{source})", store) + "\n" + untrusted
    before = store.list(kind="source")
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        _request("read_file", file_path="/report.md"),
        ToolMessage(name="read_file", content=body, tool_call_id="file-call"))
    assert checked.content.startswith(body + "\n\n")
    assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == (url,)
    assert store.list(kind="source") == before
    call = {"name": "read_file", "id": "file-call", "args": {"file_path": "/report.md"}, "type": "tool_call"}
    messages = [AIMessage(content="", tool_calls=[call]), checked]
    assert set(_seen_urls(messages)) == {normalize_url(url)}
    assert _seen_urls(messages, trust_task_results=False) == {}


def test_checker_with_only_arbitrary_literal_urls_has_empty_authority_artifact(tmp_path, monkeypatch):
    from assist.thread_datastore import BOUND_FILE_URLS_ARTIFACT
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    body = "https://untrusted.example/value"
    store.put_text(body)
    checked = ReferenceBindingMiddleware(bind_file_refs=True)._bind_result(
        _request("read_file", file_path="/report.md"),
        ToolMessage(name="read_file", content=body, tool_call_id="file-call"))
    assert checked.content == body
    assert checked.artifact[BOUND_FILE_URLS_ARTIFACT] == ()
    assert store.list(kind="source") == []


def test_source_only_file_binding_preserves_literal_url_and_adjacent_citation(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    source = register_source(store, "https://source.example/exact")
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    raw = f"Literal: https://owner.example/a[S](ref:{source})\n[Owner](https://owner.example)[S](ref:{source})\n"
    expected = raw.replace(f"\n[Owner](https://owner.example)[S](ref:{source})", "\n[Owner](https://owner.example)[S](<https://source.example/exact>)")
    calls = []
    ReferenceBindingMiddleware().wrap_tool_call(
        _request("write_file", file_path="/note.md", content=raw),
        lambda request: calls.append(request))
    assert calls[0].tool_call["args"]["content"] == expected
    assert not store.has_text_records()


def test_crlf_reference_edit_arguments_keep_exact_line_bytes(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    record = store.put_text("EXACT_ID")
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    request = _request("edit_file", file_path="/note.md",
        old_string=f"\r\nID: ref:{record}\r\n", new_string=f"\r\nChecked: ref:{record}\r\n")
    prepared = ReferenceBindingMiddleware()._bind_arguments(request)
    assert isinstance(prepared, ToolCallRequest)
    assert prepared.tool_call["args"]["old_string"] == "\r\nID: EXACT_ID\r\n"
    assert prepared.tool_call["args"]["new_string"] == "\r\nChecked: EXACT_ID\r\n"


def test_legacy_source_draft_edit_requires_literal_preimage_without_mutation(tmp_path, monkeypatch):
    store = ThreadDataStore(str(tmp_path / "thread.sqlite"))
    source = register_source(store, "https://source.example/exact")
    monkeypatch.setattr("assist.thread_datastore._current_store", lambda: store)
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    original = f"\n[S](ref:{source})\n"
    note = tmp_path / "legacy.md"
    note.write_text(original)
    def effect(request):
        args = request.tool_call["args"]
        result = backend.edit(args["file_path"], args["old_string"], args["new_string"])
        return ToolMessage(name="edit_file", content=result.error or "Updated",
                           status="error" if result.error else "success", tool_call_id="file-call")
    result = ReferenceBindingMiddleware().wrap_tool_call(
        _request("edit_file", file_path="/legacy.md", old_string=original, new_string="\nReplaced\n"), effect)
    assert result.status == "error"
    assert note.read_text() == original
