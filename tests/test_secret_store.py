"""Private thread unlock, opaque browser use, and observation redaction."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import quote, quote_plus

import pytest

from assist.browser.manager import (BrowserSession, BrowserUnavailable,
                                    _contains_hidden, _replace_hidden, browser_tools)
from assist.browser.runner import BrowserWorker
from assist.browser.redaction import MAX_PRIVATE_COMMAND
from assist.secret_store import FILE, KEY_FILE, REQUEST_FILE, SecretStore, SecretStoreError


PASSWORD = "synthetic/riv er?8642"
PHRASE = "synthetic unlock phrase only"
PAGE = "https://synthetic.example/login"


def _store(tmp_path):
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    repo.mkdir()
    thread.mkdir()
    store = SecretStore(repo, thread)
    store.unlock(PHRASE, create=True)
    ident = store.put("Synthetic site", PAGE, "synthetic-user", PASSWORD)
    return repo, thread, store, ident


def test_new_repo_secret_request_does_not_require_unlock(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    repo.mkdir()
    thread.mkdir()
    session = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                              thread_id="thread")
    tools = {tool.__name__: tool for tool in browser_tools(session)}
    assert json.loads(tools["browser_list_secrets"]()) == {
        "secrets": [], "store_status": "not_created"}

    from assist import git_sync
    from manage.web import state
    monkeypatch.setattr(git_sync, "read_state", lambda _directory: {"source": "synthetic"})
    monkeypatch.setattr(state, "DOMAINS", ["synthetic"])
    offered = json.loads(tools["browser_request_secret"]("Synthetic site", PAGE))
    assert offered["private_entry_requested"] is True
    assert offered["login_page"] == PAGE
    assert not (repo / FILE).exists()


def test_repo_ciphertext_and_independent_thread_unlock(tmp_path):
    repo, thread, store, ident = _store(tmp_path)
    assert PASSWORD.encode() not in (repo / FILE).read_bytes()
    assert PHRASE.encode() not in (repo / FILE).read_bytes()
    assert PASSWORD.encode() not in (thread / KEY_FILE).read_bytes()
    assert (thread / KEY_FILE).stat().st_mode & 0o077 == 0
    assert store.get(ident)["password"] == PASSWORD

    other = tmp_path / "other-thread"
    other.mkdir()
    independent = SecretStore(repo, other)
    assert independent.list_public() == store.list_public()
    with pytest.raises(SecretStoreError, match="Unlock this repository in this thread"):
        independent.get(ident)
    with pytest.raises(SecretStoreError, match="incorrect"):
        independent.unlock("wrong synthetic phrase")
    assert not (other / KEY_FILE).exists()
    independent.unlock(PHRASE)
    assert independent.get(ident)["password"] == PASSWORD


def test_agent_requested_private_form_binds_current_public_destination(tmp_path,
                                                                      monkeypatch):
    repo, thread, store, _ident = _store(tmp_path)
    session = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                              thread_id="thread")
    tools = {tool.__name__: tool for tool in browser_tools(session)}
    assert tools["browser_request_secret"]("Synthetic site", PAGE) == (
        "Private entry request unavailable: no configured repository.")
    from assist import git_sync
    from manage.web import state
    monkeypatch.setattr(git_sync, "read_state", lambda _directory: {"source": "synthetic"})
    monkeypatch.setattr(state, "DOMAINS", ["synthetic"])
    offered = json.loads(tools["browser_request_secret"](
        "Synthetic site", PAGE + "?temporary=public"))
    assert offered.pop("private_entry_requested") is True
    first = offered
    assert first["login_page"] == PAGE
    assert json.loads((thread / REQUEST_FILE).read_text()) == first
    assert not (repo / REQUEST_FILE).exists()
    second = store.request_input("Replacement site", "https://other.example/login")
    assert tools["browser_request_secret"]("site\nother", PAGE) == (
        "Private entry request unavailable.")
    assert tools["browser_request_secret"](
        "Spaced URL", "https://example.test/sign in") == (
            "Private entry request unavailable.")
    assert tools["browser_request_secret"](
        "Long URL", "https://example.test/" + "é" * 2000) == (
            "Private entry request unavailable.")
    assert store.pending_request() == second
    with pytest.raises(SecretStoreError, match="changed"):
        with store.selected_request(first["id"], first["label"], PAGE):
            raise AssertionError("stale form must not write")
    with pytest.raises(SecretStoreError, match="changed"):
        with store.selected_request(second["id"], second["label"], PAGE):
            raise AssertionError("changed destination must not write")
    assert store.pending_request() == second
    with store.selected_request(second["id"], second["label"],
                                second["login_page"]):
        assert store.pending_request() == second
    assert store.pending_request() is None


def test_private_submission_does_not_hold_other_threads_request_lock(tmp_path):
    first_dir = tmp_path / "first"
    other_dir = tmp_path / "other"
    first_dir.mkdir()
    other_dir.mkdir()
    first = SecretStore(tmp_path, first_dir)
    other = SecretStore(tmp_path, other_dir)
    request = first.request_input("First", PAGE)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with first.selected_request(request["id"], "First", PAGE):
            result = pool.submit(other.request_input, "Other", PAGE).result(timeout=1)
            assert result["label"] == "Other"


def test_browser_opaque_fill_and_normal_observations_hide_known_password(tmp_path):
    repo, thread, store, ident = _store(tmp_path)
    calls = []

    def command(operation, **kwargs):
        calls.append((operation, kwargs))
        if operation == "fill_secret":
            return {"result": {"filled": True}}
        encoded = quote(PASSWORD, safe="").replace("%2F", "%2f").replace("%3F", "%3f")
        form_encoded = quote_plus(PASSWORD, safe="").replace("%2F", "%2f")
        return {"result": {
            "url": PAGE + "?echo=" + encoded + "&user=synthetic-user",
            "_source_url": PAGE + "?echo=" + form_encoded,
            "snapshot": "Password: " + PASSWORD + "; user: synthetic-user",
            "targets": [{"ref": "password-ref", "name": PASSWORD},
                        {"ref": "username-ref", "name": "synthetic-user"}],
            "network_errors": [{"reason": PASSWORD}],
            "downloads": [{"name": PASSWORD + ".txt"}],
            "pages": [{"url": PAGE + "?echo=" + form_encoded}],
        }}

    session = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                              thread_id="thread", command=command)
    tools = {tool.__name__: tool for tool in browser_tools(session)}
    listing = json.loads(tools["browser_list_secrets"]())
    assert listing == {"secrets": store.list_public(), "unlocked": True}
    result = tools["browser_fill_secret"]("page", "snap", ident,
                                          "password-ref", "username-ref")
    assert result == '{"result": {"filled": true}}'
    assert calls[-1][0] == "fill_secret"
    assert calls[-1][1]["password"] == PASSWORD
    observed = tools["browser_observe"]("page")
    assert PASSWORD not in observed
    assert "synthetic-user" not in observed
    assert quote(PASSWORD, safe="") not in observed
    assert quote_plus(PASSWORD, safe="") not in observed
    assert "%2f" not in observed
    assert "[private value hidden]" in observed
    assert "source_id" not in observed
    next_turn = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                                thread_id="thread", command=command)
    restored = {tool.__name__: tool for tool in browser_tools(next_turn)}
    assert PASSWORD not in restored["browser_open"](PAGE)


def test_overlapping_credentials_and_browser_encoding_keep_handles(tmp_path):
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    repo.mkdir()
    thread.mkdir()
    store = SecretStore(repo, thread)
    store.unlock(PHRASE, create=True)
    store.put("Overlap", PAGE, "synthetic", "synthetic*a b")
    store.put("Other", PAGE, "ed", "synthetic*a b-extra")
    form_value = "synthetic*a+b"
    component_value = "synthetic*a%20b"
    session = SimpleNamespace(
        work_dir=str(repo), threads_root=str(tmp_path), thread_id="thread",
        command=lambda *_args, **_kwargs: {"result": {
            "page_id": "generation-ed-123", "snapshot_id": "ed1234567890",
            "targets": [{"ref": "ed12345678", "name": "synthetic*a b-extra"}],
            "snapshot": " / ".join(("synthetic*a b-extra", form_value,
                                      component_value)),
            "url": PAGE + "?value=" + form_value,
            "downloads": [{"download_id": "ed123", "name": "synthetic*a b.txt"}],
        }})
    tool = {item.__name__: item for item in browser_tools(session)}["browser_observe"]
    result = json.loads(tool("generation-ed-123"))["result"]
    assert result["page_id"] == "generation-ed-123"
    assert result["snapshot_id"] == "ed1234567890"
    assert result["targets"][0]["ref"] == "ed12345678"
    assert result["downloads"][0]["download_id"] == "ed123"
    assert "synthetic*a b-extra" not in json.dumps(result)
    assert form_value not in json.dumps(result)
    assert component_value not in json.dumps(result)
    assert "-extra" not in result["snapshot"]


def test_browser_url_serialization_hides_hybrid_component_encoding(tmp_path):
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    repo.mkdir()
    thread.mkdir()
    password = "synthetic!*'()~ /river8642"
    serialized = quote(password, safe="!*()")
    assert "%27" in serialized and "!" in serialized and "*" in serialized
    store = SecretStore(repo, thread)
    store.unlock(PHRASE, create=True)
    store.put("Serialized", PAGE, "synthetic-user", password)
    session = SimpleNamespace(
        work_dir=str(repo), threads_root=str(tmp_path), thread_id="thread",
        command=lambda *_args, **_kwargs: {"result": {
            "page_id": "opaque-page", "snapshot_id": "opaque-snapshot",
            "url": PAGE + "?echo=" + serialized,
            "_source_url": PAGE + "?echo=" + serialized,
            "snapshot": "Account overview; echoed " + serialized,
            "targets": [{"ref": "opaque-ref", "name": "Account overview"}],
        }})
    tool = {item.__name__: item for item in browser_tools(session)}["browser_observe"]
    observed = json.loads(tool("opaque-page"))["result"]
    assert "Account overview" in observed["snapshot"]
    assert serialized not in json.dumps(observed)
    assert "source_id" not in observed
    assert observed["targets"][0]["ref"] == "opaque-ref"


def test_hybrid_redaction_handles_percent_prefix_unicode_and_overlap():
    hidden = ("a%25", "a%2fzz", "café", "synthetic-extra", "synthetic", "hi")
    reflected = "raw a%25; encoded a%2525; a%2Fzz; caf%C3%A9; synthetic-extra; hi"
    scrubbed = _replace_hidden(reflected, hidden)
    assert scrubbed.count("[private value hidden]") == 6
    assert "25" not in scrubbed
    assert "2Fzz" not in scrubbed
    assert "-extra" not in scrubbed
    assert _contains_hidden("caf%c3%a9", hidden)
    assert _contains_hidden("a%2Fzz", hidden)
    assert not _contains_hidden("a%2424", hidden)
    assert _replace_hidden("%" * 8192 + "safe", hidden).endswith("safe")


def test_worker_host_observation_masks_before_display_limits(tmp_path, monkeypatch):
    repo, thread, store, _ident = _store(tmp_path)
    long_value = "synthetic-base-extra"
    native_value = "synthetic!*'()~ /river8642"
    store.put("Long", PAGE, "safe-user", long_value)
    store.put("Native", PAGE, "safe-user", native_value)
    encoded = quote(native_value, safe="!*()")
    url = PAGE + "?pad=" + "a" * 470 + encoded
    snapshot = "Visible account context; current synthetic-base " + "x" * 15965 + long_value

    class Link:
        def is_visible(self):
            return True

        def get_attribute(self, key):
            return {"aria-label": "L" * 119 + long_value,
                    "href": "/next?pad=" + "h" * 970 + long_value}.get(key)

        def evaluate(self, _expression):
            return "a"

        def inner_text(self):
            return ""

    class Page:
        def __init__(self):
            self.url = url

        def is_closed(self):
            return False

        def wait_for_load_state(self, **_kwargs):
            pass

        def locator(self, selector):
            if selector == "body":
                return SimpleNamespace(aria_snapshot=lambda **_kwargs: snapshot)
            if selector == 'input[type="password"]':
                return SimpleNamespace(evaluate_all=lambda _expression: ["synthetic-base"])
            return SimpleNamespace(count=lambda: 1, nth=lambda _index:
                                   SimpleNamespace(element_handle=lambda: Link()))

    worker = BrowserWorker()
    worker.pages["page"] = Page()
    other = Page()
    other.url = PAGE + "?pad=" + "b" * 215 + long_value
    worker.pages["other"] = other
    worker.downloads["download"] = SimpleNamespace(
        suggested_filename="D" * 179 + long_value)
    worker._error("E" * 254 + long_value, "document", "request_failed")
    monkeypatch.setenv("BROWSER_SESSION_TOKEN", "synthetic-token")

    def command(operation, **args):
        hidden = list(args.pop("_private_values"))
        return {"result": worker.handle({"session": "synthetic-token",
                                          "operation": operation,
                                          "args": args,
                                          "private_values": hidden})}

    session = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                              thread_id="thread", command=command)
    tool = {item.__name__: item for item in browser_tools(session)}["browser_observe"]
    observed = json.loads(tool("page"))["result"]
    visible = json.dumps(observed)
    assert "Visible account context" in visible
    assert "[password hidden]" in visible
    assert long_value not in visible and "-extra" not in visible
    assert "current synthetic-base" not in visible
    assert encoded not in visible and native_value not in visible
    assert "[password hidden]" in observed["url"]
    assert "[password hidden]" in observed["snapshot"]
    assert "[password hidden]" in observed["targets"][0]["name"]
    assert "[password hidden]" in observed["targets"][0]["href"]
    assert "[password hidden]" in observed["pages"][1]["url"]
    assert "[password hidden]" in observed["downloads"][0]["name"]
    assert "[password hidden]" in observed["network_errors"][0]["host"]
    assert observed["page_id"] == "page"
    assert observed["targets"][0]["ref"] in worker.targets["page"]
    assert worker.targets["page"][observed["targets"][0]["ref"]][2].endswith(long_value)
    assert "_source_url" not in observed and "source_id" not in observed


def test_private_browser_command_covers_maximum_valid_catalog_values():
    values = ["\0" * 320, "\0" * 1024] * 64
    public = {"session": "x" * 32, "operation": "act",
              "args": {"target": {"text": "x" * 2000},
                       "url": "https://example.test/" + "x" * 4000}}
    assert len(json.dumps(public, ensure_ascii=False).encode()) < 32768
    command = json.dumps({**public, "private_values": values},
                         ensure_ascii=False).encode()
    assert len(command) <= MAX_PRIVATE_COMMAND


def test_percent_triplet_case_is_not_registered_as_a_source(tmp_path):
    repo = tmp_path / "repo"
    thread = tmp_path / "thread"
    repo.mkdir()
    thread.mkdir()
    store = SecretStore(repo, thread)
    store.unlock(PHRASE, create=True)
    store.put("Case", PAGE, "synthetic-user", "a%2fzz")
    session = SimpleNamespace(
        work_dir=str(repo), threads_root=str(tmp_path), thread_id="thread",
        command=lambda *_args, **_kwargs: {"result": {
            "url": PAGE + "?echo=a%2Fzz",
            "_source_url": PAGE + "?echo=a%2Fzz",
            "snapshot": "Account overview; a%2Fzz",
        }})
    tool = {item.__name__: item for item in browser_tools(session)}["browser_observe"]
    result = json.loads(tool("page"))["result"]
    assert "Account overview" in result["snapshot"]
    assert "2Fzz" not in json.dumps(result)
    assert "source_id" not in result


def test_corrupt_or_unavailable_store_never_disables_existing_password_redaction(tmp_path):
    repo, thread, store, ident = _store(tmp_path)
    store.put("Other site", PAGE, "other-user", "other-password")
    catalog = json.loads((repo / FILE).read_text())
    catalog["entries"][1]["data"] = "AAAA"
    (repo / FILE).write_text(json.dumps(catalog))
    assert store.get(ident)["password"] == PASSWORD
    calls = []
    session = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                              thread_id="thread", command=lambda *_args, **_kwargs:
                              calls.append(True) or {"result": {"snapshot": PASSWORD}})
    tools = {tool.__name__: tool for tool in browser_tools(session)}
    assert "unavailable" in tools["browser_observe"]("page")
    assert "unavailable" in tools["browser_fill_secret"]("page", "snap", ident, "p")
    assert calls == []
    (repo / FILE).write_text(json.dumps({**catalog, "entries": catalog["entries"][:1]}))
    (thread / KEY_FILE).unlink()
    assert "unavailable" in tools["browser_observe"]("page")
    assert calls == []


def test_removed_record_or_catalog_cannot_erase_thread_redaction_memory(tmp_path):
    repo, thread, store, ident = _store(tmp_path)
    calls = []
    session = SimpleNamespace(work_dir=str(repo), threads_root=str(tmp_path),
                              thread_id="thread", command=lambda *_args, **_kwargs:
                              calls.append(True) or {"result": {"snapshot": PASSWORD}})
    observe = {tool.__name__: tool for tool in browser_tools(session)}["browser_observe"]
    assert PASSWORD not in observe("page")
    catalog = json.loads((repo / FILE).read_text())
    catalog["entries"] = []
    (repo / FILE).write_text(json.dumps(catalog))
    with pytest.raises(SecretStoreError, match="changed"):
        store.unlock(PHRASE)
    assert "unavailable" in observe("page")
    (repo / FILE).unlink()
    assert "unavailable" in observe("page")
    assert len(calls) == 1


@pytest.mark.parametrize("variant", ["raw", "hybrid", "percent_case"])
def test_download_rejects_known_password_before_workspace_write(
        tmp_path, monkeypatch, variant):
    repo, _thread, unused_store, _ident = _store(tmp_path)
    password = ("synthetic!*'()~ /river8642" if variant == "hybrid"
                else "a%2fzz" if variant == "percent_case" else PASSWORD)
    content = (quote(password, safe="!*()") if variant == "hybrid"
               else "a%2Fzz" if variant == "percent_case" else password).encode()
    session = BrowserSession("thread", "run", str(tmp_path), str(repo), None)
    session.identity = SimpleNamespace(generation="synthetic")
    session.command = lambda *_args, **_kwargs: {"result": {
        "path": "/run/assist-browser/downloads/synthetic", "name": "safe.txt",
        "size": len(content)}}
    monkeypatch.setattr("assist.browser.manager._docker_download",
                        lambda *_args, **_kwargs: content)
    with pytest.raises(BrowserUnavailable, match="private value"):
        session._save_download_locked("download", None, hidden=(password,))
    assert not (repo / "downloads").exists()


def test_successful_download_receipt_keeps_trusted_workspace_path(tmp_path):
    repo, _thread, store, _ident = _store(tmp_path)
    store.put("Short", PAGE, "work", "other-synthetic-password")
    session = SimpleNamespace(
        work_dir=str(repo), threads_root=str(tmp_path), thread_id="thread",
        save_download=lambda *_args: {"path": "/workspace/downloads/report.txt",
                                      "bytes": 12})
    tool = {item.__name__: item for item in browser_tools(session)}[
        "browser_save_download"]
    assert json.loads(tool("download")) == {
        "path": "/workspace/downloads/report.txt", "bytes": 12}
