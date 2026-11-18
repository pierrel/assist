"""Private thread unlock, opaque browser use, and observation redaction."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.parse import quote, quote_plus

import pytest

from assist.browser.manager import BrowserSession, BrowserUnavailable, browser_tools
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


def test_download_rejects_known_password_before_workspace_write(tmp_path, monkeypatch):
    repo, _thread, unused_store, _ident = _store(tmp_path)
    session = BrowserSession("thread", "run", str(tmp_path), str(repo), None)
    session.identity = SimpleNamespace(generation="synthetic")
    session.command = lambda *_args, **_kwargs: {"result": {
        "path": "/run/assist-browser/downloads/synthetic", "name": "safe.txt",
        "size": len(PASSWORD)}}
    monkeypatch.setattr("assist.browser.manager._docker_download",
                        lambda *_args, **_kwargs: PASSWORD.encode())
    with pytest.raises(BrowserUnavailable, match="private value"):
        session._save_download_locked("download", None, hidden=(PASSWORD,))
    assert not (repo / "downloads").exists()
