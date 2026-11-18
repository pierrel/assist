"""Native egress uses the existing cards, decisions, grants and Run queue."""
from pathlib import Path
import importlib.util
import json
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from assist.egress.client_map import ClientRecord, record_client
from assist.egress.store import EgressStore, APPROVALS_SUBDIR
from assist.egress.tools import egress_tools
from assist.hermes_provider_relay import HermesProviderRelay, HermesRelayError
from assist.hermes_runtime import HermesRuntimeManager
from assist.pi_preview import PiPreviewPolicy
from assist.thread_queue import ThreadAffinityQueue
from manage.web import egress, phone_api, state, threads


@pytest.fixture
def native_egress(tmp_path, monkeypatch):
    root = tmp_path / "threads"
    monkeypatch.setattr(threads.MANAGER, "root_dir", str(root))
    monkeypatch.setattr(threads, "DOMAINS", [])
    monkeypatch.setattr(state, "DOMAINS", [])
    store = EgressStore(str(tmp_path / "egress"))
    for module in (state, threads, egress):
        monkeypatch.setattr(module, "EGRESS_STORE", store)
    policy = PiPreviewPolicy(root, engine="hermes")
    policy.set_enabled(True)
    for module in (state, threads):
        monkeypatch.setattr(module, "HERMES_PREVIEW", policy)
    tid = threads.MANAGER.reserve_visible("hermes")
    threads._create_empty_pi_workspace(tid)
    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit",
                        lambda run_id, thread_id, **_kwargs: submitted.append((run_id, thread_id)))
    monkeypatch.setattr(threads.MANAGER, "get", lambda *_a, **_k:
                        pytest.fail("Hermes must not construct a Deep graph"))
    state.DESCRIPTION_CACHE.clear()
    yield store, tid, submitted
    threads._RUN_SERVICES_BY_ROOT.pop(str(root), None)
    state.DESCRIPTION_CACHE.clear()


@pytest.mark.parametrize("decision", ["hour", "decline"])
def test_native_request_card_decision_and_scoped_proxy_grant(
        native_egress, tmp_path, monkeypatch, decision):
    store, tid, submitted = native_egress
    projection = tmp_path / "egress" / APPROVALS_SUBDIR
    monkeypatch.setenv("APPROVALS_DIR", str(projection))
    monkeypatch.setenv("CLIENT_MAP_DIR", str(projection))
    monkeypatch.setenv("EGRESS_ALLOWLIST", "host.docker.internal")
    monkeypatch.setenv("EGRESS_SANDBOX_CIDR", "172.20.0.0/16")
    spec = importlib.util.spec_from_file_location(
        "hermes_egress_proxy_test", Path(__file__).parents[1] / "dockerfiles/egress-proxy.py")
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    # Public DNS vet is existing proxy coverage; this check exercises attribution
    # and the actual store projection without touching DNS or an external host.
    monkeypatch.setattr(proxy, "vet_resolved", lambda *_a: "93.184.216.34")
    record_client(str(projection), "172.20.0.9", ClientRecord(tid, "native-a", "hermes"))
    record_client(str(projection), "172.20.0.10", ClientRecord("other", "native-b", "hermes"))
    host, port = "releases.example.test", 443
    assert proxy.target_policy(host, port, "172.20.0.9")[1] == "host_not_approved"
    queue = ThreadAffinityQueue()
    control = tmp_path / "control"
    control.mkdir()
    with queue.acquire(tid) as handle:
        relay = HermesProviderRelay(control, "http://127.0.0.1:1/v1", "fixture", "fixture", 8192,
            lambda: queue.current_handle() is handle and not handle.expired,
            egress_tools=egress_tools(store, frozenset(), thread_dir=threads.MANAGER.thread_dir,
                                      thread_id=tid))
        relay.start()
        try:
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(relay.socket_path)),
                              timeout=3, trust_env=False) as client:
                payload = {"name": "request_egress", "arguments": {
                    "host": host, "port": port, "task": "Read the release announcement"}}
                response = client.post("http://host/egress", json=payload)
                assert response.status_code == 200 and "awaits" in response.json()["result"]
                assert "already awaiting" in client.post("http://host/egress", json=payload).json()["result"]
                assert len(store.for_thread(tid)) == 1
                assert proxy.target_policy(host, port, "172.20.0.9")[1] == "host_not_approved"
                page = TestClient(threads.app).get(f"/thread/{tid}")
                assert page.status_code == 200
                assert host in page.text and 'action="/egress/hour"' in page.text
                proposal, = phone_api._approval_collection(tid)["proposals"]
                assert proposal["kind"] == "egress" and proposal["state"] == "pending"
                token = proposal["token"]
                form = {"token": egress.EGRESS_CSRF, "tid": tid, "host": host,
                        "port": str(port), "review": token}
                assert TestClient(threads.app).post(f"/egress/{decision}", data=form,
                                                   follow_redirects=False).status_code == 303
                assert len(submitted) == 1
                followup = threads._runs().get(tid, submitted[0][0])
                assert followup.origin == "egress" and not followup.resume
                assert ("APPROVED" if decision == "hour" else "DECLINED") in followup.text
                assert not threads._hermes_messages(tid), "machine decision must not become a user message"
                result = client.post("http://host/egress", json=payload).json()["result"]
                assert ("already approved" if decision == "hour" else "DECLINED") in result
                assert TestClient(threads.app).post(f"/egress/{decision}", data=form,
                                                   follow_redirects=False).status_code == 303
                assert len(submitted) == 1
                if decision == "hour":
                    assert proxy.target_policy(host, port, "172.20.0.9") == ("93.184.216.34", None)
                else:
                    assert proxy.target_policy(host, port, "172.20.0.9")[1] == "host_not_approved"
                assert proxy.target_policy(host, 80, "172.20.0.9")[1] == "host_not_approved"
                assert proxy.target_policy(host, port, "172.20.0.10")[1] == "host_not_approved"
                assert proxy.target_policy("host.docker.internal", 8000, "172.20.0.9")[1] == "host_requires_ordinary_attribution"
        finally:
            relay.close()


def test_native_egress_denies_spoofed_identity_and_parked_or_stale_admission(native_egress, tmp_path):
    store, tid, _submitted = native_egress
    control = tmp_path / "control"
    control.mkdir()
    admitted = True
    relay = HermesProviderRelay(control, "http://127.0.0.1:1/v1", "fixture", "fixture", 8192,
                                lambda: admitted,
                                egress_tools=egress_tools(store, frozenset(), thread_id=tid))
    relay.start()
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(relay.socket_path)),
                          timeout=3, trust_env=False) as client:
            payload = {"name": "request_egress", "arguments": {
                "host": "other.example.test", "port": 443, "task": "Read the announcement"}}
            spoofed = json.loads(json.dumps(payload))
            spoofed["arguments"]["thread_id"] = "other"
            assert client.post("http://host/egress", json=spoofed).status_code == 409
            assert relay.request_pause()
            assert client.post("http://host/egress", json=payload).status_code == 409
            relay.confirm_parked()
            assert client.post("http://host/egress", json=payload).status_code == 409
            relay.resume()
            admitted = False
            assert client.post("http://host/egress", json=payload).status_code == 409
            assert not store.for_thread(tid) and not store.for_thread("other")
    finally:
        relay.close()


def test_native_egress_reports_internal_model_route_unavailable(native_egress):
    store, tid, _ = native_egress
    request, hosts, _remove, batch = egress_tools(
        store, frozenset({"host.docker.internal", "pypi.org"}), thread_id=tid)
    assert "host.docker.internal" not in hosts()
    assert "pypi.org" in hosts()
    assert "approval is unavailable" in request("http://host.docker.internal:8000", 8000, "Read")
    assert "approval is unavailable" in batch([{"host": "host.docker.internal", "port": 8000}], "Read")
    assert not store.for_thread(tid)


def test_native_egress_invalid_marker_is_conflict(native_egress):
    _store, tid, _ = native_egress
    (Path(threads.MANAGER.thread_dir(tid)) / "engine.json").write_text('{"engine":"invalid"}')
    form = {"token": egress.EGRESS_CSRF, "tid": tid, "host": "example.test", "port": "443"}
    assert TestClient(threads.app).post("/egress/hour", data=form).status_code == 409
    with pytest.raises(HTTPException) as caught:
        egress.approval_decision(tid, "a" * 64, "hour")
    assert caught.value.status_code == 409


def test_native_decision_waits_for_running_turn_and_survives_its_cancel(native_egress):
    store, tid, submitted = native_egress
    original = threads._create_run(tid, "Read the announcement")
    threads._runs().claim(tid, original.id)
    request = egress_tools(store, frozenset(), thread_id=tid)[0]
    request("releases.example.test", 443, "Read the announcement")
    proposal, = phone_api._approval_collection(tid)["proposals"]
    egress.approval_decision(tid, proposal["token"], "hour")
    threads._dispatch_egress_resolution(tid)
    assert submitted == []
    followup, = [run for run in threads._runs().list(tid) if run.origin == "egress"]
    assert followup.status == "pending"
    threads._runs().transition(tid, original.id, "cancelled")
    threads._dispatch_pending_after(tid, original.id)
    assert submitted == [(followup.id, tid)]
    threads._dispatch_egress_resolution(tid)
    assert len([run for run in threads._runs().list(tid) if run.origin == "egress"]) == 1


def test_host_egress_operation_drains_without_holding_admission_lock(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    effects = []
    def request_egress():
        entered.set()
        assert finish.wait(2)
        effects.append("pending card only")
        return "pending"
    relay = HermesProviderRelay(tmp_path, "http://127.0.0.1:1/v1", "fixture", "fixture", 8192,
                                lambda: True, egress_tools=[request_egress])
    worker = threading.Thread(target=lambda: relay.egress({"name": "request_egress", "arguments": {}}))
    worker.start()
    try:
        assert entered.wait(1)
        assert not relay.request_pause(), "host mutation must drain even without native middleware"
        relay.stop_admission()
        assert relay.stopped and worker.is_alive(), "store I/O must not prevent admission revocation"
    finally:
        finish.set()
        worker.join(2)
        relay.close()
    assert effects == ["pending card only"] and not relay.operations


def test_deletion_drains_accepted_egress_before_evicting_records(native_egress, tmp_path, monkeypatch):
    store, tid, _ = native_egress
    entered, finish, deleted = threading.Event(), threading.Event(), threading.Event()
    failures = []
    original_add = store.add_pending
    def delayed_add(*args, **kwargs):
        entered.set()
        assert finish.wait(3)
        return original_add(*args, **kwargs)
    monkeypatch.setattr(store, "add_pending", delayed_add)
    control = tmp_path / "retire-control"
    control.mkdir()
    relay = HermesProviderRelay(control, "http://127.0.0.1:1/v1", "fixture", "fixture", 8192,
                                lambda: True, egress_tools=egress_tools(store, frozenset(), thread_id=tid))
    # A stopped worker cannot send a native settled hook. Only accepted host
    # mutations may delay verified retirement, not this abandoned native entry.
    relay.boundary({"operation": "stopped-native-tool", "kind": "tool"})
    manager = HermesRuntimeManager()
    manager.turns[tid] = SimpleNamespace(relay=relay, backend=None, control_dir=control)
    monkeypatch.setattr(threads, "_HERMES_RUNTIME", manager)
    monkeypatch.setattr(threads.BrowserManager, "bounded_thread_gate", lambda _tid: nullcontext())
    monkeypatch.setattr(threads.BrowserManager, "cleanup", lambda _tid: None)
    monkeypatch.setattr(threads.browser_authority, "generation_fence", lambda *_a: nullcontext())
    def evict(deleted_tid):
        threads._evict_egress(deleted_tid)
        deleted.set()
    monkeypatch.setattr(threads, "_delete_thread_after_browser_stop", evict)
    def request():
        try:
            relay.egress({"name": "request_egress", "arguments": {
                "host": "releases.example.test", "port": 443, "task": "Read release"}})
        except BaseException as error:
            failures.append(error)
    def delete():
        try:
            threads._delete_thread_and_children(tid)
        except BaseException as error:
            failures.append(error)
    requester, deleter = threading.Thread(target=request), threading.Thread(target=delete)
    requester.start()
    try:
        assert entered.wait(1)
        deleter.start()
        deleted_while_busy = deleted.wait(.1)
    finally:
        finish.set()
        requester.join(3)
        if deleter.ident is not None:
            deleter.join(3)
        relay.close()
    assert not requester.is_alive() and not deleter.is_alive() and not failures
    records = store.for_thread(tid)
    assert not deleted_while_busy and not records, (deleted_while_busy, records)
    assert deleted.is_set() and tid not in manager.turns


def test_egress_drain_timeout_retains_cleanup_but_releases_model_slot(native_egress, tmp_path, monkeypatch):
    store, tid, _ = native_egress
    entered, finish = threading.Event(), threading.Event()
    original_add = store.add_pending
    failures = []
    def delayed_add(*args, **kwargs):
        entered.set()
        assert finish.wait(12)
        return original_add(*args, **kwargs)
    monkeypatch.setattr(store, "add_pending", delayed_add)
    control = tmp_path / "timeout-control"
    control.mkdir()
    queue = ThreadAffinityQueue()
    relay = HermesProviderRelay(control, "http://127.0.0.1:1/v1", "fixture", "fixture", 8192,
                                lambda: True, egress_tools=egress_tools(store, frozenset(), thread_id=tid))
    manager = HermesRuntimeManager()
    turn = SimpleNamespace(work_id="original", relay=relay, backend=None, control_dir=control)
    manager.turns[tid] = turn
    def request():
        try:
            relay.egress({"name": "request_egress", "arguments": {
                "host": "releases.example.test", "port": 443, "task": "Read release"}})
        except BaseException as error:
            failures.append(error)
    requester = threading.Thread(target=request)
    try:
        with pytest.raises(HermesRelayError, match="egress drain is unconfirmed"):
            with queue.acquire(tid, before_release=lambda: manager.stop(tid)) as old:
                requester.start()
                assert entered.wait(1)
        assert manager.turns[tid] is turn and queue.current_handle() is old
        assert manager.revoke_model(tid, "original")
        assert manager.turns[tid] is turn and requester.is_alive()
        assert queue._release_if_holder(old)
        with queue.acquire("competitor", wait_timeout_s=.1) as competitor:
            assert queue.current_handle() is competitor
            finish.set()
            requester.join(2)
            assert not requester.is_alive() and not failures
            manager.retire(tid)
            store.remove_thread(tid)
            assert not store.for_thread(tid) and tid not in manager.turns
            assert not queue._release_if_holder(old)
            assert queue.current_handle() is competitor
    finally:
        finish.set()
        if requester.ident is not None:
            requester.join(2)
        relay.close()
