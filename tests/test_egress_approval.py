"""Egress approval HITL — store state machine, projection, tools, proxy
functions, routes, render (docs/2026-07-21-egress-approval-hitl.org). The
live-proxy half (real containers) lives in dockerfiles/test-sandbox-egress.sh.
"""
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from assist.egress.store import (APPROVALS_SUBDIR, EgressRequest, EgressStore,
                                 PROJECTION_FILE, REVOKED_ONLY, request_key)
from assist.egress import tools as egress_tools_mod
from assist.egress.tools import egress_tools, _parse_host_port
from assist.egress.client_map import record_client, forget_client


def _store(tmp_path):
    return EgressStore(str(tmp_path))


def _projection(tmp_path) -> dict:
    with open(tmp_path / APPROVALS_SUBDIR / PROJECTION_FILE) as f:
        return json.load(f)


def _req(tid="t1", host="api.example.com", port=443, task="fetch the releases"):
    return EgressRequest(host=host, port=port, task=task, origin_tid=tid,
                         created_at=datetime.now(timezone.utc).isoformat())


# --- store state machine + projection ----------------------------------------

def test_pending_approve_hour_projects_and_expires(tmp_path):
    st = _store(tmp_path)
    st.add_pending(_req())
    assert _projection(tmp_path) == {}          # pending projects nothing
    rec = st.resolve(request_key("t1", "api.example.com", 443), "hour")
    assert rec.state == "approved"
    proj = _projection(tmp_path)
    (entry,) = proj.values()
    assert entry["host"] == "api.example.com" and entry["port"] == 443
    assert entry["origin_tid"] == "t1"
    exp = datetime.fromisoformat(entry["expires_at"])
    assert exp.tzinfo is not None and exp > datetime.now(timezone.utc)


def test_always_grant_and_revoke(tmp_path):
    st = _store(tmp_path)
    st.add_pending(_req())
    st.resolve(request_key("t1", "api.example.com", 443), "always")
    (entry,) = _projection(tmp_path).values()
    assert entry["expires_at"] == REVOKED_ONLY
    assert st.revoke(request_key("t1", "api.example.com", 443))
    assert _projection(tmp_path) == {}


def test_decline_is_persistent_and_projects_nothing(tmp_path):
    st = _store(tmp_path)
    st.add_pending(_req())
    rec = st.resolve(request_key("t1", "api.example.com", 443), "decline")
    assert rec.state == "declined"
    assert _projection(tmp_path) == {}
    # double-resolve (double-click) is a no-op
    assert st.resolve(request_key("t1", "api.example.com", 443), "hour") is None


def test_expired_grant_pruned_on_mutation(tmp_path):
    st = _store(tmp_path)
    st.add_pending(_req())
    st.resolve(request_key("t1", "api.example.com", 443), "hour")
    # force-expire by rewriting the record
    recs = st._load()
    k = request_key("t1", "api.example.com", 443)
    from dataclasses import replace
    recs[k] = replace(recs[k], expires_at=(
        datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat())
    with st._lock:
        st._mutate(recs)
    assert _projection(tmp_path) == {}
    assert st.live_grants() == []


def test_remove_thread_scopes_cleanup(tmp_path):
    st = _store(tmp_path)
    st.add_pending(_req(tid="t1"))
    st.add_pending(_req(tid="t2", host="other.example.com"))
    st.resolve(request_key("t2", "other.example.com", 443), "always")
    assert st.remove_thread("t2") == 1
    assert _projection(tmp_path) == {}
    assert [r.origin_tid for r in st.all()] == ["t1"]


def test_corrupt_files_degrade_not_raise(tmp_path):
    st = _store(tmp_path)
    (tmp_path / "egress-requests.json").write_text("{corrupt")
    assert st.all() == [] and st.peek() == []
    assert st.resolve("t1:x:443", "hour") is None


# --- tool validation + correctives -------------------------------------------

@pytest.mark.parametrize("host,port,want", [
    ("https://api.github.com/repos/x/releases", None, ("api.github.com", 443)),
    ("http://legacy.example.com/data", None, ("legacy.example.com", 80)),
    ("http://legacy.example.com/data", 0, ("legacy.example.com", 80)),
    ("api.github.com:8443", None, ("api.github.com", 8443)),
    ("API.Example.COM.", "443", ("api.example.com", 443)),
    ("xn--bcher-kva.example", 443, ("xn--bcher-kva.example", 443)),
])
def test_parse_host_port_tolerant(host, port, want):
    h, p, err = _parse_host_port(host, port)
    assert err is None and (h, p) == want


@pytest.mark.parametrize("host,port", [
    ("127.0.0.1", 443), ("0x7f.0x0.0x0.0x1", 443), ("2130706433.", 443),
    ("host.docker.internal", 443), ("svc.internal", 443), ("10.0.0.1", 80),
    ("nodots", 443), ("api.example.com", 99999), ("api.example.com", "abc"),
    ("", 443), ("a..b", 443), ("-bad.example.com", 443),
    ("bad-.example.com", 443), ("x" * 64 + ".example.com", 443),
])
def test_parse_host_port_rejects(host, port):
    h, p, err = _parse_host_port(host, port)
    assert err is not None


def test_request_flow_correctives(tmp_path, monkeypatch):
    st = _store(tmp_path)
    monkeypatch.setattr(egress_tools_mod, "_thread_id", lambda: "t1")
    request, list_hosts, remove = egress_tools(st, frozenset({"pypi.org"}))
    # events flow through the thread_dir seam when wired; unwired here — the
    # tools must still work (CLI/eval shape)
    assert "already on the base allowlist" in request("pypi.org", 443, "x")
    out = request("https://api.github.com/x", None, "fetch releases and summarize")
    assert "awaits the user's approval" in out
    assert "already awaiting" in request("api.github.com", 443, "again")
    # cap: 2 more pending fills the 3-cap
    request("b.example.com", 443, "t")
    request("c.example.com", 443, "t")
    assert "awaiting approval" in request("d.example.com", 443, "t")
    # decline → corrective, and no re-card
    st.resolve(request_key("t1", "api.github.com", 443), "decline")
    assert "DECLINED" in request("api.github.com", 443, "retry")
    # approve → "just retry"
    st.resolve(request_key("t1", "b.example.com", 443), "hour")
    assert "already approved" in request("b.example.com", 443, "t")
    # list shows base + grant with lifetime; remove drops the grant
    listing = list_hosts()
    assert "pypi.org (any port; base allowlist" in listing
    assert "b.example.com:443" in listing and "min left" in listing
    assert "Removed" in remove("b.example.com", 443)
    assert "no grant" in remove("b.example.com", 443)
    assert "operator-managed" in remove("pypi.org", 443)


def test_tools_without_thread_are_inert(tmp_path, monkeypatch):
    st = _store(tmp_path)
    monkeypatch.setattr(egress_tools_mod, "_thread_id", lambda: None)
    request, list_hosts, remove = egress_tools(st, frozenset())
    assert "no active thread" in request("a.example.com", 443, "t").lower()
    assert st.all() == []


# --- client map ---------------------------------------------------------------

def test_client_map_roundtrip(tmp_path):
    d = str(tmp_path)
    record_client(d, "172.20.0.5", "t1")
    record_client(d, "172.20.0.5", "t2")   # IP reuse: newest wins
    path = tmp_path / APPROVALS_SUBDIR / "client-map.json"
    assert json.loads(path.read_text()) == {"172.20.0.5": "t2"}
    forget_client(d, "172.20.0.5")
    assert json.loads(path.read_text()) == {}


# --- proxy pure functions (imported by path) ----------------------------------

@pytest.fixture
def proxy_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("EGRESS_ALLOWLIST", "pypi.org,host.docker.internal")
    monkeypatch.setenv("APPROVALS_DIR", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "egress_proxy_under_test",
        os.path.join(os.path.dirname(__file__), "..", "dockerfiles",
                     "egress-proxy.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_approval(tmp_path, host="api.example.com", port=443, tid="t1",
                    expires=None):
    exp = expires if expires is not None else (
        datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    (tmp_path / "approved-hosts.json").write_text(json.dumps(
        {f"{tid}:{host}:{port}": {"host": host, "port": port,
                                  "origin_tid": tid, "expires_at": exp}}))
    (tmp_path / "client-map.json").write_text(json.dumps({"172.20.0.9": tid}))


def test_proxy_approved_target_matrix(proxy_mod, tmp_path):
    _write_approval(tmp_path)
    ok = proxy_mod.approved_target
    assert ok("api.example.com", 443, "172.20.0.9")
    assert not ok("api.example.com", 22, "172.20.0.9")       # port-scoped
    assert not ok("api.example.com", 443, "172.20.0.99")     # unknown client IP
    assert not ok("evil.example.com", 443, "172.20.0.9")
    # wrong thread: same host approved for another tid
    (tmp_path / "client-map.json").write_text(json.dumps({"172.20.0.9": "t2"}))
    assert not ok("api.example.com", 443, "172.20.0.9")


def test_proxy_duration_fail_closed(proxy_mod, tmp_path):
    live = proxy_mod._grant_live
    assert live(proxy_mod.REVOKED_ONLY)
    assert live((datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat())
    assert not live((datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat())
    assert not live(None)
    assert not live("")                                     # missing ⇒ deny
    assert not live("2030-01-01T00:00:00")                  # naive ⇒ deny
    assert not live("garbage")


def test_proxy_files_fail_closed(proxy_mod, tmp_path):
    ok = proxy_mod.approved_target
    assert not ok("api.example.com", 443, "172.20.0.9")     # no files at all
    (tmp_path / "approved-hosts.json").write_text("{corrupt")
    (tmp_path / "client-map.json").write_text("[1,2]")      # wrong shape
    assert not ok("api.example.com", 443, "172.20.0.9")
    big = json.dumps({"k" + str(i): {} for i in range(9000)})
    (tmp_path / "approved-hosts.json").write_text(big)      # oversized
    _write_approval(tmp_path)  # rewrites both files validly
    (tmp_path / "approved-hosts.json").write_text(big)
    assert not ok("api.example.com", 443, "172.20.0.9")


def test_proxy_vet_resolved(proxy_mod, monkeypatch):
    def fake_gai(host, port, proto=0):
        addr = {"private.example.com": "10.1.2.3",
                "meta.example.com": "169.254.169.254",
                "loop.example.com": "127.0.0.1",
                "good.example.com": "93.184.216.34"}[host]
        return [(2, 1, 6, "", (addr, port))]
    monkeypatch.setattr(proxy_mod.socket, "getaddrinfo", fake_gai)
    assert proxy_mod.vet_resolved("good.example.com", 443) == "93.184.216.34"
    assert proxy_mod.vet_resolved("private.example.com", 443) is None
    assert proxy_mod.vet_resolved("meta.example.com", 443) is None
    assert proxy_mod.vet_resolved("loop.example.com", 443) is None


def test_base_allowlist_unaffected_by_approvals(proxy_mod, tmp_path):
    """The approvals files can only ADD narrowly — a base decision never
    consults them (allowlist membership short-circuits)."""
    assert "pypi.org" in proxy_mod.ALLOWLIST
    # corrupt approvals in place: base host membership is a plain set check
    (tmp_path / "approved-hosts.json").write_text("{corrupt")
    assert "pypi.org" in proxy_mod.ALLOWLIST


# --- denial-signature matcher -------------------------------------------------

@pytest.mark.parametrize("output,hit", [
    ("curl: (56) CONNECT tunnel failed, response 403", True),
    ("curl: (56) Received HTTP code 403 from proxy after CONNECT", True),
    ("Tunnel connection failed: 403 Forbidden", True),
    ("Proxy tunneling failed: Forbidden", True),
    ("The requested URL returned error: 403", False),   # origin's own 403
    ("HTTP error 403 while getting url", False),
    ("connection refused", False),
])
def test_denial_signature(output, hit):
    from assist.sandbox import _looks_like_egress_denial
    assert _looks_like_egress_denial(output) is hit


# --- routes + render ----------------------------------------------------------

@pytest.fixture
def wired_web(tmp_path, monkeypatch):
    from manage import web
    from manage.web import egress as egress_routes
    from manage.web import threads as threads_mod
    st = EgressStore(str(tmp_path / "egress"))
    tid = "t-egress"
    (tmp_path / tid).mkdir()
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda t: str(tmp_path / t))
    monkeypatch.setattr(egress_routes, "EGRESS_STORE", st)
    monkeypatch.setattr(threads_mod, "EGRESS_STORE", st)
    dispatched = []
    monkeypatch.setattr(threads_mod, "_scheduled_dispatch",
                        lambda t, prompt, tz: dispatched.append((t, prompt)))
    monkeypatch.setattr(threads_mod, "_mark_urgent", lambda t: None)
    return st, tid, dispatched


def test_routes_404_when_unconfigured():
    from fastapi.testclient import TestClient
    from manage import web
    from manage.web import egress as egress_routes
    if egress_routes.EGRESS_STORE is not None:
        pytest.skip("egress configured in this environment")
    assert TestClient(web.app).get("/egress").status_code == 404


def test_approve_batch_dispatches_once(wired_web, monkeypatch):
    from fastapi.testclient import TestClient
    from manage import web
    from manage.web import egress as egress_routes
    st, tid, dispatched = wired_web
    st.add_pending(_req(tid=tid, host="a.example.com", task="task A"))
    st.add_pending(_req(tid=tid, host="b.example.com", task="task B"))
    c = TestClient(web.app)
    tok = egress_routes.EGRESS_CSRF
    r = c.post("/egress/hour", data={"token": tok, "tid": tid,
                                     "host": "a.example.com", "port": "443",
                                     "redirect": f"/thread/{tid}"},
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/thread/{tid}"
    assert dispatched == []                     # b still pending — no dispatch
    c.post("/egress/decline", data={"token": tok, "tid": tid,
                                    "host": "b.example.com", "port": "443"},
           follow_redirects=False)
    assert len(dispatched) == 1                 # pending set emptied → ONE turn
    _, prompt = dispatched[0]
    assert "a.example.com:443 APPROVED" in prompt and "task A" in prompt
    assert "b.example.com:443 DECLINED" in prompt


def test_approve_requires_csrf_and_live_thread(wired_web):
    from fastapi.testclient import TestClient
    from manage import web
    from manage.web import egress as egress_routes
    st, tid, dispatched = wired_web
    st.add_pending(_req(tid="t-gone", host="a.example.com"))
    c = TestClient(web.app)
    assert c.post("/egress/hour", data={"token": "wrong", "tid": tid,
                                        "host": "a.example.com",
                                        "port": "443"}).status_code == 403
    # deleted thread: record dropped, nothing granted
    c.post("/egress/hour", data={"token": egress_routes.EGRESS_CSRF,
                                 "tid": "t-gone", "host": "a.example.com",
                                 "port": "443"}, follow_redirects=False)
    assert st.all() == [] and dispatched == []


def test_thread_card_renders_and_clears(wired_web, monkeypatch):
    from fastapi.testclient import TestClient
    from manage import web
    from manage.web.state import _set_status
    st, tid, _ = wired_web
    monkeypatch.setattr(
        web.MANAGER, "get", lambda t, sandbox_backend=None, **k:
        type("C", (), {"get_messages": lambda s: [],
                       "pending_reply": lambda s: None})())
    monkeypatch.setattr("manage.web.threads.get_cached_description",
                        lambda t: "d")
    _set_status(tid, "ready")
    st.add_pending(_req(tid=tid, host="xn--bcher-kva.example",
                        task="fetch the <catalog>"))
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "Network access request" in html_out
    assert "requested just now" in html_out          # D6's request-age line
    assert "xn--bcher-kva.example:443" in html_out
    assert "punycode" in html_out                       # IDN warning
    assert "&lt;catalog&gt;" in html_out                # task text escaped
    assert "Always allow for this thread" in html_out
    st.resolve(request_key(tid, "xn--bcher-kva.example", 443), "decline")
    assert "Network access request" not in TestClient(web.app).get(
        f"/thread/{tid}").text


def test_global_pending_cap(tmp_path):
    st = _store(tmp_path)
    for i in range(st.GLOBAL_PENDING_CAP):
        assert st.add_pending(_req(tid=f"t{i}", host=f"h{i}.example.com")) is None
    assert st.add_pending(_req(tid="t-extra", host="hx.example.com")) == "global-cap"


def test_take_undispatched_batches_once(tmp_path):
    """The resolution turn enumerates each resolution exactly ONCE — an old
    decline or a long-lived always-grant is not re-announced on later
    batches (review finding: unbounded re-enumeration re-ran old tasks)."""
    from assist.egress.store import resolution_prompt
    st = _store(tmp_path)
    st.add_pending(_req(tid="t1", host="a.example.com", task="task A"))
    st.resolve(request_key("t1", "a.example.com", 443), "always")
    batch1 = st.take_undispatched("t1")
    assert [r.host for r in batch1] == ["a.example.com"]
    assert "task A" in resolution_prompt(batch1)
    assert st.take_undispatched("t1") == []          # never re-announced
    # a later cycle only announces the NEW resolutions
    st.add_pending(_req(tid="t1", host="c.example.com", task="task C"))
    st.resolve(request_key("t1", "c.example.com", 443), "decline")
    batch2 = st.take_undispatched("t1")
    assert [r.host for r in batch2] == ["c.example.com"]
    prompt2 = resolution_prompt(batch2)
    assert "task A" not in prompt2 and "DECLINED" in prompt2


def test_malformed_tid_is_400_not_500(wired_web, monkeypatch):
    from fastapi.testclient import TestClient
    from manage import web
    from manage.web import egress as egress_routes
    from assist.thread_manager import ThreadManager
    # the fixture stubs thread_dir without validation — restore the real
    # validating implementation (bound to the patched root) for this test
    monkeypatch.setattr(web.MANAGER, "thread_dir",
                        lambda t: ThreadManager.thread_dir(web.MANAGER, t))
    c = TestClient(web.app)
    r = c.post("/egress/hour", data={"token": egress_routes.EGRESS_CSRF,
                                     "tid": "../escape", "host": "a.example.com",
                                     "port": "443"})
    assert r.status_code == 400
