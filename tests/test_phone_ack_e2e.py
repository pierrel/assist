"""End-to-end: the phone read-ack's revision check against a REAL turn.

The unit tests in tests/test_phone_api.py exercise the POST /open endpoint
directly with a hand-planted revision.  This file exercises the same endpoint
the way the phone actually hits it, and the "a new notification arrives" step
is a REAL `_process_message` turn (POST /thread/{tid}/message → the background
task → the real `_set_status("ready", mark_unseen=True)` → the real
`_mark_unseen_response`), not a flag the test planted by hand.

The revision the /open endpoint verifies is the SAME one the phone's GET
snapshot returned.  `_snapshot` computes it as a content hash of
``[stage, *message_tuples]`` where the message tuples come from the
checkpointer store the phone reads (``_checkpoint_history`` →
``state.MANAGER.checkpointer``), NOT the directory mtime.  So the only thing
that makes the phone's snapshot and the endpoint's recompute diverge is a new
response landing in that store in between — which is exactly what the real
turn does here: the fake chat persists its reply into the store through the
real checkpointer API, the way a real ``Thread.message()`` would.

What this runs FOR REAL:
  - A full `_process_message` turn (route, background task, THREAD_QUEUE,
    sandbox-backend lookup, chat.message, terminal status write)
  - The real unread ("new") marker that turn writes
  - The real checkpointer store the phone reads, and the real content-hash
    revision that bumps when the turn's response lands in it

STUBBED (the external boundaries, exactly as the sibling e2e tests do):
  - Docker (`_get_sandbox_backend` → None)
  - The LLM/graph (MANAGER.get → a fake chat whose `.message()` returns a
    canned string)
  - The domain-manager sync + description generation (no network)

The checkpointer is re-pointed at a fresh, disposable store (never the
production threads.db), so the fake chat's persisted responses are isolated.
"""
import os
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from manage import web
from manage.web import phone_api, state, threads

TID = "thread-e2e"


@pytest.fixture
def phone_client(tmp_path, monkeypatch):
    """The e2e thread fixture plus a configured phone token and a disposable
    checkpointer store, so one TestClient can hit BOTH the web
    /thread/... routes and the /api/v1/phone/... routes the phone would."""
    tdir = tmp_path / TID
    tdir.mkdir()
    from assist.browser.authority import mark_new_thread
    mark_new_thread(str(tmp_path), TID)
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(
        web.MANAGER, "thread_default_working_dir", lambda tid: str(tmp_path / tid),
    )
    from assist.location import LocationStore
    monkeypatch.setattr(threads, "LOCATION_STORE", LocationStore(str(tmp_path)))

    # Re-point the checkpointer at a fresh, disposable store.  The phone's
    # snapshot reads its `writes` table, and the fake chat persists into it —
    # so they must share one store, and it must never be the production db.
    conn = sqlite3.connect(str(tmp_path / "test-threads.db"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    monkeypatch.setattr(state.MANAGER, "checkpointer", saver)
    monkeypatch.setattr(state.MANAGER, "conn", conn)

    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, "phone-e2e-token")
    return TestClient(web.app)


def _stub_happy_path(monkeypatch, chat):
    """Stub Docker, the LLM, and the post-message network hooks so a turn
    runs end-to-end against `chat` (same shape as the sibling e2e tests)."""
    monkeypatch.setattr("manage.web.state._get_sandbox_backend", lambda tid, tz=None, **_kwargs: None)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend", lambda tid, tz=None, **_kwargs: None)
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda tid, sandbox_backend=None, on_queue_state=None, configurable=None, triage=False, continuation=False: chat,
    )
    monkeypatch.setattr(web.MANAGER, "touch", lambda tid: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda tid: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda tid: "stub")


class _FakeChat:
    """A minimal chat whose `.message()` returns a canned string AND persists
    it into the checkpointer the phone reads — so the new response is
    genuinely visible in the next snapshot and bumps its content-hash
    revision, the way a real ``Thread.message()`` would."""
    thread_id = TID
    agent = None
    on_queue_state = None

    def __init__(self):
        self._seq = 0

    def message(self, text):
        self._seq += 1
        reply = f"response {self._seq}"
        state.MANAGER.checkpointer.put_writes(
            {"configurable": {"thread_id": TID,
                              "checkpoint_id": f"cp{self._seq}", "checkpoint_ns": ""}},
            [("messages", [AIMessage(content=reply)])],
            task_id=f"task{self._seq}",
        )
        return reply

    def pending_reply(self):
        return None

    def get_messages(self):
        return []

    def get_raw_messages(self):
        return []

    def description(self):
        return "desc"


def _wait_for_terminal_status(tid: str, deadline_s: float = 5.0) -> dict:
    """Poll status.json until the BackgroundTask has written it and stage is
    terminal (ready/error).  Gated on the file existing first — `_get_status`
    defaults to {"stage": "ready"} when the file is absent, so treating that
    default as terminal would pass without the turn ever having run."""
    from manage.web.state import _get_status, _status_path
    path = _status_path(tid)
    end = time.time() + deadline_s
    while time.time() < end:
        if os.path.isfile(path):
            st = _get_status(tid)
            if st.get("stage") in ("ready", "error"):
                return st
        time.sleep(0.05)
    return _get_status(tid)


def _run_one_turn(client, chat, text="hello"):
    """POST a message through the real route and wait for the turn to settle.
    The turn's reply is persisted into the checkpointer, so a following phone
    snapshot sees it and its content-hash revision has bumped."""
    r = client.post(f"/thread/{TID}/message", data={"text": text}, follow_redirects=False)
    assert r.status_code == 303, r.text
    status = _wait_for_terminal_status(TID, deadline_s=5.0)
    assert status.get("stage") == "ready", (
        f"turn did not reach 'ready' — final status: {status!r}.  This usually "
        f"means an exception in _process_message itself (a missing import or a "
        f"stub that didn't match the call shape)."
    )
    return status


def _phone_auth():
    return {"Authorization": "Bearer phone-e2e-token"}


def _phone_revision(client, tid):
    """Fetch the phone snapshot for `tid` and return its thread revision.  A
    GET is a read: it exposes the revision and must never clear a flag."""
    snap = client.get(f"/api/v1/phone/threads/{tid}", headers=_phone_auth())
    assert snap.status_code == 200, snap.text
    return snap.json()["thread"]["revision"]


def _ack(client, tid, revision):
    return client.post(
        f"/api/v1/phone/threads/{tid}/open",
        headers=_phone_auth(), json={"revision": revision})


def test_phone_ack_for_settled_view_clears_the_real_unseen_marker(
        phone_client, monkeypatch):
    """Happy path: a real turn delivers a response and marks the thread
    unseen.  The phone renders the settled snapshot (GET — which must never
    clear), then acks with that snapshot's revision.  The server recomputes
    the revision, sees it still matches, and clears the real unseen marker
    the turn wrote."""
    chat = _FakeChat()
    _stub_happy_path(monkeypatch, chat)

    # 1. A real turn completes: the response lands in the store and the thread
    #    is marked unseen (the "new" notification the user hasn't seen yet).
    _run_one_turn(phone_client, chat)
    assert state._has_unseen_response(TID)

    # 2. The phone fetches and displays the settled snapshot.  A GET is a
    #    read: it exposes the revision and must NOT clear the flag.
    viewed_revision = _phone_revision(phone_client, TID)
    assert state._has_unseen_response(TID)  # GET never clears

    # 3. The phone acks the view it actually rendered, carrying that revision.
    response = _ack(phone_client, TID, viewed_revision)

    # 4. Nothing newer arrived, so the revision still matches and the real
    #    unseen marker the turn wrote is cleared.
    assert response.status_code == 200, response.text
    assert response.json()["cleared"] is True
    assert not state._has_unseen_response(TID)


def test_phone_ack_stale_after_a_new_real_turn_keeps_the_newer_marker(
        phone_client, monkeypatch):
    """The read-receipt race, driven by a real turn for the "new notification
    arrives" step:

      1. The phone fetches and displays view A (a settled, empty thread) and
         captures A's revision.
      2. A new response arrives — a REAL `_process_message` turn — which
         persists its reply into the store (bumping the content-hash
         revision) and marks the thread unseen.
      3. The phone's ack for A reaches the server carrying A's now-stale
         revision.

    The server recomputes the current revision, sees it no longer equals the
    acked one, and must NOT clear — the notification for the response the user
    has not seen survives."""
    chat = _FakeChat()
    _stub_happy_path(monkeypatch, chat)

    # 1. The phone fetches and displays view A.  The store is empty and the
    #    thread has no unseen marker yet, but we capture the revision of this
    #    view.
    viewed_revision = _phone_revision(phone_client, TID)
    assert not state._has_unseen_response(TID)

    # 2. A new response arrives BETWEEN the snapshot and the ack: a real turn
    #    persists its reply into the store (bumping the revision) and marks
    #    the thread unseen.
    _run_one_turn(phone_client, chat, text="a follow-up")
    assert state._has_unseen_response(TID)  # the new response is unread
    current_revision = _phone_revision(phone_client, TID)
    assert current_revision != viewed_revision, (
        "the new real turn should have bumped the content-hash revision by "
        "persisting its response; the ack race can't be exercised if the "
        "revision is stable across turns"
    )

    # 3. The phone's ack for the OLD view A reaches the server (stale revision).
    response = _ack(phone_client, TID, viewed_revision)

    # 4. The revision no longer matches: the flag for the newer, unseen
    #    response is kept.
    assert response.status_code == 200, response.text
    assert response.json()["cleared"] is False
    assert state._has_unseen_response(TID)


def test_phone_ack_after_refresh_of_newer_view_clears_it(
        phone_client, monkeypatch):
    """The inverse of the race: after the newer response arrives and the phone
    refreshes, it renders the NEW view and acks with the CURRENT revision.  The
    server verifies the match and clears the (now genuinely seen) marker."""
    chat = _FakeChat()
    _stub_happy_path(monkeypatch, chat)

    _phone_revision(phone_client, TID)  # the phone is looking at view A
    _run_one_turn(phone_client, chat)  # a new response arrives
    assert state._has_unseen_response(TID)

    # The phone refreshes: it now sees the response and its new revision.
    current_revision = _phone_revision(phone_client, TID)
    response = _ack(phone_client, TID, current_revision)

    assert response.status_code == 200, response.text
    assert response.json()["cleared"] is True
    assert not state._has_unseen_response(TID)
