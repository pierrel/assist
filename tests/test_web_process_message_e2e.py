"""End-to-end regression test for `_process_message`.

The 2026-05-30 sandbox-defer-until-queue PR (#117) initially shipped
with a `THREAD_QUEUE` NameError — the symbol was referenced in
`_process_message` but never imported in `manage/web/threads.py`.  The
existing test suite missed it because `tests/test_web_review.py` stubs
`_process_message` wholesale, and no other test actually runs the
function end-to-end.

This file plugs that gap.  It POSTs to the real
``/thread/{tid}/message`` route via FastAPI's `TestClient`, lets the
background task run, and asserts the thread reaches ``status="ready"``
without crashing inside ``_process_message``.

What this test runs FOR REAL (catches regressions in):
  - The FastAPI route handler and `BackgroundTasks` scheduling
  - `_process_message`'s top-level flow (imports, status writes,
    exception handling)
  - The `THREAD_QUEUE.acquire(...)` block (catches missing-import
    and contextvar-handling regressions)
  - The post-acquire status sequence

What is STUBBED (NOT exercised here — would need an integration test
with real Docker + a real LLM):
  - `_get_sandbox_backend` — stubbed to None (the same shape it
    returns when Docker is unavailable)
  - `MANAGER.get` — returns a `_FakeChat` (Thread / agent / LLM stack
    is not exercised)
  - The domain-manager sync and description-generation paths
"""
import os
import time

import pytest
from fastapi.testclient import TestClient

from manage import web
from manage.web import threads
from manage.web.state import _get_status, _set_status


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Repoint the singleton ThreadManager at a tmp dir + create one
    thread directory.  Match the pattern from test_web_review.py.
    """
    tdir = tmp_path / "thread-e2e"
    tdir.mkdir()
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(
        web.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid)
    )
    # _process_message also calls MANAGER.thread_default_working_dir
    # (in the SandboxContainerLostError branch); point it at the same dir.
    monkeypatch.setattr(
        web.MANAGER, "thread_default_working_dir", lambda tid: str(tmp_path / tid),
    )
    return TestClient(web.app)


def _wait_for_terminal_status(tid: str, deadline_s: float = 5.0) -> dict:
    """Poll status.json until stage in {ready, error}, or fail.

    *Important:* `_get_status` returns ``{"stage": "ready"}`` as its
    DEFAULT when the file doesn't exist yet (see
    ``manage/web/state.py``).  Treating that default as "terminal" would
    let this test pass without `_process_message` ever having run — a
    false positive that defeats the regression purpose.  So gate on
    ``os.path.isfile(_status_path(tid))`` first; only when the
    BackgroundTask has actually written status.json do we consider
    ``stage`` terminal."""
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


def test_post_message_runs_process_message_without_crashing(
    client, monkeypatch
):
    """The full POST → BackgroundTask → _process_message → THREAD_QUEUE
    acquire → sandbox-backend lookup → chat.message → status="ready"
    path runs without raising.  Regression: PR #117's initial commit
    broke this with a `THREAD_QUEUE` NameError that no existing test
    caught."""
    # Stub the sandbox backend lookup so the test doesn't require Docker.
    # Returning None is the same shape `_get_sandbox_backend` returns when
    # Docker is unavailable on the host.  Patch both the source binding
    # AND the threads-module's already-imported reference; _process_message
    # calls the latter.
    monkeypatch.setattr(
        "manage.web.state._get_sandbox_backend", lambda tid, tz=None: None,
    )
    monkeypatch.setattr(
        "manage.web.threads._get_sandbox_backend", lambda tid, tz=None: None,
    )

    # Stub MANAGER.get to return a minimal fake chat whose `.message()`
    # returns a canned response without touching the LLM or langgraph.
    class _FakeChat:
        thread_id = "thread-e2e"
        agent = None
        on_queue_state = None
        def message(self, text):
            return "ok"
        def description(self):
            return "desc"
        def pending_reply(self):
            return None

        def get_messages(self):
            return [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"}]

    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda tid, sandbox_backend=None, on_queue_state=None, configurable=None, triage=False: _FakeChat(),
    )
    monkeypatch.setattr(web.MANAGER, "touch", lambda tid: None)

    # Stub the domain-manager hook so the post-message sync block is a no-op.
    monkeypatch.setattr(
        "manage.web.threads._get_domain_manager", lambda tid: None,
    )

    # Stub description generation so it doesn't try to network.
    # IMPORTANT: `_process_message` does `from manage.web.state import
    # ... get_cached_description`, so the live binding is the one in
    # `manage.web.threads`'s module namespace — patching only the source
    # module (`manage.web.state.get_cached_description`) would leave the
    # imported reference unchanged.
    monkeypatch.setattr(
        "manage.web.threads.get_cached_description", lambda tid: "stub",
    )

    r = client.post(
        "/thread/thread-e2e/message",
        data={"text": "hello"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text

    # The BackgroundTask runs in the same TestClient request cycle once
    # the response is consumed (anyio.to_thread.run_sync semantics under
    # the test client are synchronous for our purposes).  Give it a
    # short poll window in case of timing.
    status = _wait_for_terminal_status("thread-e2e", deadline_s=5.0)
    assert status.get("stage") == "ready", (
        f"_process_message did not reach 'ready' — final status: {status!r}. "
        f"This usually means an exception in _process_message itself, e.g. "
        f"a missing import (NameError) or a stub that didn't match the "
        f"call shape."
    )


# --- per-turn container teardown ------------------------------------------------
#
# One container per turn: _process_message must tear down the turn's sandbox in
# a `finally`, so it runs on the success path AND every error path.  These pin
# the wiring (the symptom — a real container being reaped — is covered un-mocked
# by tests/test_sandbox_per_turn.py's real-Docker test).


def _stub_happy_path(monkeypatch, chat):
    """Stub the sandbox lookup, MANAGER.get, and the post-message hooks so
    _process_message runs end-to-end against `chat`."""
    monkeypatch.setattr("manage.web.state._get_sandbox_backend", lambda tid, tz=None: None)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend", lambda tid, tz=None: None)
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda tid, sandbox_backend=None, on_queue_state=None, configurable=None, triage=False: chat,
    )
    monkeypatch.setattr(web.MANAGER, "touch", lambda tid: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda tid: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda tid: "stub")


def _spy_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(
        threads.SandboxManager, "cleanup",
        classmethod(lambda cls, work_dir: calls.append(work_dir)),
    )
    return calls


def test_process_message_kills_container_at_turn_end_on_success(client, monkeypatch):
    class _Chat:
        def message(self, text):
            return "ok"

        def pending_reply(self):
            return None
    _stub_happy_path(monkeypatch, _Chat())
    calls = _spy_cleanup(monkeypatch)

    r = client.post("/thread/thread-e2e/message", data={"text": "hi"},
                    follow_redirects=False)
    assert r.status_code == 303, r.text
    assert _wait_for_terminal_status("thread-e2e").get("stage") == "ready"

    assert len(calls) == 1, f"expected exactly one per-turn teardown, got {calls}"
    assert calls[0].endswith("thread-e2e"), f"teardown targeted the wrong work_dir: {calls}"


def test_process_message_kills_container_even_when_turn_errors(client, monkeypatch):
    """The teardown is in a `finally`, so a crash mid-turn still reaps the
    container — otherwise an erroring turn would leak its sandbox."""
    class _BoomChat:
        def message(self, text):
            raise RuntimeError("boom mid-turn")
    _stub_happy_path(monkeypatch, _BoomChat())
    calls = _spy_cleanup(monkeypatch)

    r = client.post("/thread/thread-e2e/message", data={"text": "hi"},
                    follow_redirects=False)
    assert r.status_code == 303, r.text
    assert _wait_for_terminal_status("thread-e2e").get("stage") == "error"

    assert len(calls) == 1, f"erroring turn must still tear down its container, got {calls}"


def test_process_message_reaps_registered_container_when_creation_then_raises(
        client, monkeypatch, tmp_path):
    """The exact round-1 gap: sandbox creation registers a container and THEN
    raises.  Because the creation is inside the try, the teardown `finally`
    runs, and because cleanup keys on work_dir (not the `sandbox` handle that
    was never returned), the registered container is reaped — no leak until the
    backstop TTL.  (Copilot review, PR #139.)

    Exercised un-mocked: cleanup is NOT stubbed.  A real container handle is put
    in the registry to stand in for "creation registered it", then creation
    raises; we assert the turn actually removed it (SIGKILL + dropped)."""
    from unittest.mock import MagicMock
    from assist.sandbox_manager import SandboxManager

    work_dir = str(tmp_path / "thread-e2e")  # == MANAGER.thread_default_working_dir
    registered = MagicMock()
    SandboxManager._containers[work_dir] = registered

    def _register_then_boom(tid, tz=None):
        # The container is already in the registry (as get_sandbox_backend
        # leaves it); creation now fails before returning a usable backend.
        raise RuntimeError("sandbox creation failed after registering a container")
    monkeypatch.setattr("manage.web.state._get_sandbox_backend", _register_then_boom)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend", _register_then_boom)
    monkeypatch.setattr(web.MANAGER, "touch", lambda tid: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda tid: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda tid: "stub")

    try:
        r = client.post("/thread/thread-e2e/message", data={"text": "hi"},
                        follow_redirects=False)
        assert r.status_code == 303, r.text
        assert _wait_for_terminal_status("thread-e2e").get("stage") == "error"

        registered.kill.assert_called_once()  # reaped (SIGKILL), not leaked
        assert work_dir not in SandboxManager._containers
    finally:
        SandboxManager._containers.pop(work_dir, None)


# --- _mark_pending: synchronous feedback so a queued message isn't lost -------
#
# Regression: the thread page has no polling, so feedback is gated on the
# status being a BUSY_STAGE at redirect-render time.  `post_message` left the
# first status write to the background task, which races (and under load
# loses to) the redirect render — the message vanished from the UI with no
# "waiting in queue" feedback.  `_mark_pending` writes it synchronously.


def test_mark_pending_sets_queued_when_another_thread_holds_slot(client, monkeypatch):
    monkeypatch.setattr(
        threads.THREAD_QUEUE, "peek_holder", lambda: "other-thread",
    )
    threads._mark_pending("thread-e2e", "hello there")
    st = _get_status("thread-e2e")
    assert st.get("stage") == "queued", st
    assert st.get("pending_message") == "hello there", st


def test_mark_pending_sets_processing_when_slot_free(client, monkeypatch):
    # Free slot -> "processing" (a BUSY but NON-INIT stage): the existing
    # thread's history and input must stay visible on the redirect render.
    from manage.web.state import INIT_STAGES
    monkeypatch.setattr(threads.THREAD_QUEUE, "peek_holder", lambda: None)
    threads._mark_pending("thread-e2e", "hello")
    st = _get_status("thread-e2e")
    assert st.get("stage") == "processing", st
    assert st.get("stage") not in INIT_STAGES, st
    assert st.get("pending_message") == "hello", st


def test_mark_pending_noop_when_thread_already_busy(client, monkeypatch):
    # An in-flight turn must not be clobbered by a second submission.
    _set_status("thread-e2e", "processing", pending_message="first turn")
    monkeypatch.setattr(
        threads.THREAD_QUEUE, "peek_holder", lambda: "other-thread",
    )
    threads._mark_pending("thread-e2e", "second turn")
    st = _get_status("thread-e2e")
    assert st.get("stage") == "processing", st
    assert st.get("pending_message") == "first turn", st


def test_post_message_writes_busy_status_synchronously(client, monkeypatch):
    """The POST handler must persist a BUSY_STAGE + pending_message before it
    returns the redirect — so the redirect-GET renders feedback even when the
    background task hasn't run yet.  Stub `_process_message` to a no-op so we
    observe the endpoint's synchronous write, not a later overwrite."""
    monkeypatch.setattr(
        threads.THREAD_QUEUE, "peek_holder", lambda: "other-thread",
    )
    monkeypatch.setattr("manage.web.threads._process_message",
                        lambda tid, text, rider=None: None)

    r = client.post(
        "/thread/thread-e2e/message",
        data={"text": "queued message"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    st = _get_status("thread-e2e")
    assert st.get("stage") == "queued", st
    assert st.get("pending_message") == "queued message", st


def test_pending_message_renders_at_top_as_latest(client, monkeypatch):
    """The just-submitted (pending) message must render at the TOP — as the
    latest message, right under the in-progress "..." placeholder — not
    stranded at the bottom under the prior conversation.  The page is
    newest-at-top and get_messages() is chronological, so the pending bubble
    must be appended (rendered first after the reverse), not inserted at 0."""
    class _FakeChat:
        def pending_reply(self):
            return None

        def get_messages(self):
            return [
                {"role": "user", "content": "OLD question"},
                {"role": "assistant", "content": "OLD answer"},
            ]
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda tid, sandbox_backend=None, on_queue_state=None, configurable=None, triage=False: _FakeChat(),
    )
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda tid: None)

    _set_status("thread-e2e", "queued", pending_message="NEW pending message")
    html = client.get("/thread/thread-e2e").text

    pos_new = html.find("NEW pending message")
    pos_old = html.find("OLD question")
    assert pos_new != -1, "pending message not rendered at all"
    assert pos_old != -1, "prior conversation not rendered"
    # newest-at-top: the just-sent message must appear ABOVE the old one.
    assert pos_new < pos_old, (
        "pending message rendered below the prior conversation — it should be "
        "the latest message at the top"
    )


def test_pending_bubble_not_duplicated_when_already_persisted(client, monkeypatch):
    """Once the agent persists the just-submitted message into the
    conversation, the pending bubble must dedup against it — even when the
    persisted text carries trailing whitespace the stripped `pending` does not
    (review submissions end with a newline).  Otherwise the busy render shows
    the message twice."""
    persisted = "## Change review\n\nLooks solid to me\n"  # trailing newline, as _format_review_message emits
    class _FakeChat:
        def pending_reply(self):
            return None

        def get_messages(self):
            return [{"role": "user", "content": persisted}]
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda tid, sandbox_backend=None, on_queue_state=None, configurable=None, triage=False: _FakeChat(),
    )
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda tid: None)

    # _mark_pending stores the message unstripped; the thread is mid-turn.
    _set_status("thread-e2e", "processing", pending_message=persisted)
    html = client.get("/thread/thread-e2e").text

    assert html.count("Looks solid to me") == 1, (
        f"duplicate pending bubble: the message rendered "
        f"{html.count('Looks solid to me')} times"
    )


def test_rider_flows_to_sandbox_tz_and_configurable(client, monkeypatch):
    """Symptom test: a POST carrying sent_at+tz reaches BOTH rider consumers —
    the sandbox TZ (so `date` is the user's local time) and the turn's
    `configurable` (so the middleware can render the model line)."""
    from assist.context_rider import CONTEXT_RIDER_KEY
    captured = {}
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend",
                        lambda tid, tz=None: captured.update(tz=tz) or None)

    class _FakeChat:
        def message(self, text):
            return "ok"
        def pending_reply(self):
            return None

        def get_messages(self):
            return [{"role": "user", "content": "hi"}]

    def _get(tid, sandbox_backend=None, on_queue_state=None, configurable=None, triage=False):
        captured["configurable"] = configurable
        return _FakeChat()

    monkeypatch.setattr(web.MANAGER, "get", _get)
    monkeypatch.setattr(web.MANAGER, "touch", lambda tid: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda tid: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description", lambda tid: "stub")

    r = client.post(
        "/thread/thread-e2e/message",
        data={"text": "what's today?",
              "sent_at": "2026-06-29T21:05:00.000Z",  # real browser toISOString() format
              "tz": "America/Los_Angeles",
              "lat": "37.7749", "lon": "-122.4194"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    _wait_for_terminal_status("thread-e2e", deadline_s=5.0)

    assert captured.get("tz") == "America/Los_Angeles"
    rider = (captured.get("configurable") or {}).get(CONTEXT_RIDER_KEY)
    assert rider is not None and rider.tz == "America/Los_Angeles"
    assert rider.lat == 37.7749 and rider.lon == -122.4194  # coords flow to the rider too


def test_new_thread_form_flows_rider(client, monkeypatch):
    """The index new-thread form (/threads/with-message) now carries the rider too —
    sent_at/tz/lat/lon reach _initialize_thread (→ the first message's _process_message)."""
    from assist.context_rider import ContextRider  # noqa: F401
    captured = {}
    monkeypatch.setattr("manage.web.threads._initialize_thread",
                        lambda tid, text, domain, rider=None: captured.update(rider=rider))

    class _FakeChat:
        thread_id = "thread-e2e"
    monkeypatch.setattr(web.MANAGER, "new", lambda: _FakeChat())

    r = client.post(
        "/threads/with-message",
        data={"text": "find a restaurant nearby",
              "sent_at": "2026-06-29T21:05:00.000Z", "tz": "America/Los_Angeles",
              "lat": "37.7749", "lon": "-122.4194"},
        follow_redirects=False,
    )
    assert r.status_code == 303, r.text
    rider = captured.get("rider")
    assert rider is not None
    assert rider.tz == "America/Los_Angeles" and rider.lat == 37.7749 and rider.lon == -122.4194


def test_process_message_supersedes_pending_reply(client, monkeypatch):
    """A new inbound message while a reply is awaiting approval rejects that draft and runs
    ONE turn with the fold-into-a-single-reply rider (Pierre's preference), checked after the
    queue acquire so a quick second text can't race a still-processing turn."""
    class _Chat:
        def __init__(self):
            self.resumed, self.messaged, self._pending = [], [], True

        def pending_reply(self):
            return {"text": "old draft"} if self._pending else None

        def resume_reply(self, decision):
            self.resumed.append(decision); self._pending = False; return ""

        def message(self, text):
            self.messaged.append(text); return "combined reply"

    chat = _Chat()
    _stub_happy_path(monkeypatch, chat)
    # The pending draft is for the SAME sender → fold into one combined reply.
    _set_status("thread-e2e", "awaiting_approval", pending_sender="+15551234567")
    threads._process_message("thread-e2e", "second message", sender="+15551234567")
    assert chat.resumed and chat.resumed[0]["type"] == "reject"     # pending draft rejected
    assert chat.messaged and "ONE reply" in chat.messaged[0]        # fold rider present
    assert "second message" in chat.messaged[0]                     # + the new message


def test_process_message_supersede_different_sender_does_not_fold(client, monkeypatch):
    """A catch-all subscription funneling different senders must NOT mix conversations: the
    old draft is rejected but the new sender's message runs clean (no fold rider)."""
    class _Chat:
        def __init__(self):
            self.resumed, self.messaged, self._pending = [], [], True

        def pending_reply(self):
            return {"text": "draft for A"} if self._pending else None

        def resume_reply(self, decision):
            self.resumed.append(decision); self._pending = False; return ""

        def message(self, text):
            self.messaged.append(text); return "reply to B"

    chat = _Chat()
    _stub_happy_path(monkeypatch, chat)
    _set_status("thread-e2e", "awaiting_approval", pending_sender="+1AAAAAAAAAA")  # sender A
    threads._process_message("thread-e2e", "hi from B", sender="+1BBBBBBBBBB")      # sender B
    assert chat.resumed and chat.resumed[0]["type"] == "reject"     # A's draft dropped
    assert chat.messaged and "ONE reply" not in chat.messaged[0]    # NO fold (cross-sender)
    assert "hi from B" in chat.messaged[0]


def test_process_message_no_supersede_when_nothing_pending(client, monkeypatch):
    class _Chat:
        def __init__(self):
            self.resumed, self.messaged = [], []

        def pending_reply(self):
            return None

        def resume_reply(self, decision):
            self.resumed.append(decision); return ""

        def message(self, text):
            self.messaged.append(text); return "ok"

    chat = _Chat()
    _stub_happy_path(monkeypatch, chat)
    threads._process_message("thread-e2e", "hello", sender="+15551234567")
    assert chat.resumed == []                                       # nothing to supersede
    assert chat.messaged == ["hello"]                               # plain turn, no rider


def test_process_message_supersede_gives_up_after_cap_without_dropping(client, monkeypatch):
    """If the reject-turn keeps re-proposing past the cap, don't run a fresh turn on the
    still-interrupted graph (which langgraph would drop) — leave the draft pending + status
    awaiting_approval; the new message stays durably archived."""
    class _StubbornChat:
        def __init__(self):
            self.messaged = []
        def pending_reply(self):
            return {"text": "keeps re-proposing"}     # never clears
        def resume_reply(self, decision):
            return ""
        def message(self, text):
            self.messaged.append(text); return "should not run"
    chat = _StubbornChat()
    _stub_happy_path(monkeypatch, chat)
    _set_status("thread-e2e", "awaiting_approval", pending_sender="+15551234567")
    threads._process_message("thread-e2e", "new msg", sender="+15551234567")
    assert chat.messaged == []                          # never ran a turn on the stuck graph
    assert _get_status("thread-e2e").get("stage") == "awaiting_approval"
