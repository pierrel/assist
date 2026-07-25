"""Progressive-response migration, rendering, and failure surfacing."""
import contextlib
import json
from types import SimpleNamespace

import pytest

from manage import web
from manage.web import threads
from manage.web.state import MESSAGE_BACKLOG, _get_status, _set_status, _thread_title
from assist.backlog import PendingMessage
from assist.events.thread_log import append_event, read_events

CHAIN_CAP = threads.CHAIN_CAP


@pytest.fixture
def wired(tmp_path, monkeypatch):
    tid = "t-prog"
    (tmp_path / tid).mkdir()
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_dir", lambda t: str(tmp_path / t))
    monkeypatch.setattr(MESSAGE_BACKLOG, "_root", str(tmp_path))
    monkeypatch.setattr(web.MANAGER, "thread_default_working_dir",
                        lambda t: str(tmp_path / t))
    monkeypatch.setattr(web.MANAGER, "touch", lambda t: None)
    monkeypatch.setattr("manage.web.threads._get_sandbox_backend",
                        lambda t, tz=None: None)
    monkeypatch.setattr("manage.web.threads._get_domain_manager", lambda t: None)
    monkeypatch.setattr("manage.web.threads.get_cached_description",
                        lambda t: "real description")
    monkeypatch.setattr("manage.web.state.get_cached_description",
                        lambda t: "real description")
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd, expected=None: None)
    with contextlib.suppress(Exception):
        while True:
            threads._RESUME_SCHEDULER._q.get_nowait()
    return tid, tmp_path


class _Chat:
    def __init__(self, tid, calls, reply=None):
        self.thread_id = tid
        self._calls = calls
        self._reply = reply

    def message(self, text):
        self._calls.append(("message", text))
        return "done"

    def resume(self):
        self._calls.append(("resume",))
        return "resumed"

    def pending_reply(self):
        return self._reply

    def get_messages(self):
        return []


def _wire_chat(monkeypatch, tid, calls, reply=None):
    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda t, sandbox_backend=None, on_queue_state=None, configurable=None,
        triage=False, continuation=False: _Chat(t, calls, reply))


# --- the event log ------------------------------------------------------------

def test_event_log_roundtrip_and_torn_line(tmp_path):
    d = str(tmp_path)
    append_event(d, "continuation_scheduled", id="a", task="t1")
    append_event(d, "continuation_dispatched", id="a")
    # a torn final line (reader racing an append) is skipped, never fatal
    with open(tmp_path / "events.jsonl", "a") as f:
        f.write('{"ts": "2026-')
    evs = read_events(d)
    assert [e["kind"] for e in evs] == ["continuation_scheduled",
                                       "continuation_dispatched"]
    assert read_events(d, kind="continuation_dispatched")[0]["id"] == "a"
    assert read_events(str(tmp_path / "missing")) == []


# --- chain-length derivation --------------------------------------------------

def test_chain_len_counts_trailing_markers_plus_pending(wired, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    tid, _ = wired
    R = threads._CONTINUATION_RIDER
    msgs = [HumanMessage("user asks"), AIMessage("a1"),
            HumanMessage(R + "job 1"), AIMessage("a2"),
            HumanMessage(R + "job 2"), AIMessage("a3")]
    tup = SimpleNamespace(checkpoint={"channel_values": {"messages": msgs}})
    monkeypatch.setattr(web.MANAGER, "checkpointer",
                        SimpleNamespace(get_tuple=lambda cfg: tup))
    threads._create_run(tid, "job 3", origin="continuation")
    threads._create_run(tid, "user follow-up")
    assert threads._continuation_chain_len(tid) == 3   # 2 trailing + 1 pending

    # A USER message after the chain resets the trailing run.
    msgs2 = msgs + [HumanMessage("user interjects"), AIMessage("a4")]
    tup.checkpoint = {"channel_values": {"messages": msgs2}}
    assert threads._continuation_chain_len(tid) == 1   # only the pending one


def test_chain_len_fails_closed_on_unreadable_state(wired, monkeypatch):
    tid, _ = wired
    monkeypatch.setattr(web.MANAGER, "checkpointer", SimpleNamespace(
        get_tuple=lambda cfg: (_ for _ in ()).throw(RuntimeError("boom"))))
    assert threads._continuation_chain_len(tid) >= CHAIN_CAP


# --- dispatch at the ready exit ----------------------------------------------

def test_continuation_turn_gets_marker_and_agent_note_render(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    rec = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="bg job",
                                             origin="continuation"))
    threads._process_message(tid, "bg job", origin="continuation")
    # the self-message the agent ran carried the durable attribution marker
    assert calls == [("message", threads._CONTINUATION_RIDER + "bg job")]


def test_peek_is_side_effect_free_on_corruption(tmp_path):
    """The loop-side read (render) must never mutate: a corrupt journal returns
    [] from peek() with the file left untouched — the move-aside belongs to the
    LOCKED read path only (Copilot #198 rd1)."""
    import os
    from assist.backlog import MessageBacklog
    store = MessageBacklog(str(tmp_path))
    (tmp_path / "t1").mkdir()
    (tmp_path / "t1" / "pending_messages.json").write_text('[{bad')
    assert store.peek("t1") == []
    assert (tmp_path / "t1" / "pending_messages.json").exists()
    assert not (tmp_path / "t1" / "pending_messages.json.corrupt").exists()
    # parseable-but-wrong-shaped entries must also degrade to [] on the render
    # path, never raise (Copilot #198 rd2)
    (tmp_path / "t1" / "pending_messages.json").write_text('[1, "x", {"weird": 1}]')
    assert store.peek("t1") == []


def test_user_message_preserves_unclaimed_continuations(wired, monkeypatch):
    tid, tmp_path = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    stale = threads._create_run(tid, "stale plan", origin="continuation")
    threads._process_message(tid, "new user direction", origin=None)
    assert threads._runs().get(tid, stale.id).status == "pending"
    kinds = [e["kind"] for e in read_events(str(tmp_path / tid))]
    assert "continuation_cancelled" not in kinds


def test_system_turns_do_not_clear_continuations(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="planned work",
                                       origin="continuation"))
    threads._process_message(tid, "scheduled prompt", origin="system")
    assert len(MESSAGE_BACKLOG.for_thread(tid)) == 1


def test_continuation_defers_while_reply_awaits_approval(wired, monkeypatch):
    """The supersede guard: a continuation must never reject a pending HITL
    draft — it returns WITHOUT claiming; the entry stays journaled for the
    approve-resume's ready exit."""
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls, reply={"text": "draft awaiting approval"})
    rec = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="bg job",
                                             origin="continuation"))
    threads._process_message(tid, "bg job", origin="continuation")
    assert calls == []                                          # nothing ran
    assert [r.id for r in MESSAGE_BACKLOG.for_thread(tid)] == [rec.id]


# --- failure surfacing --------------------------------------------------------

def test_continuation_failure_is_loud_and_attributed(wired, monkeypatch):
    tid, tmp_path = wired

    class _Boom(_Chat):
        def message(self, text):
            raise RuntimeError("research exploded")
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _Boom(t, []))
    unseen = []
    monkeypatch.setattr(threads, "_mark_unseen_response",
                        lambda t: unseen.append(t))
    rec = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="bg job",
                                             origin="continuation"))
    threads._process_message(tid, "bg job", origin="continuation")
    st = _get_status(tid)
    assert st["stage"] == "error"
    assert "background follow-up" in st["error"]        # not "your message"
    assert unseen == [tid]                              # loud, not bottom-band
    kinds = [e["kind"] for e in read_events(str(tmp_path / tid))]
    assert "continuation_failed" in kinds


# --- render surfaces ----------------------------------------------------------

def test_persisted_continuation_message_renders_as_agent_note(wired, monkeypatch):
    from fastapi.testclient import TestClient
    tid, _ = wired
    R = threads._CONTINUATION_RIDER

    class _Hist(_Chat):
        def get_messages(self):
            return [{"role": "user", "content": "real user question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": R + "research the rest"},
                    {"role": "assistant", "content": "follow-up answer"}]
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _Hist(t, []))
    _set_status(tid, "ready")
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "following up: research the rest" in html_out
    assert f">{R}" not in html_out       # raw marker text never renders as a bubble
    # the agent-note is NOT a user bubble
    assert 'class="msg continuation"' in html_out


def test_busy_banner_and_title_for_continuation_turn(wired, monkeypatch):
    from fastapi.testclient import TestClient
    from manage.web import state as st_mod
    tid, _ = wired
    _wire_chat(monkeypatch, tid, [])
    # The continuation title path reads the CACHE/file only (never the
    # generating path — it runs on the loop while the slot is busy).
    monkeypatch.setitem(st_mod.DESCRIPTION_CACHE, tid, "real description")
    R = threads._CONTINUATION_RIDER
    _set_status(tid, "processing", origin="continuation",
                pending_message=R + "research bike accessories")
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "Following up: research bike accessories" in html_out
    assert "Processing your message" not in html_out
    assert _thread_title(tid) == "real description"     # not the task snippet


def test_ready_thread_shows_will_follow_up_note(wired, monkeypatch):
    from fastapi.testclient import TestClient
    tid, _ = wired
    _wire_chat(monkeypatch, tid, [])
    _set_status(tid, "ready")
    threads._create_run(tid, "find tire sizes", origin="continuation")
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "will follow up: find tire sizes" in html_out


def test_locked_read_skips_malformed_entries_keeps_good(tmp_path):
    """The LOCKED readers (for_thread — used inside turn error handlers) must
    never raise on wrong-shaped entries: a raise there masks the original turn
    failure and strands the thread busy. Bad entries skip loudly; good ones
    survive (Copilot #198 rd3)."""
    import json as _json
    from assist.backlog import MessageBacklog, PendingMessage
    store = MessageBacklog(str(tmp_path))
    (tmp_path / "t1").mkdir()
    good = PendingMessage(thread_id="t1", text="keep me").to_dict()
    (tmp_path / "t1" / "pending_messages.json").write_text(
        _json.dumps([1, "junk", good, {"weird": True}]))
    recs = store.for_thread("t1")
    assert [r.text for r in recs] == ["keep me"]


def test_user_message_with_marker_prefix_is_neutralized(wired, monkeypatch):
    """A user pasting/quoting the continuation marker must not be misattributed
    as an agent note or count toward the chain (Copilot #198 rd4): a leading
    space breaks the startswith keying."""
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    R = threads._CONTINUATION_RIDER
    threads._process_message(tid, R + "just quoting you", origin=None)
    assert calls == [("message", " " + R + "just quoting you")]


def test_resume_of_continuation_turn_runs(wired, monkeypatch):
    """A fair-scheduling resume continues the checkpointed continuation."""
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    # the journal entry is GONE (claimed by the original turn start)
    assert MESSAGE_BACKLOG.for_thread(tid) == []
    _set_status(tid, "paused", origin="continuation",
                pending_message="[Continuing my earlier work — background follow-up] x")
    threads._process_message(tid, None, resume=True, origin="continuation")
    assert ("resume",) in calls, "the resume was swallowed by the entry-gone gate"
    assert _get_status(tid)["stage"] == "ready"


def test_waiting_resume_keeps_paused_status(wired, monkeypatch):
    """Same seam as the gate fix, un-mocked: a RESUME waiting behind another
    thread's turn must NOT overwrite its `paused` status with "queued" — the
    paused record is the resume's durable home, and follow-up routing keys on
    it (a "queued" stage misroutes a follow-up off the serial scheduler and
    onto the mid-flight checkpoint)."""
    import threading as _threading
    import time as _time
    from assist.thread_queue import THREAD_QUEUE
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    _set_status(tid, "paused", accumulated_active_ms=1234.0,
                pending_message="mid-flight message")
    release = _threading.Event()
    held = _threading.Event()

    def _holder():
        with THREAD_QUEUE.acquire("t-other-holder"):
            held.set()
            release.wait(timeout=10)
    h = _threading.Thread(target=_holder, daemon=True)
    h.start()
    assert held.wait(timeout=5)
    w = _threading.Thread(target=threads._process_message,
                          args=(tid, None), kwargs={"resume": True}, daemon=True)
    w.start()
    deadline = _time.time() + 2.0
    while _time.time() < deadline:      # the whole wait window: never "queued"
        assert _get_status(tid)["stage"] == "paused"
        _time.sleep(0.05)
    release.set()
    w.join(timeout=10)
    h.join(timeout=10)
    assert ("resume",) in calls
    assert _get_status(tid)["stage"] == "ready"


def test_turn_observer_fires_on_ready(wired, monkeypatch):
    tid, _ = wired
    _wire_chat(monkeypatch, tid, [])
    seen = []
    monkeypatch.setattr(threads, "_TURN_OBSERVERS", [lambda *a: seen.append(a)])
    threads._process_message(tid, "hi")
    assert seen == [(tid, "ready", None, "done", None)]


def test_turn_observer_isolated_and_reports_error(wired, monkeypatch):
    tid, _ = wired

    class _Boom:
        def message(self, text):
            raise RuntimeError("boom")

        def pending_reply(self):
            return None

        def get_messages(self):
            return []
    monkeypatch.setattr(web.MANAGER, "get", lambda t, **k: _Boom())
    good = []
    monkeypatch.setattr(threads, "_TURN_OBSERVERS", [
        lambda *a: (_ for _ in ()).throw(ValueError("observer boom")),  # raises
        lambda *a: good.append(a),                                      # still fires
    ])
    threads._process_message(tid, "hi")                 # must not raise
    assert good == [(tid, "error", None, None, None)]   # error stage, reply None
    assert _get_status(tid)["stage"] == "error"         # turn still terminalized


def test_turn_observer_registration_during_notify_is_isolated(wired, monkeypatch):
    # Snapshot semantics: turns run concurrently over the shared _TURN_OBSERVERS
    # global, so a registration racing a notify pass must not extend that pass. An
    # observer that registers a new one mid-pass must not make the newcomer fire now.
    tid, _ = wired
    _wire_chat(monkeypatch, tid, [])
    late = []
    monkeypatch.setattr(threads, "_TURN_OBSERVERS",
                        [lambda *a: threads.register_turn_observer(lambda *b: late.append(b))])
    threads._process_message(tid, "hi")
    assert late == []                       # the mid-pass registrant did NOT fire this pass


def test_turn_observer_fires_on_supersede_cap_awaiting_approval(wired, monkeypatch):
    # The supersede-cap path: a new message can't clear a stuck pending draft after
    # the reject-loop cap, so it writes a terminal awaiting_approval and returns
    # EARLY — before the common notify at the function end. The observer must still
    # fire there, or this real terminal outcome is invisible to a registered client.
    tid, _ = wired

    class _Stuck(_Chat):
        def resume_reply(self, decision):
            self._calls.append(("resume_reply", decision.get("type")))
            return "still proposing"        # the reject never clears the interrupt

    monkeypatch.setattr(
        web.MANAGER, "get",
        lambda t, sandbox_backend=None, **k: _Stuck(t, [], reply={"text": "stuck draft"}))
    _set_status(tid, "awaiting_approval", pending_reply="stuck draft",
                pending_sender="senderX")
    seen, holder_at_fire = [], []

    def _obs(*a):
        seen.append(a)
        holder_at_fire.append(threads.THREAD_QUEUE.peek_holder())
    monkeypatch.setattr(threads, "_TURN_OBSERVERS", [_obs])

    threads._process_message(tid, "a superseding message", origin=None)

    assert _get_status(tid)["stage"] == "awaiting_approval"
    assert seen == [(tid, "awaiting_approval", None, "stuck draft", None)]
    # The whole point of the controlled unwind: the observer fires AFTER the queue
    # is released (Copilot rd2), never while this turn still holds the slot.
    assert holder_at_fire == [None]
