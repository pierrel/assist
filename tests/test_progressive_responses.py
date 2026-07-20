"""Progressive responses — the continue_later tool, chain cap, dispatch,
render keying, and failure surfacing (docs/2026-07-19-progressive-responses-
design.org). LLM-judgment behaviors (does the model split well) live in
edd/eval/test_progressive_responses.py; everything mechanical is pinned here.
"""
import contextlib
import json
from types import SimpleNamespace

import pytest

from manage import web
from manage.web import threads
from manage.web.state import MESSAGE_BACKLOG, _get_status, _set_status, _thread_title
from assist.backlog import PendingMessage
from assist.events.continuations import CHAIN_CAP, continuation_tools
from assist.events.thread_log import append_event, read_events


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
    monkeypatch.setattr("manage.web.threads.SandboxManager.cleanup", lambda wd: None)
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


# --- the tool: cap, journal, corrective strings -------------------------------

def test_continue_later_journals_and_directs(monkeypatch):
    journaled, out = [], []
    tools = continuation_tools(lambda tid, task: journaled.append((tid, task)),
                               lambda tid: 0)
    tool = tools[0]
    monkeypatch.setattr("assist.events.continuations._thread_id", lambda: "t1")
    msg = tool("  research   bike accessories  ")
    assert journaled == [("t1", "research bike accessories")]   # whitespace collapsed
    assert "finish your answer" in msg.lower() and "end this turn" in msg.lower()


def test_continue_later_cap_refusal_schedules_nothing(monkeypatch):
    journaled = []
    tools = continuation_tools(lambda tid, task: journaled.append(task),
                               lambda tid: CHAIN_CAP)
    monkeypatch.setattr("assist.events.continuations._thread_id", lambda: "t1")
    msg = tools[0]("more research")
    assert journaled == []
    assert "cap reached" in msg.lower() and "do not promise" in msg.lower()


def test_continue_later_rejects_empty_task(monkeypatch):
    journaled = []
    tools = continuation_tools(lambda tid, task: journaled.append(task),
                               lambda tid: 0)
    monkeypatch.setattr("assist.events.continuations._thread_id", lambda: "t1")
    assert "nothing scheduled" in tools[0]("   ").lower()
    assert journaled == []


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
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="job 3",
                                       origin="continuation"))
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="user follow-up"))
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

def test_ready_exit_dispatches_continuations_with_origin(wired, monkeypatch):
    """The real flow: the TURN ITSELF schedules the continuation mid-run (the
    continue_later tool journals it); the ready exit then dispatches it. (An
    entry journaled BEFORE a user turn is a stale plan and is cleared instead —
    pinned by test_user_message_clears_unclaimed_continuations.)"""
    tid, tmp_path = wired

    class _Scheduling(_Chat):
        def message(self, text):
            threads._journal_continuation(self.thread_id, "bg job")
            return "fast answer"
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _Scheduling(t, []))
    submitted = []
    monkeypatch.setattr(
        threads._RESUME_SCHEDULER, "submit_message",
        lambda t, text, rider, sender, backlog_id=None, origin=None:
            submitted.append((t, text, backlog_id, origin)))

    threads._process_message(tid, "user question", origin=None)

    recs = MESSAGE_BACKLOG.for_thread(tid)
    assert len(recs) == 1 and recs[0].text == "bg job"   # durable while queued
    assert submitted == [(tid, "bg job", recs[0].id, "continuation")]
    kinds = [e["kind"] for e in read_events(str(tmp_path / tid))]
    assert kinds.count("continuation_scheduled") == 1
    assert kinds.count("continuation_dispatched") == 1


def test_continuation_turn_gets_marker_and_agent_note_render(wired, monkeypatch):
    tid, _ = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    rec = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="bg job",
                                             origin="continuation"))
    threads._process_message(tid, "bg job", backlog_id=rec.id,
                             origin="continuation")
    # the self-message the agent ran carried the durable attribution marker
    assert calls == [("message", threads._CONTINUATION_RIDER + "bg job")]
    assert MESSAGE_BACKLOG.for_thread(tid) == []   # claimed


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


def test_continuation_turn_does_not_redispatch_siblings(wired, monkeypatch):
    """A continuation turn's ready exit must not re-submit still-queued sibling
    entries — they already have dispatchers from the scheduling turn; re-adding
    jobs is redundant traffic + duplicate dispatched events (Copilot #198 rd1)."""
    tid, tmp_path = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    # sibling B still journaled (its job queued elsewhere) while continuation A runs
    a = MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="job A",
                                           origin="continuation"))
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="job B",
                                       origin="continuation"))
    submitted = []
    monkeypatch.setattr(
        threads._RESUME_SCHEDULER, "submit_message",
        lambda t, text, rider, sender, backlog_id=None, origin=None:
            submitted.append(backlog_id))

    threads._process_message(tid, "job A", backlog_id=a.id, origin="continuation")

    assert submitted == []          # no re-dispatch from the continuation turn
    kinds = [e["kind"] for e in read_events(str(tmp_path / tid))]
    assert kinds.count("continuation_dispatched") == 0


def test_user_message_clears_unclaimed_continuations(wired, monkeypatch):
    tid, tmp_path = wired
    calls = []
    _wire_chat(monkeypatch, tid, calls)
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="stale plan",
                                       origin="continuation"))
    threads._process_message(tid, "new user direction", origin=None)
    assert MESSAGE_BACKLOG.for_thread(tid) == []
    kinds = [e["kind"] for e in read_events(str(tmp_path / tid))]
    assert "continuation_cancelled" in kinds


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
    threads._process_message(tid, "bg job", backlog_id=rec.id,
                             origin="continuation")
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
    threads._process_message(tid, "bg job", backlog_id=rec.id,
                             origin="continuation")
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
    MESSAGE_BACKLOG.add(PendingMessage(thread_id=tid, text="find tire sizes",
                                       origin="continuation"))
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
