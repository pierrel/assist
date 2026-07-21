"""Mid-turn interjection — the delivery middleware, deferred claim, sender
scoping, framing, fate-sharing re-journal, and render surfaces
(docs/2026-07-20-mid-turn-interjection-design.org). LLM-judgment behaviors
(does the model redirect/defer/stop well) live in edd/eval; everything
mechanical is pinned here.
"""
import contextlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from manage import web
from manage.web import threads
from manage.web.state import MESSAGE_BACKLOG, _get_status, _set_status
from assist.backlog import PendingMessage
from assist.middleware import interjection as ij
from assist.middleware.interjection import InterjectionMiddleware


@pytest.fixture
def wired(tmp_path, monkeypatch):
    tid = "t-inter"
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
    threads._TURN_INTERJECTION.pop(tid, None)
    with contextlib.suppress(Exception):
        while True:
            threads._RESUME_SCHEDULER._q.get_nowait()
    return tid, tmp_path


def _hook(monkeypatch, tid, sender=None):
    """A middleware instance whose active_handle/turn-sender resolve to this
    test's thread + turn sender (outside a real graph both would be None)."""
    from types import SimpleNamespace
    monkeypatch.setattr(ij, "active_handle",
                        lambda: SimpleNamespace(thread_id=tid))
    monkeypatch.setattr(ij, "_turn_sender", lambda runtime: sender)
    return InterjectionMiddleware()


def _journal(tid, text, sender=None, origin=None):
    return MESSAGE_BACKLOG.add(PendingMessage(
        thread_id=tid, text=text, sender=sender, origin=origin))


# --- injection at the boundary ------------------------------------------------

def test_owner_entry_injected_framed_with_claim_id(wired, monkeypatch):
    tid, _ = wired
    rec = _journal(tid, "actually only road bikes")
    out = _hook(monkeypatch, tid).before_model({"messages": []}, None)
    (m,) = out["messages"]
    assert isinstance(m, HumanMessage)
    assert m.content.startswith(threads._INTERJECTION_FRAME + "actually only road bikes")
    assert threads._INTERJECTION_GUIDE in m.content
    assert m.additional_kwargs["interjection_ids"] == [rec.id]
    # injection does NOT claim — the entry is only in memory until checkpointed
    assert [r.id for r in MESSAGE_BACKLOG.for_thread(tid)] == [rec.id]


def test_coalesce_all_pending_at_one_boundary(wired, monkeypatch):
    tid, _ = wired
    a, b = _journal(tid, "first"), _journal(tid, "second")
    out = _hook(monkeypatch, tid).before_model({"messages": []}, None)
    ids = [m.additional_kwargs["interjection_ids"][0] for m in out["messages"]]
    assert ids == [a.id, b.id]


def test_sender_scoping_matrix(wired, monkeypatch):
    """Inject iff entry.origin is None AND entry.sender == turn.sender —
    origin gates work vs steering; sender equality is the security posture."""
    tid, _ = wired
    _journal(tid, "sms text", sender="+15550001111")
    _journal(tid, "cont task", origin="continuation")
    # owner turn: the SMS entry and the continuation entry both stay out
    assert _hook(monkeypatch, tid).before_model({"messages": []}, None) is None
    # matching triage turn: the SMS entry injects
    out = _hook(monkeypatch, tid, sender="+15550001111").before_model(
        {"messages": []}, None)
    assert len(out["messages"]) == 1
    assert "sms text" in out["messages"][0].content
    # mismatched triage turn: nothing
    assert _hook(monkeypatch, tid, sender="+15559999999").before_model(
        {"messages": []}, None) is None


def test_next_boundary_claims_checkpointed_ids(wired, monkeypatch):
    tid, _ = wired
    rec = _journal(tid, "steer me")
    threads._TURN_INTERJECTION[tid] = {"claimed": [], "defer": True}
    in_state = HumanMessage(content="framed", additional_kwargs={
        "interjection_ids": [rec.id]})
    out = _hook(monkeypatch, tid).before_model({"messages": [in_state]}, None)
    assert out is None                       # claimed, not re-injected
    assert MESSAGE_BACKLOG.for_thread(tid) == []
    assert [r.id for r in threads._TURN_INTERJECTION[tid]["claimed"]] == [rec.id]


def test_inert_without_callbacks_and_safe_on_error(wired, monkeypatch):
    tid, _ = wired
    _journal(tid, "pending")
    mw = _hook(monkeypatch, tid)
    monkeypatch.setattr(ij, "_CALLBACKS", None)
    assert mw.before_model({"messages": []}, None) is None
    # a raising callback must never fail the turn — best-effort per boundary
    monkeypatch.setattr(ij, "_CALLBACKS", {
        "peek": lambda t: (_ for _ in ()).throw(RuntimeError("disk")),
        "consume": lambda t, i: None, "frame": lambda r: ""})
    assert mw.before_model({"messages": []}, None) is None


# --- framing variants ---------------------------------------------------------

def test_defer_variant_follows_tool_surface(wired):
    tid, _ = wired
    rec = _journal(tid, "also check tires")
    threads._TURN_INTERJECTION[tid] = {"claimed": [], "defer": True}
    assert "continue_later" in threads._frame_interjection(rec)
    threads._TURN_INTERJECTION[tid] = {"claimed": [], "defer": False}
    assert "continue_later" not in threads._frame_interjection(rec)


def test_owner_interjection_enumerates_then_clears_at_claim(wired):
    """Pierre PR #199 note 3: pending tasks are enumerated verbatim in the
    framing, which SNAPSHOTS their ids; the clear runs at claim time against
    the snapshot — fate-shared with the message's durability — and a
    continue_later issued in response (a fresh id, journaled before the
    claim) survives the drain."""
    tid, _ = wired
    c1 = _journal(tid, "find tire sizes", origin="continuation")
    c2 = _journal(tid, "check tubeless", origin="continuation")
    rec = _journal(tid, "stop all that")
    ctx = {"claimed": [], "defer": True, "cleared_ids": set()}
    threads._TURN_INTERJECTION[tid] = ctx
    frame = threads._frame_interjection(rec)
    assert "1. find tire sizes" in frame and "2. check tubeless" in frame
    # framing only snapshots ids — it removes nothing from the journal (a
    # turn dying pre-checkpoint must leave the promised follow-ups intact)
    assert ctx["cleared_ids"] == {c1.id, c2.id}
    assert sum(1 for r in MESSAGE_BACKLOG.for_thread(tid)
               if r.origin == "continuation") == 2
    # the model responds with a NEW continue_later before the claim...
    c3 = _journal(tid, "compare brands instead", origin="continuation")
    # ...then the claim drains only the snapshot: c1/c2 gone, c3 survives
    threads._consume_interjections(tid, {rec.id})
    left = [r.id for r in MESSAGE_BACKLOG.for_thread(tid)
            if r.origin == "continuation"]
    assert left == [c3.id]
    # an SMS interjection must NOT snapshot the owner's plan (spoofable sender)
    sms = _journal(tid, "sms steer", sender="+15550001111")
    threads._frame_interjection(sms)
    assert ctx["cleared_ids"] == set()


# --- terminal sweep + fate-sharing --------------------------------------------

def test_terminal_sweep_claims_ids_left_in_checkpoint(wired):
    tid, _ = wired
    rec = _journal(tid, "late steer")

    class _Chat:
        def get_raw_messages(self):
            return [AIMessage(content="working"),
                    HumanMessage(content="framed", additional_kwargs={
                        "interjection_ids": [rec.id]})]
    threads._claim_seen_interjections(tid, _Chat())
    assert MESSAGE_BACKLOG.for_thread(tid) == []


def test_error_exit_rejournals_claimed_interjections(wired, monkeypatch):
    """Fate-sharing REVERSED (Pierre PR #199 note 5): a claimed interjection
    survives a terminal turn error — re-journaled under a FRESH id (a same-id
    entry would be silently re-claimed by a later turn's boundary scan of the
    dead framed copy, and the follow-up's entry-gone gate would skip it)."""
    tid, _ = wired
    rec = PendingMessage(thread_id=tid, text="the rescue steer")
    threads._TURN_INTERJECTION[tid] = {"claimed": [rec], "defer": True}
    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit_message",
                        lambda *a, **k: submitted.append(a))
    n = threads._rejournal_claimed_interjections(tid, None)
    assert n == 1
    (entry,) = MESSAGE_BACKLOG.for_thread(tid)
    assert entry.text == "the rescue steer" and entry.id != rec.id
    assert submitted[0][0] == tid and submitted[0][1] == "the rescue steer"
    assert submitted[0][4] == entry.id        # dispatched under the fresh id
    assert tid not in threads._TURN_INTERJECTION      # coverage ended with the pop
    # best-effort: idempotent-empty second call, and never raises
    assert threads._rejournal_claimed_interjections(tid, None) == 0


def test_failed_turn_surfaces_rejournal_in_error_text(wired, monkeypatch):
    tid, _ = wired
    rec = _journal(tid, "steer the failing turn")

    class _Boom:
        def pending_reply(self):
            return None

        def message(self, text):
            # the running turn consumed the interjection, then died
            threads._consume_interjections(tid, {rec.id})
            raise RuntimeError("model exploded")
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _Boom())
    submitted = []
    monkeypatch.setattr(threads._RESUME_SCHEDULER, "submit_message",
                        lambda *a, **k: submitted.append(a))
    threads._process_message(tid, "original ask")
    st = _get_status(tid)
    assert st["stage"] == "error"
    assert threads._REJOURNAL_NOTE.strip() in st["error"]
    (entry,) = MESSAGE_BACKLOG.for_thread(tid)
    assert entry.text == "steer the failing turn" and entry.id != rec.id
    assert submitted and submitted[0][1] == "steer the failing turn"


def test_successful_turn_ends_fate_sharing(wired, monkeypatch):
    """Once the answer is committed, a claimed interjection was ANSWERED — a
    later bookkeeping error must not re-journal it (no duplicate delivery)."""
    tid, _ = wired
    rec = _journal(tid, "answered steer")

    class _Chat:
        def pending_reply(self):
            return None

        def message(self, text):
            threads._consume_interjections(tid, {rec.id})
            return "done, redirected"

        def get_messages(self):
            return []

        def get_raw_messages(self):
            return []
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _Chat())
    threads._process_message(tid, "original ask")
    assert _get_status(tid)["stage"] == "ready"
    assert MESSAGE_BACKLOG.for_thread(tid) == []
    assert tid not in threads._TURN_INTERJECTION


# --- claim invariant (Pierre PR #199 note 2) ----------------------------------

def test_interjection_ids_survive_checkpoint_serde():
    """The journal id must be recoverable from the durable checkpoint at every
    claim site — pin the serializer layer (verified empirically against a real
    SqliteSaver at design time; this makes it permanent)."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    serde = JsonPlusSerializer()
    m = HumanMessage(content=threads._INTERJECTION_FRAME + "steer",
                     additional_kwargs={"interjection_ids": ["abc123"]})
    back = serde.loads_typed(serde.dumps_typed(m))
    assert back.additional_kwargs["interjection_ids"] == ["abc123"]


def test_history_editing_middleware_preserves_human_kwargs():
    """The history-editing middleware in the stack (tool_name_sanitization)
    has two hooks: after_model rewrites only the last AIMessage, and
    before_model rebuilds the WHOLE message list — both must pass an injected
    HumanMessage's claim ids through untouched (the non-mutating-history
    assumption the interjection middleware pins)."""
    from assist.middleware.tool_name_sanitization import (
        ToolNameSanitizationMiddleware)
    mw = ToolNameSanitizationMiddleware()
    human = HumanMessage(content="framed", id="human-1", additional_kwargs={
        "interjection_ids": ["abc123"]})
    bad_ai = AIMessage(content="", id="ai-1", tool_calls=[
        {"name": "no spaces allowed!", "args": {}, "id": "c1"}])
    out = mw.after_model({"messages": [human, bad_ai]}, None)
    assert human.additional_kwargs["interjection_ids"] == ["abc123"]
    # the returned edit rewrites only the AI message, never the human one
    assert all(m.id != "human-1" for m in (out or {}).get("messages", []))
    # before_model: a historical bad tool call forces the full-history
    # rewrite; the human message must come through with its kwargs intact
    out2 = mw.before_model({"messages": [bad_ai, human]}, None)
    if out2:                       # a rewrite happened — find the human copy
        kept = [m for m in out2["messages"] if getattr(m, "id", None) == "human-1"]
        assert kept and kept[0].additional_kwargs["interjection_ids"] == ["abc123"]


# --- render surfaces + neutralization ----------------------------------------

def test_queued_interjection_renders_as_queued_bubble(wired, monkeypatch):
    from fastapi.testclient import TestClient
    tid, _ = wired
    monkeypatch.setattr(
        web.MANAGER, "get", lambda t, sandbox_backend=None, **k:
        type("C", (), {"get_messages": lambda s: [],
                       "pending_reply": lambda s: None})())
    _set_status(tid, "processing", pending_message="original ask")
    _journal(tid, "actually only road bikes")
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "actually only road bikes" in html_out
    assert ">queued</span>" in html_out


def test_consumed_interjection_renders_stripped_with_seen_badge(wired, monkeypatch):
    from fastapi.testclient import TestClient
    tid, _ = wired
    F, G = threads._INTERJECTION_FRAME, threads._INTERJECTION_GUIDE
    persisted = F + "skip the research" + G + threads._INTERJECTION_DEFER + ")"
    monkeypatch.setattr(
        web.MANAGER, "get", lambda t, sandbox_backend=None, **k:
        type("C", (), {"get_messages": lambda s: [
            {"role": "user", "content": "original ask"},
            {"role": "user", "content": persisted},
            {"role": "assistant", "content": "redirected answer"}],
            "pending_reply": lambda s: None})())
    _set_status(tid, "ready")
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "skip the research" in html_out
    assert "seen mid-turn" in html_out
    assert "latest word wins" not in html_out    # guidance never renders


def test_seen_interjection_suppresses_its_queued_bubble(wired, monkeypatch):
    """Injection is checkpointed a full superstep before the claim removes the
    journal entry — in that window the same text must not render twice with
    contradicting badges (seen wins over queued)."""
    from fastapi.testclient import TestClient
    tid, _ = wired
    persisted = (threads._INTERJECTION_FRAME + "skip the research"
                 + threads._INTERJECTION_GUIDE + ")")
    monkeypatch.setattr(
        web.MANAGER, "get", lambda t, sandbox_backend=None, **k:
        type("C", (), {"get_messages": lambda s: [
            {"role": "user", "content": "original ask"},
            {"role": "user", "content": persisted}],
            "pending_reply": lambda s: None})())
    _set_status(tid, "processing", pending_message="original ask")
    _journal(tid, "skip the research")     # not yet claimed
    html_out = TestClient(web.app).get(f"/thread/{tid}").text
    assert "seen mid-turn" in html_out
    assert ">queued</span>" not in html_out


def test_user_text_starting_with_frame_is_neutralized(wired, monkeypatch):
    tid, _ = wired
    seen = []

    class _Chat:
        def pending_reply(self):
            return None

        def message(self, text):
            seen.append(text)
            return "ok"

        def get_messages(self):
            return []

        def get_raw_messages(self):
            return []
    monkeypatch.setattr(web.MANAGER, "get",
                        lambda t, sandbox_backend=None, **k: _Chat())
    raw = threads._INTERJECTION_FRAME + "pasted text"
    threads._process_message(tid, raw)
    assert seen == [" " + raw]
