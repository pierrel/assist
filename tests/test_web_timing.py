"""Message-timing indicators (web UI): completed-turn elapsed badges + live WIP timer.

Symptom-level render tests (per CLAUDE.md): render the actual HTML and assert the badge
lands on the CORRECT assistant bubble, the live timer is present when busy, and a
pre-feature thread (no sidecar) renders with no badge and no crash. Plus the sidecar
read/write round-trip and the formatter boundaries.
"""
import pytest

from manage.web import state as st
from manage.web import threads as th
from manage.web.threads import _format_elapsed, _human_ordinal, render_thread


def test_format_elapsed_boundaries():
    assert _format_elapsed(0) == "0s"
    assert _format_elapsed(14) == "14s"
    assert _format_elapsed(59) == "59s"
    assert _format_elapsed(60) == "1m 0s"
    assert _format_elapsed(125) == "2m 5s"
    assert _format_elapsed(3600) == "1h 0m"
    assert _format_elapsed(3785) == "1h 3m"


def test_get_timings_missing_and_corrupt(tmp_path, monkeypatch):
    d = tmp_path / "t"
    d.mkdir()
    monkeypatch.setattr(st.MANAGER, "thread_dir", lambda tid: str(d))
    assert st._get_timings("t") == {}          # missing → {}
    (d / "turn_timings.json").write_text("{ not json")
    assert st._get_timings("t") == {}          # corrupt → {} (never raises)
    st._append_timing("t", 1, 14.4)
    st._append_timing("t", 2, 125.6)
    assert st._get_timings("t") == {"1": 14, "2": 126}  # rounded ints


def test_human_ordinal_counts_user_bubbles():
    msgs = [{"role": "user"}, {"role": "assistant"}, {"role": "tools"},
            {"role": "user"}, {"role": "assistant"}]
    assert _human_ordinal(msgs) == 2


class _FakeThread:
    def __init__(self, msgs):
        self._m = msgs

    def get_messages(self):
        return self._m


@pytest.fixture
def thread_env(tmp_path, monkeypatch):
    d = tmp_path / "t1"
    d.mkdir()
    monkeypatch.setattr(st.MANAGER, "thread_dir", lambda tid: str(d))
    # render_thread derives the title (would otherwise hit the model) — stub it.
    monkeypatch.setattr(th, "_thread_title", lambda tid: "Test thread")
    return "t1"


def _two_turns():
    # Turn 1: q1 → a1. Turn 2: q2 → (tool bubble) → a2 (the concluding reply).
    return [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "ANSWER-ONE"},
        {"role": "user", "content": "q2"},
        {"role": "tools", "content": "did a thing", "names": ["read_url"]},
        {"role": "assistant", "content": "ANSWER-TWO"},
    ]


def test_badge_lands_on_the_correct_assistant_bubble(thread_env):
    st._set_status("t1", "ready")
    st._append_timing("t1", 1, 14)
    st._append_timing("t1", 2, 125)
    html = render_thread("t1", _FakeThread(_two_turns()))
    assert html.count("elapsed-badge") == 2
    # Rendered newest-first: turn-2 bubble (2m 5s + ANSWER-TWO) before turn-1 (14s + ANSWER-ONE).
    # This ordering pins each badge to ITS turn's reply, not just "both appear somewhere".
    assert html.index("2m 5s") < html.index("ANSWER-TWO") < html.index("14s") < html.index("ANSWER-ONE")


def test_badge_ordinal_counts_a_review_submission_as_a_user_turn(thread_env):
    # A review submission renders as a "user" bubble (different render branch, same role),
    # so the ordinal count must include it and the badge must land on ITS reply — the
    # design's C1 "span a review submission" case.
    from manage.web.review import _REVIEW_HEADER
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "ANSWER-ONE"},
        {"role": "user", "content": _REVIEW_HEADER + "\nlooks good, ship it"},
        {"role": "assistant", "content": "ANSWER-REVIEW"},
    ]
    st._set_status("t1", "ready")
    st._append_timing("t1", 1, 10)
    st._append_timing("t1", 2, 200)  # the review turn is ordinal 2
    html = render_thread("t1", _FakeThread(msgs))
    assert html.count("elapsed-badge") == 2
    assert html.index("3m 20s") < html.index("ANSWER-REVIEW") < html.index("10s") < html.index("ANSWER-ONE")


def test_badge_only_on_last_assistant_when_a_turn_has_multiple(thread_env):
    # A single turn can produce >1 content-only assistant bubble (a retry / empty-response
    # recovery). Only the LAST (concluding) reply gets the badge — no double-badge.
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "FIRST-DRAFT"},
        {"role": "assistant", "content": "FINAL-ANSWER"},
    ]
    st._set_status("t1", "ready")
    st._append_timing("t1", 1, 30)
    html = render_thread("t1", _FakeThread(msgs))
    assert html.count("elapsed-badge") == 1                       # not two
    # Rendered newest-first: FINAL-ANSWER (with the 30s badge) then FIRST-DRAFT (no badge).
    assert html.index("30s") < html.index("FINAL-ANSWER") < html.index("FIRST-DRAFT")


def test_no_sidecar_renders_no_badge_and_does_not_crash(thread_env):
    st._set_status("t1", "ready")  # no _append_timing → no turn_timings.json
    html = render_thread("t1", _FakeThread(_two_turns()))
    assert "elapsed-badge" not in html
    assert "ANSWER-ONE" in html and "ANSWER-TWO" in html  # still renders the conversation


def test_busy_thread_shows_live_timer_with_baseline_and_ticker(thread_env, monkeypatch):
    # Freeze the clock so the baseline is deterministic (real wall-clock would let a slow
    # render floor to 6+ under load — the flaky assertion Copilot flagged).
    monkeypatch.setattr(th, "_now_ms", lambda: 1_000_000)
    st._set_status("t1", "processing", started_at=1_000_000 - 5000)  # exactly 5s ago
    html = render_thread("t1", _FakeThread(_two_turns()))
    assert 'class="elapsed"' in html            # the ticking span
    assert 'data-baseline="5"' in html          # server-computed baseline (deterministic)
    assert "setInterval" in html                # the JS ticker is included


def test_get_timings_ignores_valid_but_non_dict_json(tmp_path, monkeypatch):
    # A syntactically-valid but wrong-shaped file must degrade to {} (no badges), not
    # crash the on-loop render on `str(ord) in timings` / indexing.
    d = tmp_path / "t"
    d.mkdir()
    monkeypatch.setattr(st.MANAGER, "thread_dir", lambda tid: str(d))
    for bad in ("5", "true", "[]", '"oops"', "null"):
        (d / "turn_timings.json").write_text(bad)
        assert st._get_timings("t") == {}
    # A dict is fine, but non-numeric VALUES are dropped (they'd crash _format_elapsed):
    (d / "turn_timings.json").write_text('{"1": 14, "2": "oops", "3": null}')
    assert st._get_timings("t") == {"1": 14}


def test_busy_without_started_at_degrades_to_spinner_no_timer(thread_env):
    st._set_status("t1", "processing")          # pre-feature in-flight turn: no started_at
    html = render_thread("t1", _FakeThread(_two_turns()))
    assert "status-banner" in html and 'class="elapsed"' not in html
