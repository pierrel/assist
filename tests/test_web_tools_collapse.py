"""The non-response (tool-call) turns collapse into a subtle <details>; the human
and AI messages keep their existing flat-bubble treatment.  Symptom-level — assert
against the rendered HTML.  No model/GPU.

The collapsed-turn summary names come from the STRUCTURED tool calls (the dict's
``names``, populated by ``_messages_to_dicts``), not parsed out of the rendered text —
so arg/prose content can't inject a spurious name.
"""
import os

import pytest

from manage import web
from manage.web import state
from manage.web.threads import render_thread, _tools_summary


@pytest.fixture
def threads_root(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    state.DESCRIPTION_CACHE.clear()
    yield tmp_path
    state.DESCRIPTION_CACHE.clear()


class _Chat:
    def __init__(self, msgs, web_msgs=None):
        self._msgs = msgs
        self._web_msgs = web_msgs or msgs

    def get_messages(self):
        return self._msgs

    def get_web_messages(self):
        return self._web_msgs


def _render(root, msgs, web_msgs=None):
    os.makedirs(root / "t1", exist_ok=True)
    state.DESCRIPTION_CACHE["t1"] = "T"
    return render_thread("t1", _Chat(msgs, web_msgs))


def test_tools_turn_is_collapsed_details(threads_root):
    html = _render(threads_root, [
        {"role": "tools", "names": ["read_url", "grep"],
         "content": "Calling read_url with {'url': 'x'} -- Calling grep with {'pattern': 'y'}"},
    ])
    # a subtle collapsed <details> (no `open` attr → collapsed by default)
    assert '<details class="msg tools">' in html
    assert "<summary>read_url, grep</summary>" in html   # tool names, legible without expanding
    assert '<details class="msg tools" open' not in html


def test_summary_from_structured_names_not_content(threads_root):
    # even if the rendered content mentions " -- Calling read_url", the summary is
    # driven by the structured names → no spurious name injected.
    html = _render(threads_root, [
        {"role": "tools", "names": ["write_file"],
         "content": "Calling write_file with {'content': 'x -- Calling read_url with y'}"},
    ])
    assert "<summary>write_file</summary>" in html
    assert "read_url" not in html.split("</summary>")[0]   # not in the summary


def test_human_and_ai_messages_untouched(threads_root):
    html = _render(threads_root, [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "the answer"},
    ])
    # still flat bubbles with their role label — NOT wrapped in <details>
    assert '<div class="msg user">' in html and '<div class="msg assistant">' in html
    assert '<details class="msg user"' not in html
    assert '<details class="msg assistant"' not in html


def test_tool_call_prose_stays_visible_without_expanding_activity(threads_root):
    from assist.thread import _messages_to_dicts
    from langchain_core.messages import AIMessage

    summary = "Bring the full medication and exercise summary."
    raw = [
        AIMessage(content=summary, tool_calls=[
            {"name": "write_todos", "args": {"todos": []}, "id": "call-1"},
        ]),
        AIMessage(content="Good luck."),
    ]
    ordinary = _messages_to_dicts(raw)
    web = _messages_to_dicts(raw, split_tool_call_content=True)

    html = _render(threads_root, ordinary, web)
    activity = html[html.index('<details class="msg tools">'):html.index("</details>")]
    assert '<div class="msg assistant">' in html
    assert summary in html and summary not in activity
    assert '<summary>write_todos</summary>' in activity
    assert '<details class="msg tools" open' not in html
    assert html.index("Good luck.") < html.index(summary) < html.index(activity)


class TestToolsSummary:
    def test_names_deduped_in_order(self):
        assert _tools_summary(["grep", "read_url", "grep"]) == "grep, read_url"

    def test_empty_names_fallback(self):
        assert _tools_summary([]) == "tool call"
        assert _tools_summary(None) == "tool call"
