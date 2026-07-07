"""The non-response (tool-call) turns collapse into a subtle <details>; the human
and AI messages keep their existing flat-bubble treatment.  Symptom-level — assert
against the rendered HTML.  No model/GPU.
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
    def __init__(self, msgs):
        self._msgs = msgs

    def get_messages(self):
        return self._msgs


def _render(root, msgs):
    os.makedirs(root / "t1", exist_ok=True)
    state.DESCRIPTION_CACHE["t1"] = "T"
    return render_thread("t1", _Chat(msgs))


def test_tools_turn_is_collapsed_details(threads_root):
    html = _render(threads_root, [
        {"role": "tools", "content": "Calling read_url with {'url': 'x'} -- Calling grep with {'pattern': 'y'}"},
    ])
    # a subtle collapsed <details> (no `open` attr → collapsed by default)
    assert '<details class="msg tools">' in html
    assert "<summary>read_url, grep</summary>" in html   # tool names, legible without expanding
    assert '<details class="msg tools" open' not in html


def test_human_and_ai_messages_untouched(threads_root):
    html = _render(threads_root, [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "the answer"},
    ])
    # still flat bubbles with their role label — NOT wrapped in <details>
    assert '<div class="msg user">' in html and '<div class="msg assistant">' in html
    assert '<details class="msg user"' not in html
    assert '<details class="msg assistant"' not in html


class TestToolsSummary:
    def test_names_deduped_in_order(self):
        assert _tools_summary("Calling grep with {} -- Calling read_url with {} -- Calling grep with {}") \
            == "grep, read_url"

    def test_subagent_task_call(self):
        assert _tools_summary("Calling subagent research with {'x': 1}") == "research"

    def test_fallback_when_unmatched(self):
        assert _tools_summary("some prose with no call") == "tool call"

    def test_arg_text_containing_calling_does_not_inject(self):
        # a write_file whose content mentions "Calling read_url ..." must not add it
        raw = "Calling write_file with {'content': 'Calling read_url with the URL'}"
        assert _tools_summary(raw) == "write_file"
