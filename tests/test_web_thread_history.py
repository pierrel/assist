"""Thread history windows and browser-local response-pin render contracts."""
import re

import pytest

from fastapi.testclient import TestClient

from manage.web import state
from manage.web import threads


def _turn(number: int) -> list[dict]:
    return [
        {"role": "user", "content": f"question {number}", "message_id": f"u{number}"},
        {"role": "tools", "content": "activity", "names": ["read_file"], "message_id": f"a{number}"},
        {"role": "assistant", "content": f"answer {number}", "message_id": f"a{number}"},
    ]


def test_history_windows_keep_complete_turns_and_use_an_anchored_cursor():
    messages = [message for number in range(1, 13) for message in _turn(number)]
    newest, cursor = threads._turn_page(messages)

    assert [message["content"] for message in newest if message["role"] == "user"] == [
        f"question {number}" for number in range(3, 13)
    ]
    assert cursor == "u3"
    assert newest[1]["role"] == "tools"  # activity stays with its user turn

    older, older_cursor = threads._turn_page(messages, cursor)
    assert [message["content"] for message in older if message["role"] == "user"] == [
        "question 1", "question 2"
    ]
    assert older_cursor is None

    # A newer turn cannot shift an anchored older cursor into a duplicate or gap.
    later = messages + _turn(13)
    anchored, _ = threads._turn_page(later, cursor)
    assert anchored == older


def test_history_cursor_rejects_unknown_or_malformed_identity():
    messages = _turn(1)
    for cursor in ("", "not allowed/", "missing"):
        try:
            threads._turn_page(messages, cursor)
        except ValueError:
            pass
        else:
            assert False, cursor


def test_history_fragment_marker_is_valid_inside_an_html_comment():
    marker = threads._history_marker()

    assert re.fullmatch(r"[0-9a-f]{48}", marker)
    assert "--" not in marker


@pytest.mark.parametrize("message_id", [None, "unsafe/value"])
def test_checkpoint_messages_without_a_safe_langchain_id_still_page(message_id):
    from assist.thread import _messages_to_dicts
    from langchain_core.messages import AIMessage, HumanMessage

    raw = [message for number in range(12) for message in (
        HumanMessage(content=f"question {number}", id=message_id),
        AIMessage(content=f"answer {number}", id=message_id),
    )]
    messages = _messages_to_dicts(raw, split_tool_call_content=True,
                                  include_message_ids=True)

    newest, cursor = threads._turn_page(messages)

    assert cursor == "message:4"
    assert [message["content"] for message in newest if message["role"] == "user"] == [
        f"question {number}" for number in range(2, 12)
    ]


class _Chat:
    def __init__(self, messages):
        self.messages = messages

    def get_messages(self):
        return self.messages

    def get_web_messages(self):
        return self.messages


def test_rendered_assistant_is_pinnable_but_tools_are_not(tmp_path, monkeypatch):
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(threads, "_thread_title", lambda tid: "Thread")
    state._set_status("t", "ready")

    html = threads.render_thread("t", _Chat(_turn(1)))

    assert 'data-response-id="a1"' in html
    assert 'data-response-id="u1"' not in html
    assert 'class="pin-response" type="button" data-response-id=' not in html
    assert html.count('>Pin</button>') == 1
    assert "assist:pins:" in html
    assert "You can pin up to 50 responses." in html
    assert "saveCaptures(items.slice(0, 20))" in html
    assert "function saveCaptures(items) { try { localStorage.setItem" in html
    assert "button.closest('.msg[data-response-id]')" in html
    assert "if (!writePins(ids)) { localStorage.removeItem(pinMarkupKey(id)); throw new Error('storage unavailable'); }" in html
    assert "if (!writePins(readPins().filter(function(item) { return item !== id; }))) throw new Error('storage unavailable');" in html
    assert 'id="loadMore"><button' not in html
    assert "overflow-x: clip" in html
    assert ".msg pre, .msg .content table" in html
    assert ".msg .content img, .msg .content video, .msg .content svg, .msg .content canvas" in html


def test_history_render_omits_current_busy_placeholder(tmp_path, monkeypatch):
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(threads, "_thread_title", lambda tid: "Thread")
    state._set_status("t", "processing", pending_message="new question")
    messages = [message for number in range(1, 13) for message in _turn(number)]

    html = threads.render_thread("t", _Chat(messages), history_before="u3",
                                 history_marker="fresh-marker")

    assert "question 1" in html and "question 2" in html
    assert "question 3" not in html
    assert 'class="msg assistant placeholder"' not in html
    assert "new question" not in html
    assert "<!-- fresh-marker-start -->" in html


def test_history_route_validates_cursor_and_returns_the_next_window(tmp_path, monkeypatch):
    (tmp_path / "t").mkdir()
    monkeypatch.setattr(state.MANAGER, "thread_dir", lambda tid: str(tmp_path / tid))
    monkeypatch.setattr(state.MANAGER, "get", lambda tid, sandbox_backend=None: _Chat(
        [message for number in range(1, 13) for message in _turn(number)]))
    monkeypatch.setattr(threads, "_thread_title", lambda tid: "Thread")
    diff_calls = []

    class _DiffManager:
        def main_diff(self):
            diff_calls.append(True)
            return []

    monkeypatch.setattr(threads, "_get_domain_manager", lambda tid: _DiffManager())
    state._set_status("t", "ready")
    client = TestClient(threads.app)

    response = client.get("/thread/t/history", params={"before": "u3"})

    assert response.status_code == 200
    assert "question 1" in response.json()["html"]
    assert "question 3" not in response.json()["html"]
    assert response.json()["before"] is None
    assert not diff_calls
    assert client.get("/thread/t/history", params={"before": "unknown"}).status_code == 404
    assert client.get("/thread/t/history", params={"before": "bad/value"}).status_code == 404
