from unittest.mock import patch

from assist.tools import filesystem
from langchain_core.messages import AIMessage


class DummyLLM:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def invoke(self, messages, *args, **kwargs):  # pragma: no cover - trivial
        resp = self._responses[self.calls]
        self.calls += 1
        return AIMessage(content=resp)


def test_short_file_returns_content(tmp_path):
    file = tmp_path / "short.txt"
    file.write_text("hello")

    dummy = DummyLLM([])
    with patch("assist.study_agent.select_chat_model", return_value=dummy), \
         patch("assist.study_agent.get_context_limit", return_value=1000):
        out = filesystem.read_file.invoke({"absolute_path": str(file)})

    assert out == "hello"
    assert dummy.calls == 0


def test_long_file_uses_study_agent(tmp_path):
    file = tmp_path / "long.txt"
    file.write_text("a" * 50)

    dummy = DummyLLM(["s1", "s2", "s3", "s4"])
    with patch("assist.study_agent.select_chat_model", return_value=dummy), \
         patch("assist.study_agent.get_context_limit", return_value=20):
        out = filesystem.read_file.invoke({"absolute_path": str(file), "task": "t", "request": "r"})

    assert out == "s4"
    assert dummy.calls == 4


def test_relative_path_raises(tmp_path):
    file = tmp_path / "rel.txt"
    file.write_text("hi")
    rel = file.relative_to(tmp_path)
    try:
        filesystem.read_file.invoke({"absolute_path": str(rel)})
    except ValueError as e:
        assert "absolute" in str(e)
    else:  # pragma: no cover - ensure exception
        assert False, "Expected ValueError for relative path"

