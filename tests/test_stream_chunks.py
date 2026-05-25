"""Tests for ``assist.stream_chunks.unwrap_messages``.

Pins the real langgraph 1.2 ``Overwrite`` shape (imported, not faked) so the
test fails loudly if a future upgrade changes the wrapper.
"""
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Overwrite

from assist.stream_chunks import unwrap_messages


class TestUnwrapMessages:
    def test_bare_list_passes_through(self):
        m1, m2 = AIMessage(content="a"), AIMessage(content="b")
        assert unwrap_messages([m1, m2]) == [m1, m2]

    def test_tuple_becomes_list(self):
        m = AIMessage(content="a")
        assert unwrap_messages((m,)) == [m]

    def test_overwrite_is_unwrapped(self):
        # The crash shape: a node overwriting the messages channel.
        tm = ToolMessage(content="ok", tool_call_id="t1", name="apply_config")
        assert unwrap_messages(Overwrite(value=[tm])) == [tm]

    def test_overwrite_with_empty_value(self):
        assert unwrap_messages(Overwrite(value=[])) == []

    def test_none_returns_empty(self):
        assert unwrap_messages(None) == []

    def test_single_message_is_wrapped(self):
        m = AIMessage(content="lonely")
        assert unwrap_messages(m) == [m]

    def test_unknown_scalar_returns_empty_not_unwrapped(self):
        # A non-message object with a `.value` must NOT be silently unwrapped
        # (we key on `isinstance(Overwrite)`, not duck-typing on `.value`).
        class HasValue:
            value = "surprise"

        assert unwrap_messages(HasValue()) == []
        assert unwrap_messages(123) == []
        assert unwrap_messages("text") == []
