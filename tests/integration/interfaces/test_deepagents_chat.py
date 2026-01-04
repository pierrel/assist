import os
import time
import pytest

from assist.deepagents_agent import DeepAgentsThread

@pytest.mark.integration
def test_deepagents_chat_basic():
    # Ensure environment key is present for real execution (may fail if missing)
    assert os.getenv("TAVILY_API_KEY") is not None, "TAVILY_API_KEY must be set for integration"
    chat = DeepAgentsThread("/")
    reply = chat.message("Hello! Please introduce yourself briefly.")
    assert isinstance(reply, str) and reply.strip() != ""
    msgs = chat.get_messages()
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
