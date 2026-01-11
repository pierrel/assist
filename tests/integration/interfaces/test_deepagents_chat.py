import os
import time
import pytest

from assist.agent import Thread

@pytest.mark.integration
def test_deepagents_chat_basic():
    # Ensure environment key is present for real execution (may fail if missing)
    chat = Thread("/")
    reply = chat.message("Hello! Please introduce yourself briefly.")
    assert isinstance(reply, str) and reply.strip() != ""
    msgs = chat.get_messages()
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "user"
    assert msgs[-1]["role"] == "assistant"
