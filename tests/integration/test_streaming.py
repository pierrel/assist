import json
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, AIMessage
from tests.utils import make_test_agent
from assist import server


def test_streaming_chat_completions():
    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
    ]
    agent = make_test_agent([msgs])
    with patch("assist.server.get_agent", return_value=agent):
        client = TestClient(server.app)
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        with client.stream("POST", "/chat/completions", json=payload) as resp:
            events = []
            for line in resp.iter_lines():
                if line:
                    assert line.startswith("data:")
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    events.append(json.loads(data))
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events[1:-1]
    )
    assert content == "Hi"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
