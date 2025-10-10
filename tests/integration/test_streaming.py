import json
import time
from threading import Thread
from unittest.mock import patch

import pytest
import requests
import uvicorn
from langchain_core.messages import AIMessage, HumanMessage

from assist import server
from tests.utils import make_test_agent


@pytest.fixture(scope="module")
def run_server():
    """Start the FastAPI server in a background thread and stop it after tests."""
    port = 5001

    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="info")
    srv = uvicorn.Server(config)
    thread = Thread(target=srv.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    timeout = time.time() + 5
    while True:
        try:
            requests.get(base_url)
            break
        except Exception:
            if time.time() > timeout:
                raise RuntimeError("Server failed to start")
            time.sleep(0.1)

    yield base_url

    srv.should_exit = True
    thread.join()

def test_streaming_agent():
    agent, plan_llm = server.get_agent(0.2)

    payload = {"messages": [
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
    ]}
    saved = []
    for ch, metadata in agent.stream(payload,
                                     stream_mode="messages"):
        saved.append(ch)

    assert len(saved) > 20


def test_streaming_chat_completions(run_server):
    url = f"{run_server}/chat/completions"
    payload = {
        "model": "qwen3:8b",
        "messages": [
            {
                "role": "user",
                "content": "What kinds of things can you help me with?",
            }
        ],
        "stream": True,
    }
    with requests.post(url, json=payload, stream=True) as resp:
        print("Sent request...")
        events = []
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                assert line.startswith("data:")
                data = line[len("data: ") :]
                print(data)
                if data == "[DONE]":
                    break
                events.append(json.loads(data))

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events[1:-1]
    )
    print(content)
    print(f'Totla events: {len(events)}')
    assert len(events) > 20
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
