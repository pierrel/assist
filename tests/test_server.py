from unittest import TestCase
from unittest.mock import patch
import asyncio
import json
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from fastapi.responses import StreamingResponse
from .utils import make_test_agent
from assist import server  # under test


class TestServer(TestCase):
    def test_simple_work_messages(self):
        msgs = [
            HumanMessage(content="What is 2 plus 2?"),
            AIMessage(content="thinking"),
            AIMessage(content="4"),
        ]
        agent = make_test_agent([msgs])
        resp = agent.invoke({"messages": [msgs[0]]})
        work_messages = server.work_messages(resp["messages"])

        work_message_types = [type(m) for m in work_messages]
        self.assertListEqual(work_message_types, [AIMessage])
        self.assertEqual(len(work_messages), 1)

    def test_tool_work_messages(self):
        msgs = [
            HumanMessage(content="What is the size of the capital of Colorado in the united states?"),
            AIMessage(content="I'll look that up"),
            ToolMessage(content="search result", tool_call_id="1"),
            AIMessage(content="The area is about 154 square kilometers"),
            AIMessage(content="final answer"),
        ]
        agent = make_test_agent([msgs])
        resp = agent.invoke({"messages": [msgs[0]]})
        work_messages = server.work_messages(resp["messages"])
        work_message_types = [type(m) for m in work_messages]
        self.assertListEqual(work_message_types, [AIMessage, ToolMessage, AIMessage])
        self.assertEqual(len(work_messages), 3)

    def test_streaming_response(self):
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ]
        agent = make_test_agent([msgs])
        req = server.ChatCompletionRequest(
            model="test-model",
            messages=[server.ChatMessage(role="user", content="Hello")],
            stream=True,
        )
        with patch("assist.server.get_agent", return_value=agent):
            resp = server.chat_completions(req)
        self.assertIsInstance(resp, StreamingResponse)

        async def _collect(gen):
            return [chunk async for chunk in gen]

        chunks = list(asyncio.run(_collect(resp.body_iterator)))
        events = []
        for chunk in chunks:
            self.assertTrue(chunk.startswith("data:"))
            payload = chunk[len("data: "):].strip()
            if payload == "[DONE]":
                break
            events.append(json.loads(payload))

        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        content = "".join(
            e["choices"][0]["delta"].get("content", "") for e in events[1:-1]
        )
        self.assertEqual(content, "Hi")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
