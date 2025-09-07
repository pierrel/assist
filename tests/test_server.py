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

    def test_streaming_only_new_messages(self):
        class DummyAgent:
            def stream(self, args, stream_mode="messages"):
                messages = args["messages"]
                for m in messages + [AIMessage(content="new answer")]:
                    yield m, {}

        messages = [
            server.ChatMessage(role="user", content="hi"),
            server.ChatMessage(role="assistant", content="hello"),
            server.ChatMessage(role="user", content="how are you?"),
        ]

        with patch("assist.server.get_agent", return_value=DummyAgent()):
            req = server.ChatCompletionRequest(model="mock", messages=messages, stream=True)
            resp = server.chat_completions(req)

            async def collect(r: StreamingResponse):
                data = []
                async for chunk in r.body_iterator:
                    if isinstance(chunk, bytes):
                        data.append(chunk.decode())
                    else:
                        data.append(chunk)
                return "".join(data)

            stream = asyncio.run(collect(resp))

        self.assertIn("new answer", stream)
        self.assertNotIn("hello", stream)
