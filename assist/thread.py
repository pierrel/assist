import os
import time
import tempfile
from datetime import datetime
from typing import Literal, Dict, Any, List, Iterator

import sqlite3
from langchain.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.language_models.chat_models import BaseChatModel

from assist.promptable import base_prompt_for
from assist.model_manager import select_chat_model
from assist.agent import create_research_agent, create_agent
from assist.checkpoint_rollback import invoke_with_rollback

def render_tool_calls(message: AIMessage) -> str:
    calls = getattr(message, "tool_calls", None)
    if calls:
        calls_str = " -- ".join(map(lambda c: render_tool_call(c), calls))
        if getattr(message, "content", None):
            return f"{calls_str} \n> {message.content}"

        return calls_str
    return ""


def render_tool_call(call: dict) -> str:
    name = call.get("name", "none")
    args = call.get("args", {})
    if name == "task" and call.get("args", None):
        subagent = args.get("subagent_type", "none")
        return f"Calling subagent {subagent} with {args}"
    else:
        return f"Calling {name} with {args}"

class Thread:
    """Reusable chat-like interface that mimics the CLI back-and-forth.

    Initialize with a working directory; it derives a thread id from cwd + timestamp,
    keeps a rolling messages list, and exposes a message() method that returns the
    assistant reply as a string.
    """

    def __init__(self,
                 working_dir: str,
                 thread_id: str | None = None,
                 checkpointer=None,
                 model: BaseChatModel | None = None,
                 max_concurrency: int = 5,
                 sandbox_backend=None):
        self.working_dir = working_dir
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.thread_id = thread_id or f"{working_dir}:{ts}"
        self.model = model or select_chat_model("mistral-nemo", 0.1)
        self.max_concurrency = max_concurrency
        self.runconfig = {
            "configurable": {"thread_id": self.thread_id},
            "max_concurrency": self.max_concurrency
        }

        self.agent = create_agent(self.model,
                                  working_dir=working_dir,
                                  checkpointer=checkpointer,
                                  sandbox_backend=sandbox_backend)

    def message(self, text: str) -> str:
        """Continue the thread and return the last response"""
        result = invoke_with_rollback(
            self.agent,
            {"messages": [{"role": "user", "content": text}]},
            self.runconfig,
        )
        # Extract content from the last AIMessage
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                return last_msg.content
        return ""

    def stream_message(self, text: str) -> Iterator[dict[str, Any] | Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # Continue the thread by sending only the latest human message; prior state is in the checkpointer.
        return self.agent.stream(
            {"messages": [{"role": "user", "content": text}]},
            self.runconfig,
            stream_mode=["messages", "updates"]
        )

    def get_messages(self) -> list[dict]:
        """Return user/assistant messages from checkpointer state as role/content dicts."""
        state = self.agent.get_state(self.runconfig)
        msgs = []
        for m in state.values.get("messages", []):
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                calls = getattr(m, "tool_calls", None)
                if calls:
                    msgs.append({"role": "assistant", "content": render_tool_calls(m)})
                else:
                    msgs.append({"role": "assistant", "content": m.content})
            else:
                msgs.append({"role": m.type, "content": m.content})
        return msgs

    def get_message_count(self) -> int:
        """Return the number of messages in the thread."""
        state = self.agent.get_state(self.runconfig)
        return len(state.values.get("messages", []))

    def clear_messages(self) -> None:
        """Clear all messages from the thread."""
        self.agent.update_state(self.runconfig, {"messages": []})

    def get_last_message(self) -> dict | None:
        """Return the last message in the thread."""
        messages = self.get_messages()
        return messages[-1] if messages else None

    def get_first_message(self) -> dict | None:
        """Return the first message in the thread."""
        messages = self.get_messages()
        return messages[0] if messages else None

    def get_conversation_history(self) -> list[dict]:
        """Return the full conversation history."""
        return self.get_messages()

    def get_thread_id(self) -> str:
        """Return the thread ID."""
        return self.thread_id

    def get_working_dir(self) -> str:
        """Return the working directory."""
        return self.working_dir

    def get_model(self) -> BaseChatModel:
        """Return the model used by this thread."""
        return self.model

    def get_runconfig(self) -> Dict[str, Any]:
        """Return the run configuration."""
        return self.runconfig

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the thread."""
        return self.agent.get_state(self.runconfig)

    def update_state(self, values: Dict[str, Any]) -> None:
        """Update the state of the thread."""
        self.agent.update_state(self.runconfig, values)