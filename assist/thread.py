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

        self.agent = create_agent(self.model,
                                  working_dir=working_dir,
                                  checkpointer=checkpointer,
                                  sandbox_backend=sandbox_backend)

    def message(self, text: str) -> str:
        """Continue the thread and return the last response"""
        result = invoke_with_rollback(
            self.agent,
            {"messages": [{"role": "user", "content": text}]},
            {
                "configurable": {"thread_id": self.thread_id},
                "max_concurrency": self.max_concurrency
            },
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
            {
                "configurable": {"thread_id": self.thread_id},
                "max_concurrency": self.max_concurrency
            },
            stream_mode=["messages", "updates"]
        )

    def get_messages(self) -> list[dict]:
        """Return user/assistant messages from checkpointer state as role/content dicts."""
        state = self.agent.get_state({"configurable": {"thread_id": self.thread_id}})
        msgs = []
        for m in state.values.get("messages", []):
            if isinstance(m, HumanMessage):
                msgs.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                calls = getattr(m, "tool_calls", None)
                if calls:
                    msgs.append({"role": "tools",
                                 "content": render_tool_calls(m)})
                elif m.content:
                    msgs.append({"role": "assistant", "content": m.content})
        return msgs


    def get_raw_messages(self) -> List[AnyMessage]:
        state = self.agent.get_state({"configurable": {"thread_id": self.thread_id}})
        return state.values.get("messages", [])

    
    def description(self) -> str:
        """Return a short (<=5 words) description of the conversation so far.

        Uses the underlying chat model directly. Raises ValueError if there
        are no messages yet.
        If description.txt exists in the thread directory, return it; otherwise compute and cache.
        """
        msgs = self.get_messages()
        if not msgs:
            raise ValueError("no messages to describe")

        # Filter out "tools" role messages - LangChain doesn't recognize this role
        # Only include user/assistant messages for description generation
        filtered_msgs = [m for m in msgs if m.get("role") in ("user", "assistant")]

        if not filtered_msgs:
            raise ValueError("no user/assistant messages to describe")

        prompt = {
            "role": "system",
            "content": base_prompt_for("deepagents/describe_system.md.j2"),
        }
        request = {
            "role": "user",
            "content": "Describe the conversation up until now",
        }
        resp = self.model.invoke([prompt] + filtered_msgs + [request])
        desc = resp.content.strip()

        return desc


class ThreadManager:
    """Manage DeepAgentsThread instances persisted under a directory tree.

    At the root directory, a sqlite DB named 'threads.db' is used for LangGraph
n    checkpointing via SqliteSaver.
    """

    DEFAULT_THREAD_WORKING_DIRECTORY = "domain"

    def __init__(self, root_dir: str | None = None):
        if root_dir:
            self.root_dir = root_dir
        else:
            self.root_dir = tempfile.mkdtemp()
            
        os.makedirs(self.root_dir, exist_ok=True)
        self.db_path = os.path.join(self.root_dir, "threads.db")
        # Ensure DB file exists upfront
        if not os.path.exists(self.db_path):
            open(self.db_path, "a").close()
        # SqliteSaver expects a sqlite3.Connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        # Create and reuse one chat model for all threads
        self.model = select_chat_model("mistral-nemo", 0.1)

    def list(self) -> list[str]:
        return [name for name in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, name)) and name != "__pycache__"]

    def get(self,
            thread_id: str,
            working_dir: str | None = None,
            sandbox_backend=None) -> Thread:
        tdir = os.path.join(self.root_dir, thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"thread directory not found: {thread_id}, {tdir}")
        if not working_dir:
            working_dir = self.make_default_working_dir(tdir)

        return Thread(working_dir,
                      thread_id=thread_id,
                      checkpointer=self.checkpointer,
                      model=self.model,
                      sandbox_backend=sandbox_backend)

    def remove(self, thread_id: str) -> None:
        tdir = os.path.join(self.root_dir, thread_id)
        if os.path.isdir(tdir):
            # Best-effort delete
            for root, dirs, files in os.walk(tdir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(tdir)
            except Exception:
                pass

    def new(self, working_dir: str|None = None, sandbox_backend=None) -> Thread:
        # Derive a clean ID for directory: prefer timestamp+rand
        tid = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + os.urandom(4).hex()
        tdir = os.path.join(self.root_dir, tid)
        os.makedirs(tdir, exist_ok=True)
        if not working_dir:
            working_dir = self.make_default_working_dir(tdir)

        return Thread(working_dir, thread_id=tid, checkpointer=self.checkpointer,
                      model=self.model, sandbox_backend=sandbox_backend)

    def close(self) -> None:
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass

    def thread_dir(self, tid: str) -> str:
        return os.path.join(self.root_dir, tid)

    def thread_default_working_dir(self, tid: str) -> str:
        return os.path.join(os.path.join(self.root_dir, tid),
                            self.DEFAULT_THREAD_WORKING_DIRECTORY)

    def make_default_working_dir(self, tdir: str) -> str:
        wdir = self.DEFAULT_THREAD_WORKING_DIRECTORY
        working_dir = os.path.join(tdir, wdir)
        os.makedirs(working_dir, exist_ok=True)
        
        return working_dir




    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
