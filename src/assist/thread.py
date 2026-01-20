import os
import re
import requests
import time
from urllib.parse import urlparse, parse_qs, unquote
from typing import Literal, Dict, Any, List
from pprint import pformat

from langchain.messages import HumanMessage, AIMessage, ToolMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.language_models.chat_models import BaseChatModel

from assist.promptable import base_prompt_for
from assist.model_manager import select_chat_model
from datetime import datetime

from assist.agent import create_research_agent, create_agent
from assist.domain_manager import DomainManager

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
                 domain: str | None = None,
                 thread_id: str | None = None,
                 checkpointer=None,
                 model: BaseChatModel | None = None):
        self.working_dir = working_dir
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.thread_id = thread_id or f"{working_dir}:{ts}"
        self.model = model or select_chat_model("mistral-nemo", 0.1)
        self.domain_manager = DomainManager(working_dir,
                                            domain)
        self.agent = create_agent(self.model,
                                  working_dir=self.domain_manager.domain(),
                                  checkpointer=checkpointer)

    def message(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # Continue the thread by sending only the latest human message; prior state is in the checkpointer.
        resp = self.agent.invoke({"messages": [{"role": "user", "content": text}]},
                                 {"configurable": {"thread_id": self.thread_id}})
        # The agent appends the AI reply to the persisted messages channel; return the last assistant content.
        # But callers should read final_report.md via get_messages for the final answer.
        return resp["messages"][-1].content

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
        # If a there were any changes, then summarize them at the end
        changes = self.domain_manager.changes()
        if changes:
            changes_content = "\n".join([f"{c.path}\n{c.diff}\n" for c in changes])
            return msgs + [{"role": "diff",
                            "content": changes_content}]
        else:
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
        desc_path = os.path.join(self.working_dir, "description.txt")
        try:
            if os.path.exists(desc_path):
                with open(desc_path, "r", encoding="utf-8") as f:
                    cached = f.read().strip()
                    if cached:
                        return cached
        except Exception:
            pass

        msgs = self.get_messages()
        if not msgs:
            raise ValueError("no messages to describe")
        prompt = {
            "role": "system",
            "content": base_prompt_for("deepagents/describe_system.md.j2"),
        }
        resp = self.model.invoke([prompt] + msgs)
        desc = resp.content.strip()

        try:
            os.makedirs(self.working_dir, exist_ok=True)
            with open(desc_path, "w", encoding="utf-8") as f:
                f.write(desc)
        except Exception:
            pass

        return desc


class ThreadManager:
    """Manage DeepAgentsThread instances persisted under a directory tree.

    At the root directory, a sqlite DB named 'threads.db' is used for LangGraph
    checkpointing via SqliteSaver.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
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

    def get(self, thread_id: str) -> Thread:
        tdir = os.path.join(self.root_dir, thread_id)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"thread directory not found: {thread_id}")
        return Thread(tdir, thread_id=thread_id, checkpointer=self.checkpointer, model=self.model)

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

    def new(self, domain: str) -> Thread:
        # Derive a clean ID for directory: prefer timestamp+rand
        tid = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + os.urandom(4).hex()
        tdir = os.path.join(self.root_dir, tid)
        os.makedirs(tdir, exist_ok=True)
        return Thread(tdir, domain, thread_id=tid, checkpointer=self.checkpointer, model=self.model)

    def close(self) -> None:
        try:
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
