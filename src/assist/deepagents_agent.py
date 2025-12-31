import os
from typing import Literal

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from tavily import TavilyClient

from assist.promptable import base_prompt_for
from langgraph.graph.state import CompiledStateGraph
from datetime import datetime

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = ChatOpenAI(model="gpt-4o-mini")

def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance", "software", "shopping"] = "general",
        include_raw_content: bool = False,
):
    """Used to search the internet for information on a given topic using a query string."""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs


def deepagents_agent() -> CompiledStateGraph:
    """Create a DeepAgents-based agent suitable for general-purpose research replies.

    Includes Tavily web search and a critique/research subagent pair. The main agent
    should respond to the user with findings rather than only writing to files.
    """
    research_sub_agent = {
        "name": "research-agent",
        "description": "Used to research more in depth questions. Only give this researcher one topic at a time.",
        "system_prompt": base_prompt_for("deepagents/sub_research.txt.j2"),
        "tools": [internet_search],
    }

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Used to critique the final report.",
        "system_prompt": base_prompt_for("deepagents/sub_critique.txt.j2"),
    }

    return create_deep_agent(
        model=model,
        tools=[internet_search],
        checkpointer=InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/research_instructions.txt.j2"),
        subagents=[critique_sub_agent, research_sub_agent],
    )


class DeepAgentsChat:
    """Reusable chat-like interface that mimics the CLI back-and-forth.

    Initialize with a working directory; it derives a thread id from cwd + timestamp,
    keeps a rolling messages list, and exposes a message() method that returns the
    assistant reply as a string.
    """

    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.thread_id = f"{working_dir}:{ts}"
        self.messages = []
        self.agent = deepagents_agent()

    def message(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self.messages.append({"role": "user", "content": text})
        resp = self.agent.invoke({"messages": self.messages}, {"configurable": {"thread_id": self.thread_id}})
        content = resp["messages"][-1].content
        self.messages.append({"role": "assistant", "content": content})
        return content

    def get_messages(self) -> list[dict]:
        """Return all messages in this chat (role/content dicts)."""
        return list(self.messages)
