import os
from typing import Literal

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from tavily import TavilyClient

from assist.promptable import base_prompt_for
from langgraph.graph.state import CompiledStateGraph

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
    """Create a DeepAgents-based agent with default tools plus HTTP and Tavily search.

    This wraps deepagents.graph.create_deep_agent and ensures the agent has the
    standard DeepAgents toolset along with:
      - http_request: generic HTTP tool
      - web_search: Tavily-powered web search
      - fetch_url: fetch and convert HTML to markdown

    Additional kwargs are forwarded to create_deep_agent (e.g., middleware, backend, etc.).
    """
    research_sub_agent = {
        "name": "research-agent",
        "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
        "system_prompt": base_prompt_for("deepagents/sub_research.txt.j2"),
        "tools": [internet_search],
    }
    
    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
        "system_prompt": base_prompt_for("deepagents/sub_critique.txt.j2"),
    }

    return create_deep_agent(
        model=model,
        tools=[internet_search],
        checkpointer=InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/research_instructions.txt.j2"),
        subagents=[critique_sub_agent, research_sub_agent],
    )
