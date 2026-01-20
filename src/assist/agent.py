from deepagents import create_deep_agent, CompiledSubAgent
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models.chat_models import BaseChatModel

from assist.promptable import base_prompt_for
from langgraph.graph.state import CompiledStateGraph
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelRetryMiddleware

from openai import InternalServerError

from assist.tools import read_url, search_internet

def create_agent(model: BaseChatModel,
                 working_dir: str,
                 checkpointer=None) -> CompiledStateGraph:
    retry_middle = ModelRetryMiddleware(max_retries=4,
                                        backoff_factor=1.5)
    mw = [retry_middle]
    
    research_sub = CompiledSubAgent(
        name="research-agent",
        description= "Used to conduct thorough research. The result of the research will be placed in a file and the file name/path will be returned. Provide a filename for more control.",
        runnable= create_research_agent(model,
                                        working_dir,
                                        checkpointer,
                                        mw)
    )

    fs = FilesystemBackend(root_dir=working_dir,
                           virtual_mode=True)

    return create_deep_agent(
        model=model,
        tools=[search_internet],
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/general_instructions.md.j2"),
        middleware=mw,
        backend=fs,
        subagents=[research_sub]
    )


def create_research_agent(model: BaseChatModel,
                          working_dir: str,
                          checkpointer=None,
                          middleware=[]) -> CompiledStateGraph:
    """Create a DeepAgents-based agent suitable for general-purpose research replies.

    Includes Tavily web search and a critique/research/fact-check subagent trio.
    """
    
    
    research_sub_agent = {
        "name": "research-agent",
        "description": "Used to research more in depth questions. Only give this researcher one topic at a time. It will return research results.",
        "system_prompt": base_prompt_for("deepagents/sub_research.txt.j2"),
        "tools": [search_internet, read_url],
    }

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Used to critique the final report. You MUST provide the file it should critique.",
        "system_prompt": base_prompt_for("deepagents/sub_critique.txt.j2"),
    }

    fact_check_sub_agent = {
        "name": "fact-check-agent",
        "description": "Used to check all references for alignment with claims and statements. You MUST provide the file it should fact-check.",
        "system_prompt": base_prompt_for("deepagents/fact_checker.md.j2"),
        "tools": [read_url],
    }

    
    fs = FilesystemBackend(root_dir=working_dir,
                           virtual_mode=True)
    return create_deep_agent(
        model=model,
        tools=[search_internet, read_url],
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/research_instructions.txt.j2"),
        backend=fs,
        middleware=middleware,
        subagents=[critique_sub_agent,
                   research_sub_agent,
                   fact_check_sub_agent]
    )
