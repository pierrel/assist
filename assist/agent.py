import uuid
import logging

from deepagents import create_deep_agent, CompiledSubAgent
from langchain.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.agents.middleware import ModelRetryMiddleware
from openai import InternalServerError

from assist.promptable import base_prompt_for
from assist.tools import read_url, search_internet
from assist.backends import create_composite_backend
from assist.middleware.model_logging_middleware import ModelLoggingMiddleware
from assist.middleware.json_validation_middleware import JsonValidationMiddleware
from assist.middleware.context_aware_tool_eviction import ContextAwareToolEvictionMiddleware


logger = logging.getLogger(__name__)


def _create_standard_backend(working_dir: str):
    """Create the standard composite backend with state exclusions.

    This backend excludes ephemeral files like question.txt and large_tool_results/
    from the stateful filesystem, using StateBackend instead.
    """
    return create_composite_backend(
        working_dir,
        ["/question.txt",
         "question.txt",
         "/large_tool_results/",
         "large_tool_results/",
         "large_tool_results"]
    )


class AgentHarness:
    """Makes it easier to have conversations"""
    
    def __init__(self, agent: CompiledStateGraph, thread_id: str | None = None):
        self.agent = agent
        self.thread_id = thread_id or uuid.uuid1()

    def message(self, text: str) -> AIMessage:
        resp = self.agent.invoke({"messages": [{"role": "user", "content": text}]},
                                 {"configurable": {"thread_id": self.thread_id}})
        return resp["messages"][-1].content

    def all_messages(self) -> list[AnyMessage]:
        state = self.agent.get_state({
            "configurable": {"thread_id": self.thread_id}
        })
        return state.values.get("messages", [])
        



def create_agent(model: BaseChatModel,
                 working_dir: str,
                 checkpointer=None) -> CompiledStateGraph:
    # Core middleware: retry, tool call limiting, JSON validation, and logging
    retry_middle = ModelRetryMiddleware(max_retries=6,
                                        backoff_factor=2)
    # Validate and fix JSON in tool call arguments
    json_validation_mw = JsonValidationMiddleware(strict=False)
    logging_mw = ModelLoggingMiddleware("general-agent")

    # Context-aware tool eviction: evict results to filesystem if they would cause overflow
    context_eviction_mw = ContextAwareToolEvictionMiddleware(
        trigger_fraction=0.75,  # Evict if context would reach 75%
    )

    mw = [retry_middle, json_validation_mw, context_eviction_mw]

    backend = _create_standard_backend(working_dir)

    research_sub = CompiledSubAgent(
        name="research-agent",
        description= "Used to conduct thorough research. The result of the research will be placed in a file and the file name/path will be returned. Provide a filename for more control.",
        runnable=create_research_agent(model,
                                       working_dir,
                                       checkpointer,
                                       [retry_middle, json_validation_mw])
    )

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/general_instructions.md.j2"),
        middleware=mw + [logging_mw],
        backend=backend,
        subagents=[research_sub]
    )

    # ContextAwareToolEvictionMiddleware handles context overflow prevention
    # by monitoring cumulative context usage. FilesystemMiddleware (built-in)
    # provides backup eviction for extremely large individual results (>20k tokens default).

    return agent

def create_context_agent(model: BaseChatModel,
                         working_dir: str,
                         checkpointer=None,
                         middleware=[]) -> CompiledStateGraph:
    # Only add JSON validation if not already provided
    has_json_validation = any(isinstance(m, JsonValidationMiddleware) for m in middleware)

    base_mw = []
    if not has_json_validation:
        base_mw.append(JsonValidationMiddleware(strict=False))

    # Context-aware tool eviction
    context_eviction_mw = ContextAwareToolEvictionMiddleware(
        trigger_fraction=0.75,
    )
    base_mw.append(context_eviction_mw)

    backend = _create_standard_backend(working_dir)
    logging_mw = ModelLoggingMiddleware("context-agent")

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/context_agent.md.j2"),
        backend=backend,
        middleware=base_mw + middleware + [logging_mw],
    )

    return agent


def create_research_agent(model: BaseChatModel,
                          working_dir: str,
                          checkpointer=None,
                          middleware=[]) -> CompiledStateGraph:
    """Create a DeepAgents-based agent suitable for general-purpose research replies.

    Includes DuckDuckGo web search and a critique/research/fact-check subagent trio.
    """
    # Only add JSON validation if not already provided
    has_json_validation = any(isinstance(m, JsonValidationMiddleware) for m in middleware)

    base_mw = []
    if not has_json_validation:
        base_mw.append(JsonValidationMiddleware(strict=False))

    # Context-aware tool eviction for research agents (more aggressive thresholds)
    context_eviction_mw = ContextAwareToolEvictionMiddleware(
        trigger_fraction=0.70,  # Evict at 70% for research (more aggressive)
    )
    base_mw.append(context_eviction_mw)

    backend = _create_standard_backend(working_dir)
    logging_mw = ModelLoggingMiddleware("research-agent")

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

    agent = create_deep_agent(
        model=model,
        tools=[search_internet, read_url],
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/research_instructions.txt.j2"),
        backend=backend,
        middleware=base_mw + middleware + [logging_mw],
        subagents=[critique_sub_agent,
                   research_sub_agent,
                   fact_check_sub_agent]
    )

    return agent
