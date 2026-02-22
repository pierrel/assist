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
from assist.backends import create_composite_backend, create_sandbox_composite_backend, STATEFUL_PATHS
from assist.checkpoint_rollback import invoke_with_rollback, RollbackRunnable
from assist.middleware.model_logging_middleware import ModelLoggingMiddleware
from assist.middleware.json_validation_middleware import JsonValidationMiddleware
from assist.middleware.context_aware_tool_eviction import ContextAwareToolEvictionMiddleware
from assist.middleware.tool_name_sanitization import ToolNameSanitizationMiddleware
from assist.middleware.bad_request_retry import BadRequestRetryMiddleware


logger = logging.getLogger(__name__)


def _create_standard_backend(working_dir: str):
    """Create the standard composite backend with state exclusions.

    This backend excludes ephemeral files like question.txt and large_tool_results/
    from the stateful filesystem, using StateBackend instead.
    """
    return create_composite_backend(working_dir, STATEFUL_PATHS)


class AgentHarness:
    """Makes it easier to have conversations"""
    
    def __init__(self, agent: CompiledStateGraph, thread_id: str | None = None):
        self.agent = agent
        self.thread_id = thread_id or uuid.uuid1()

    def message(self, text: str) -> AIMessage:
        resp = invoke_with_rollback(
            self.agent,
            {"messages": [{"role": "user", "content": text}]},
            {"configurable": {"thread_id": self.thread_id}},
        )
        return resp["messages"][-1].content

    def all_messages(self) -> list[AnyMessage]:
        state = self.agent.get_state({
            "configurable": {"thread_id": self.thread_id}
        })
        return state.values.get("messages", [])
        



def create_agent(model: BaseChatModel,
                 working_dir: str,
                 checkpointer=None,
                 sandbox_backend=None) -> CompiledStateGraph:
    # Core middleware: retry, tool call limiting, JSON validation, and logging
    # Only retry on transient server errors (5xx, timeouts, connection issues).
    # BadRequestError (400) is handled by invoke_with_rollback via checkpoint rollback.
    retry_middle = ModelRetryMiddleware(max_retries=3,
                                        retry_on=(InternalServerError, TimeoutError, ConnectionError),
                                        backoff_factor=2)
    # Validate and fix JSON in tool call arguments
    json_validation_mw = JsonValidationMiddleware(strict=False)
    # Strip tool calls with invalid names (e.g. '[]' hallucinated by small models)
    tool_name_mw = ToolNameSanitizationMiddleware()
    logging_mw = ModelLoggingMiddleware("general-agent")

    # Context-aware tool eviction: evict results to filesystem if they would cause overflow
    context_eviction_mw = ContextAwareToolEvictionMiddleware(
        trigger_fraction=0.75,  # Evict if context would reach 75%
    )

    mw = [retry_middle, json_validation_mw, tool_name_mw, context_eviction_mw]

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
        backend = _create_standard_backend(working_dir)

    context_sub = CompiledSubAgent(
        name="context-agent",
        description="Discovers and surfaces relevant context from the user's local filesystem. Use this agent to find files, read content, and understand the user's file structure before taking action. It is read-only — it will not modify files.",
        runnable=create_context_agent(model,
                                      working_dir,
                                      checkpointer,
                                      [retry_middle, json_validation_mw, tool_name_mw],
                                      sandbox_backend=sandbox_backend)
    )

    research_sub = CompiledSubAgent(
        name="research-agent",
        description="Used to conduct thorough research on external topics. The result of the research will be placed in a file and the file name/path will be returned. Provide a filename for more control.",
        runnable=create_research_agent(model,
                                       working_dir,
                                       checkpointer,
                                       [retry_middle, json_validation_mw, tool_name_mw],
                                       sandbox_backend=sandbox_backend)
    )

    dev_sub = CompiledSubAgent(
        name="dev-agent",
        description="Handles ALL software development tasks: writing code, editing code, fixing bugs, adding features, changing behaviour, updating tests, and modifying configuration files. Use this agent whenever the user's request requires creating or changing any source code, tests, or config.",
        runnable=create_dev_agent(model,
                                  working_dir,
                                  checkpointer,
                                  sandbox_backend=sandbox_backend)
    )

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/general_instructions.md.j2"),
        middleware=mw + [logging_mw],
        backend=backend,
        subagents=[context_sub, research_sub, dev_sub]
    )

    return agent

def create_context_agent(model: BaseChatModel,
                         working_dir: str,
                         checkpointer=None,
                         middleware=[],
                         sandbox_backend=None) -> RollbackRunnable:
    """Create a read-only context agent for codebase exploration.

    Returns a RollbackRunnable-wrapped agent — on BadRequestError the agent
    rolls back to a previous checkpoint rather than crashing.  This is safe
    because the context-agent is read-only (no filesystem side effects).
    """
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

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
        backend = _create_standard_backend(working_dir)
    logging_mw = ModelLoggingMiddleware("context-agent")

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/context_agent.md.j2"),
        backend=backend,
        middleware=base_mw + middleware + [logging_mw],
    )

    return RollbackRunnable(agent)


def create_research_agent(model: BaseChatModel,
                          working_dir: str,
                          checkpointer=None,
                          middleware=[],
                          sandbox_backend=None) -> RollbackRunnable:
    """Create a DeepAgents-based agent suitable for general-purpose research replies.

    Includes DuckDuckGo web search and a critique/research/fact-check subagent trio.

    Returns a RollbackRunnable-wrapped agent — on BadRequestError the agent
    rolls back to a previous checkpoint.  Research agents only write additive
    report files, so rollback is low-risk.
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

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
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

    return RollbackRunnable(agent)


def create_dev_agent(model: BaseChatModel,
                     working_dir: str,
                     checkpointer=None,
                     sandbox_backend=None) -> CompiledStateGraph:
    """Create a software development agent for writing, testing, and improving code.

    Runs inside a Docker sandbox and follows TDD practices.
    Uses BadRequestRetryMiddleware instead of checkpoint rollback — the dev-agent
    writes to the real filesystem and Docker sandbox, so rollback would leave
    orphaned side effects.  On BadRequestError the middleware sanitizes messages
    and retries, keeping the agent moving forward.

    Sub-agents:
    - context-agent: Read-only codebase exploration (same as general agent)
    - critique-agent: Reviews diffs for bugs, missing tests, and style issues
    """
    # Only retry on transient server errors (5xx, timeouts, connection issues).
    retry_middle = ModelRetryMiddleware(max_retries=3,
                                        retry_on=(InternalServerError, TimeoutError, ConnectionError),
                                        backoff_factor=2)
    json_validation_mw = JsonValidationMiddleware(strict=False)
    tool_name_mw = ToolNameSanitizationMiddleware()
    logging_mw = ModelLoggingMiddleware("dev-agent")
    context_eviction_mw = ContextAwareToolEvictionMiddleware(trigger_fraction=0.75)
    # BadRequestError (400) — sanitize messages and retry instead of rollback.
    bad_request_mw = BadRequestRetryMiddleware(max_retries=3)

    mw = [retry_middle, bad_request_mw, json_validation_mw, tool_name_mw, context_eviction_mw]

    if sandbox_backend:
        backend = create_sandbox_composite_backend(sandbox_backend)
    else:
        backend = _create_standard_backend(working_dir)

    context_sub = CompiledSubAgent(
        name="context-agent",
        description="Discovers and surfaces relevant context from the project filesystem. Use this to understand project structure, find files, read code, and discover conventions. Read-only — will not modify files.",
        runnable=create_context_agent(model,
                                      working_dir,
                                      checkpointer,
                                      [retry_middle, json_validation_mw, tool_name_mw],
                                      sandbox_backend=sandbox_backend)
    )

    critique_sub_agent = {
        "name": "critique-agent",
        "description": "Reviews code diffs for bugs, missing tests, style issues, and security concerns. Provide the full git diff output when calling this agent.",
        "system_prompt": base_prompt_for("deepagents/dev_critique.md.j2"),
    }

    agent = create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("deepagents/dev_agent_instructions.md.j2"),
        middleware=mw + [logging_mw],
        backend=backend,
        subagents=[context_sub, critique_sub_agent],
    )

    return agent
