"""
EDD Agent for intelligent test case generation.
"""
from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from deepagents.backends import FilesystemBackend

from edd.promptable import base_prompt_for


def create_capture_agent(model: BaseChatModel,
                        working_dir: str,
                        checkpointer=None,
                        middleware=None) -> CompiledStateGraph:
    """
    Create an agent that intelligently generates test cases from conversations.

    This agent analyzes a conversation and:
    - Identifies relevant files
    - Generates focused test cases
    - Creates minimal reproducible examples
    - Writes all necessary files

    Args:
        model: The language model to use
        working_dir: Directory to work in (will be the capture output directory)
        checkpointer: Optional checkpointer for state persistence
        middleware: Optional middleware list

    Returns:
        A compiled LangGraph agent
    """
    middleware = middleware or []

    fs = FilesystemBackend(root_dir=working_dir,
                           virtual_mode=True)  # Agent works in sandboxed capture directory

    return create_deep_agent(
        model=model,
        checkpointer=checkpointer or InMemorySaver(),
        system_prompt=base_prompt_for("capture_agent.md.j2"),
        backend=fs,
        middleware=middleware,
    )
