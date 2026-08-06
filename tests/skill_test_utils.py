"""Shared test helpers for skill middleware contracts."""
import inspect
from types import SimpleNamespace

from langgraph.types import Command


def load_skill(middleware, name: str) -> str:
    """Invoke either the legacy or progressive test loader by exact name."""
    update = middleware.before_agent({}, SimpleNamespace(), {})
    state = {
        "skills_metadata": update["skills_metadata"],
        "loaded_skill_tools": frozenset(),
    }
    arguments = [name]
    if len(inspect.signature(middleware.tools[0].func).parameters) > 1:
        arguments.append(SimpleNamespace(
            state=state, config={}, tool_call_id="test-load"))
    result = middleware.tools[0].func(*arguments)
    return (result.update["messages"][0].content
            if isinstance(result, Command) else result)
