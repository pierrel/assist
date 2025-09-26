from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from typing import Any, List
import time

from assist.model_manager import get_context_limit

TRUNCATE_MSG = "\n[Message truncated to fit context window]"
# Limit tool output to 90% of the context window to leave headroom for other messages
CONTEXT_BUFFER_RATIO = 0.9


def _truncate(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + TRUNCATE_MSG


def _guard_tool(tool: BaseTool, limit: int) -> BaseTool:
    class GuardedTool(BaseTool):
        name: str = tool.name
        description: str = tool.description
        args_schema: Any = tool.args_schema
        return_direct: bool = tool.return_direct

        def _run(self, *args: Any, config: Any | None = None, run_manager: Any = None, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
            res = tool._run(*args, config=config, run_manager=run_manager, **kwargs)
            if isinstance(res, str):
                return _truncate(res, limit)
            return res

        async def _arun(self, *args: Any, config: Any | None = None, run_manager: Any = None, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
            res = await tool._arun(*args, config=config, run_manager=run_manager, **kwargs)
            if isinstance(res, str):
                return _truncate(res, limit)
            return res

    return GuardedTool()

def general_agent(
    llm: Runnable[Any, Any],
    tools: List[BaseTool] | None = None,
) -> Runnable[Any, Any]:
    """Return a ReAct agent configured with useful tools."""
    limit = get_context_limit(llm)
    limit = int(limit * CONTEXT_BUFFER_RATIO) - len(TRUNCATE_MSG)
    limit = max(limit, 0)
    guarded = [_guard_tool(t, limit) for t in tools or []]
    agent_executor = create_react_agent(llm, guarded)
    return agent_executor
