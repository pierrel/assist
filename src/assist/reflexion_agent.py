"""Planner and reflexion agents.

``PlannerAgent`` generates a short plan describing how to complete a task using
the available tools. ``ReflexionAgent`` composes this planning step with the
existing ReAct agent returned by :func:`general_agent`.  Both stages log their
activity with :mod:`loguru` and can emit LangChain tracing callbacks via
``ConsoleCallbackHandler``.
"""

from typing import List, Optional

from loguru import logger
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from .general_agent import general_agent


class PlannerAgent:
    """Generate a tool-based plan for a user request."""

    def __init__(self, llm: Runnable, tools: List[BaseTool], callbacks: Optional[List] = None):
        self.llm = llm
        self.tools = tools
        self.callbacks = callbacks or [ConsoleCallbackHandler()]

    def make_plan(self, user_request: str) -> str:
        logger.debug("Generating plan for request: %s", user_request)
        tool_list = ", ".join(t.name for t in self.tools)
        messages = [
            SystemMessage(
                content=(
                    "You are a planning assistant. Given a task and available tools, "
                    "produce a numbered list describing how to accomplish the task using those tools."
                )
            ),
            HumanMessage(content=f"Tools: {tool_list}\nTask: {user_request}\nPlan:")
        ]
        response = self.llm.invoke({"messages": messages}, {"callbacks": self.callbacks})
        if isinstance(response, dict) and "messages" in response:
            msg = response["messages"][-1]
            plan = getattr(msg, "content", str(msg))
        else:
            plan = getattr(response, "content", str(response))
        logger.debug("Plan generated:\n%s", plan)
        return plan


class ReflexionAgent:
    """Compose planning with the existing ReAct agent."""

    def __init__(self, llm: Runnable, tools: List[BaseTool], callbacks: Optional[List] = None):
        self.callbacks = callbacks or [ConsoleCallbackHandler()]
        self.planner = PlannerAgent(llm, tools, callbacks=self.callbacks)
        self.agent = general_agent(llm, tools)

    def invoke(self, inputs: dict) -> dict:
        messages = inputs.get("messages", [])
        user_msg = messages[-1]
        logger.debug("Invoking agent for message: %s", getattr(user_msg, "content", user_msg))
        plan = self.planner.make_plan(getattr(user_msg, "content", user_msg))
        plan_msg = SystemMessage(content=f"Follow this plan:\n{plan}")
        logger.debug("Executing plan via ReAct agent")
        result = self.agent.invoke({"messages": messages + [plan_msg]}, {"callbacks": self.callbacks})
        return result


def reflexion_agent(
    llm: Runnable,
    tools: List[BaseTool],
    callbacks: Optional[List] = None,
) -> Runnable:
    """Create a reflexion agent using ``llm`` and ``tools``.

    The agent first asks ``llm`` for a plan referencing the given tools, then
    executes that plan using the ReAct agent returned by :func:`general_agent`.
    Additional ``callbacks`` are passed to both the planning and execution
    stages, allowing integration with LangChain tracing utilities.
    """
    return ReflexionAgent(llm, tools, callbacks=callbacks)

__all__ = ["reflexion_agent", "PlannerAgent", "ReflexionAgent"]
