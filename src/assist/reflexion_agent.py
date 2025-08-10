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

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

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
    """Plan execution and summarization orchestrated with LangGraph."""

    class _State(TypedDict):
        messages: List[BaseMessage]
        plan_steps: List[str]
        step_index: int

    def __init__(self, llm: Runnable, tools: List[BaseTool], callbacks: Optional[List] = None):
        self.callbacks = callbacks or [ConsoleCallbackHandler()]
        self.planner = PlannerAgent(llm, tools, callbacks=self.callbacks)
        # Separate agents for execution and summarization
        self.agent = general_agent(llm, tools)
        self.summarizer = general_agent(llm, tools)
        self.graph = self._build_graph()

    def _build_graph(self) -> Runnable:
        graph = StateGraph(self._State)

        def plan_node(state: ReflexionAgent._State) -> dict:
            user_msg = state["messages"][-1]
            content = getattr(user_msg, "content", str(user_msg))
            logger.debug("Generating plan for message: %s", content)
            plan = self.planner.make_plan(content)
            steps = [s.split(".", 1)[-1].strip() for s in plan.splitlines() if s.strip()]
            logger.debug("Plan steps: %s", steps)
            return {"plan_steps": steps, "step_index": 0}

        def execute_node(state: ReflexionAgent._State) -> dict:
            idx = state["step_index"]
            step = state["plan_steps"][idx]
            logger.debug("Executing step %d: %s", idx + 1, step)
            step_msg = HumanMessage(content=step)
            resp = self.agent.invoke(
                {"messages": state["messages"] + [step_msg]},
                {"callbacks": self.callbacks},
            )
            result_msg = resp["messages"][-1]
            logger.debug("Step %d result: %s", idx + 1, result_msg.content)
            return {
                "messages": state["messages"] + [step_msg, result_msg],
                "step_index": idx + 1,
            }

        def summarize_node(state: ReflexionAgent._State) -> dict:
            logger.debug("Summarizing final response")
            summary_prompt = SystemMessage(content="Summarize the conversation so far.")
            resp = self.summarizer.invoke(
                {"messages": state["messages"] + [summary_prompt]},
                {"callbacks": self.callbacks},
            )
            final_msg = resp["messages"][-1]
            logger.debug("Final response: %s", final_msg.content)
            return {"messages": state["messages"] + [summary_prompt, final_msg]}

        def continue_or_finish(state: ReflexionAgent._State) -> str:
            if state["step_index"] < len(state["plan_steps"]):
                return "execute"
            return "summarize"

        graph.add_node("plan", plan_node)
        graph.add_node("execute", execute_node)
        graph.add_node("summarize", summarize_node)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_conditional_edges("execute", continue_or_finish, {"execute": "execute", "summarize": "summarize"})
        graph.set_finish_point("summarize")

        return graph.compile()

    def invoke(self, inputs: dict) -> dict:
        state: ReflexionAgent._State = {
            "messages": inputs.get("messages", []),
            "plan_steps": [],
            "step_index": 0,
        }
        result = self.graph.invoke(state, {"callbacks": self.callbacks})
        return {"messages": result["messages"]}


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
