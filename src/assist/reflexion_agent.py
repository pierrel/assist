"""Reflexion graph built from planning and execution steps."""

from typing import List, Optional, TypedDict

from loguru import logger
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .general_agent import general_agent
from .promptable import prompt_for


def reflexion_agent(
    llm: Runnable, tools: List[BaseTool], callbacks: Optional[List] = None
) -> Runnable:
    """Compose planning, step execution and summarisation using LangGraph."""

    callbacks = callbacks or [ConsoleCallbackHandler()]
    agent = general_agent(llm, tools)

    class ReflexionState(TypedDict):
        messages: List[BaseMessage]
        plan: List[str]
        step_index: int
        history: List[str]

    graph = StateGraph(ReflexionState)

    class Plan(BaseModel):
        goal: str = Field(
            description="A concise description of the goal to be achieved by following the steps"
        )
        steps: List[str] = Field(
            description="The list of steps to follow to achieve a goal"
        )

    def plan_node(state: ReflexionState):
        user_msg = state["messages"][-1]
        request = getattr(user_msg, "content", user_msg)
        logger.debug(f"Generating plan for request: {request}")
        tool_list = ", ".join(t.name for t in tools)
        messages = [
            SystemMessage(content=prompt_for("make_plan_system.txt")),
            HumanMessage(
                content=prompt_for(
                    "make_plan_user.txt", tools=tool_list, task=request
                )
            ),
        ]
        plan = llm.with_structured_output(Plan).invoke(messages, {"callbacks": callbacks})
        steps = plan.steps
        logger.debug("Plan generated:\n" + "\n".join(steps))
        return {"plan": steps, "step_index": 0, "history": []}

    graph.add_node("plan", plan_node)

    def execute_node(state: ReflexionState):
        step = state["plan"][state["step_index"]]
        history_text = "\n".join(state["history"])
        logger.debug(f"Executing step {state['step_index'] + 1}: {step}")
        messages = [
            SystemMessage(content=prompt_for("execute_step_system.txt")),
            *state["messages"],
            HumanMessage(
                content=prompt_for(
                    "execute_step_user.txt", history=history_text, step=step
                )
            ),
        ]
        result = agent.invoke({"messages": messages}, {"callbacks": callbacks})
        output_msg = result["messages"][-1]
        new_hist = state["history"] + [f"{step}: {output_msg.content}"]
        logger.debug(f"Step result: {output_msg.content}")
        return {"history": new_hist, "step_index": state["step_index"] + 1}

    graph.add_node("execute", execute_node)

    def continue_cond(state: ReflexionState):
        return state["step_index"] < len(state["plan"])

    graph.add_conditional_edges("execute", continue_cond, {True: "execute", False: "summarize"})

    def summarize_node(state: ReflexionState):
        history_text = "\n".join(state["history"])
        messages = [
            SystemMessage(content=prompt_for("summarize_system.txt")),
            HumanMessage(
                content=prompt_for("summarize_user.txt", history=history_text)
            ),
        ]
        summary_invocation = llm.invoke(messages)
        summary_msg = summary_invocation["messages"][0]
        logger.debug(f"Summary: {summary_msg.content}")
        return {"messages": state["messages"] + [summary_msg]}

    graph.add_node("summarize", summarize_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("summarize", END)

    return graph.compile()
