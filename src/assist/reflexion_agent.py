"""Reflexion graph built from planning and execution steps."""

from typing import List, Optional, TypedDict

from loguru import logger
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from assist.general_agent import general_agent
from assist.promptable import prompt_for

class Step(BaseModel):
    step: str = Field(
        description="A concrete description of what to *do* to accompliash an objective that, together with other steps will resolve the ultimate goal."
    )
    objective: str = Field(
        description="Objective of the step - how does this particular step get the user closer to their goal?"
    )

class StepResolution(Step):
    resolution: str = Field(
        description="The resolution of the step"
    )

class Plan(BaseModel):
    goal: str = Field(
        description="A concise description of the goal to be achieved by following the steps"
    )
    steps: List[Step] = Field(
        description="The list of steps to follow to achieve a goal."
    )
    assumptions: List[str] = Field(
        description="A list of assumptions being made that can be reversed."
    )
    risks: List[str] = Field(
        description="A list of gaps, hazards, or decisions needed"
    )

class ReflexionState(TypedDict):
    messages: List[BaseMessage]
    plan: Plan
    step_index: int
    history: List[StepResolution]


def tool_list_item(tool: BaseTool):
    return f"- {tool.name}: {tool.description}"


def build_reflexion_graph(
    llm: Runnable,
    tools: List[BaseTool],
    callbacks: Optional[List] = None
) -> Runnable:
    """Compose planning, step execution and summarization using LangGraph."""

    callbacks = callbacks or [ConsoleCallbackHandler()]
    agent = general_agent(llm, tools)

    graph = StateGraph(ReflexionState)

    def plan_node(state: ReflexionState):
        user_msg = state["messages"][-1]
        request = getattr(user_msg, "content", user_msg)
        logger.debug(f"Generating plan for request: {request}")
        tool_list = "\n".join(tool_list_item(t) for t in tools)
        messages = [
            SystemMessage(content=prompt_for("make_plan_system.txt")),
            HumanMessage(
                content=prompt_for(
                    "make_plan_user.txt", tools=tool_list, task=request
                )
            ),
        ]
        plan = llm.with_structured_output(Plan).invoke(
            messages,
            {"callbacks": callbacks}
        )
        logger.debug(f"Plan generated:\n{plan}")
        return {"plan": plan, "step_index": 0, "history": []}

    graph.add_node("plan", plan_node)

    def execute_node(state: ReflexionState):
        step = state["plan"].steps[state["step_index"]]
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
        summary = llm.invoke(messages)
        logger.debug(f"Summary: {summary.content}")
        return {"messages": state["messages"] + [summary]}

    graph.add_node("summarize", summarize_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("summarize", END)

    return graph.compile()
