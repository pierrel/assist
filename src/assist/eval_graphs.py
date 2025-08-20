"""Graph builders for evaluation harness."""
from __future__ import annotations

import os
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END

from assist.general_agent import general_agent
from assist.promptable import base_prompt_for
from assist.reflexion_agent import (
    build_reflexion_graph,
    Plan,
    PlanRetrospective,
    Step,
)


try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - fallback in environments without openai
    ChatOpenAI = None  # type: ignore


def _default_llm() -> Runnable:
    """Return a default LLM for tests or fall back to a fake runnable."""
    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    # Fallback to simple fake runnable with empty responses
    from langchain_core.messages import AIMessage
    from assist.fake_runnable import FakeRunnable

    return FakeRunnable([[AIMessage(content="")]])


def reflexion_graph_v1() -> Runnable:
    """Full reflexion agent graph."""
    llm = _default_llm()
    return build_reflexion_graph(llm, tools=[])


def planner_graph_v1() -> Runnable:
    """Graph that only performs planning."""
    llm = _default_llm()
    graph = StateGraph(dict)

    def plan_node(state: Dict[str, object]) -> Dict[str, object]:
        request = state.get("user", "")
        messages: List[BaseMessage] = [
            SystemMessage(content=base_prompt_for("reflexion_agent/make_plan_system.txt")),
            HumanMessage(
                content=base_prompt_for(
                    "reflexion_agent/make_plan_user.txt", tools="", task=request
                )
            ),
        ]
        plan = llm.with_structured_output(Plan).invoke(messages)
        return {"output": plan.model_dump_json()}

    graph.add_node("plan", plan_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    return graph.compile()


def plan_checker_graph_v1() -> Runnable:
    """Graph that evaluates if replanning is required."""
    llm = _default_llm()
    graph = StateGraph(dict)

    def check_node(state: Dict[str, object]) -> Dict[str, object]:
        human_prompt = base_prompt_for(
            "reflexion_agent/plan_check_user.txt",
            step_resolutions=state.get("step_resolutions", []),
            remaining_steps=state.get("remaining_steps", []),
            goal=state.get("goal", ""),
        )
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/plan_check_system.txt")),
            HumanMessage(content=human_prompt),
        ]
        retro: PlanRetrospective = llm.with_structured_output(PlanRetrospective).invoke(messages)
        return {"output": retro.model_dump_json()}

    graph.add_node("check", check_node)
    graph.set_entry_point("check")
    graph.add_edge("check", END)
    return graph.compile()


def step_executor_graph_v1() -> Runnable:
    """Graph that executes a single step."""
    llm = _default_llm()
    agent = general_agent(llm, [])
    graph = StateGraph(dict)

    def exec_node(state: Dict[str, object]) -> Dict[str, object]:
        step = Step(**state.get("step", {}))
        history = state.get("history", "")
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/execute_step_system.txt")),
            HumanMessage(content=str(history)),
            HumanMessage(
                content=base_prompt_for("reflexion_agent/execute_step_user.txt", history=history, step=step)
            ),
        ]
        result_raw = agent.invoke({"messages": messages})
        output_msg = result_raw["messages"][-1]
        return {"output": output_msg.content}

    graph.add_node("exec", exec_node)
    graph.set_entry_point("exec")
    graph.add_edge("exec", END)
    return graph.compile()


def summarizer_graph_v1() -> Runnable:
    """Graph that summarizes a list of step resolutions."""
    llm = _default_llm()
    graph = StateGraph(dict)

    def sum_node(state: Dict[str, object]) -> Dict[str, object]:
        history = state.get("history", [])
        history_text = "\n".join(history)
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/summarize_system.txt")),
            HumanMessage(
                content=base_prompt_for("reflexion_agent/summarize_user.txt", history=history_text)
            ),
        ]
        summary = llm.invoke(messages)
        return {"output": summary.content}

    graph.add_node("sum", sum_node)
    graph.set_entry_point("sum")
    graph.add_edge("sum", END)
    return graph.compile()
