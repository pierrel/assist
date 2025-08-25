"""Reflexion graph built from planning and execution steps."""
import math
import os
import time

from typing import Dict, List, Optional, TypedDict, Literal, Callable

from loguru import logger
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from assist.general_agent import general_agent
from assist.promptable import base_prompt_for
from assist.agent_types import AgentInvokeResult


class Step(BaseModel):
    action: str = Field(
        description="A concise and concrete description of what to *do* to accompliash an objective that, together with other steps will resolve the ultimate goal."
    )
    objective: str = Field(
        description="Objective of the step - how does this particular step get the user closer to their goal?"
    )


class StepResolution(Step):
    resolution: str = Field(
        description="The resolution of the step that achieves the objective."
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


class PlanRetrospective(BaseModel):
    needs_replan: bool = Field(
        description="Whether or not the user requires a new plan to achieve the goal"
    )
    learnings: Optional[str] = Field(
        description="If a new plan is required, then the learnings that should be incorporated into that new plan so that it has a better chance of achieving the goal."
    )


class ReflexionState(TypedDict):
    messages: List[BaseMessage]
    plan: Plan
    step_index: int
    history: List[StepResolution]
    needs_replan: bool
    learnings: List[str]


def tool_list_item(tool: BaseTool) -> str:
    return f"- {tool.name}: {tool.description}"


def _default_llm() -> Runnable:
    """Return a default LLM or a fake runnable for offline tests."""
    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    from langchain_core.messages import AIMessage

    class _StaticLLM(Runnable):
        def __init__(self) -> None:
            self._schema: type[BaseModel] | None = None

        def with_structured_output(self, schema: type[BaseModel]) -> "_StaticLLM":
            self._schema = schema
            return self

        def invoke(self, _messages, _opts=None):
            if self._schema is Plan:
                self._schema = None
                return Plan(goal="", steps=[], assumptions=[], risks=[])
            if self._schema is PlanRetrospective:
                self._schema = None
                return PlanRetrospective(needs_replan=False, learnings=None)
            return AIMessage(content="")

    return _StaticLLM()


def build_plan_node(llm: BaseChatModel,
                    tools: List[BaseTool],
                    callbacks: Optional[List]) -> Callable:
    def plan_node(state: ReflexionState) -> Dict[str, object]:
        user_msg = state["messages"][-1]
        request = getattr(user_msg, "content", user_msg)
        logger.debug(f"Generating plan for request: {request}")
        tool_list = "\n".join(tool_list_item(t) for t in tools)
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/make_plan_system.txt")),
            HumanMessage(
                content=base_prompt_for(
                    "reflexion_agent/make_plan_user.txt", tools=tool_list, task=request
                )
            ),
        ]
        start = time.time()
        plan = llm.with_structured_output(Plan).invoke(
            messages,
            {"callbacks": callbacks}
        )
        logger.debug(f"Plan generated in {time.time() - start}s:\n{plan}")
        return {"plan": plan,
                "step_index": 0,
                "history": [],
                "needs_replan": False,
                "learnings": state.get("learnings", [])}
    return plan_node


def build_execute_node(agent: Runnable,
                       tools: List[BaseTool],
                       callbacks: Optional[List]) -> Callable:
    def execute_node(state: ReflexionState) -> Dict[str, object]:
        step = state["plan"].steps[state["step_index"]]
        history_text = "\n".join(state["history"])
        logger.debug(f"Executing step {state['step_index'] + 1}: {step}")
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/execute_step_system.txt")),
            *state["messages"],
            HumanMessage(
                content=base_prompt_for(
                    "reflexion_agent/execute_step_user.txt", history=history_text, step=step
                )
            ),
        ]
        result_raw = agent.invoke({"messages": messages}, {"callbacks": callbacks})
        result = AgentInvokeResult.model_validate(result_raw)
        output_msg = result.messages[-1]
        new_hist = state["history"] + [f"{step}: {output_msg.content}"]
        logger.debug(f"Step result: {output_msg.content}")
        return {"history": new_hist, "step_index": state["step_index"] + 1}

    return execute_node

def build_plan_check_node(llm: BaseChatModel,
                          callbacks: Optional[List]) -> Callable:
    def plan_check_node(state: ReflexionState) -> Dict[str, object]:
        """Asks an llm if a replan is required. If so, updates learnings and the replan bit."""
        plan = state["plan"]
        human_prompt = base_prompt_for(
            "reflexion_agent/plan_check_user.txt",
            step_resolutions=state["history"],
            remaining_steps=plan.steps[len(state["history"]):],
            goal=plan.goal
        )
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/plan_check_system.txt")),
            HumanMessage(content=human_prompt),
        ]

        retro: PlanRetrospective = llm.with_structured_output(PlanRetrospective).invoke(messages, {"callbacks": callbacks})
        logger.debug(f"Retrospected with:\n{retro}")
        all_learnings = state.get("learnings", [])
        if retro.needs_replan:
            all_learnings = all_learnings + [retro.learnings]
        return {"needs_replan": retro.needs_replan,
                "learnings": all_learnings}

    return plan_check_node


def build_summarize_node(llm: BaseChatModel,
                         callbacks: Optional[List]) -> Callable:
    def summarize_node(state: ReflexionState) -> Dict[str, List[BaseMessage]]:
        history_text = "\n".join(state["history"])
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/summarize_system.txt")),
            HumanMessage(
                content=base_prompt_for("reflexion_agent/summarize_user.txt", history=history_text)
            ),
        ]
        summary = llm.invoke(messages)
        logger.debug(f"Summary: {summary.content}")
        return {"messages": state["messages"] + [summary]}
    return summarize_node


def checkpoints(total_steps: int) -> set[int]:
    """Return step indices where a plan check should occur."""
    return {
        total_steps,
        math.ceil(total_steps / 3),
        math.ceil(2 * total_steps / 3),
    }


def after_execute(state: ReflexionState) -> Literal["plan_check", "execute", "summarize"]:
    """Determine next node after executing a step."""
    total = len(state["plan"].steps)
    idx = state["step_index"]
    if idx in checkpoints(total):
        return "plan_check"
    elif idx < total:
        return "execute"
    else:
        return "summarize"


def replan_cond(state: ReflexionState) -> bool:
    """Check the state for the need to replan."""
    return state["needs_replan"]


def continue_cond(state: ReflexionState) -> bool:
    return state["step_index"] < len(state["plan"].steps)


def big_condition(state: ReflexionState) -> Literal["execute", "plan", "summarize"]:
    if replan_cond(state):
        return "plan"
    elif continue_cond(state):
        return "execute"
    else:
        return "summarize"


def build_reflexion_graph(
    llm: Runnable,
    tools: List[BaseTool],
    callbacks: Optional[List] = None
) -> Runnable:
    """Compose planning, step execution and summarization using LangGraph."""

    callbacks = callbacks or [ConsoleCallbackHandler()]
    agent = general_agent(llm, tools)
    graph = StateGraph(ReflexionState)

    graph.add_node("plan", build_plan_node(llm,
                                           tools,
                                           callbacks))
    graph.add_node("execute", build_execute_node(agent,
                                                 tools,
                                                 callbacks))
    graph.add_node("plan_check", build_plan_check_node(llm,
                                                       callbacks))
    graph.add_node("summarize", build_summarize_node(llm,
                                                     callbacks))

    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("plan_check", big_condition)
    graph.add_conditional_edges("execute", after_execute)
    graph.set_entry_point("plan")
    graph.add_edge("summarize", END)

    return graph.compile()

def single_node_graph

def reflexion_graph_v1() -> Runnable:
    """Default reflexion graph using a lightweight LLM."""
    return build_reflexion_graph(_default_llm(), tools=[])


def planner_graph_v1() -> Runnable:
    """Graph that only performs planning."""
    llm = _default_llm()
    graph = StateGraph(dict)
    
    
    graph.add_node("plan", build_plan_node(llm, [], []))
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    return graph.compile()


def plan_checker_graph_v1() -> Runnable:
    """Graph that evaluates if replanning is required."""
    llm = _default_llm()
    graph = StateGraph(dict)

    graph.add_node("check", build_plan_check_node(llm, []))
    graph.set_entry_point("check")
    graph.add_edge("check", END)
    return graph.compile()


def step_executor_graph_v1() -> Runnable:
    """Graph that executes a single step."""
    llm = _default_llm()
    agent = general_agent(llm, [])
    graph = StateGraph(dict)

    graph.add_node("exec", build_execute_node(agent,
                                              [],
                                              []))
    graph.set_entry_point("exec")
    graph.add_edge("exec", END)
    return graph.compile()


def summarizer_graph_v1() -> Runnable:
    """Graph that summarizes a list of step resolutions."""
    llm = _default_llm()
    graph = StateGraph(dict)

    graph.add_node("sum", build_summarize_node(llm,
                                               [],
                                               []))
    graph.set_entry_point("sum")
    graph.add_edge("sum", END)
    return graph.compile()
