"""Reflexion graph built from planning and execution steps."""
import pdb
import math
import os
import time

from typing import Any, Callable, Dict, List, Optional, TypedDict, Literal, cast

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


def _extract_system_and_context(messages: List[BaseMessage]) -> tuple[list[str], List[BaseMessage]]:
    """Return system message contents and non-system messages."""
    system_msgs = [
        m.content
        for m in messages
        if isinstance(m, SystemMessage) and isinstance(m.content, str)
    ]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    return system_msgs, non_system


def _combine_system_prompt(base_prompt: str, system_msgs: list[str]) -> str:
    """Append server-supplied system prompts to the base prompt."""
    if not system_msgs:
        return base_prompt
    guidance = "\n\nHere is guidance from the user:\n" + "\n".join(system_msgs)
    return f"{base_prompt}{guidance}"


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


def build_plan_node(
    llm: BaseChatModel,
    tools: List[BaseTool],
    callbacks: Optional[List[Any]],
) -> Callable[[ReflexionState], Dict[str, object]]:
    def plan_node(state: ReflexionState) -> Dict[str, object]:
        user_msg = state["messages"][-1]
        request = getattr(user_msg, "content", user_msg)
        logger.debug(f"Generating plan for request: {request}")
        tool_list = "\n".join(tool_list_item(t) for t in tools)
        project_root = os.environ.get("ASSIST_SERVER_PROJECT_ROOT", "")
        prior_messages = state["messages"][:-1]
        sys_msgs, context_msgs = _extract_system_and_context(prior_messages)
        system_prompt = _combine_system_prompt(
            base_prompt_for("reflexion_agent/make_plan_system.txt",
                            tools=tool_list,
                            project_root=project_root),
            sys_msgs
        )
        messages = [
            SystemMessage(content=system_prompt),
            *context_msgs,
            HumanMessage(
                content=base_prompt_for(
                    "reflexion_agent/make_plan_user.txt",
                    task=request,
                    learnings=state.get("learnings", [])
                )
            ),
        ]
        logger.debug(f"Full messaging for plan:\n{messages}")
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


def build_execute_node(
    agent: Runnable[Any, Any],
    callbacks: Optional[List[Any]],
) -> Callable[[ReflexionState], ReflexionState]:
    def execute_node(state: ReflexionState) -> ReflexionState:
        step = state["plan"].steps[state["step_index"]]
        step_index = state['step_index']
        history_text = "\n".join([str(h) for h in state['history']])
        logger.debug(f"Executing step {step_index}: {step}")
        project_root = os.environ.get("ASSIST_SERVER_PROJECT_ROOT", "")
        messages = [
            SystemMessage(content=base_prompt_for("reflexion_agent/execute_step_system.txt")),
            HumanMessage(
                content=base_prompt_for(
                    "reflexion_agent/execute_step_user.txt",
                    history=history_text,
                    step=step,
                    goal=state["plan"].goal,
                    project_root=project_root
                )
            ),
        ]
        result_raw = agent.invoke({"messages": messages},
                                  {"callbacks": callbacks})
        result = AgentInvokeResult.model_validate(result_raw)
        output_msg = result.messages[-1]
        res = StepResolution(action=step.action,
                             objective=step.objective,
                             resolution=output_msg.content)
        new_hist = state["history"] + [res]
        state["history"] = new_hist
        state["step_index"] = step_index + 1
        return state

    return execute_node


def build_plan_check_node(
    llm: BaseChatModel,
    callbacks: Optional[List[Any]],
) -> Callable[[ReflexionState], Dict[str, object]]:
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

        retro_raw = llm.with_structured_output(PlanRetrospective).invoke(
            messages, {"callbacks": callbacks}
        )
        retro = PlanRetrospective.model_validate(retro_raw)
        logger.debug(f"Retrospected with:\n{retro}")
        all_learnings = state.get("learnings", [])
        if retro.needs_replan and retro.learnings is not None:
            all_learnings = all_learnings + [retro.learnings]
        return {"needs_replan": retro.needs_replan, "learnings": all_learnings}

    return plan_check_node


def build_summarize_node(
    llm: BaseChatModel,
    callbacks: Optional[List[Any]],
) -> Callable[[ReflexionState], Dict[str, List[BaseMessage]]]:
    def summarize_node(state: ReflexionState) -> Dict[str, List[BaseMessage]]:
        history_text = "\n".join(h.resolution for h in state["history"])
        sys_msgs, context_msgs = _extract_system_and_context(state["messages"])
        system_prompt = _combine_system_prompt(
            base_prompt_for("reflexion_agent/summarize_system.txt"), sys_msgs
        )
        messages = [
            SystemMessage(content=system_prompt),
            *context_msgs,
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
    llm: BaseChatModel,
    tools: List[BaseTool],
    callbacks: Optional[List[Any]] = None,
    execution_llm: Optional[BaseChatModel] = None,
) -> Runnable[Any, Any]:
    """Compose planning, step execution and summarization using LangGraph.

    ``llm`` is used for planning related tasks (planning, plan checking and
    summarization) while ``execution_llm`` – if provided – is used for step
    execution. When ``execution_llm`` is ``None`` the planner ``llm`` is also
    used for execution.
    """

    callbacks = callbacks or [ConsoleCallbackHandler()]
    exec_llm = execution_llm or llm
    agent = general_agent(exec_llm, tools)
    graph = StateGraph(ReflexionState)

    graph.add_node("plan", cast(Any, build_plan_node(llm, tools, callbacks)))
    graph.add_node("execute", cast(Any, build_execute_node(agent, callbacks)))
    graph.add_node("plan_check", cast(Any, build_plan_check_node(llm, callbacks)))
    graph.add_node("summarize", cast(Any, build_summarize_node(llm, callbacks)))

    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("plan_check", big_condition)
    graph.add_conditional_edges("execute", after_execute)
    graph.set_entry_point("plan")
    graph.add_edge("summarize", END)

    return graph.compile()
