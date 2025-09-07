"""Demonstrate a tool raising an exception during agent execution."""
from typing import List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def always_fail() -> str:
    """Raise a ``RuntimeError`` to demonstrate tool errors."""
    raise RuntimeError("always_fail tool was invoked")


class AgentState(TypedDict):
    """State for the simple graph."""
    messages: List


def model_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([always_fail])
    ai = llm_with_tools.invoke(
        state["messages"],
        tool_choice={"type": "function", "function": {"name": "always_fail"}},
    )
    return {"messages": state["messages"] + [ai]}


def tool_node(state: AgentState) -> AgentState:
    ai: AIMessage = state["messages"][-1]
    call = ai.tool_calls[0]
    try:
        result = always_fail.invoke(call["args"])
    except Exception as exc:  # pragma: no cover - demonstration
        result = f"Error: {exc}"
    tool_msg = ToolMessage(content=result, name=call["name"], tool_call_id=call["id"])
    return {"messages": state["messages"] + [tool_msg]}


workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.add_node("tool", tool_node)
workflow.add_edge("model", "tool")
workflow.add_edge("tool", END)
workflow.set_entry_point("model")
agent = workflow.compile()


if __name__ == "__main__":
    result = agent.invoke({"messages": [HumanMessage(content="trigger the tool")]})
    for msg in result["messages"]:
        print(msg)
