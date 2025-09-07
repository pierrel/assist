"""Demonstrate a tool raising an exception during agent execution."""
from typing import List, TypedDict

from langgraph.prebuilt import create_react_agent

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def always_fail() -> str:
    """Raise a ``RuntimeError`` to demonstrate tool errors."""
    raise RuntimeError("always_fail tool was invoked")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(llm, [always_fail])

result = agent.invoke({"messages": [HumanMessage(content="trigger the tool")]})
for msg in result["messages"]:
    print(msg)
