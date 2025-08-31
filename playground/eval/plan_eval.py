from typing import Optional, Callable
from pydantic import BaseModel
from assist.reflexion_agent import build_plan_node, ReflexionState
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from assist.tools.base import base_tools
from langchain.callbacks.tracers.stdout import ConsoleCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage

def graphiphy(node: Callable,
              state: BaseModel) -> CompiledStateGraph:
    graph = StateGraph(state)

    graph.add_node("node", node)
    graph.set_entry_point("node")
    graph.add_edge("node", END)
    return graph.compile()


def plan_eval_simple():
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.8)
    node = build_plan_node(llm,
                           base_tools("~/.cache/assist/dbs"),
                           [ConsoleCallbackHandler()])
    graph = graphiphy(node, ReflexionState)
    state = ReflexionState(messages=[SystemMessage("You are a helpful assistant"),
                                     HumanMessage("What's the 200th digit of pi?")],
                           plan=[],
                           step_index=0,
                           history=[],
                           needs_replan=False,
                           learnings=[])
    return graph.invoke(state)

plan_eval_simple()
