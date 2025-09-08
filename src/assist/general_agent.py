from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from typing import Any, List
import time
import os

def general_agent(
    llm: Runnable[Any, Any],
    tools: List[BaseTool] | None = None,
) -> Runnable[Any, Any]:
    """Return a ReAct agent configured with useful tools."""
    agent_executor = create_react_agent(llm, tools or [])
    return agent_executor

def test_agents() -> None:
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI
    nontool_message = HumanMessage(content="Hello. How are you doing today? What kinds of things can you help me with?")
    tool_message = HumanMessage(content="What's the weather like in San Francisco today?")
    sys = SystemMessage(content="You are a helpful assistant")
    for llm in [ChatOllama(model="llama3.2", temperature=0.5),
                ChatOllama(model="qwen3", temperature=0.5),
                ChatOllama(model="mistral", tempareature=0.5)]:
        print(f"Using {llm.model}\n")
        agent = general_agent(llm)
        for message in [nontool_message, tool_message]:
            print(f"Message: {message.content}")
            start = time.time()
            res = agent.invoke({"messages": [sys, message]})
            elapsed = time.time() - start
            messages = res['messages']
            print(f"Response ({elapsed}): {messages[-1].content}")
        print("=====================\n\n")
