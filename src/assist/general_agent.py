from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from typing import List
from .tools import filesystem as fstools
from .tools import project_index
import time
import os
from pathlib import Path

@tool
def date() -> str:
    """Returns the current date formatted like [month] [day of month],
    [year]"""
    dt = datetime.now()
    return dt.strftime("%B %d, %Y")

def check_tavily_api_key():
    if not os.getenv('TAVILY_API_KEY'):
        raise RuntimeError('Please define the environment variable TAVILY_API_KEY')
    

def general_agent(llm: Runnable, extra_tools: List[BaseTool] = []):
    """Returns an Agent as described in
    https://python.langchain.com/docs/tutorials/agents/"""
    check_tavily_api_key()
    search = TavilySearchResults(max_results=10)
    proj_tool = project_index.project_search
    tools = [
        search,
        date,
        fstools.list_files,
        fstools.file_contents,
        proj_tool,
    ]
    agent_executor = create_react_agent(llm,
                                        tools + extra_tools)
    return agent_executor

def test_agents():
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
