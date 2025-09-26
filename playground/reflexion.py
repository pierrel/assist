from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from assist.tools.base import base_tools
from assist.reflexion_agent import build_reflexion_graph, ReflexionState
import time
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

f = open('playground/server-config.txt', 'r')
config = f.readline().strip()
f.close()


class HowyaDoingResponse(BaseModel):
    well: bool = Field(description="Whether you're doing well")
    not_well: bool = Field(description="Whether you're not doing well")
    explanation: str = Field(description="More details about how you're doing")

tools = base_tools("~/.cache/assist/dbs/")

llm = ChatOpenAI(base_url=config,
                 api_key="sk-local",
                 model="/models/mistral.gguf")

# Can handle normal questions
llm.invoke([HumanMessage("Hello, how are you?")])

# Can handle structured output
llm.with_structured_output(HowyaDoingResponse).invoke(
    [HumanMessage("Hello, how are you?")])

# Can use tools
llm.bind_tools(tools).invoke([HumanMessage("What's in the file ~/src/bin/em ?")])

# Can handle a react agent
agent = create_react_agent(llm, tools)

agent.invoke({"messages": [HumanMessage("What's in the file ~/src/bin/em?")]})

# Can handle plan and execute
agent = build_reflexion_graph(llm,
                              base_tools("~/.cache/assist/dbs/"))

request = HumanMessage(content="Help me understand how python iterators work\nThis is the project directory within the context of this request: /home/pierre/src/chat_ollama_streaming")

start = time.time()
resp = agent.invoke({"messages": [request]})
total_time = time.time() - start
print(resp)

print(f"\n\nTook {total_time}s")
