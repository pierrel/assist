from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from assist.tools import project_index, filesystem
from assist.reflexion_agent import build_reflexion_graph, ReflexionState
from IPython.display import Image, display
import time

#llm = ChatOllama(model="qwen3:4b", temperature=0.8)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
pi = project_index.ProjectIndex()
proj_tool = pi.search_tool()
search = TavilySearchResults(max_results=10)
agent = build_reflexion_graph(llm,
                              [filesystem.file_contents,
                               filesystem.list_files,
                               proj_tool,
                               search])

#Image(agent.get_graph().draw_mermaid_png(output_file_path="./agent_graph.png"))

request = HumanMessage(content="Help me understand how python iterators work\nThis is the project directory within the context of this request: /home/pierre/src/assist/")

start = time.time()
resp = agent.invoke({"messages": [request]})
total_time = time.time() - start
print(resp)

print(f"\n\nTook {total_time}s")
