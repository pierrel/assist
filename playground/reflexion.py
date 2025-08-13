from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from assist.tools import project_index, filesystem
from assist.reflexion_agent import build_reflexion_graph, ReflexionState
from IPython.display import Image, display

llm = ChatOllama(model="qwen3:8b", temperature=0.8)
pi = project_index.ProjectIndex()
proj_tool = pi.search_tool()
search = TavilySearchResults(max_results=10)
agent = build_reflexion_graph(llm,
                              [filesystem.file_contents,
                               filesystem.list_files,
                               proj_tool,
                               search])

Image(agent.get_graph().draw_mermaid_png(output_file_path="./agent_graph.png"))

request = HumanMessage(content="What’s the best practice for organizing langgraph agents? Do I create a class to house the construction of the graph? Just a function to generate and return the graph? Something else?")
resp = agent.invoke({"messages": [request]})
print(resp)



