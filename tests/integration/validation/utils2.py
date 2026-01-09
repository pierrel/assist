from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from langgraph.graph.state import CompiledStateGraph

def send_message(agent: CompiledStateGraph,
                 message: str,
                 thread_id: str) -> list[BaseMessage]:
    resp = agent.invoke({"messages": [{"role": "user",
                                       "content": message}]},
                        {"configurable": {"thread_id": thread_id}})
    return resp["messages"]
