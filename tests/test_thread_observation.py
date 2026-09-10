"""Regression coverage for phone-visible Thread observation."""
from typing import Annotated

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from assist.thread import Thread


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _streamed_message(model, messages) -> AIMessage:
    """Run the fake chat model through LangChain's real chunk callback path."""
    chunks = list(model.stream(messages))
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    return AIMessage(content="".join(chunk.content for chunk in chunks))


def _message_events(events):
    """Yield LangGraph message payloads from either top-level event shape."""
    for event in events:
        if isinstance(event, tuple) and len(event) >= 2 and event[-2] == "messages":
            yield event[-1]


def test_observe_message_publishes_model_namespace_and_excludes_child_graph():
    """The real stream keeps a ``model:<uuid>`` chunk but omits child output.

    This uses a compiled parent/checkpointer and an invoked child graph.  With
    ``subgraphs=True`` the child emits its real nested model chunks; the same
    graph through ``Thread._observe`` uses ``stream_with_rollback`` and its
    explicit ``subgraphs=False`` boundary, so only the parent's model chunks
    reach the phone callback.
    """
    child_model = FakeListChatModel(responses=["child"])

    def child_node(state: _State):
        return {"messages": [_streamed_message(child_model, state["messages"])]}

    child_builder = StateGraph(_State)
    child_builder.add_node("model", child_node)
    child_builder.add_edge(START, "model")
    child_builder.add_edge("model", END)
    child = child_builder.compile()

    top_model = FakeListChatModel(responses=["top"])

    def run_child(state: _State):
        child.invoke(state)
        return {}

    def top_node(state: _State):
        return {"messages": [_streamed_message(top_model, state["messages"])]}

    parent_builder = StateGraph(_State)
    parent_builder.add_node("child", run_child)
    parent_builder.add_node("model", top_node)
    parent_builder.add_edge(START, "child")
    parent_builder.add_edge("child", "model")
    parent_builder.add_edge("model", END)
    agent = parent_builder.compile(checkpointer=MemorySaver())

    nested_events = list(agent.stream(
        {"messages": [{"role": "user", "content": "hello"}]},
        {"configurable": {"thread_id": "subgraphs-visible"}},
        stream_mode=["messages"], durability="sync", subgraphs=True,
    ))
    assert any(
        message.content == "child"
        and metadata["langgraph_checkpoint_ns"].startswith("child:")
        for message, metadata in _message_events(nested_events)
    )
    assert any(
        isinstance(message, AIMessageChunk)
        and message.content == "t"
        and metadata["langgraph_checkpoint_ns"].startswith("model:")
        for message, metadata in _message_events(nested_events)
    )

    chat = object.__new__(Thread)
    chat.agent = agent
    chat.thread_id = "top-level-model-namespace"
    chat.runconfig = {"configurable": {"thread_id": chat.thread_id}}
    chat.on_queue_state = None
    deltas: list[str] = []

    assert chat.observe_message("hello", deltas.append) == "top"
    assert "".join(deltas).startswith("top")
    assert "child" not in "".join(deltas)
