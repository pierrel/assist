"""Exercise real LangGraph accounting, checkpoint reuse and terminal bounds."""
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.config import get_config
from langgraph.types import Command

from assist.middleware.execution_limit import (
    EXECUTION_STEP_LIMIT, LIMIT_WARNING, ExecutionLimitMiddleware,
)


class RecordingModel(FakeMessagesListChatModel):
    prompts: list = []
    steps: list = []
    stop_after: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.prompts.append(messages)
        self.steps.append(get_config()["metadata"]["langgraph_step"])
        if self.stop_after and len(self.prompts) >= self.stop_after:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Done."))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="", tool_calls=[{"id": f"inspect-{len(self.prompts)}",
                                     "name": "inspect_item", "args": {}}]))])


@tool
def inspect_item() -> str:
    """Inspect another item."""
    return "The item is still unresolved."


def make_agent():
    model = RecordingModel(responses=[AIMessage(content="", tool_calls=[{
        "id": "inspect", "name": "inspect_item", "args": {},
    }])])
    graph = create_agent(model, tools=[inspect_item],
                         middleware=[ExecutionLimitMiddleware()],
                         checkpointer=InMemorySaver(), system_prompt="Be honest.")
    return model, graph


@pytest.mark.parametrize("stream", [False, True])
def test_relative_budget_warning_once_and_hard_stop(stream):
    model, graph = make_agent()
    config = {"configurable": {"thread_id": "budget"},
              "recursion_limit": EXECUTION_STEP_LIMIT}
    for turn in range(2):
        model.prompts.clear()
        model.steps.clear()
        with pytest.raises(GraphRecursionError):
            if stream:
                list(graph.stream({"messages": [("user", "Inspect the items.")]}, config))
            else:
                graph.invoke({"messages": [("user", "Inspect the items.")]}, config)
        warned = [LIMIT_WARNING in str(p[0].content) for p in model.prompts]
        assert any(warned)
        first = warned.index(True)
        assert first > 150  # Prior checkpoint steps cannot trigger an early warning.
        invocation_start = turn * (EXECUTION_STEP_LIMIT + 2)
        assert 500 <= model.steps[first] - invocation_start <= 504
        assert warned == [False] * first + [True] * (len(warned) - first)
        assert all(str(p[0].content).count(LIMIT_WARNING) <= 1 for p in model.prompts)
        state = graph.get_state(config)
        assert state.values["execution_limit_warned"] is True
        assert not any(LIMIT_WARNING in str(m.content) for m in state.values["messages"])
        assert state.metadata["step"] == (turn + 1) * (EXECUTION_STEP_LIMIT + 2) - 2


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_warning_preserves_approval_and_resume(decision):
    effects = []

    @tool
    def inspect_item() -> str:
        """Perform the requested action."""
        effects.append("executed")
        return "Action completed."

    model = RecordingModel(responses=[], stop_after=2)
    graph = create_agent(model, tools=[inspect_item], checkpointer=InMemorySaver(),
                         middleware=[ExecutionLimitMiddleware(),
                                     HumanInTheLoopMiddleware(interrupt_on={"inspect_item": True})])
    cfg = {"configurable": {"thread_id": "approval"}, "recursion_limit": 50}
    graph.invoke({"messages": [("user", "Perform the action.")]}, cfg)
    assert not effects
    assert graph.get_state(cfg).interrupts
    graph.invoke(Command(resume={"decisions": [{"type": decision}]}), cfg)
    assert effects == (["executed"] if decision == "approve" else [])
    assert all(str(p[0].content).count(LIMIT_WARNING) == 1 for p in model.prompts)
    assert graph.get_state(cfg).values["messages"][-1].content == "Done."
