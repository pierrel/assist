import pytest
from langchain_core.messages import HumanMessage, AIMessage

from assist import reflexion_agent
from assist.reflexion_agent import build_reflexion_graph, Plan, Step, PlanRetrospective


class DummyLLM:
    def __init__(self):
        self.plan_calls = 0
        self.retro_calls = 0
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, _messages, _opts=None):
        if self.schema is Plan:
            self.plan_calls += 1
            self.schema = None
            step = Step(action=f"step{self.plan_calls}", objective="obj")
            return Plan(goal="goal", steps=[step], assumptions=[], risks=[])
        elif self.schema is PlanRetrospective:
            self.retro_calls += 1
            self.schema = None
            if self.retro_calls == 1:
                return PlanRetrospective(needs_replan=True, learnings="learn")
            else:
                return PlanRetrospective(needs_replan=False, learnings=None)
        else:
            return AIMessage(content="summary")


class DummyAgent:
    def __init__(self):
        self.count = 0

    def invoke(self, _inputs, _opts=None):
        self.count += 1
        return {"messages": [AIMessage(content=f"result{self.count}")]}


def test_replanning_flow(monkeypatch):
    llm = DummyLLM()
    dummy_agent = DummyAgent()

    def fake_general_agent(_llm, _tools):
        return dummy_agent

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    final_state = graph.invoke({"messages": [HumanMessage(content="do task")]})

    assert dummy_agent.count == 2
    assert final_state["learnings"] == ["learn"]
