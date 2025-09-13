import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.errors import GraphRecursionError

from assist import reflexion_agent
from assist.reflexion_agent import build_reflexion_graph, Plan, Step, PlanRetrospective, StepResolution


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


def test_plan_check_intervals(monkeypatch):
    class IntervalLLM:
        def __init__(self):
            self.schema = None
            self.retro_calls = 0

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, _messages, _opts=None):
            if self.schema is Plan:
                self.schema = None
                steps = [Step(action=f"step{i}", objective="obj") for i in range(6)]
                return Plan(goal="goal", steps=steps, assumptions=[], risks=[])
            elif self.schema is PlanRetrospective:
                self.retro_calls += 1
                self.schema = None
                return PlanRetrospective(needs_replan=False, learnings=None)
            else:
                return AIMessage(content="summary")

    llm = IntervalLLM()
    dummy_agent = DummyAgent()

    def fake_general_agent(_llm, _tools):
        return dummy_agent

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    graph.invoke({"messages": [HumanMessage(content="do task")]})

    assert llm.retro_calls == 3
    assert dummy_agent.count == 6


def test_plan_node_outputs_initial_state(monkeypatch):
    class LLM:
        def __init__(self):
            self.schema = None
            self.calls: list[list] = []

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages, _opts=None):
            self.calls.append(messages)
            step = Step(action="a", objective="o")
            return Plan(goal="g", steps=[step], assumptions=[], risks=[])

    llm = LLM()

    def fake_general_agent(_llm, _tools):
        return DummyAgent()

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    plan_node = graph.builder.nodes["plan"].runnable.func

    state = {"messages": [HumanMessage(content="do task")], "learnings": []}
    out = plan_node(state)

    assert isinstance(out["plan"], Plan)
    assert out["step_index"] == 0
    assert out["history"] == []
    assert out["needs_replan"] is False
    assert out["plan_check_needed"] is False
    assert out["learnings"] == []
    assert "do task" in llm.calls[0][1].content


def test_execute_node_passes_history_to_agent(monkeypatch):
    class TwoStepLLM:
        def __init__(self):
            self.schema = None

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, _messages, _opts=None):
            steps = [
                Step(action="step1", objective="obj1"),
                Step(action="step2", objective="obj2"),
            ]
            return Plan(goal="g", steps=steps, assumptions=[], risks=[])

    class SpyAgent:
        def __init__(self):
            self.calls = []

        def invoke(self, inputs, _opts=None):
            self.calls.append(inputs["messages"])
            idx = len(self.calls)
            return {"messages": [AIMessage(content=f"result{idx}")]}

    llm = TwoStepLLM()
    agent = SpyAgent()

    def fake_general_agent(_llm, _tools):
        return agent

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    plan_node = graph.builder.nodes["plan"].runnable.func
    execute_node = graph.builder.nodes["execute"].runnable.func

    state = {"messages": [HumanMessage(content="task")], "learnings": []}
    state.update(plan_node(state))

    out1 = execute_node(state)
    state.update(out1)

    out2 = execute_node(state)

    assert len(agent.calls) == 2
    first_call = agent.calls[0][-1].content
    assert len(agent.calls[0]) == 2
    assert "step1" in first_call
    second_call = agent.calls[1][-1].content
    assert "result1" in second_call
    assert out1["history"][0].resolution.endswith("result1")
    assert out2["history"][1].resolution.endswith("result2")


def test_plan_check_updates_state(monkeypatch):
    class RetroLLM:
        def __init__(self):
            self.schema = None
            self.calls = []

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages, _opts=None):
            self.calls.append(messages)
            return PlanRetrospective(needs_replan=True, learnings="learn")

    llm = RetroLLM()

    def fake_general_agent(_llm, _tools):
        return DummyAgent()

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    plan_check = graph.builder.nodes["plan_check"].runnable.func
    plan = Plan(goal="g", steps=[Step(action="a", objective="o")], assumptions=[], risks=[])
    state = {
        "messages": [HumanMessage(content="task")],
        "plan": plan,
        "history": ["Step(action='a', objective='o'): result"],
        "step_index": 1,
        "needs_replan": False,
        "learnings": [],
    }

    out = plan_check(state)

    assert out["needs_replan"] is True
    assert out["learnings"] == ["learn"]
    assert "overall user task" in llm.calls[0][1].content


def test_summarize_node_appends_message(monkeypatch):
    class SummLLM:
        def __init__(self):
            self.calls = []

        def invoke(self, messages, _opts=None):
            self.calls.append(messages)
            return AIMessage(content="summary")

        def with_structured_output(self, schema):
            return self

    llm = SummLLM()

    def fake_general_agent(_llm, _tools):
        return DummyAgent()

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    summarize = graph.builder.nodes["summarize"].runnable.func

    state = {
        "messages": [HumanMessage(content="task")],
        "history": [StepResolution(action='a', objective='o', resolution='result')]
    }

    out = summarize(state)

    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == "summary"
    assert "result" in llm.calls[0][-1].content


def test_build_reflexion_graph_allows_separate_execution_llm(monkeypatch):
    plan_llm = object()
    exec_llm = object()

    class DummyAgent:
        pass

    def fake_general_agent(llm, _tools):
        fake_general_agent.called_with = llm
        return DummyAgent()

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(plan_llm, [], execution_llm=exec_llm)

    assert isinstance(graph, Runnable)
    assert fake_general_agent.called_with is exec_llm


def test_context_and_system_message_routing(monkeypatch):
    class SpyLLM:
        def __init__(self):
            self.schema = None
            self.plan_messages = None
            self.retro_messages = None
            self.summary_messages = None

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages, _opts=None):
            if self.schema is Plan:
                self.plan_messages = messages
                step = Step(action="a", objective="o")
                self.schema = None
                return Plan(goal="g", steps=[step], assumptions=[], risks=[])
            elif self.schema is PlanRetrospective:
                self.retro_messages = messages
                self.schema = None
                return PlanRetrospective(needs_replan=False, learnings=None)
            else:
                self.summary_messages = messages
                return AIMessage(content="summary")

    class SpyAgent:
        def __init__(self):
            self.calls = []

        def invoke(self, inputs, _opts=None):
            self.calls.append(inputs["messages"])
            return {"messages": [AIMessage(content="result")]} 

    llm = SpyLLM()
    agent = SpyAgent()

    def fake_general_agent(_llm, _tools):
        return agent

    monkeypatch.setattr(reflexion_agent, "general_agent", fake_general_agent)

    graph = build_reflexion_graph(llm, [])
    plan_node = graph.builder.nodes["plan"].runnable.func
    execute_node = graph.builder.nodes["execute"].runnable.func
    plan_check = graph.builder.nodes["plan_check"].runnable.func
    summarize = graph.builder.nodes["summarize"].runnable.func

    state = {
        "messages": [
            SystemMessage(content="sys"),
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
            HumanMessage(content="task"),
        ],
        "learnings": [],
    }

    state.update(plan_node(state))
    execute_node(state)
    plan_check(state)
    summarize(state)

    # Planner sees system message and prior context
    plan_sys = llm.plan_messages[0].content
    assert "sys" in plan_sys
    assert "Here is guidance from the user:" in plan_sys
    assert any(isinstance(m, HumanMessage) and m.content == "hello" for m in llm.plan_messages)
    assert any(isinstance(m, AIMessage) and m.content == "hi" for m in llm.plan_messages)

    # Executor does not receive conversation context
    exec_text = " ".join(m.content for m in agent.calls[0])
    assert "hello" not in exec_text and "sys" not in exec_text

    # Plan check also ignores conversation context
    retro_text = " ".join(m.content for m in llm.retro_messages)
    assert "hello" not in retro_text and "sys" not in retro_text

    # Summarizer sees system message and prior context
    summary_sys = llm.summary_messages[0].content
    assert "sys" in summary_sys
    assert "Here is guidance from the user:" in summary_sys
    assert any(isinstance(m, HumanMessage) and m.content == "hello" for m in llm.summary_messages)
    assert any(isinstance(m, AIMessage) and m.content == "hi" for m in llm.summary_messages)


def test_execute_node_handles_recursion(monkeypatch):
    class LLM:
        def __init__(self):
            self.schema = None

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, _messages, _opts=None):
            step = Step(action="run", objective="obj")
            return Plan(goal="g", steps=[step], assumptions=[], risks=[])

    class FailingAgent:
        def invoke(self, _inputs, _opts=None):
            raise GraphRecursionError("boom")

    monkeypatch.setattr(
        reflexion_agent, "general_agent", lambda _llm, _tools: FailingAgent()
    )

    graph = build_reflexion_graph(LLM(), [])
    plan_node = graph.builder.nodes["plan"].runnable.func
    execute_node = graph.builder.nodes["execute"].runnable.func

    state = {"messages": [HumanMessage(content="task")], "learnings": []}
    state.update(plan_node(state))
    out = execute_node(state)

    assert out["plan_check_needed"] is True
    assert out["step_index"] == 1
    assert len(out["history"]) == 1
    assert "run" in out["history"][0].resolution
    assert reflexion_agent.after_execute(out) == "plan_check"


def test_reflexion_graph_handles_recursion_error(monkeypatch):
    class LoopingLLM:
        def __init__(self):
            self.schema = None

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, _messages, _opts=None):
            if self.schema is Plan:
                self.schema = None
                step = Step(action="run", objective="obj")
                return Plan(goal="goal", steps=[step], assumptions=[], risks=[])
            elif self.schema is PlanRetrospective:
                self.schema = None
                return PlanRetrospective(needs_replan=True, learnings=None)
            else:
                return AIMessage(content="Need more info?")

    dummy_agent = DummyAgent()
    monkeypatch.setattr(
        reflexion_agent, "general_agent", lambda _llm, _tools: dummy_agent
    )

    graph = build_reflexion_graph(LoopingLLM(), [])
    result = graph.invoke({"messages": [HumanMessage(content="task")]})
    last = result["messages"][-1]

    assert isinstance(last, AIMessage)
    assert last.content.endswith("?")
