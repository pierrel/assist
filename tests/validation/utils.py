import re
from typing import Any

from langchain_core.messages import AIMessage

from assist.reflexion_agent import Plan, Step, PlanRetrospective
from eval.types import Validation


class DummyLLM:
    """Minimal stand‑in for chat models used in validation tests."""

    def __init__(self, message: str = "ok") -> None:
        self.schema: Any = None
        self.message = message

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, _messages, _opts: Any | None = None):
        if self.schema is Plan:
            self.schema = None
            steps = [
                Step(action="tavily_search", objective="find info"),
                Step(action="other", objective="second"),
                Step(action="more", objective="third"),
            ]
            return Plan(
                goal="goal", steps=steps, assumptions=["assumption"], risks=["risk"]
            )
        if self.schema is PlanRetrospective:
            self.schema = None
            return PlanRetrospective(needs_replan=False, learnings=None)
        self.schema = None
        return AIMessage(content=self.message)


class DummyAgent:
    """Return a canned message to simulate step execution."""

    def __init__(self, message: str = "result") -> None:
        self.message = message

    def invoke(self, _inputs, _opts: Any | None = None):
        return {"messages": [AIMessage(content=self.message)]}


def run_validation(graph, validation: Validation) -> bool:
    """Execute ``graph`` with ``validation.input`` and evaluate ``validation.check``.

    ``graph`` can be a simple callable or an object exposing ``invoke``.
    Returns ``True`` if the check passes, otherwise ``False``.
    """
    runner = getattr(graph, "invoke", graph)
    output = runner(validation.input)
    check = validation.check
    if isinstance(check, re.Pattern):
        return bool(check.search(output))
    return bool(check(output))
