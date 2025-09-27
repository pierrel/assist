"""Render the reflexion agent state graph to a Mermaid diagram."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from assist import reflexion_agent
from assist.reflexion_agent import Plan, PlanRetrospective


class _StructuredDummyLLM:
    """Return minimal objects expected by ``with_structured_output`` calls."""

    def __init__(self, schema: type[BaseModel]):
        self._schema = schema

    def invoke(self, _messages: Iterable[Any], _options: Any | None = None) -> BaseModel:
        schema = self._schema
        if schema is Plan:
            return Plan(goal="graph", steps=[], assumptions=[], risks=[])
        if schema is PlanRetrospective:
            return PlanRetrospective(needs_replan=False, learnings=None)
        return schema()  # type: ignore[call-arg]


class DummyLLM:
    """Lightweight chat model stub sufficient for graph compilation."""

    model = "reflexion-graph-renderer"

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredDummyLLM:
        return _StructuredDummyLLM(schema)

    def invoke(self, _messages: Iterable[Any], _options: Any | None = None) -> AIMessage:
        return AIMessage(content="stub")


class DummyAgent:
    """Minimal agent used to satisfy ``build_reflexion_graph`` dependencies."""

    def invoke(self, _inputs: Any, _options: Any | None = None) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content="stub")]} 


def render_reflexion_graph(output_dir: Path) -> Path:
    """Generate the reflexion agent graph diagram into ``output_dir``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "reflexion_graph.mmd"

    dummy_llm = DummyLLM()
    dummy_agent = DummyAgent()

    original_general_agent = reflexion_agent.general_agent
    try:
        reflexion_agent.general_agent = lambda _llm, _tools: dummy_agent  # type: ignore[assignment]
        graph = reflexion_agent.build_reflexion_graph(dummy_llm, tools=[], callbacks=[])
    finally:
        reflexion_agent.general_agent = original_general_agent

    mermaid = graph.get_graph().draw_mermaid()
    output_path.write_text(mermaid, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the reflexion agent graph.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/reflexion_graph"),
        help="Directory where the Mermaid diagram should be written.",
    )
    args = parser.parse_args()

    output_path = render_reflexion_graph(args.output_dir)
    print(f"Reflexion graph written to {output_path}")


if __name__ == "__main__":
    main()
