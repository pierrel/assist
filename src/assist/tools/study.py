from __future__ import annotations

"""Tool wrapper around the study agent."""

from pathlib import Path
from typing import List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable

from assist.study_agent import build_study_graph


class StudyToolInput(BaseModel):
    objective: str = Field(description="Objective guiding the study")
    resources: List[str] = Field(description="File paths or URLs to study")
    filename: str = Field(
        description="Name of the file to write notes to", default="study_notes.md"
    )


class StudyTool(BaseTool):
    name: str = "study"
    description: str = (
        "Study a list of resources and write notes relevant to an objective to a file"
    )
    args_schema = StudyToolInput

    def __init__(self, llm: Runnable, output_dir: Path):
        super().__init__()
        self._llm = llm
        self._output_dir = Path(output_dir)

    def _run(self, objective: str, resources: List[str], filename: str = "study_notes.md") -> str:
        graph = build_study_graph(self._llm)
        out_path = self._output_dir / filename
        state = {
            "objective": objective,
            "resources": resources,
            "index": 0,
            "notes": [],
            "output_path": str(out_path),
        }
        graph.invoke(state)
        return str(out_path)

    async def _arun(self, *args, **kwargs):  # pragma: no cover - sync only
        raise NotImplementedError("StudyTool does not support async")


__all__ = ["StudyTool", "StudyToolInput"]
