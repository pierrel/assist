import copy
import json
import re
import sys
import time
import uuid
import importlib
import pathlib
from dataclasses import dataclass, asdict
from typing import Any

# Ensure the src/ directory is on the import path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from .types import Validation, Check

MODULES = [
    "eval.reflexion",
    "eval.planner",
    "eval.plan_checker",
    "eval.step_executor",
    "eval.summarizer",
]


def _convert(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, BaseMessage):
        return {"type": obj.type, "content": obj.content}
    if isinstance(obj, list):
        return [_convert(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    return obj


def run_one(graph, state: dict) -> tuple[str, float, str | None]:
    t0 = time.perf_counter()
    try:
        out_state = graph.invoke(copy.deepcopy(state))
        wall_ms = (time.perf_counter() - t0) * 1000
        output_text = json.dumps(_convert(out_state))
        return output_text, wall_ms, None
    except Exception as e:  # pragma: no cover - safety
        return "", (time.perf_counter() - t0) * 1000, str(e)


def score_output(output: str, check: Check) -> float:
    if isinstance(check, re.Pattern):
        return 1.0 if check.search(output) else 0.0
    return float(check(output))


@dataclass
class EvalRecord:
    run_id: str
    node: str
    test_id: str
    score: float
    wall_ms: float
    raw_output: str
    error: str | None


def run(out_path: pathlib.Path) -> None:
    run_id = uuid.uuid4().hex
    with out_path.open("a") as outf:
        for mod_name in MODULES:
            mod = importlib.import_module(mod_name)
            graph = getattr(mod, "GRAPH")
            validations: list[Validation] = getattr(mod, "VALIDATIONS")
            for idx, val in enumerate(validations):
                output, wall_ms, error = run_one(graph, val.input)
                score = score_output(output, val.check)
                rec = EvalRecord(
                    run_id=run_id,
                    node=mod_name.split(".")[-1],
                    test_id=str(idx),
                    score=score,
                    wall_ms=wall_ms,
                    raw_output=output,
                    error=error,
                )
                outf.write(json.dumps(asdict(rec)) + "\n")
                print(f"[{mod_name}:{idx}] score={score:.2f} time={wall_ms:.0f}ms")


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def main(out_file: pathlib.Path):
        run(out_file)

    app()
