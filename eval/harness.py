import json
import time
import uuid
import importlib
import pathlib
import yaml
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Callable, Optional

from .validators import VALIDATORS
from .hydrate import hydrate


def _load_from_path(path: pathlib.Path) -> Any:
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        data = path.read_text()
    return _resolve_refs(data, path.parent)


def _resolve_refs(obj: Any, base_dir: pathlib.Path) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k.endswith("_path") and isinstance(v, str):
                ref = base_dir / v
                out[k[:-5]] = _load_from_path(ref)
            else:
                out[k] = _resolve_refs(v, base_dir)
        return out
    if isinstance(obj, list):
        return [_resolve_refs(i, base_dir) for i in obj]
    return obj


def load_graph(dotted: str):
    mod_name, fn = dotted.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn)()



@dataclass
class EvalRecord:
    run_id: str
    dataset: str
    node: str
    test_id: str
    ok: bool
    score: float
    wall_ms: float
    steps: int
    tool_calls: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    raw_output: str
    error: Optional[str]
    meta: Dict[str, Any]


def run_one(graph, in_obj: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out_state = graph.invoke(in_obj)
        wall_ms = (time.perf_counter() - t0) * 1000
        output_text = out_state.get("answer") or out_state.get("output") or json.dumps(out_state)
        steps = out_state.get("_trace_steps", 0)
        tool_calls = out_state.get("_trace_tool_calls", 0)
        usage = out_state.get("_llm_usage", {})
        return dict(output=output_text, wall_ms=wall_ms, steps=steps, tool_calls=tool_calls, usage=usage, error=None)
    except Exception as e:
        return dict(output="", wall_ms=(time.perf_counter()-t0)*1000, steps=0, tool_calls=0, usage={}, error=str(e))


def grade(output: str, expect: Dict[str, Any]) -> tuple[bool, float, List[str]]:
    notes = []
    results = []
    for v in expect.get("validators", []):
        vtype = v["type"]
        fn = VALIDATORS[vtype]
        args = {k: v[k] for k in v.keys() if k not in ("type")}
        ok, note = fn(output, **args) if args else fn(output)
        results.append(ok)
        notes.append(f"{'✓' if ok else '✗'} {note}")
    ok = all(results) if results else True
    score = sum(results)/len(results) if results else 1.0
    return ok, score, notes


GRAPH_MAP = {
    "reflexion": "assist.reflexion_agent:reflexion_graph_v1",
    "planner": "assist.reflexion_agent:planner_graph_v1",
    "plan_check": "assist.reflexion_agent:plan_checker_graph_v1",
    "step_executor": "assist.reflexion_agent:step_executor_graph_v1",
    "summarizer": "assist.reflexion_agent:summarizer_graph_v1",
}


def _infer_graph(dataset: pathlib.Path) -> tuple[str, str]:
    stem = dataset.stem
    for prefix, graph in GRAPH_MAP.items():
        if stem.startswith(prefix):
            node = prefix
            if prefix == "plan_check":
                node = "plan_checker"
            return node, graph
    raise ValueError(f"Cannot infer graph for dataset {dataset.name}")


def run(
    dataset: pathlib.Path,
    out: pathlib.Path = pathlib.Path("eval_results.jsonl"),
):
    node, graph_dotted = _infer_graph(dataset)
    graph = load_graph(graph_dotted)
    tests = yaml.safe_load(dataset.read_text())
    tests = [_resolve_refs(t, dataset.parent) for t in tests]
    run_id = uuid.uuid4().hex
    dataset_name = dataset.stem
    outf = out.open("a")
    for t in tests:
        base_state = t["input"]
        state = hydrate(base_state)
        res = run_one(graph, state)
        ok, score, notes = grade(res["output"], t.get("expect", {}))
        rec = EvalRecord(
            run_id=run_id,
            dataset=dataset_name,
            node=node,
            test_id=t["id"],
            ok=ok,
            score=score,
            wall_ms=res["wall_ms"],
            steps=res["steps"],
            tool_calls=res["tool_calls"],
            prompt_tokens=res["usage"].get("prompt_tokens"),
            completion_tokens=res["usage"].get("completion_tokens"),
            raw_output=res["output"],
            error=res["error"],
            meta={"notes": notes, "graph": graph_dotted},
        )
        outf.write(json.dumps(asdict(rec)) + "\n")
        print(f"[{t['id']}] {'OK' if ok else 'FAIL'} score={score:.2f} time={rec.wall_ms:.0f}ms {' | '.join(notes)}")
    outf.close()


if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def cli(dataset: pathlib.Path, out: pathlib.Path = pathlib.Path("eval_results.jsonl")):
        run(dataset=dataset, out=out)

    app()
