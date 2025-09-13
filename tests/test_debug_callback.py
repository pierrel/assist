import importlib.util
import json
from pathlib import Path
from uuid import uuid4

from langchain_core.outputs import LLMResult, Generation
from assist.reflexion_agent import Plan, Step

SPEC = importlib.util.spec_from_file_location(
    "debug_callback", Path(__file__).resolve().parents[1] / "src" / "assist" / "debug_callback.py"
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)
ReadableConsoleCallbackHandler = module.ReadableConsoleCallbackHandler


def test_pretty_print_json(capsys):
    handler = ReadableConsoleCallbackHandler()
    run_id = uuid4()
    handler.on_llm_start({"name": "test"}, ["hi"], run_id=run_id, tags=["demo"])
    data = {"foo": {"bar": 1}}
    result = LLMResult(generations=[[Generation(text=json.dumps(data))]])
    handler.on_llm_end(result, run_id=run_id)
    out = capsys.readouterr().out
    assert json.dumps(data, indent=2) in out


def test_plan_and_execute_and_tool(capsys):
    handler = ReadableConsoleCallbackHandler()
    # Plan output
    plan = {"steps": [{"action": "a", "objective": "b"}]}
    handler.on_chain_end({"plan": plan}, run_id=uuid4(), tags=["plan"])
    # Tool usage
    tool_run = uuid4()
    handler.on_tool_start({"name": "tool-a"}, json.dumps({"x": 1}), run_id=tool_run)
    handler.on_tool_end({"result": 2}, run_id=tool_run)
    # Final response from execute node
    handler.on_chain_end({"history": [{"resolution": "done"}]}, run_id=uuid4(), tags=["execute"])
    out = capsys.readouterr().out
    assert "Plan:" in out
    assert json.dumps(plan, indent=2) in out
    assert "tool-a" in out and '"x": 1' in out and '"result": 2' in out
    assert "Final Response:" in out and "done" in out


def test_plan_object_handled(capsys):
    handler = ReadableConsoleCallbackHandler()
    plan = Plan(goal="g", steps=[Step(action="a", objective="b")], assumptions=[], risks=[])
    handler.on_chain_end(plan, run_id=uuid4(), tags=["plan"])
    out = capsys.readouterr().out
    assert "Plan:" in out and "g" in out
