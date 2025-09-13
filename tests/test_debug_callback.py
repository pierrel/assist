import importlib.util
import json
from pathlib import Path
from uuid import uuid4

from langchain_core.outputs import LLMResult, Generation

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
