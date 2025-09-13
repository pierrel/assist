"""Custom callback handler for readable prompt/response logging."""
from __future__ import annotations

import json
from pprint import pformat
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class ReadableConsoleCallbackHandler(BaseCallbackHandler):
    """Print prompts and responses with node and LLM information.

    The handler captures each LLM call and prints the prompt and final
    response grouped together. Node names are taken from the first tag in
    the run configuration.
    """

    def __init__(self) -> None:  # noqa: D401 - short and simple
        self._runs: Dict[UUID, Dict[str, Any]] = {}
        self._tools: Dict[UUID, Dict[str, Any]] = {}

    @staticmethod
    def _pretty(obj: Any) -> str:
        """Return a human-friendly representation of *obj*.

        If *obj* is a JSON string or a JSON-serializable structure, it is
        formatted with indentation. Otherwise ``str(obj)`` or ``pformat`` is
        used. This keeps LLM outputs like plans or reflexion states readable.
        """

        if obj is None:
            return ""
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except Exception:
                return obj
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except Exception:
            return pformat(obj)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,  # noqa: D417 - required by protocol
        metadata: Optional[Dict[str, Any]] = None,  # noqa: D417 - required by protocol
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "tool")
        self._tools[run_id] = {"name": name, "input": input_str}

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._tools.pop(run_id, {"name": "tool", "input": ""})
        print("Tool:")
        print(f"  Name: {info['name']}")
        print("  Args:")
        print(self._pretty(info["input"]))
        print("  Output:")
        print(self._pretty(output))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,  # noqa: D417 - required by protocol
        **kwargs: Any,
    ) -> None:
        tags = tags or []
        node = tags[0] if tags else "unknown"
        model = (
            serialized.get("kwargs", {}).get("model")
            or serialized.get("kwargs", {}).get("model_name")
            or serialized.get("name", "llm")
        )
        self._runs[run_id] = {"node": node, "model": model, "prompts": prompts}

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._runs.pop(
            run_id,
            cast(Dict[str, Any], {"node": "unknown", "model": "llm", "prompts": []}),
        )
        node = info["node"]
        model = info["model"]
        prompts = info["prompts"]

        print(f"\n===== Node: {node} | LLM: {model} =====")
        for prompt in prompts:
            print("Prompt:")
            print(prompt)
            print()

        print("Response:")
        for gen_list in response.generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                content = getattr(message, "content", getattr(gen, "text", ""))
                formatted = self._pretty(content)
                if formatted:
                    print(formatted)
        print("===== End =====\n")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        tags = kwargs.get("tags") or []
        node = tags[0] if tags else ""
        if node == "plan":
            plan_obj: Any = outputs
            if isinstance(outputs, dict):
                plan_obj = outputs.get("plan", outputs)
            if hasattr(plan_obj, "model_dump"):
                plan_obj = plan_obj.model_dump()
            print("Plan:")
            print(self._pretty(plan_obj))
        elif node == "execute":
            history = outputs.get("history", [])
            resolution: Any = ""
            if history:
                last = history[-1]
                if isinstance(last, dict):
                    resolution = last.get("resolution", "")
                else:
                    resolution = getattr(last, "resolution", "")
            if resolution:
                print("Final Response:")
                print(self._pretty(resolution))
