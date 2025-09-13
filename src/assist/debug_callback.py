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
