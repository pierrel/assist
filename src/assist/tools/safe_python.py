from __future__ import annotations

"""Execute Python code in a restricted, read-only environment."""

import builtins
import contextlib
import io
from typing import Dict, Any
from langchain_core.tools import BaseTool


class SafePythonTool(BaseTool):
    """Execute Python code without filesystem side effects."""

    name: str = "python"
    description: str = ""

    def __init__(self) -> None:
        allowed_builtins = {
            name: getattr(builtins, name)
            for name in [
                "abs",
                "min",
                "max",
                "sum",
                "len",
                "range",
                "enumerate",
                "zip",
                "map",
                "filter",
                "sorted",
                "round",
                "all",
                "any",
                "print",
                "float",
                "int",
                "str",
                "bool",
                "dict",
                "list",
                "tuple",
                "set",
            ]
        }
        allowed_modules = {"math", "statistics", "random"}
        for mod in ("numpy",):
            try:
                __import__(mod)
            except Exception:  # pragma: no cover - optional dependency
                continue
            allowed_modules.add(mod)

        def _safe_import(
            name: str,
            globals: Any | None = None,
            locals: Any | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name in allowed_modules:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Module '{name}' not allowed")

        allowed_builtins["__import__"] = _safe_import

        builtins_str = ", ".join(sorted(b for b in allowed_builtins if b != "__import__"))
        modules_str = ", ".join(sorted(allowed_modules))
        description = (
            "Execute Python code in a sandboxed environment. "
            "No filesystem or network access. "
            f"Builtins: {builtins_str}. "
            f"Modules: {modules_str}."
        )
        super().__init__(description=description)

        self._globals: Dict[str, Any] = {"__builtins__": allowed_builtins}
        for mod in allowed_modules:
            try:
                self._globals[mod] = __import__(mod)
            except Exception:  # pragma: no cover - import guard
                pass

    def _run(self, code: str) -> str:
        locals_dict: Dict[str, Any] = {}
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, self._globals, locals_dict)
        except Exception as exc:  # pragma: no cover - errors handled
            return f"Error: {exc}"
        output = stdout.getvalue().strip()
        result = locals_dict.get("result")
        if result is not None:
            result_str = repr(result)
            return f"{output}\n{result_str}" if output else result_str
        return output or "None"

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - sync only
        raise NotImplementedError


__all__ = ["SafePythonTool"]
