"""Compatibility shim to expose the `src/assist` package without installation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _PACKAGE_ROOT / "src" / "assist"
_INIT_FILE = _SRC_PACKAGE / "__init__.py"

_spec = importlib.util.spec_from_file_location(
    __name__, str(_INIT_FILE), submodule_search_locations=[str(_SRC_PACKAGE)]
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load assist package from {_INIT_FILE!s}")

globals()["__file__"] = str(_INIT_FILE)
if _spec.submodule_search_locations is not None:
    globals()["__path__"] = list(_spec.submodule_search_locations)
globals()["__spec__"] = _spec

_spec.loader.exec_module(sys.modules[__name__])
