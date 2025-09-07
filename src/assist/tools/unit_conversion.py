from __future__ import annotations

from langchain_core.tools import BaseTool

try:
    import pint
except Exception:  # pragma: no cover - import guard
    pint = None


class UnitConversionTool(BaseTool):
    """Convert quantities between common units.

    Uses the ``pint`` library to handle conversions for temperature, volume,
    mass, and time units. Inputs are ``value``, ``from_unit`` and ``to_unit``.
    Returns the converted value and units as a string.
    """

    name: str = "unit_convert"
    description: str = (
        "Convert quantities between units. "
        "Args: value (float), from_unit (str), to_unit (str)."
    )

    def __init__(self) -> None:
        super().__init__()
        self._ureg = pint.UnitRegistry() if pint else None

    def _run(self, value: float, from_unit: str, to_unit: str) -> str:
        if self._ureg is None:
            return "pint library not available"
        try:
            qty = value * self._ureg(from_unit)
            converted = qty.to(to_unit)
        except Exception as exc:  # pragma: no cover - error path
            return f"Error: {exc}"
        return f"{converted.magnitude} {converted.units}"

    async def _arun(self, *args, **kwargs) -> str:  # pragma: no cover - sync tool
        raise NotImplementedError


__all__ = ["UnitConversionTool"]
