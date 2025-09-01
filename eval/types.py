from dataclasses import dataclass
from typing import Any, Callable, Dict, Pattern, Union

Check = Union[Pattern[str], Callable[[str], float]]

@dataclass
class Validation:
    """Input and validation pair for a node evaluation."""
    input: Dict[str, Any]
    check: Check
