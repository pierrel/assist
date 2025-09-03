import re
import pytest

from eval.types import Validation
from .utils import run_validation


def greet_node(state: dict) -> str:
    return f"Hello, {state['name']}!"


GRAPH = greet_node

VALIDATIONS = [
    Validation(input={"name": "Alice"}, check=re.compile(r"Hello, Alice!")),
    Validation(input={"name": "Bob"}, check=lambda out: out.endswith("Bob!")),
]


@pytest.mark.validation
@pytest.mark.parametrize("validation", VALIDATIONS)
def test_greet_node(validation: Validation) -> None:
    assert run_validation(GRAPH, validation)
