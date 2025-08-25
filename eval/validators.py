import json
import re
from jsonschema import validate as json_validate, ValidationError
from typing import Any, Dict, Tuple


def v_exact(output: str, target: str) -> Tuple[bool, str]:
    ok = output.strip() == target.strip()
    return ok, "exact match"


def v_regex(output: str, pattern: str) -> Tuple[bool, str]:
    ok = re.search(pattern, output.strip()) is not None
    return ok, f"regex:{pattern}"


def v_max_tokens(output: str, value: int) -> Tuple[bool, str]:
    ok = len(output.split()) <= value
    return ok, f"<= {value} words"


def v_jsonschema(output: str, schema: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        obj = json.loads(output)
        json_validate(instance=obj, schema=schema)
        return True, "jsonschema ok"
    except (json.JSONDecodeError, ValidationError) as e:
        return False, f"jsonschema fail: {e}"


VALIDATORS = {
    "exact": v_exact,
    "regex": v_regex,
    "max_tokens": v_max_tokens,
    "jsonschema": v_jsonschema,
}
