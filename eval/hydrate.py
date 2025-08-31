from typing import Any, Dict, List

from assist.reflexion_agent import Plan, StepResolution

HYDRATORS: Dict[str, Any] = {
    "plan": Plan,
    "history": StepResolution,
}


def hydrate(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state.items():
        model = HYDRATORS.get(k)
        if model is None:
            out[k] = v
            continue
        if isinstance(v, list):
            out[k] = [model.model_validate(i) for i in v]
        elif isinstance(v, dict):
            out[k] = model.model_validate(v)
        else:
            out[k] = v
    return out
