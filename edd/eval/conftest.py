import os
from functools import wraps

import pytest

from assist.promptable import env


def pytest_configure():
    """Optionally force one reasoning mode for a focused model comparison.

    The production-facing model factory defaults reasoning off.  A migration
    experiment can set ``ASSIST_EVAL_REASONING_MODE=on`` to make eval modules
    imported afterwards use the otherwise identical reasoning-on factory.
    This is opt-in and eval-only: ordinary tests and Assist itself retain the
    production default.
    """
    mode = os.environ.get("ASSIST_EVAL_REASONING_MODE")
    if mode is None:
        return
    if mode not in {"off", "on"}:
        raise pytest.UsageError(
            "ASSIST_EVAL_REASONING_MODE must be 'off' or 'on'"
        )

    import assist.model_manager as model_manager

    original = model_manager.select_assistant_model
    enabled = mode == "on"

    @wraps(original)
    def select_assistant_model(temperature, *, enable_thinking=False):
        return original(temperature, enable_thinking=enabled)

    # Eval modules import this factory during collection, after this hook.
    # Rebinding the module here therefore changes only this pytest process.
    model_manager.select_assistant_model = select_assistant_model
    print(f"Forcing eval reasoning mode: {mode}")


@pytest.fixture(autouse=True)
def fixed_eval_datetime(monkeypatch):
    """Freeze rendered prompt time only for an explicitly controlled eval run."""
    value = os.environ.get("ASSIST_EVAL_FIXED_DATETIME")
    if value is not None:
        monkeypatch.setitem(env.globals, "current_datetime", lambda: value)
