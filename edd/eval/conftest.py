import os

import pytest

from assist.promptable import env


@pytest.fixture(autouse=True)
def fixed_eval_datetime(monkeypatch):
    """Freeze rendered prompt time only for an explicitly controlled eval run."""
    value = os.environ.get("ASSIST_EVAL_FIXED_DATETIME")
    if value is None:
        yield
        return

    monkeypatch.setitem(env.globals, "current_datetime", lambda: value)
    yield
