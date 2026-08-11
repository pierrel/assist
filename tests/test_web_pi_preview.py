"""The visible Pi choice is host-gated before a thread can be reserved."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from manage.web import threads


class _Preview:
    def __init__(self, admits: bool) -> None:
        self._admits = admits

    def claim_admits(self, engine: str) -> bool:
        assert engine == "pi"
        return self._admits


def test_new_thread_engine_defaults_to_deep_and_rejects_unknown(monkeypatch) -> None:
    assert threads._require_new_thread_engine("deepagents") == "deepagents"
    with pytest.raises(HTTPException) as error:
        threads._require_new_thread_engine("anything")
    assert error.value.status_code == 400


def test_new_pi_thread_requires_fresh_host_admission(monkeypatch) -> None:
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(False))
    with pytest.raises(HTTPException) as error:
        threads._require_new_thread_engine("pi")
    assert error.value.status_code == 503
    monkeypatch.setattr(threads, "PI_PREVIEW", _Preview(True))
    assert threads._require_new_thread_engine("pi") == "pi"


def test_merge_refuses_pi_before_constructing_a_deep_thread(monkeypatch) -> None:
    monkeypatch.setattr(threads, "_is_pi_thread", lambda tid: True)
    monkeypatch.setattr(
        threads.MANAGER, "get",
        lambda *args, **kwargs: pytest.fail("Pi must not construct a Deep thread"))

    with pytest.raises(HTTPException) as error:
        threads.merge_thread("pi-thread")

    assert error.value.status_code == 409
