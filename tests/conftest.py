"""Shared pytest fixtures for the unit-test suite.

These fixtures protect tests from accidentally networking to the live
LLM endpoint.  ``assist.model_manager`` caches an ``OpenAIConfig`` in a
module-level global; without this fixture, a test that mutates the
cache (e.g. via ``invalidate_config_cache``) would leak state into the
next test, and any test that triggers ``select_chat_model`` through an
agent factory would attempt an HTTP probe against the real endpoint.

The fixture is autouse so every test in ``tests/`` gets the clean
state.  Tests that *want* to exercise the real probe path can
monkeypatch ``_probe_endpoint`` themselves.
"""
from __future__ import annotations

import pytest

from assist import model_manager


@pytest.fixture(autouse=True)
def _clean_model_manager_cache(monkeypatch):
    """Stub the probe and clean the cache after each test.

    The stub returns a synthetic ``OpenAIConfig`` so any code path that
    transitively reaches ``select_chat_model`` gets a deterministic
    result without networking.  Tests that need to assert on probe
    behavior should override the monkeypatch with their own.
    """

    def _fake_probe(url: str, api_key: str) -> model_manager.OpenAIConfig:
        return model_manager.OpenAIConfig(
            url=url,
            model="test-model",
            api_key=api_key,
            context_len=32768,
        )

    monkeypatch.setattr(model_manager, "_probe_endpoint", _fake_probe)
    yield
    model_manager._cached_config = None
