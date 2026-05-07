"""Tests for ``assist.agent._make_retry_middleware``.

Pins the ``ASSIST_LLM_MAX_RETRIES`` env override and the default of 3
retries — the project's single source of truth for transient-error
retry counts after the OpenAI SDK's own ``max_retries=0`` is set in
``assist/model_manager.py``.
"""
import os
from unittest.mock import patch

from assist.agent import _make_retry_middleware


def test_default_max_retries_is_three():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ASSIST_LLM_MAX_RETRIES", None)
        mw = _make_retry_middleware()
    assert mw.max_retries == 3


def test_env_override_lowers_max_retries():
    with patch.dict(os.environ, {"ASSIST_LLM_MAX_RETRIES": "1"}):
        mw = _make_retry_middleware()
    assert mw.max_retries == 1


def test_env_override_zero_disables_retries():
    with patch.dict(os.environ, {"ASSIST_LLM_MAX_RETRIES": "0"}):
        mw = _make_retry_middleware()
    assert mw.max_retries == 0


def test_garbage_env_falls_back_to_default():
    # A typo in .deploy.env (e.g. "thee" instead of "three") should
    # silently fall back to the default rather than crash the agent
    # factory on first thread build.
    with patch.dict(os.environ, {"ASSIST_LLM_MAX_RETRIES": "thee"}):
        mw = _make_retry_middleware()
    assert mw.max_retries == 3
