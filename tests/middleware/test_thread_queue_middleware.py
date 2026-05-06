"""Tests for ThreadQueueMiddleware."""
from unittest.mock import MagicMock

import pytest

from assist.middleware.thread_queue_middleware import ThreadQueueMiddleware
from assist.thread_queue import ThreadHoldExpired, _Handle, _active_handle


def test_after_model_passes_through_when_no_active_handle():
    mw = ThreadQueueMiddleware()
    # Default ContextVar state: no handle.
    result = mw.after_model(MagicMock(), MagicMock())
    assert result is None


def test_after_model_passes_through_when_handle_not_expired():
    mw = ThreadQueueMiddleware()
    handle = _Handle("A")
    token = _active_handle.set(handle)
    try:
        result = mw.after_model(MagicMock(), MagicMock())
        assert result is None
    finally:
        _active_handle.reset(token)


def test_after_model_raises_when_handle_expired():
    mw = ThreadQueueMiddleware()
    handle = _Handle("A")
    handle.expired = True
    token = _active_handle.set(handle)
    try:
        with pytest.raises(ThreadHoldExpired) as ctx:
            mw.after_model(MagicMock(), MagicMock())
        assert "A" in str(ctx.value)
    finally:
        _active_handle.reset(token)
