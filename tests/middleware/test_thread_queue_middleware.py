"""Tests for ThreadQueueMiddleware."""
from unittest.mock import MagicMock

import pytest

from assist.middleware.thread_queue_middleware import ThreadQueueMiddleware
from assist.thread_queue import (
    ThreadHoldExpired,
    ThreadPauseRequested,
    _Handle,
    _active_handle,
)


def _handle(tid="A"):
    return _Handle(tid, quantum_s=600.0, hold_timeout_s=7200.0)


def test_after_model_passes_through_when_no_active_handle():
    mw = ThreadQueueMiddleware()
    # Default ContextVar state: no handle.
    result = mw.after_model(MagicMock(), MagicMock())
    assert result is None


def test_after_model_passes_through_when_handle_not_expired():
    mw = ThreadQueueMiddleware()
    handle = _handle()
    token = _active_handle.set(handle)
    try:
        result = mw.after_model(MagicMock(), MagicMock())
        assert result is None
    finally:
        _active_handle.reset(token)


def test_after_model_raises_when_handle_expired():
    mw = ThreadQueueMiddleware()
    handle = _handle()
    handle.expired = True
    token = _active_handle.set(handle)
    try:
        with pytest.raises(ThreadHoldExpired) as ctx:
            mw.after_model(MagicMock(), MagicMock())
        assert "A" in str(ctx.value)
    finally:
        _active_handle.reset(token)


def test_after_model_raises_pause_when_pause_requested():
    mw = ThreadQueueMiddleware()
    handle = _handle()
    handle.pause_requested = True
    token = _active_handle.set(handle)
    try:
        with pytest.raises(ThreadPauseRequested) as ctx:
            mw.after_model(MagicMock(), MagicMock())
        assert "A" in str(ctx.value)
    finally:
        _active_handle.reset(token)


def test_after_model_expired_wins_over_pause():
    # The tick never sets both, but expired is checked first as belt-and-suspenders:
    # a kill must never be demoted to a resumable pause.
    mw = ThreadQueueMiddleware()
    handle = _handle()
    handle.expired = True
    handle.pause_requested = True
    token = _active_handle.set(handle)
    try:
        with pytest.raises(ThreadHoldExpired):
            mw.after_model(MagicMock(), MagicMock())
    finally:
        _active_handle.reset(token)
