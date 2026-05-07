"""Tests for ``assist.env`` helpers."""
import os
from unittest.mock import patch

from assist.env import env_float, env_int


class TestEnvFloat:
    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"FOO": "1.5"}):
            assert env_float("FOO", 99.0) == 1.5

    def test_returns_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOO", None)
            assert env_float("FOO", 99.0) == 99.0

    def test_returns_default_when_invalid(self):
        with patch.dict(os.environ, {"FOO": "garbage"}):
            assert env_float("FOO", 99.0) == 99.0

    def test_returns_default_when_empty_string(self):
        with patch.dict(os.environ, {"FOO": ""}):
            assert env_float("FOO", 99.0) == 99.0


class TestEnvInt:
    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"FOO": "42"}):
            assert env_int("FOO", 99) == 42

    def test_returns_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOO", None)
            assert env_int("FOO", 99) == 99

    def test_returns_default_when_invalid(self):
        with patch.dict(os.environ, {"FOO": "garbage"}):
            assert env_int("FOO", 99) == 99

    def test_returns_default_on_float_string(self):
        # int("1.5") raises; should fall back.
        with patch.dict(os.environ, {"FOO": "1.5"}):
            assert env_int("FOO", 99) == 99
