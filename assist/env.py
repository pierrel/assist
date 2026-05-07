"""Load development environment variables from ``.dev.env``."""
from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    """Read a float-valued env var with silent fallback on missing/invalid.

    Used for tuning knobs in ``.deploy.env``: a typo in the value
    should not hard-fail every thread on first model build with a
    bare ``ValueError``.  The default wins on both unset and
    unparseable values.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Read an int-valued env var with silent fallback on missing/invalid.

    See :func:`env_float` for the rationale.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_dev_env() -> None:
    """Load ``.dev.env`` from the project root if it exists.

    Only variables that are **not** already set in the environment are applied,
    so real environment variables always take precedence.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".dev.env")

    if not os.path.isfile(env_path):
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
