import os
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Optional

CONFIG_FILENAME = "config.yml"

def _project_root() -> Path:
    """Locate the project root by searching for ``pyproject.toml``."""

    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start.parent

def config_path() -> Path:
    return _project_root() / CONFIG_FILENAME

@lru_cache(maxsize=1)
def get_config() -> dict:
    """Get configuration from config.yml or environment variables.

    Environment variables take precedence over config.yml values.
    If config.yml doesn't exist, only environment variables are used.
    """
    path = config_path()

    # Start with empty config
    config = {}

    # Load from config.yml if it exists
    if path.exists():
        try:
            config = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - filesystem/env
            raise RuntimeError(f"Unable to read {CONFIG_FILENAME}: {exc}") from exc

    # Override with environment variables (if set)
    if os.getenv("ASSIST_MODEL_URL"):
        config["url"] = os.getenv("ASSIST_MODEL_URL")
    if os.getenv("ASSIST_MODEL_NAME"):
        config["model"] = os.getenv("ASSIST_MODEL_NAME")
    if os.getenv("ASSIST_API_KEY"):
        config["api_key"] = os.getenv("ASSIST_API_KEY")
    if os.getenv("ASSIST_CONTEXT_LEN"):
        config["context_len"] = int(os.getenv("ASSIST_CONTEXT_LEN"))
    if os.getenv("ASSIST_DOMAIN"):
        config["domain"] = os.getenv("ASSIST_DOMAIN")

    return config


def get_domain() -> Optional[str]:
    """Get domain from config or environment variable.

    Returns None if domain is not configured (making it optional).
    """
    config = get_config()
    return config.get("domain", None)
