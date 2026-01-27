import yaml
from pathlib import Path
from functools import lru_cache

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
    path = config_path()
    if not path.exists():
        raise RuntimeError(f"{CONFIG_FILENAME} does not exist")

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - filesystem/env
        raise RuntimeError(f"Unable to read {CONFIG_FILENAME}: {exc}") from exc
    return raw


def get_domain() -> str:
    config = get_config()
    domain = config.get("domain", None)
    if not domain:
        raise RuntimeError(f"Domain not available in {CONFIG_FILENAME}: {config}")

    return domain
