"""
Template loading utilities for EDD module.
"""
from typing import Any
from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("edd"),
    autoescape=select_autoescape(),
)


def base_prompt_for(prompt_path: str, **kwargs: Any) -> str:
    """Load and render a template from the edd templates directory."""
    template = env.get_template(prompt_path)
    return template.render(**kwargs)
