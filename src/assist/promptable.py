import inspect

from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("assist"),
    autoescape=select_autoescape(),
)


def _folder_from_module(module: str) -> str:
    """Return the template folder name for a module."""
    return module.split(".")[-1]


def _infer_module() -> str:
    """Return the module name of the first caller outside this module."""
    for frame_info in inspect.stack():
        module = inspect.getmodule(frame_info.frame)
        if module and module.__name__ != __name__:
            return module.__name__
    raise RuntimeError("Could not infer caller module")


def base_prompt_for(prompt_path: str, **kwargs) -> str:
    template = env.get_template(prompt_path)
    return template.render(**kwargs)


def prompt_for(prompt_name: str, *, module: str | None = None, **kwargs) -> str:
    """Render ``prompt_name`` for ``module`` using optional ``kwargs``.

    If ``module`` is not provided, it is inferred from the caller's module.
    """
    module = module or _infer_module()
    name = _folder_from_module(module)
    path = f"{name}/{prompt_name}"
    return base_prompt_for(path, **kwargs)


class Promptable:
    """Convenience mixin for class-based prompt access."""

    def prompts_folder(self) -> str:
        return _folder_from_module(self.__module__)

    def prompt_for(self, prompt_name: str, **kwargs) -> str:
        return prompt_for(prompt_name, **kwargs)
