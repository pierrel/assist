import re

from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("assist"),
    autoescape=select_autoescape()
)

class Promptable:
    """Provides utility functions to make it easy to write and
    retrieve prompts using template files"""
    def prompts_folder(self) -> str:
        """Returns the snake_case-version of the class name"""
        name = self.__class__.__name__
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def prompt_for(self,
                   prompt_name: str,
                   **kwargs):
        name = self.prompts_folder()
        path = f"{name}/{prompt_name}"
        template = env.get_template(path)
        
        return template.render(**kwargs)
