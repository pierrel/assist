from assist.promptable import Promptable


class MyClass(Promptable):
    pass


def test_prompts_folder():
    assert MyClass().prompts_folder() == "test_promptable"


def test_prompt_for_renders_template():
    mc = MyClass()
    result = mc.prompt_for("test_template.md.jinja", here="somewhere")
    assert "somewhere" in result
