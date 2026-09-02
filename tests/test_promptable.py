from assist.promptable import Promptable, env


class MyClass(Promptable):
    pass


def test_prompts_folder():
    assert MyClass().prompts_folder() == "test_promptable"


def test_prompt_for_renders_template():
    mc = MyClass()
    result = mc.prompt_for("test_template.txt", here="somewhere")
    assert "somewhere" in result


def test_static_agent_templates_do_not_render_a_clock():
    static_prompts = (
        "general_instructions.md.j2", "context_agent.md.j2", "dev_critique.md.j2",
        "fact_checker.md.j2", "research_instructions.txt.j2",
        "sub_critique.txt.j2", "sub_research.txt.j2",
    )
    for name in static_prompts:
        text, _, _ = env.loader.get_source(env, f"deepagents/{name}")
        assert "Current date/time:" not in text
        assert "current_datetime" not in text
