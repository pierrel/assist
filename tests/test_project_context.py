from assist.tools.filesystem import project_context


def test_project_context(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("hello readme")
    sub = tmp_path / "docs"
    sub.mkdir()
    agents = sub / "AGENTS.md"
    agents.write_text("agent info")

    result = project_context(str(tmp_path))

    assert "hello readme" in result
    assert "agent info" in result
