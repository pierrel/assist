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


def test_project_context_respects_ignore(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("root readme")
    hidden = tmp_path / ".secret"
    hidden.mkdir()
    hidden_readme = hidden / "README"
    hidden_readme.write_text("hidden")
    docs = tmp_path / "docs"
    docs.mkdir()
    ignored_agents = docs / "AGENTS.md"
    ignored_agents.write_text("ignored")
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("docs/\n")

    result = project_context(str(tmp_path))

    assert "root readme" in result
    assert "hidden" not in result
    assert "ignored" not in result
