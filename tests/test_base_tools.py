from assist.tools.base import base_tools
from assist.tools import filesystem


def test_base_tools_includes_python(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    tools = base_tools(tmp_path)
    names = [t.name for t in tools]
    assert "python" in names


def test_base_tools_includes_write_file_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    tools = base_tools(tmp_path)
    names = [t.name for t in tools]
    assert filesystem.write_file_user.name in names
    assert filesystem.write_file_tmp.name in names
