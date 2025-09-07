import importlib

from assist.tools.safe_python import SafePythonTool


def test_executes_math():
    tool = SafePythonTool()
    output = tool.run("result = sum(i*i for i in range(5))")
    assert "30" in output


def test_blocks_file_write(tmp_path):
    tool = SafePythonTool()
    code = "open('x.txt', 'w').write('hi')"
    out = tool.run(code)
    assert "Error" in out


def test_blocks_os_import():
    tool = SafePythonTool()
    out = tool.run("import os")
    assert "not allowed" in out


def test_description_mentions_limits():
    tool = SafePythonTool()
    desc = tool.description
    assert "builtins" in desc and "modules" in desc
    assert "abs" in desc and "math" in desc


def test_numpy_mention_matches_installation():
    tool = SafePythonTool()
    try:
        importlib.import_module("numpy")
    except Exception:
        assert "numpy" not in tool.description
    else:
        assert "numpy" in tool.description
