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
    assert "Builtins" in desc and "Modules" in desc
    assert "abs" in desc and "pow" in desc and "math" in desc


def test_pow_builtin():
    tool = SafePythonTool()
    out = tool.run("result = pow(2, 3)")
    assert out.strip().endswith("8")


def test_explicit_code_argument():
    tool = SafePythonTool()
    out = tool.invoke({"code": "result = 1 + 1"})
    assert out.strip().endswith("2")


def test_numpy_mention_matches_installation():
    tool = SafePythonTool()
    try:
        importlib.import_module("numpy")
    except Exception:
        assert "numpy" not in tool.description
    else:
        assert "numpy" in tool.description
