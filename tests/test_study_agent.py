import pathlib

from langchain_core.messages import AIMessage

from assist.study_agent import build_study_graph
from assist.tools.study import StudyTool


class DummyLLM:
    def invoke(self, messages, _opts=None):
        content = messages[-1].content
        if "alpha" in content:
            return AIMessage(content="note alpha")
        if "beta" in content:
            return AIMessage(content="note beta")
        return AIMessage(content="note")


def test_study_graph_writes_notes(tmp_path):
    file1 = tmp_path / "a.txt"
    file2 = tmp_path / "b.txt"
    file1.write_text("alpha text")
    file2.write_text("beta text")
    output = tmp_path / "notes.md"

    graph = build_study_graph(DummyLLM())
    state = {
        "objective": "letters",
        "resources": [str(file1), str(file2)],
        "index": 0,
        "notes": [],
        "output_path": str(output),
    }
    graph.invoke(state)

    text = output.read_text()
    assert "note alpha" in text
    assert "note beta" in text


def test_study_tool_invokes_graph(tmp_path):
    file1 = tmp_path / "a.txt"
    file1.write_text("alpha text")

    tool = StudyTool(DummyLLM(), tmp_path)
    path = tool._run("letters", [str(file1)], filename="out.md")

    out_text = pathlib.Path(path).read_text()
    assert "note alpha" in out_text
    assert pathlib.Path(path).name == "out.md"
