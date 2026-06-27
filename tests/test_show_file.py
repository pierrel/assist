"""Tests for the show_file tool + the /thread/{tid}/show viewer route.

CPU/no-model: the tool returns a string; the route renders md (markdown lib),
pdf (FileResponse bytes), and org (emacs, skipped if emacs is absent). The
agent-driven end-to-end (agent calls show_file) needs the model — out of scope.
"""
import os
import shutil

import pytest
from fastapi.testclient import TestClient

from assist.tools import show_file
from manage import web
from manage.web.threads import _render_show_file, _safe_workspace_file


class TestShowFileTool:
    def test_supported_extensions_confirm(self):
        for p in ("report.org", "notes.md", "a.markdown", "doc.pdf"):
            assert show_file(p).startswith("Showing")

    def test_unsupported_extension_explains(self):
        out = show_file("data.txt")
        assert "can't display" in out
        assert ".org, .md, and .pdf" in out


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    wd = tmp_path / "t1" / "domain"   # == thread_default_working_dir("t1")
    wd.mkdir(parents=True)
    return wd


class TestSafeWorkspaceFile:
    def test_resolves_inside_workspace(self, workspace):
        (workspace / "a.md").write_text("hi")
        got = _safe_workspace_file("t1", "a.md")
        assert got == os.path.realpath(str(workspace / "a.md"))

    @pytest.mark.parametrize("path", ["../../etc/passwd", "../secret", "/etc/passwd"])
    def test_rejects_traversal(self, workspace, path):
        # ../ escapes the workspace -> None; an absolute path resolves outside too.
        assert _safe_workspace_file("t1", path) is None


class TestRenderShowFile:
    def test_pdf_uses_embed(self):
        h = _render_show_file("t1", "doc.pdf")
        assert "<embed" in h and 'type="application/pdf"' in h
        assert "/thread/t1/show?path=doc.pdf" in h

    def test_md_uses_iframe(self):
        h = _render_show_file("t1", "notes.md")
        assert "<iframe" in h
        assert "/thread/t1/show?path=notes.md" in h

    def test_path_is_url_quoted(self):
        h = _render_show_file("t1", "my report.org")
        assert "my%20report.org" in h


class TestShowRoute:
    def test_markdown_renders_to_html(self, workspace):
        (workspace / "n.md").write_text("# Title\n\n- a\n- b\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "n.md"})
        assert r.status_code == 200
        assert "<h1>Title</h1>" in r.text
        assert "<li>a</li>" in r.text

    def test_pdf_served_as_bytes(self, workspace):
        (workspace / "d.pdf").write_bytes(b"%PDF-1.4 fake bytes")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "d.pdf"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content == b"%PDF-1.4 fake bytes"

    def test_missing_file_404(self, workspace):
        r = TestClient(web.app, raise_server_exceptions=False).get(
            "/thread/t1/show", params={"path": "nope.md"})
        assert r.status_code == 404

    def test_traversal_404(self, workspace):
        (workspace.parent.parent / "secret.md").write_text("secret")  # outside the workspace
        r = TestClient(web.app, raise_server_exceptions=False).get(
            "/thread/t1/show", params={"path": "../../secret.md"})
        assert r.status_code == 404

    def test_unsupported_extension_415(self, workspace):
        (workspace / "x.txt").write_text("plain")
        r = TestClient(web.app, raise_server_exceptions=False).get(
            "/thread/t1/show", params={"path": "x.txt"})
        assert r.status_code == 415


@pytest.mark.skipif(not shutil.which("emacs"), reason="emacs not available for org export")
class TestOrgRender:
    def test_org_exports_to_html(self, workspace):
        (workspace / "r.org").write_text("* Heading\n\nSome *bold* text.\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "r.org"})
        assert r.status_code == 200
        assert "Heading" in r.text
        assert "<b>bold</b>" in r.text or "bold" in r.text  # org emphasis -> bold

    def test_org_include_directive_is_stripped(self, workspace):
        # A #+INCLUDE pointing at a host file must NOT be expanded into the output.
        secret = workspace.parent.parent / "secret.txt"
        secret.write_text("TOPSECRET")
        (workspace / "evil.org").write_text(
            f"* Doc\n#+INCLUDE: \"{secret}\"\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "evil.org"})
        assert r.status_code == 200
        assert "TOPSECRET" not in r.text


class TestMessagesToDicts:
    """The get_messages logic that turns a show_file tool call into a structured
    render directive (vs other calls staying as the 'tools' text line)."""

    def _ai(self, content="", tool_calls=None):
        from langchain_core.messages import AIMessage
        return AIMessage(content=content, tool_calls=tool_calls or [])

    def test_show_file_call_becomes_directive(self):
        from assist.thread import _messages_to_dicts
        m = self._ai(tool_calls=[{"name": "show_file", "args": {"path": "r.org"}, "id": "1"}])
        assert _messages_to_dicts([m]) == [{"role": "show_file", "path": "r.org"}]

    def test_show_file_alongside_other_call(self):
        from assist.thread import _messages_to_dicts
        m = self._ai(tool_calls=[
            {"name": "read_file", "args": {"path": "x"}, "id": "1"},
            {"name": "show_file", "args": {"path": "r.md"}, "id": "2"}])
        out = _messages_to_dicts([m])
        tools = next(d for d in out if d["role"] == "tools")
        assert "read_file" in tools["content"] and "show_file" not in tools["content"]
        assert {"role": "show_file", "path": "r.md"} in out

    def test_plain_user_and_assistant_unchanged(self):
        from assist.thread import _messages_to_dicts
        from langchain_core.messages import HumanMessage
        out = _messages_to_dicts([HumanMessage(content="hi"), self._ai(content="hello")])
        assert out == [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}]

    def test_show_file_empty_path_skipped(self):
        from assist.thread import _messages_to_dicts
        assert _messages_to_dicts([self._ai(tool_calls=[
            {"name": "show_file", "args": {}, "id": "1"}])]) == []
