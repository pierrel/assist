"""Tests for the show_file tool + the /thread/{tid}/show viewer route.

CPU/no-model: the tool returns a string; the route renders md (markdown lib),
pdf (FileResponse bytes), and org (a pure-Python renderer — no emacs/eval). The
agent-driven end-to-end (agent calls show_file) needs the model — out of scope.
"""
import os

import pytest
from fastapi.testclient import TestClient

from assist.tools import show_file
from manage import web
from manage.web.threads import _render_show_file, _safe_workspace_file


class TestShowFileTool:
    def test_supported_extensions_confirm(self):
        for p in ("report.org", "notes.md", "doc.pdf"):
            assert show_file(p).startswith("Showing")

    def test_unsupported_extension_explains(self):
        out = show_file("data.txt")
        assert "can't display" in out
        assert ".org, .md, and .pdf" in out

    @pytest.mark.parametrize("bad", [None, ["a.md"], {"path": "a.md"}, ""])
    def test_non_string_or_empty_path_returns_guidance(self, bad):
        # untrusted model args: must not raise (os.path.splitext TypeError).
        out = show_file(bad)
        assert "needs a single file path" in out


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(web.MANAGER, "root_dir", str(tmp_path))
    # == thread_default_working_dir("t1"); use the constant so the test tracks
    # the default working-dir name if it ever changes.
    wd = tmp_path / "t1" / web.MANAGER.DEFAULT_THREAD_WORKING_DIRECTORY
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

    def test_embedded_nul_is_none_not_error(self, workspace):
        # An embedded NUL makes realpath raise ValueError; it must resolve to
        # None -> 404, never bubble a 500.
        assert _safe_workspace_file("t1", "a\x00.md") is None


class TestRenderShowFile:
    def test_pdf_uses_embed(self):
        h = _render_show_file("t1", "doc.pdf")
        assert "<embed" in h and 'type="application/pdf"' in h
        assert "/thread/t1/show?path=doc.pdf" in h

    def test_md_uses_iframe(self):
        h = _render_show_file("t1", "notes.md")
        assert "<iframe" in h
        assert "/thread/t1/show?path=notes.md" in h

    def test_iframe_is_sandboxed_no_scripts(self):
        # The md/org iframe must carry a sandbox WITHOUT allow-scripts so
        # agent-generated content can't run JS in it (the md path emits raw HTML).
        h = _render_show_file("t1", "notes.md")
        assert "sandbox=" in h and "allow-scripts" not in h

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

    def test_md_response_has_script_blocking_csp(self, workspace):
        # The caption link opens this route as a top-level doc in the app origin;
        # the md path passes raw HTML through, so the response must carry a CSP
        # with no script source so scripts can't run even standalone.
        (workspace / "x.md").write_text("# hi\n<script>alert(1)</script>\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "x.md"})
        csp = r.headers.get("content-security-policy", "")
        assert "default-src 'none'" in csp and "script-src" not in csp
        assert r.headers.get("x-content-type-options") == "nosniff"

    def test_unsupported_extension_415(self, workspace):
        (workspace / "x.txt").write_text("plain")
        r = TestClient(web.app, raise_server_exceptions=False).get(
            "/thread/t1/show", params={"path": "x.txt"})
        assert r.status_code == 415


class TestOrgRender:
    """Pure-Python org renderer (no emacs — see the security note in threads.py)."""

    def test_org_headings_emphasis_lists(self, workspace):
        (workspace / "r.org").write_text(
            "* Heading\n\nSome *bold* and /italic/ text.\n\n- a\n- b\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "r.org"})
        assert r.status_code == 200
        assert "<h1>Heading</h1>" in r.text
        assert "<b>bold</b>" in r.text and "<i>italic</i>" in r.text
        assert "<li>a</li>" in r.text

    def test_org_star_bullets(self, workspace):
        # Indented '* ' is an org list bullet (column-0 '*' stays a heading).
        (workspace / "s.org").write_text("* Top\n\n  * one\n  * two\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "s.org"})
        assert "<h1>Top</h1>" in r.text            # column-0 * -> heading
        assert "<li>one</li>" in r.text and "<li>two</li>" in r.text  # indented * -> list

    def test_org_macro_eval_does_not_execute(self, workspace):
        # The reason org is NOT rendered via emacs: org export would eval a
        # macro's (eval ...) form (host RCE).  The pure renderer must not run it.
        (workspace / "evil.org").write_text(
            '#+MACRO: pwn (eval (shell-command-to-string "id"))\n{{{pwn}}}\n')
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "evil.org"})
        assert r.status_code == 200
        assert "uid=" not in r.text          # the `id` command never ran
        assert "shell-command-to-string" not in r.text  # #+MACRO line dropped

    def test_org_include_directive_not_expanded(self, workspace):
        # A #+INCLUDE pointing at a host file must NOT pull its contents in.
        secret = workspace.parent.parent / "secret.txt"
        secret.write_text("TOPSECRET")
        (workspace / "inc.org").write_text(f"* Doc\n#+INCLUDE: \"{secret}\"\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "inc.org"})
        assert r.status_code == 200
        assert "TOPSECRET" not in r.text

    def test_org_content_is_escaped(self, workspace):
        (workspace / "x.org").write_text("Plain <script>alert(1)</script> text\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "x.org"})
        assert "<script>alert(1)</script>" not in r.text
        assert "&lt;script&gt;" in r.text


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

    def _no_directive(self, args):
        from assist.thread import _messages_to_dicts
        out = _messages_to_dicts([self._ai(tool_calls=[
            {"name": "show_file", "args": args, "id": "1"}])])
        assert not any(d["role"] == "show_file" for d in out)
        return out

    def test_show_file_empty_path_no_directive(self):
        self._no_directive({})

    def test_show_file_non_string_path_no_directive(self):
        # untrusted model output; a non-string path must not become a directive
        # (it would crash the renderer's urllib.quote).
        self._no_directive({"path": ["a", "b"]})

    def test_show_file_unsupported_ext_falls_back_to_tools_line(self):
        # An out-of-spec show_file (e.g. .txt) must NOT embed the viewer route
        # (which would render a 415) — it shows as a normal tool-call line.
        out = self._no_directive({"path": "notes.txt"})
        assert any(d["role"] == "tools" and "show_file" in d["content"] for d in out)

    def test_render_tool_calls_empty_when_no_calls(self):
        # The CLI prints this per AIMessage; a plain assistant message must
        # render empty so its content isn't duplicated.
        from assist.thread import render_tool_calls
        assert render_tool_calls(self._ai(content="hello, no tools")) == ""
