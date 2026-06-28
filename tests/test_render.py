"""Tests for the render skill's web layer: parsing ```render blocks out of
assistant content, the file embed, and the /thread/{tid}/show viewer route.

CPU/no-model: the agent-driven behavior (model emits a render block) is an eval
(edd/eval/test_render_agent.py). Here we test the parser + renderer + route.
"""
import os

import pytest
from fastapi.testclient import TestClient

from manage import web
from manage.web.threads import (
    _safe_workspace_file,
    _file_embed_html,
    _render_file_block,
    _render_assistant_content,
    _RENDER_DISPATCH,
    _SHOWABLE_EXTS,
)


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

    @pytest.mark.parametrize("path", ["../../etc/passwd", "../secret", "../../secret.md"])
    def test_rejects_traversal(self, workspace, path):
        assert _safe_workspace_file("t1", path) is None

    @pytest.mark.parametrize("path", ["a.md", "/a.md", "/workspace/a.md"])
    def test_maps_agent_workspace_paths(self, workspace, path):
        # The agent addresses files in /workspace space; all three forms name the
        # same host file under the working dir.
        (workspace / "a.md").write_text("hi")
        assert _safe_workspace_file("t1", path) == os.path.realpath(str(workspace / "a.md"))

    def test_embedded_nul_is_none_not_error(self, workspace):
        assert _safe_workspace_file("t1", "a\x00.md") is None


class TestFileEmbed:
    def test_pdf_uses_embed(self):
        h = _file_embed_html("t1", "doc.pdf")
        assert "<embed" in h and 'type="application/pdf"' in h
        assert "/thread/t1/show?path=doc.pdf" in h

    def test_md_uses_sandboxed_iframe(self):
        h = _file_embed_html("t1", "notes.md")
        assert "<iframe" in h and "/thread/t1/show?path=notes.md" in h
        # sandbox WITHOUT allow-scripts so embedded content can't run JS.
        assert "sandbox=" in h and "allow-scripts" not in h

    def test_path_is_url_quoted(self):
        assert "my%20report.org" in _file_embed_html("t1", "my report.org")

    def test_render_file_block_rejects_unshowable_ext(self):
        assert _render_file_block("t1", {"path": "data.txt"}) is None

    def test_render_file_block_rejects_empty_path(self):
        assert _render_file_block("t1", {}) is None

    def test_render_file_block_renders_showable(self):
        assert "<iframe" in _render_file_block("t1", {"path": "/workspace/r.org"})


class TestRenderAssistantContent:
    def test_render_block_becomes_embed(self):
        raw = "Here's your file:\n\n```render\ntype: file\npath: /workspace/fitness.org\n```\n"
        out = _render_assistant_content("t1", raw)
        assert "show-embed" in out and "/thread/t1/show?path=" in out
        assert "Here&#39;s your file" in out or "Here's your file" in out
        assert "```render" not in out  # the block was lifted, not shown as code

    def test_unknown_type_left_as_code(self):
        raw = "```render\ntype: bogus\npath: x\n```"
        out = _render_assistant_content("t1", raw)
        assert "show-embed" not in out
        assert "<code" in out  # markdown rendered the fence as a code block

    def test_unshowable_file_left_as_code(self):
        raw = "```render\ntype: file\npath: notes.txt\n```"
        out = _render_assistant_content("t1", raw)
        assert "show-embed" not in out and "<code" in out

    def test_plain_markdown_untouched(self):
        out = _render_assistant_content("t1", "# Hi\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
        assert "<h1>Hi</h1>" in out and "<table>" in out and "show-embed" not in out

    def test_dispatch_is_single_source_of_truth(self):
        assert set(_RENDER_DISPATCH) == {"file"}


class TestShowRoute:
    def test_markdown_renders_to_html(self, workspace):
        (workspace / "n.md").write_text("# Title\n\n- a\n- b\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "n.md"})
        assert r.status_code == 200
        assert "<h1>Title</h1>" in r.text and "<li>a</li>" in r.text

    def test_workspace_prefixed_path_renders(self, workspace):
        (workspace / "n.md").write_text("# Title\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "/workspace/n.md"})
        assert r.status_code == 200 and "<h1>Title</h1>" in r.text

    def test_md_response_has_script_blocking_csp(self, workspace):
        (workspace / "x.md").write_text("# hi\n<script>alert(1)</script>\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "x.md"})
        csp = r.headers.get("content-security-policy", "")
        assert "default-src 'none'" in csp and "script-src" not in csp
        assert r.headers.get("x-content-type-options") == "nosniff"

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
        (workspace.parent.parent / "secret.md").write_text("secret")
        r = TestClient(web.app, raise_server_exceptions=False).get(
            "/thread/t1/show", params={"path": "../../secret.md"})
        assert r.status_code == 404

    def test_unsupported_extension_415(self, workspace):
        (workspace / "x.txt").write_text("plain")
        r = TestClient(web.app, raise_server_exceptions=False).get(
            "/thread/t1/show", params={"path": "x.txt"})
        assert r.status_code == 415

    def test_route_renders_every_showable_ext(self, workspace):
        # Drift guard: _SHOWABLE_EXTS is the allow-list for the file embed AND
        # this route; if the set lists an ext the route can't handle it would 415
        # inside the embed. Assert the route renders (never 415s) each member.
        for ext in _SHOWABLE_EXTS:
            name = f"drift{ext}"
            (workspace / name).write_text("# hi\n" if ext != ".pdf" else "%PDF-1.4 x")
            r = TestClient(web.app, raise_server_exceptions=False).get(
                "/thread/t1/show", params={"path": name})
            assert r.status_code != 415, f"{ext} -> {r.status_code}"


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
        (workspace / "s.org").write_text("* Top\n\n  * one\n  * two\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "s.org"})
        assert "<h1>Top</h1>" in r.text
        assert "<li>one</li>" in r.text and "<li>two</li>" in r.text

    def test_org_macro_eval_does_not_execute(self, workspace):
        (workspace / "evil.org").write_text(
            '#+MACRO: pwn (eval (shell-command-to-string "id"))\n{{{pwn}}}\n')
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "evil.org"})
        assert r.status_code == 200
        assert "uid=" not in r.text and "shell-command-to-string" not in r.text

    def test_org_content_is_escaped(self, workspace):
        (workspace / "x.org").write_text("Plain <script>alert(1)</script> text\n")
        r = TestClient(web.app).get("/thread/t1/show", params={"path": "x.org"})
        assert "<script>alert(1)</script>" not in r.text and "&lt;script&gt;" in r.text


class TestMessagesToDicts:
    """_messages_to_dicts no longer special-cases show_file; a render block rides
    in the assistant content (the web layer lifts it at render time)."""

    def _ai(self, content="", tool_calls=None):
        from langchain_core.messages import AIMessage
        return AIMessage(content=content, tool_calls=tool_calls or [])

    def test_render_block_stays_in_assistant_content(self):
        from assist.thread import _messages_to_dicts
        content = "Here:\n```render\ntype: file\npath: /workspace/r.org\n```\n"
        out = _messages_to_dicts([self._ai(content=content)])
        assert out == [{"role": "assistant", "content": content}]

    def test_tool_call_becomes_tools_line(self):
        from assist.thread import _messages_to_dicts
        m = self._ai(tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}])
        out = _messages_to_dicts([m])
        assert out[0]["role"] == "tools" and "read_file" in out[0]["content"]

    def test_plain_user_and_assistant(self):
        from assist.thread import _messages_to_dicts
        from langchain_core.messages import HumanMessage
        out = _messages_to_dicts([HumanMessage(content="hi"), self._ai(content="hello")])
        assert out == [{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}]

    def test_render_tool_calls_empty_when_no_calls(self):
        from assist.thread import render_tool_calls
        assert render_tool_calls(self._ai(content="hello, no tools")) == ""
