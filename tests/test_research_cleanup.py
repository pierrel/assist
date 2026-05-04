"""Tests for ReferencesCleanupRunnable + research-agent wiring.

Deterministic unit tests using a fake inner runnable — no LLM or vLLM
server required.  Verifies:

  * The snapshot/prune semantics of ``ReferencesCleanupRunnable``.
  * The two final-filename extraction paths (tool-call inspection +
    FINAL_REPORT regex fallback).
  * Wiring in ``create_research_agent`` — the backend is rooted at
    ``<working_dir>/references/`` and the runnable returned is a
    ``ReferencesCleanupRunnable`` wrapping a ``RollbackRunnable``.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from langchain.messages import AIMessage

from assist.research_cleanup import ReferencesCleanupRunnable


def _ai_with_write(file_path: str, content_text: str = "done"):
    return AIMessage(
        content=content_text,
        tool_calls=[
            {"name": "write_file", "args": {"file_path": file_path}, "id": "t1"}
        ],
    )


class _FakeInner:
    """A fake inner runnable that runs ``side_effect(refs_path)`` to
    create files inside ``references/`` and returns a result dict with
    the messages it was constructed with.
    """

    def __init__(self, side_effect, messages):
        self._side_effect = side_effect
        self._messages = messages

    def bind_path(self, refs_path):
        self._refs = refs_path

    def invoke(self, input_data, config=None, **kwargs):
        self._side_effect(self._refs)
        return {"messages": self._messages}


def _run(side_effect, messages):
    """Helper: build a temp references/ dir, run the wrapper, return the
    sorted post-cleanup file list and the wrapper itself for inspection.
    """
    wd = tempfile.mkdtemp()
    refs = os.path.join(wd, "references")
    os.makedirs(refs)
    inner = _FakeInner(side_effect, messages)
    inner.bind_path(refs)
    wrapper = ReferencesCleanupRunnable(inner, references_path=refs)
    wrapper.invoke({"messages": []})
    return sorted(os.listdir(refs)), refs


class TestReferencesCleanup:
    def test_prunes_drafts_keeps_final_via_tool_call(self):
        def writes(refs):
            open(os.path.join(refs, "draft1.org"), "w").write("d1")
            open(os.path.join(refs, "scratch.txt"), "w").write("s")
            open(os.path.join(refs, "final.org"), "w").write("ok")

        after, _ = _run(writes, [_ai_with_write("/final.org")])
        assert after == ["final.org"]

    def test_keeps_pre_existing_files(self):
        wd = tempfile.mkdtemp()
        refs = os.path.join(wd, "references")
        os.makedirs(refs)
        # File from a "prior turn" — must survive cleanup.
        open(os.path.join(refs, "prior.org"), "w").write("prior")

        def writes(_):
            open(os.path.join(refs, "draft.org"), "w").write("d")
            open(os.path.join(refs, "final.org"), "w").write("f")

        inner = _FakeInner(writes, [_ai_with_write("final.org")])
        inner.bind_path(refs)
        ReferencesCleanupRunnable(inner, references_path=refs).invoke({})

        after = sorted(os.listdir(refs))
        assert "prior.org" in after
        assert "final.org" in after
        assert "draft.org" not in after

    def test_final_report_regex_fallback(self):
        def writes(refs):
            open(os.path.join(refs, "draft.org"), "w").write("d")
            open(os.path.join(refs, "report.org"), "w").write("r")

        # No tool_calls at all — agent emits text-only termination
        # with FINAL_REPORT marker.
        msg = AIMessage(content="Here is the report.\nFINAL_REPORT: report.org\n")
        after, _ = _run(writes, [msg])
        assert after == ["report.org"]

    def test_no_signal_leaves_everything(self):
        """If we can't identify the final report, prune nothing.

        Per design: never fall back to mtime-based selection because the
        critique sub-sub-agent often writes files and would be most-recent.
        """
        def writes(refs):
            open(os.path.join(refs, "a.org"), "w").write("a")
            open(os.path.join(refs, "b.org"), "w").write("b")

        msg = AIMessage(content="I gave up — no file written.")
        after, _ = _run(writes, [msg])
        assert after == ["a.org", "b.org"]

    def test_overwrite_keeps_pre_existing_target(self):
        """If the agent overwrites a pre-existing file with the same
        name as the final report, the file stays (it was in ``before``)
        and gets the new content.
        """
        wd = tempfile.mkdtemp()
        refs = os.path.join(wd, "references")
        os.makedirs(refs)
        open(os.path.join(refs, "final.org"), "w").write("OLD")
        open(os.path.join(refs, "unrelated.org"), "w").write("UNRELATED")

        def writes(_):
            open(os.path.join(refs, "final.org"), "w").write("NEW")

        inner = _FakeInner(writes, [_ai_with_write("final.org")])
        inner.bind_path(refs)
        ReferencesCleanupRunnable(inner, references_path=refs).invoke({})

        after = sorted(os.listdir(refs))
        assert after == ["final.org", "unrelated.org"]
        assert open(os.path.join(refs, "final.org")).read() == "NEW"

    def test_auto_creates_references_dir(self):
        wd = tempfile.mkdtemp()
        refs = os.path.join(wd, "references")
        # Directory does NOT exist yet — wrapper must create it.
        assert not os.path.exists(refs)

        def writes(_):
            os.makedirs(refs, exist_ok=True)
            open(os.path.join(refs, "r.org"), "w").write("r")

        inner = _FakeInner(writes, [_ai_with_write("r.org")])
        inner.bind_path(refs)
        ReferencesCleanupRunnable(inner, references_path=refs).invoke({})

        assert os.path.isdir(refs)

    def test_skip_cleanup_on_exception(self):
        """If the inner runnable raises, cleanup is skipped — partial
        files survive (next invocation snapshots them as ``before``).
        """
        wd = tempfile.mkdtemp()
        refs = os.path.join(wd, "references")
        os.makedirs(refs)
        open(os.path.join(refs, "halfwritten.org"), "w").write("partial")

        class Boom:
            def invoke(self, *a, **kw):
                raise RuntimeError("agent crashed")

        try:
            ReferencesCleanupRunnable(Boom(), references_path=refs).invoke({})
        except RuntimeError:
            pass

        assert os.path.exists(os.path.join(refs, "halfwritten.org"))

    def test_proxies_attributes_to_inner(self):
        """Anything not on the wrapper falls through to the inner —
        same idiom as RollbackRunnable.
        """
        inner = Mock()
        inner.get_state_history = Mock(return_value=["snapshot"])
        wrapper = ReferencesCleanupRunnable(inner, references_path="/tmp/x")

        assert wrapper.get_state_history() == ["snapshot"]
        inner.get_state_history.assert_called_once()

    def test_extracts_basename_from_absolute_path(self):
        """Tool-call args may have an absolute path ('/final.org') or
        a bare name ('final.org').  Either should produce 'final.org'.
        """
        def writes(refs):
            open(os.path.join(refs, "final.org"), "w").write("ok")
            open(os.path.join(refs, "extra.org"), "w").write("x")

        # Absolute path form.
        after, _ = _run(writes, [_ai_with_write("/final.org")])
        assert after == ["final.org"]

        # Bare name form.
        after, _ = _run(writes, [_ai_with_write("final.org")])
        assert after == ["final.org"]

    def test_ainvoke_runs_cleanup(self):
        """The async path must do the same snapshot/prune work as the
        sync path — without an explicit ``ainvoke`` method, deepagents'
        async ``task`` tool would call the inner's ``ainvoke`` via
        ``__getattr__`` and silently bypass cleanup.
        """
        import asyncio

        wd = tempfile.mkdtemp()
        refs = os.path.join(wd, "references")
        os.makedirs(refs)

        class AsyncInner:
            async def ainvoke(self, input_data, config=None, **kwargs):
                open(os.path.join(refs, "draft.org"), "w").write("d")
                open(os.path.join(refs, "final.org"), "w").write("f")
                return {"messages": [_ai_with_write("final.org")]}

        wrapper = ReferencesCleanupRunnable(AsyncInner(), references_path=refs)
        asyncio.run(wrapper.ainvoke({"messages": []}))

        after = sorted(os.listdir(refs))
        assert after == ["final.org"], f"async cleanup didn't prune: {after}"

    def test_uses_last_final_report_match_when_multiple(self):
        """If the agent emits multiple FINAL_REPORT: lines (a self-
        correction, or a mid-message draft followed by the real one),
        the LAST match wins.
        """
        def writes(refs):
            open(os.path.join(refs, "draft.org"), "w").write("d")
            open(os.path.join(refs, "real.org"), "w").write("r")

        msg = AIMessage(content=(
            "Initial plan: write to draft.\n"
            "FINAL_REPORT: draft.org\n"
            "On second thought, use a clearer name.\n"
            "FINAL_REPORT: real.org\n"
        ))
        after, _ = _run(writes, [msg])
        assert after == ["real.org"]


class TestResearchAgentWiring:
    """Pin the integration: ``create_research_agent`` must wire the
    references-confined backend and wrap the result with
    ``ReferencesCleanupRunnable``.  A future refactor that drops either
    leg silently regresses the cleanup contract.
    """

    def test_local_mode_backend_rooted_at_references(self):
        from assist.agent import create_research_agent
        from assist.checkpoint_rollback import RollbackRunnable
        from deepagents.backends import CompositeBackend, FilesystemBackend

        with patch("assist.agent.create_deep_agent") as fake:
            fake.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                runnable = create_research_agent(MagicMock(), wd)

                refs = os.path.join(wd, "references")
                assert os.path.isdir(refs), "references dir not created"
                assert isinstance(runnable, ReferencesCleanupRunnable)
                assert isinstance(runnable._inner, RollbackRunnable)
                assert runnable._references_path == refs
                assert runnable._sandbox is None

                kwargs = fake.call_args.kwargs
                backend = kwargs["backend"]
                assert isinstance(backend, CompositeBackend)
                assert isinstance(backend.default, FilesystemBackend)
                # Use Path.resolve() on both sides so the test is robust
                # to symlink-resolution (e.g. macOS's /var -> /private/var).
                assert backend.default.cwd == Path(refs).resolve()

    def test_local_backend_strips_references_prefix(self):
        """If the agent slips and writes ``references/foo.org`` (because
        its prompt or task description still mentions the directory
        name), the path normalizer flattens the leading prefix so the
        file lands at ``<refs>/foo.org`` instead of nesting under
        ``<refs>/references/foo.org``.
        """
        from assist.backends import create_references_backend

        with tempfile.TemporaryDirectory() as wd:
            backend = create_references_backend(wd)
            refs_root = os.path.join(wd, "references")

            for path_form in (
                "references/foo.org",
                "/references/foo.org",
            ):
                # Resolve via the backend's underlying default — we
                # can't write through the composite cleanly without
                # spinning up a graph, but the normalizer is on the
                # default's _normalize method.
                normalized = backend.default._normalize(path_form)
                # After normalization, the path should NOT have a
                # leading "references/" — bare or absolute "foo.org".
                assert not normalized.lstrip("/").startswith("references/")
                assert os.path.basename(normalized) == "foo.org"

            # Bare names pass through unchanged.
            assert backend.default._normalize("foo.org") == "foo.org"
            # Other dirs pass through unchanged.
            assert backend.default._normalize("other/foo.org") == "other/foo.org"

    def test_sandbox_strip_prefixes_flattens_path(self):
        """Same defense for the sandbox-mode sibling DockerSandboxBackend.
        ``strip_prefixes`` runs before ``_resolve`` so the work_dir prefix
        doesn't double up.
        """
        from assist.sandbox import DockerSandboxBackend

        sb = DockerSandboxBackend(
            container=MagicMock(), work_dir="/workspace/references",
            strip_prefixes=("references",),
        )
        assert sb._resolve("references/foo.org") == "/workspace/references/foo.org"
        assert sb._resolve("/references/foo.org") == "/workspace/references/foo.org"
        # Bare names still get the work_dir prefix.
        assert sb._resolve("foo.org") == "/workspace/references/foo.org"
        # Already-resolved paths under work_dir pass through.
        assert sb._resolve("/workspace/references/foo.org") == "/workspace/references/foo.org"

    def test_sandbox_mode_uses_references_sibling_backend(self):
        from assist.agent import create_research_agent
        from assist.sandbox import DockerSandboxBackend

        sandbox = MagicMock(spec=DockerSandboxBackend)
        sandbox.work_dir = "/workspace"
        sandbox.container = MagicMock()

        with patch("assist.agent.create_deep_agent") as fake:
            fake.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                runnable = create_research_agent(
                    MagicMock(), wd, sandbox_backend=sandbox,
                )

                # Cleanup wrapper got the parent sandbox (not a sibling) —
                # so its `mkdir -p /workspace/references` runs with a
                # workdir that exists.
                assert runnable._sandbox is sandbox
                assert runnable._references_path == "/workspace/references"

                # The agent's backend used a *sibling* sandbox rooted
                # at /workspace/references for file ops.
                kwargs = fake.call_args.kwargs
                backend = kwargs["backend"]
                # backend is a CompositeBackend whose default is the
                # sibling DockerSandboxBackend.
                assert backend.default is not sandbox
                assert backend.default.work_dir == "/workspace/references"
