"""Exercise the sandbox file program and transfer contract without a model."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.messages import AIMessage

from assist.research_cleanup import ReferencesCleanupRunnable
from assist.thread_datastore import ThreadDataStore, register_source


class LocalSandboxTransport:
    """Run the exact sandbox program and transfer bytes over its public API."""

    def execute(self, command):
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return SimpleNamespace(exit_code=result.returncode,
                               output=result.stdout + result.stderr)

    def download_files(self, paths, *, max_bytes=None):
        assert max_bytes is not None
        return [SimpleNamespace(error=None, content=Path(path).read_bytes())
                for path in paths]

    def upload_files(self, files):
        for path, data in files:
            Path(path).write_bytes(data)
        return [SimpleNamespace(error=None) for _ in files]


class WritesReport:
    def __init__(self, write):
        self.write = write

    def invoke(self, *_args, **_kwargs):
        self.write()
        return {"messages": [AIMessage(content="FINAL_REPORT: final.md")]}


def test_sandbox_report_binding_prune_and_transfer(tmp_path):
    refs = tmp_path / "references"
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    url = "https://example.test/exact?x=1&y=2"
    record_id = register_source(store, url)
    (refs).mkdir()
    (refs / "prior.md").write_text("keep")

    def write():
        (refs / "draft.md").write_text("delete")
        (refs / "final.md").write_text(
            f"[Source](ref:{record_id})\n[Owner](https://owner.test/a)\n")

    runner = ReferencesCleanupRunnable(WritesReport(write), str(refs),
                                       LocalSandboxTransport())
    runner.invoke({}, config={"configurable": {"thread_datastore_path": store.path}})
    assert sorted(p.name for p in refs.iterdir()) == ["final.md", "prior.md"]
    assert (refs / "final.md").read_text() == (
        f"[Source](<{url}>)\n[Owner](https://owner.test/a)\n")
    assert not list(refs.glob(".source-*"))


def test_sandbox_directory_symlink_cannot_redirect_cleanup(tmp_path):
    refs = tmp_path / "references"
    victim = tmp_path / "victim"
    refs.mkdir()
    victim.mkdir()
    (victim / "final.md").write_text("outside")
    (victim / "draft.md").write_text("outside")

    def replace_directory():
        refs.rename(tmp_path / "old-references")
        refs.symlink_to(victim, target_is_directory=True)

    runner = ReferencesCleanupRunnable(WritesReport(replace_directory), str(refs),
                                       LocalSandboxTransport())
    with pytest.raises(RuntimeError, match="research sandbox exists failed"):
        runner.invoke({})
    assert (victim / "final.md").read_text() == "outside"
    assert (victim / "draft.md").read_text() == "outside"


def test_sandbox_failed_replace_removes_staged_file(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    store = ThreadDataStore(str(tmp_path / "thread-data.sqlite"))
    record_id = register_source(store, "https://example.test/exact")

    class FailingTransport(LocalSandboxTransport):
        def execute(self, command):
            if ' replace ' in command:
                return SimpleNamespace(exit_code=1, output="replacement failed")
            return super().execute(command)

    def write():
        (refs / "final.md").write_text(f"[Source](ref:{record_id})")

    runner = ReferencesCleanupRunnable(WritesReport(write), str(refs),
                                       FailingTransport())
    with pytest.raises(RuntimeError, match="replacement failed"):
        runner.invoke({}, config={"configurable": {"thread_datastore_path": store.path}})
    assert (refs / "final.md").read_text() == f"[Source](ref:{record_id})"
    assert not list(refs.glob(".source-*"))


def test_sandbox_cleanup_with_listing_beyond_argv_limit(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    for index in range(5000):
        (refs / f"long-retained-report-{index:04d}.md").write_text("x")

    def write():
        (refs / "draft.md").write_text("delete")
        (refs / "final.md").write_text("keep")

    runner = ReferencesCleanupRunnable(WritesReport(write), str(refs),
                                       LocalSandboxTransport())
    runner.invoke({})
    assert len(list(refs.iterdir())) == 5001
    assert not (refs / "draft.md").exists()
    assert (refs / "final.md").read_text() == "keep"


def test_sandbox_prunes_quote_heavy_names_via_transfer(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()

    def write():
        for index in range(60):
            (refs / (f"{index:02d}-" + "'" * 245)).write_text("draft")
        (refs / "final.md").write_text("keep")

    runner = ReferencesCleanupRunnable(WritesReport(write), str(refs),
                                       LocalSandboxTransport())
    runner.invoke({})
    assert [p.name for p in refs.iterdir()] == ["final.md"]


def test_sandbox_rejects_tampered_cleanup_transfer(tmp_path):
    refs = tmp_path / "references"
    refs.mkdir()
    victim = tmp_path / "victim.md"
    victim.write_text("outside")

    class TamperingTransport(LocalSandboxTransport):
        def upload_files(self, files):
            result = super().upload_files(files)
            if b"draft.md" in files[0][1]:
                Path(files[0][0]).write_text('["../victim.md"]')
            return result

    def write():
        (refs / "draft.md").write_text("draft")
        (refs / "final.md").write_text("keep")

    runner = ReferencesCleanupRunnable(WritesReport(write), str(refs),
                                       TamperingTransport())
    with pytest.raises(RuntimeError, match="research cleanup list changed"):
        runner.invoke({})
    assert victim.read_text() == "outside"
