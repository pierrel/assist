"""Source binding, operator preservation and restricted clean-probe contracts."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from assist.git_sync import (GitDirtyWorktreeError, GitSyncError, _write_state,
                             detach_legacy_object_hardlinks, enroll_legacy,
                             identity, read_state, require_clean, workspace)
from tests.test_local_git import git, repositories  # noqa: F401


def test_legacy_enrollment_preserves_dirty_files_and_uses_compact_binding(
        repositories, tmp_path):
    source, server, _ = repositories
    thread = tmp_path / "thread"
    thread.mkdir()
    (server / "draft.txt").write_text("uncommitted user work\n")
    expected = identity(str(server))
    enroll_legacy(str(thread), str(server), str(source), expected)
    assert read_state(str(thread)) == {"version": 2, "source": str(source),
                                      "branch": expected[0], "floor": expected[1]}
    assert (server / "draft.txt").read_text() == "uncommitted user work\n"
    assert workspace(str(thread), str(server))["branch"] == expected[0]
    with pytest.raises(GitSyncError, match="already authorized"):
        enroll_legacy(str(thread), str(server), str(source), expected)


def test_enrollment_rejects_shared_object_inode_before_writing_binding(
        repositories, tmp_path):
    source, server, _ = repositories
    thread = tmp_path / "thread"
    thread.mkdir()
    object_root = server / ".git" / "objects"
    loose = next(file for file in object_root.rglob("*")
                 if file.is_file() and len(file.name) == 38)
    alias = tmp_path / "linked-object"
    os.link(loose, alias)
    with pytest.raises(GitSyncError, match="hardlinks"):
        enroll_legacy(str(thread), str(server), str(source), identity(str(server)))
    assert read_state(str(thread)) is None


def test_operator_detachment_keeps_branch_and_files_and_allows_enrollment(
        repositories, tmp_path):
    source, _, _ = repositories
    linked = tmp_path / "linked"
    git("clone", "-b", "assist/thread", source, linked)
    thread = tmp_path / "thread"
    thread.mkdir()
    (linked / "draft.txt").write_text("user work\n")
    expected = identity(str(linked))
    remote = git("rev-parse", "refs/heads/assist/thread", cwd=source)
    detach_legacy_object_hardlinks(
        str(thread), str(linked), source=str(source), expected=expected,
        expected_remote=remote, verify_stopped=lambda: None)
    enroll_legacy(str(thread), str(linked), str(source), expected)
    assert identity(str(linked)) == expected
    assert (linked / "draft.txt").read_text() == "user work\n"


def test_compact_binding_rejects_extra_publication_fields(tmp_path):
    thread = tmp_path / "thread"
    thread.mkdir()
    _write_state(str(thread), {"version": 2, "source": "source.git",
                               "branch": "assist/test", "floor": "a" * 40,
                               "published_revision": "a" * 40})
    with pytest.raises(GitSyncError, match="binding"):
        read_state(str(thread))


@pytest.mark.parametrize("exit_code,truncated,exception", [
    (0, False, None), (42, False, GitDirtyWorktreeError),
    (42, True, GitSyncError), (1, False, GitSyncError),
])
def test_only_complete_clean_probe_can_classify_dirty(exit_code, truncated, exception):
    class Backend:
        def execute(self, _command):
            return SimpleNamespace(exit_code=exit_code, truncated=truncated)

    if exception is None:
        require_clean(Backend())
    else:
        with pytest.raises(exception):
            require_clean(Backend())
