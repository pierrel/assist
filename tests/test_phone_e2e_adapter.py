"""Pin the simulator's observed exact-generation cleanup contracts."""
from types import SimpleNamespace

import pytest

from tests import phone_e2e_server as adapter


@pytest.fixture
def shell(monkeypatch):
    shell = SimpleNamespace(container=object(), owner=None)
    monkeypatch.setattr(adapter, "GENERATIONS", {"/fixture": shell})
    return shell


@pytest.mark.parametrize("expected", [None, "stale"])
def test_cleanup_preserves_current_generation_for_absent_or_stale_capture(shell, expected):
    adapter.cleanup("/fixture", expected)
    assert adapter.GENERATIONS["/fixture"] is shell


def test_cleanup_omitted_capture_removes_current_generation(shell):
    adapter.cleanup("/fixture")
    assert "/fixture" not in adapter.GENERATIONS


@pytest.mark.parametrize("expected", [None, "stale", "absent"])
def test_verified_cleanup_rejects_unavailable_capture(shell, expected):
    if expected == "absent":
        adapter.GENERATIONS.clear()
        expected = shell.container
    with pytest.raises(RuntimeError, match="generation is unavailable"):
        adapter.cleanup_verified("/fixture", expected)
    if adapter.GENERATIONS:
        assert adapter.GENERATIONS["/fixture"] is shell


def test_verified_cleanup_removes_exact_generation(shell):
    adapter.cleanup_verified("/fixture", shell.container)
    assert "/fixture" not in adapter.GENERATIONS


def test_verified_cleanup_rejects_replacement_during_teardown(shell, monkeypatch):
    replacement = SimpleNamespace(container=object(), owner=None)
    cleanup = adapter.cleanup

    def replace(worktree, expected):
        cleanup(worktree, expected)
        adapter.GENERATIONS[worktree] = replacement

    monkeypatch.setattr(adapter, "cleanup", replace)
    with pytest.raises(RuntimeError, match="generation changed during teardown"):
        adapter.cleanup_verified("/fixture", shell.container)
    assert adapter.GENERATIONS["/fixture"] is replacement
