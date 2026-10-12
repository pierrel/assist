"""Pin the simulator's observed exact-generation cleanup contracts."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests import phone_e2e_server as adapter


@pytest.fixture
def shell(monkeypatch):
    shell = SimpleNamespace(container=SimpleNamespace(id="fixture-generation"), owner=None)
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


@pytest.mark.parametrize("failure", ["begin", "add_generation"])
def test_failed_authority_setup_preserves_prior_registry_generation(shell, monkeypatch, failure):
    def operation(name):
        def run(*args):
            if name == failure:
                raise OSError("authority write failed")
        return run

    @contextmanager
    def fence(*args):
        yield SimpleNamespace(begin=operation("begin"), add_generation=operation("add_generation"))

    monkeypatch.setattr(adapter.authority, "fence", fence)
    monkeypatch.setattr(adapter.threads, "_runs", lambda: SimpleNamespace(list=lambda tid: []))
    with pytest.raises(OSError, match="authority write failed"):
        adapter.Shell("/fixture", ("/scope", "thread"), "owner")
    assert adapter.GENERATIONS["/fixture"] is shell


def test_failed_authority_teardown_retains_generation_obligation(shell, monkeypatch):
    shell.owner, shell.scope = "owner", ("/scope", "thread")

    @contextmanager
    def fence(*args):
        raise OSError("authority write failed")
        yield

    monkeypatch.setattr(adapter.authority, "fence", fence)
    with pytest.raises(OSError, match="authority write failed"):
        adapter.cleanup_verified("/fixture", shell.container)
    assert adapter.GENERATIONS["/fixture"] is shell


def test_cleanup_preserves_replacement_registered_during_authority_teardown(shell, monkeypatch):
    shell.owner, shell.scope = "owner", ("/scope", "thread")
    replacement = SimpleNamespace(container=object(), owner=None)

    @contextmanager
    def fence(*args):
        yield SimpleNamespace(remove_generation=lambda *args: None,
                              clear=lambda owner: adapter.GENERATIONS.update({"/fixture": replacement}))

    monkeypatch.setattr(adapter.authority, "fence", fence)
    adapter.cleanup("/fixture", shell.container)
    assert adapter.GENERATIONS["/fixture"] is replacement
