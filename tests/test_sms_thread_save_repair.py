"""Synthetic finalization regression: no messages or private thread data."""
import sys
from types import SimpleNamespace

import pytest

from assist.browser import authority, manager as browser
from assist.events.model import Subscription
from assist.events.store import SubscriptionStore
from manage.web.thread_git_lifecycle import ThreadGitLifecycle
from tests.test_thread_git_simple import LocalSandbox, bound_thread, git, repositories  # noqa: F401


def snapshot_client(monkeypatch, entries):
    def containers(**kwargs):
        assert kwargs == {'all': True, 'filters': {'label': 'assist.sandbox=true'}}
        return entries
    monkeypatch.setattr(browser.SandboxManager, '_get_docker_client',
                        lambda: SimpleNamespace(api=SimpleNamespace(containers=containers)))


def test_harmless_finalization_after_unrelated_container_auto_removal(
        bound_thread, tmp_path, monkeypatch):
    owner, source, worktree, _ = bound_thread
    authority.mark_new_thread(str(tmp_path), 'thread')
    store = SubscriptionStore(str(tmp_path))
    store.add(Subscription('synthetic-subscription', 'thread', '.*', '{text}'))
    subscription_bytes = (tmp_path/'thread'/'subscriptions.json').read_bytes()
    # A listed unrelated container has disappeared by any later inspect.
    # Exercise the real bounded subprocess failure if the old two-call scan returns.
    shim = tmp_path/'docker-shim.py'
    shim.write_text("import sys\nif sys.argv[1]=='ps': print('removed-other')\nelse: sys.stderr.write('error: no such object: removed-other\\n'); sys.exit(1)\n")
    bounded = browser._bounded_cli
    cli_calls = []
    def cli(argv, **kwargs):
        cli_calls.append(argv)
        return bounded([sys.executable, str(shim), *argv[1:]], **kwargs)
    monkeypatch.setattr(browser, '_bounded_cli', cli)
    monkeypatch.setattr(browser, 'configured_directory', lambda: None)
    snapshot_client(monkeypatch, [{'Id': 'removed-other', 'Mounts': [
        {'Source': str(tmp_path/'unrelated'/'domain'), 'Destination': '/workspace'}]}])

    lifecycle = ThreadGitLifecycle(owner, str(worktree), (str(tmp_path), 'thread'))
    def backend(_timezone):
        browser.BrowserManager.confirm_owner_stopped(
            str(tmp_path), 'thread', None, before_replacement=True)
        generation = SimpleNamespace(id='synthetic-verifier', kill=lambda: None,
                                     wait=lambda timeout: None)
        browser.SandboxManager._containers[str(worktree)] = generation
        sandbox = LocalSandbox(); sandbox.container = generation
        return sandbox
    monkeypatch.setattr(lifecycle, '_backend', backend)
    monkeypatch.setattr(browser.SandboxManager, 'get_git_verification_backend',
                        lambda *args, **kwargs: backend(kwargs.get('tz')))
    before = git('rev-parse', 'HEAD', cwd=worktree)
    assert lifecycle.finish_visible('No workspace changes', None) is None
    assert git('rev-parse', 'HEAD', cwd=worktree) == before
    assert git('rev-parse', 'refs/heads/assist/thread', cwd=source) == before
    assert (tmp_path/'thread'/'subscriptions.json').read_bytes() == subscription_bytes
    assert store.route('+15555550100').id == 'synthetic-subscription'
    assert cli_calls == []  # No second scope query exists to race auto-removal.


def test_mount_snapshot_keeps_legacy_alias_and_exact_destination(monkeypatch, tmp_path):
    real = tmp_path/'thread'/'domain'; real.mkdir(parents=True)
    alias = tmp_path/'legacy-alias'; alias.symlink_to(real)
    snapshot_client(monkeypatch, [
        {'Id': 'legacy', 'Mounts': [{'Source': str(alias), 'Destination': '/workspace'}]},
        {'Id': 'different-destination', 'Mounts': [{'Source': str(real), 'Destination': '/reference'}]},
        {'Id': 'different-thread', 'Mounts': [{'Source': str(tmp_path/'other'/'domain'), 'Destination': '/workspace'}]},
    ])
    assert browser._sandbox_generations(str(tmp_path), 'thread') == {'legacy'}


@pytest.mark.parametrize('entries', [None, [{'Id': 'unknown'}],
    [{'Id': 'unknown', 'Mounts': {}}],
    [{'Id': 'unknown', 'Mounts': [{'Source': 'x'*262145}]}]])
def test_unknown_or_oversized_snapshot_is_not_clean_proof(monkeypatch, tmp_path, entries):
    snapshot_client(monkeypatch, entries)
    with pytest.raises(browser.BrowserUnavailable):
        browser._sandbox_generations(str(tmp_path), 'thread')


def test_unknown_scan_preserves_owner_lease_and_prevents_finalization(
        bound_thread, tmp_path, monkeypatch):
    owner, source, worktree, _ = bound_thread
    authority.mark_new_thread(str(tmp_path), 'thread')
    with authority.fence(str(tmp_path), 'thread') as state:
        state.begin('previous-run', 1)
        state.add_generation('previous-run', 'unknown-generation')
    def unavailable(**kwargs):
        raise RuntimeError('synthetic daemon unavailable')
    monkeypatch.setattr(browser.SandboxManager, '_get_docker_client',
        lambda: SimpleNamespace(api=SimpleNamespace(containers=unavailable)))
    monkeypatch.setattr(browser, 'configured_directory', lambda: None)
    lifecycle = ThreadGitLifecycle(owner, str(worktree), (str(tmp_path), 'thread'))
    monkeypatch.setattr(lifecycle, '_backend', lambda _: browser.BrowserManager.confirm_owner_stopped(
        str(tmp_path), 'thread', None, before_replacement=True))
    before = git('rev-parse', 'refs/heads/assist/thread', cwd=source)
    with pytest.raises(browser.BrowserUnavailable, match='unavailable'):
        lifecycle.finish_visible('Must not publish', None)
    assert git('rev-parse', 'refs/heads/assist/thread', cwd=source) == before
    with authority.fence(str(tmp_path), 'thread') as state:
        assert state.lease['generations'] == ['unknown-generation']



def test_auto_removed_matching_generation_uses_existing_exact_stop_proof(monkeypatch, tmp_path):
    (tmp_path/'thread').mkdir()
    authority.mark_new_thread(str(tmp_path), 'thread')
    with authority.fence(str(tmp_path), 'thread') as state:
        state.begin('previous-run', 1)
        state.add_generation('previous-run', 'already-removed')
    snapshot_client(monkeypatch, [{'Id': 'already-removed', 'Mounts': [
        {'Source': str(tmp_path/'thread'/'domain'), 'Destination': '/workspace'}]}])
    monkeypatch.setattr(browser, 'configured_directory', lambda: None)
    monkeypatch.setattr(browser, '_bounded_cli', lambda *args, **kwargs: b'')
    monkeypatch.setattr(browser.SandboxManager, '_ensure_egress_proxy_bounded', lambda: None)
    calls=[]
    def missing(argv, **kwargs):
        calls.append(argv[:2])
        return browser.subprocess.CompletedProcess(argv, 1, b'', b'error: no such object: already-removed')
    monkeypatch.setattr(browser.subprocess, 'run', missing)
    assert browser.BrowserManager.confirm_owner_stopped(str(tmp_path), 'thread', None, before_replacement=True)
    assert calls==[['docker','kill'],['docker','inspect']]
    with authority.fence(str(tmp_path), 'thread') as state:
        assert state.lease is None
