"""Host credentials work without checkout config or global rewrites/hooks."""
from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit

import pytest

from assist.git_sync import identity
from assist.local_git import LocalGit
from assist.pi_preview import PiPreviewPolicy
from manage.web import phone_api, state, threads
from tests.test_local_git import git, repositories  # noqa: F401


@pytest.fixture
def credentialed_source(tmp_path, repositories, monkeypatch):
    source, server_checkout, _ = repositories
    git('config', 'http.receivepack', 'true', cwd=source)
    backend = Path(git('--exec-path')) / 'git-http-backend'
    expected = 'Basic ' + base64.b64encode(b'fixture:synthetic-password').decode()
    observations = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.serve_git()

        def do_POST(self):
            self.serve_git()

        def serve_git(self):
            parsed = urlsplit(self.path)
            writing = 'git-receive-pack' in self.path
            authenticated = self.headers.get('Authorization') == expected
            observations.append((self.command, writing, authenticated))
            if writing and not authenticated:
                self.send_response(401)
                self.send_header('WWW-Authenticate', 'Basic realm="private-fixture"')
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith('GIT_')}
            environment.update(GIT_PROJECT_ROOT=str(source.parent), GIT_HTTP_EXPORT_ALL='1',
                               PATH_INFO=parsed.path, QUERY_STRING=parsed.query,
                               REQUEST_METHOD=self.command,
                               CONTENT_TYPE=self.headers.get('Content-Type', ''),
                               CONTENT_LENGTH=str(len(body)), REMOTE_USER='fixture',
                               GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1')
            result = subprocess.run([str(backend)], input=body, env=environment,
                                    capture_output=True, check=True, timeout=10)
            headers, payload = result.stdout.split(b'\r\n\r\n', 1)
            lines = [line.decode().split(':', 1) for line in headers.split(b'\r\n')]
            status = next((int(value.strip().split()[0]) for key, value in lines
                           if key.lower() == 'status'), 200)
            self.send_response(status)
            for key, value in lines:
                if key.lower() != 'status':
                    self.send_header(key, value.strip())
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}/{source.name}'
    home = tmp_path / 'host-home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(home / 'xdg'))
    monkeypatch.delenv('GIT_CONFIG_GLOBAL', raising=False)
    marker = tmp_path / 'untrusted-helper-ran'
    credentials = home / 'credentials.conf'
    # A scoped empty helper resets the generic one. Both following helpers
    # are needed, and the password helper requires useHttpPath to survive.
    credentials.write_text(
        '[credential]\n helper = "!touch ' + str(marker) + '"\n'
        '[credential "' + url + '"]\n helper =\n'
        ' helper = "!f() { test \\"$1\\" != get || '
        'printf \'username=fixture\\n\'; }; f"\n'
        ' helper = "!f() { data=$(cat); test \\"$1\\" != get || '
        '{ printf \'%s\\n\' \\"$data\\" | grep -qx path=source.git && '
        'printf \'password=synthetic-password\\n\'; }; }; f"\n'
        ' useHttpPath\n'
        '[credential "https://unrelated.example.test"]\n'
        ' helper = "!touch ' + str(marker) + '"\n')
    (home / '.gitconfig').write_text('[include]\n path = ' + str(credentials) + '\n')
    try:
        yield url, source, server_checkout, marker, observations, home
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_host_helper_chain_publishes_without_checkout_or_global_rewrites(
        credentialed_source, tmp_path):
    url, source, checkout, marker, observations, home = credentialed_source
    # LocalGit must neither run the checkout helper nor follow a global rewrite.
    git('config', 'credential.helper', '!touch ' + str(marker), cwd=checkout)
    git('config', 'url.http://127.0.0.1:1/.insteadOf', url, cwd=checkout)
    with (home / '.gitconfig').open('a') as config:
        config.write('[url "http://127.0.0.1:1/"]\n insteadOf = ' + url + '\n')
    with LocalGit(url, str(checkout)) as host:
        branch, revision = host.same_tip()
        assert host.publish('assist/credential-proof', revision, create_only=True) == revision
    assert git('rev-parse', 'refs/heads/assist/credential-proof', cwd=source) == revision
    assert any(writing and auth for _, writing, auth in observations)
    assert not marker.exists()


def test_missing_host_credentials_never_fall_back_to_checkout_helper(
        credentialed_source):
    url, source, checkout, marker, _, home = credentialed_source
    (home / '.gitconfig').unlink()
    git('config', 'credential.helper', '!touch ' + str(marker), cwd=checkout)
    from assist.git_sync import GitSyncError
    with LocalGit(url, str(checkout)) as host:
        _, revision = host.same_tip()
        with pytest.raises(GitSyncError, match='publication is not verified'):
            host.publish('assist/missing-credential', revision, create_only=True)
    assert git('for-each-ref', '--format=%(objectname)',
               'refs/heads/assist/missing-credential', cwd=source) == ''
    assert not marker.exists()


@pytest.mark.parametrize('engine', ['hermes', 'deepagents'])
def test_phone_git_initializer_publishes_with_host_credentials(
        credentialed_source, tmp_path, monkeypatch, engine):
    url, source, _, marker, observations, _ = credentialed_source
    root = tmp_path / 'threads'
    root.mkdir()
    monkeypatch.setattr(state.MANAGER, 'root_dir', str(root))
    monkeypatch.setattr(state, 'DOMAINS', [url])
    monkeypatch.setattr(threads, 'DOMAINS', [url])
    monkeypatch.setenv('ASSIST_DOMAINS', url)
    policy = PiPreviewPolicy(root, engine='hermes')
    policy.set_enabled(True)
    monkeypatch.setattr(state, 'HERMES_PREVIEW', policy)
    monkeypatch.setattr(threads, 'HERMES_PREVIEW', policy)
    monkeypatch.setattr(threads.BrowserManager, 'confirm_owner_stopped', lambda *a, **k: False)
    monkeypatch.setattr(threads, '_dispatch_pending_after', lambda *_: None)
    executed = []
    monkeypatch.setattr(threads, '_execute_run', lambda rid, tid: executed.append((rid, tid)))
    try:
        tid, run, domain, replay = phone_api._create_and_submit(
            phone_api._CreateThread(message='Tell me what is in this repository.',
                                   repo_key=phone_api._repo_key(url), harness=engine),
            'private-credential-fixture-' + engine)
        threads._initialize_thread_active(tid, run.id, domain)
        branch, revision = identity(state.MANAGER.thread_default_working_dir(tid))
        assert not replay and executed == [(run.id, tid)]
        assert git('rev-parse', 'refs/heads/' + branch, cwd=source) == revision
        assert phone_api._thread_git_status(tid)['state'] == 'available'
        assert any(writing and auth for _, writing, auth in observations)
        assert not marker.exists()
    finally:
        threads._RUN_SERVICES_BY_ROOT.pop(str(root), None)
