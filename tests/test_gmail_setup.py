"""Enrollment is an operator workflow; tests use a fake consent flow and no network."""
import json
import sys
from types import SimpleNamespace

import pytest

from assist import gmail, gmail_setup


def test_enrollment_fixed_scope_loopback_and_private_new_token(monkeypatch,tmp_path,capsys):
    client=tmp_path/'client.json'
    client.write_text(json.dumps({'installed':{'client_id':'synthetic','client_secret':'secret',
                                             'auth_uri':'https://accounts.google.com/o/oauth2/auth',
                                             'token_uri':gmail._TOKEN_URL}}))
    client.chmod(0o600)
    target=tmp_path/'private'/'token.json'
    calls=[]
    class Flow:
        @staticmethod
        def from_client_config(config,scopes):
            calls.append((config,scopes))
            return Flow()
        def run_local_server(self,**kwargs):
            calls.append(kwargs)
            value={'refresh_token':'synthetic-refresh','client_id':'synthetic','client_secret':'secret',
                   'token_uri':gmail._TOKEN_URL,'scopes':[gmail.GMAIL_SCOPE]}
            return SimpleNamespace(refresh_token='synthetic-refresh',to_json=lambda:json.dumps(value))
    monkeypatch.setitem(sys.modules,'google_auth_oauthlib',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'google_auth_oauthlib.flow',SimpleNamespace(InstalledAppFlow=Flow))
    monkeypatch.setenv('ASSIST_THREADS_DIR',str(tmp_path/'threads'))
    gmail_setup.main(['--client-file',str(client),'--token-file',str(target),'--no-browser'])
    assert calls[0][1]==[gmail.GMAIL_SCOPE]
    assert calls[1]['host']=='127.0.0.1' and calls[1]['port']==8765
    assert calls[1]['timeout_seconds']==300 and not calls[1]['open_browser']
    assert target.stat().st_mode & 0o777==0o600
    assert target.parent.stat().st_mode & 0o777==0o700
    output=capsys.readouterr().out
    assert '7 days' in output and 'synthetic-refresh' not in output and 'secret' not in output
    # A second enrollment cannot overwrite a saved token.
    with pytest.raises(SystemExit):
        gmail_setup.main(['--client-file',str(client),'--token-file',str(target)])


def test_enrollment_refuses_token_under_agent_mount_before_consent(monkeypatch,tmp_path):
    monkeypatch.setenv('ASSIST_THREADS_DIR',str(tmp_path))
    with pytest.raises(SystemExit):
        gmail_setup.main(['--client-file','unused','--token-file',str(tmp_path/'token.json')])
