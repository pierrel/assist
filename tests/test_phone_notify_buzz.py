"""Phone-only notify: durable urgent reason, exact acknowledgments and narrow tools."""
import json

import pytest
from fastapi.testclient import TestClient

from assist.events import notify as notify_mod
from assist import thread_manager
from manage.web import state, phone_api
from manage.web.app import app


@pytest.fixture
def phone_attention(tmp_path, monkeypatch):
    monkeypatch.setattr(state.MANAGER, 'root_dir', str(tmp_path))
    (tmp_path / 'thread-a').mkdir()
    state._URGENT.clear()
    state._NOTIFICATION_REVISIONS.clear()
    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, 'synthetic-phone-token')
    monkeypatch.setattr(notify_mod, '_thread_id', lambda: 'thread-a')
    yield tmp_path
    state._URGENT.clear()
    state._NOTIFICATION_REVISIONS.clear()


def test_notify_queues_only_phone_reason_and_coalesces_until_open(phone_attention, monkeypatch):
    # Legacy settings must not restore a removed SMS fallback.
    monkeypatch.setenv('URGENT_SMS_RECIPIENT', '+15555550100')
    monkeypatch.setenv('URGENT_SMS_THREAD_URL_BASE', 'https://example.test')
    monkeypatch.setenv('ASSIST_SMS_OUTBOUND_URL', 'https://example.test/outbound/sms')
    monkeypatch.setenv('ASSIST_SMS_SECRET', 'synthetic-unused')
    outgoing = []
    monkeypatch.setattr('assist.events.reply.requests.post',
                        lambda *a, **k: outgoing.append((a, k)))
    notify = notify_mod.notify_tools(state._mark_urgent)[0]
    assert 'queued' in notify('Lease response needed before 4pm').lower()
    first = json.loads((phone_attention / 'thread-a' / 'urgent_response').read_text())
    assert first['reason'] == 'Lease response needed before 4pm'
    assert first['phone_ack'] is False
    assert 'pending' in notify('Lease response now needed before 3pm').lower()
    updated = json.loads((phone_attention / 'thread-a' / 'urgent_response').read_text())
    assert updated['id'] == first['id']
    assert updated['reason'] == 'Lease response now needed before 3pm'
    assert outgoing == []


def test_authenticated_attention_ack_preserves_urgent_and_rejects_old_cycle(phone_attention):
    state._mark_urgent('thread-a', 'Synthetic time-sensitive event')
    client = TestClient(app)
    assert client.get('/api/v1/phone/notifications').status_code == 401
    headers = {'Authorization': 'Bearer synthetic-phone-token'}
    response = client.get('/api/v1/phone/notifications', headers=headers)
    assert response.status_code == 200
    alert = response.json()['notifications'][0]
    assert alert['thread_id'] == 'thread-a'
    assert alert['reason'] == 'Synthetic time-sensitive event'
    path = f"/api/v1/phone/notifications/thread-a/{alert['id']}/ack"
    assert client.post(path, headers=headers).json()['acknowledged']
    assert state._has_urgent('thread-a')
    assert client.get('/api/v1/phone/notifications', headers=headers).json() == {'notifications': []}
    state._clear_urgent('thread-a')
    state._mark_urgent('thread-a', 'Another urgent cycle')
    newer = client.get('/api/v1/phone/notifications', headers=headers).json()['notifications'][0]
    assert newer['id'] != alert['id']
    assert client.post(path, headers=headers).json() == {'acknowledged': False}
    assert client.get('/api/v1/phone/notifications', headers=headers).json()['notifications'] == [newer]


def test_notify_requires_reason_and_reports_storage_failure(phone_attention, monkeypatch):
    notify = notify_mod.notify_tools(state._mark_urgent)[0]
    for reason in ('', '  ', 'x' * 201, 'bad\x1breason'):
        assert 'unavailable' in notify(reason).lower()
    assert not state._has_urgent('thread-a')
    assert not (phone_attention / 'thread-a' / 'urgent_response').exists()
    def unavailable(*args, **kwargs):
        raise OSError('synthetic private filesystem detail')
    monkeypatch.setattr(state, '_atomic_write', unavailable)
    result = notify('A useful bounded reason')
    assert 'unavailable' in result.lower()
    assert 'private filesystem' not in result
    assert not state._has_urgent('thread-a')


def test_sms_turn_only_gains_notify_not_owner_configuration_tools():
    names = {tool.__name__ for tool in thread_manager._web_triage_tools}
    assert names == {'send_reply', 'notify'}


def test_unacknowledged_first_batch_cannot_starve_later_threads(phone_attention):
    for index in range(20):
        tid = f'thread-{index:02d}'
        (phone_attention / tid).mkdir()
        state._mark_urgent(tid, 'Synthetic bounded attention')
    first = state._phone_notifications()['notifications']
    second = state._phone_notifications()['notifications']
    assert len(first) == len(second) == 16
    assert len({row['thread_id'] for row in first + second}) == 20


def test_maximum_unicode_reason_round_trips_pending_ack_and_readback(phone_attention):
    reason = '😀' * 200
    assert state._mark_urgent('thread-a', reason) == 'queued'
    pending = state._phone_notifications()['notifications']
    assert pending[0]['reason'] == reason
    alert_id = pending[0]['id']
    assert state._ack_phone_notification('thread-a', alert_id)['acknowledged']
    restored = state._phone_alert('thread-a')
    assert restored['reason'] == reason
    assert restored['id'] == alert_id and restored['phone_ack']


def test_legacy_badge_only_write_failure_remains_best_effort(phone_attention, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('synthetic storage failure')
    monkeypatch.setattr(state, '_atomic_write', unavailable)
    state._mark_urgent('thread-a')
    assert state._has_urgent('thread-a')
    assert not (phone_attention / 'thread-a' / 'urgent_response').exists()
