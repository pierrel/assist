"""Synthetic effect-boundary tests; no phone/network SMS is contacted."""
import pytest
import requests
from assist.events import reply
from assist.events.sms_archive import configured_archive


@pytest.fixture
def outgoing(tmp_path, monkeypatch):
    monkeypatch.setenv('ASSIST_SMS_ARCHIVE_DIR', str(tmp_path / 'archive'))
    monkeypatch.setenv('ASSIST_THREADS_DIR', str(tmp_path / 'threads'))
    monkeypatch.setenv('ASSIST_SMS_OUTBOUND_URL', 'https://phone.invalid/outbound/sms/v1')
    monkeypatch.setenv('ASSIST_SMS_SECRET', 'synthetic-secret')


def test_lost_response_preserves_unknown_and_repeated_confirmation_never_resends(outgoing, monkeypatch):
    sent = []
    def lost(url, **kwargs):
        payload = kwargs['json']
        assert configured_archive().read(payload['message_id'])['status'] == 'unknown'
        assert kwargs['timeout'] == (5, 70)
        sent.append(payload)
        raise requests.Timeout('response lost AFTER effect')
    monkeypatch.setattr(reply.requests, 'post', lost)
    first = reply.send_outbound_sms('+15555550100', 'exact "quote"', attempt_id='confirmed1')
    second = reply.send_outbound_sms('+15555550100', 'exact "quote"', attempt_id='confirmed1')
    assert first.status == second.status == 'unknown' and len(sent) == 1
    row, _ = configured_archive().record('confirmed1', 'outgoing', '+15555550100', 'exact "quote"', 'sent')
    assert reply.send_outbound_sms('+15555550100', 'exact "quote"', attempt_id='confirmed1').status == 'sent'
    assert len(sent) == 1 and row['status'] == 'sent'


def test_receipt_identity_and_conflict_do_not_send(outgoing, monkeypatch):
    sent = []
    def post(url, **kwargs):
        payload = kwargs['json']; sent.append(payload)
        return type('R', (), {'status_code': 200, 'json': lambda self: {'message_id': payload['message_id'], 'status': 'sent'}})()
    monkeypatch.setattr(reply.requests, 'post', post)
    assert reply.send_outbound_sms('+15555550100', 'same', attempt_id='one').status == 'sent'
    assert reply.send_outbound_sms('+15555550100', 'same', attempt_id='two').status == 'sent'
    assert reply.send_outbound_sms('+15555550100', 'CHANGED', attempt_id='one').status == 'not_sent'
    assert len(sent) == 2
    assert configured_archive().read('one')['occurred_at'] == sent[0]['occurred_at']


def test_reply_confirmation_identity_survives_repeated_tool_execution(outgoing, monkeypatch):
    monkeypatch.setattr(reply, 'get_config', lambda: {'configurable': {'thread_id': 't1', 'sms_reply_review_id': 'approval1', 'sms_sender': '+15555550100'}})
    sent = []
    def post(url, **kwargs):
        payload = kwargs['json']; sent.append(payload)
        return type('R', (), {'status_code': 200, 'json': lambda self: {'message_id': payload['message_id'], 'status': 'not_sent'}})()
    monkeypatch.setattr(reply.requests, 'post', post)
    assert 'not sent' in reply.send_reply('approved')
    assert 'not sent' in reply.send_reply('approved')
    assert len(sent) == 1


def test_malformed_success_does_not_claim_sent(outgoing, monkeypatch):
    monkeypatch.setattr(reply.requests, 'post', lambda *a, **k: type('R', (), {'status_code': 200, 'json': lambda self: {'status': 'sent'}})())
    assert reply.send_outbound_sms('+15555550100', 'hi', attempt_id='one').status == 'unknown'




def test_legacy_endpoint_is_refused_before_any_sms_effect(outgoing, monkeypatch):
    monkeypatch.setenv('ASSIST_SMS_OUTBOUND_URL', 'https://phone.invalid/outbound/sms')
    monkeypatch.setattr(reply.requests, 'post', lambda *args, **kwargs: pytest.fail('legacy SMS effect attempted'))
    result = reply.send_outbound_sms('+15555550100', 'hi', attempt_id='one')
    assert result.status == 'not_sent' and 'no send attempted' in result.detail


@pytest.mark.parametrize('status', ['sent', 'unknown'])
def test_replay_outcome_survives_removed_transport_configuration(outgoing, monkeypatch, status):
    configured_archive().record('existing', 'outgoing', '+15555550100', 'hi', status)
    monkeypatch.delenv('ASSIST_SMS_SECRET')
    monkeypatch.delenv('ASSIST_SMS_OUTBOUND_URL')
    monkeypatch.setattr(reply.requests, 'post', lambda *a, **k: pytest.fail('existing attempt repeated'))
    assert reply.send_outbound_sms('+15555550100', 'hi', attempt_id='existing').status == status


@pytest.mark.parametrize('malformed', [[], {}, None, 1])
def test_malformed_status_never_escapes_after_possible_effect(outgoing, monkeypatch, malformed):
    monkeypatch.setattr(reply.requests, 'post', lambda *a, **k: type('R', (), {'status_code':200, 'json':lambda self:{'message_id':'one','status':malformed}})())
    assert reply.send_outbound_sms('+15555550100', 'hi', attempt_id='one').status == 'unknown'


@pytest.mark.parametrize('recipient',['(unknown caller)','Anonymous','12;drop','+1555\n123',''])
def test_unsafe_raw_caller_never_reaches_transport(outgoing,monkeypatch,recipient):
    monkeypatch.setattr(reply.requests,'post',lambda *a,**k:pytest.fail('unsafe recipient reached transport'))
    assert reply.send_outbound_sms(recipient,'approved text',attempt_id='one').status=='not_sent'




def test_unknown_send_logs_category_without_private_payload(outgoing, monkeypatch, caplog):
    def lost(*args, **kwargs):
        raise requests.Timeout('private-secret private-body +15555550100')
    monkeypatch.setattr(reply.requests, 'post', lost)
    assert reply.send_outbound_sms('+15555550100', 'private-body', attempt_id='private-id').status == 'unknown'
    assert 'event=send_unknown' in caplog.text
    assert all(value not in caplog.text for value in ['private-id', 'private-body', '+15555550100', 'private-secret'])
