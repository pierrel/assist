from types import SimpleNamespace
import pytest
from assist.events.sms_outbox import SmsOutbox,sms_sending_tools
from manage.web import phone_api,state

@pytest.fixture
def box(tmp_path,monkeypatch):
    (tmp_path/'thread').mkdir()
    monkeypatch.setenv('ASSIST_SMS_ARCHIVE_DIR',str(tmp_path/'archive'))
    monkeypatch.setenv('ASSIST_THREADS_DIR',str(tmp_path/'workspaces'))
    monkeypatch.setattr(state.MANAGER,'root_dir',str(tmp_path))
    monkeypatch.setattr(phone_api,'_thread_dir',lambda tid: str(tmp_path/tid))
    return SmsOutbox(str(tmp_path))

def test_normal_tool_returns_pending_before_approval_and_stable_replay(box,monkeypatch):
    from assist.events import sms_outbox
    monkeypatch.setattr(sms_outbox,'get_config',lambda:{'configurable':{'thread_id':'thread'}})
    tool=sms_sending_tools(box)[0]
    result=tool('+15555550100','hey now',SimpleNamespace(tool_call_id='operation'))
    assert 'if approved' in result and 'Nothing sent' in result
    assert box.for_thread('thread')[0].state=='pending'
    assert tool('+15555550100','hey now',SimpleNamespace(tool_call_id='operation'))==result
    assert len(box.for_thread('thread'))==1
    assert 'changed' in tool('+15555550100','different',SimpleNamespace(tool_call_id='operation'))
    assert box.for_thread('thread')[0].text=='hey now'

def test_existing_approval_collection_binds_exact_sms_and_never_sends(box,monkeypatch):
    record=box.stage('thread','operation','+15555550100','quote "one"\nline 👋')
    value=phone_api._sms_previews('thread')[0]
    assert (value['number'],value['text'])==(record.number,record.text)
    invalid=phone_api._ApprovalDecision(kind='send_sms',token='wrong',decision='approve')
    with pytest.raises(Exception):phone_api._sms_decision('thread',invalid)
    accepted=phone_api._sms_decision('thread',phone_api._ApprovalDecision(kind='send_sms',token=value['token'],decision='approve'))
    assert accepted['proposal']['state']=='approved'
    assert box.for_thread('thread')[0].state=='approved'
    outcome=phone_api._sms_decision('thread',phone_api._ApprovalDecision(kind='send_sms',token=value['token'],decision='unknown'))
    assert outcome['proposal']['state']=='unknown'
    assert phone_api._sms_decision('thread',phone_api._ApprovalDecision(kind='send_sms',token=value['token'],decision='approve'))['proposal']['state']=='unknown'
    with pytest.raises(Exception):
        phone_api._sms_decision('thread',phone_api._ApprovalDecision(kind='send_sms',token=value['token'],decision='reject'))

def test_rejection_is_durable_and_does_not_create_sms_archive_attempt(box):
    from assist.events.sms_archive import configured_archive
    record=box.stage('thread','reject','+15555550100','do not send')
    box.decide('thread',record.id,'rejected')
    assert SmsOutbox(box._root).stage('thread','reject',record.number,record.text).state=='rejected'
    assert configured_archive().read(record.id) is None


def test_authenticated_http_discovery_and_rejection_never_send(box, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from manage.web import egress, geo, threads

    monkeypatch.setenv(phone_api.PHONE_API_TOKEN_ENV, 'synthetic-only-token')
    monkeypatch.setattr(threads, '_is_pi_thread', lambda tid: False)
    monkeypatch.setattr(phone_api, '_approval_preview', lambda tid: {'proposal': None})
    monkeypatch.setattr(threads, 'reply_approval_preview', lambda tid: None)
    monkeypatch.setattr(egress, 'approval_previews', lambda tid: [])
    monkeypatch.setattr(geo, 'approval_previews', lambda tid: [])
    def forbidden(*args, **kwargs):
        raise AssertionError('Approval metadata must never send SMS')
    monkeypatch.setattr('requests.post', forbidden)
    draft = box.stage('thread', 'http-operation', '+15555550100', 'hey now')
    app = FastAPI()
    app.include_router(phone_api.router)
    client = TestClient(app)
    url = phone_api.PHONE_API_PREFIX + '/threads/thread/approvals'
    assert client.get(url).status_code == 401
    headers = {'Authorization': 'Bearer synthetic-only-token'}
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    proposal = response.json()['proposals'][0]
    assert (proposal['id'], proposal['number'], proposal['text'], proposal['state']) == (
        draft.id, draft.number, draft.text, 'pending')
    decision = {'kind': 'send_sms', 'token': proposal['token'], 'decision': 'reject'}
    response = client.post(url.removesuffix('s'), headers=headers, json=decision)
    assert response.status_code == 200
    assert response.json()['proposal']['state'] == 'rejected'
    assert box.for_thread('thread')[0].state == 'rejected'


def test_terminal_history_does_not_hide_pending_or_consume_capacity(box):
    from assist.events.sms_archive import configured_archive
    archive = configured_archive()
    for index in range(20):
        draft = box.stage('thread', str(index), '+15555550100', '\x01' * 4096)
        box.decide('thread', draft.id, 'approved')
        archive.record(draft.id, 'outgoing', draft.number, draft.text, 'sent')
        # Only the trusted phone result (or its existing manual Review) finishes a draft.
        box.decide('thread', draft.id, 'sent')
    pending = box.stage('thread', 'new', '+15555550100', 'new pending')
    previews = phone_api._sms_previews('thread')
    assert len(previews) == 11
    assert previews[0]['id'] == pending.id
    assert len(box.for_thread('thread')) == 21  # durable identities remain available
    assert len(__import__('json').dumps({'proposals': previews}).encode()) < phone_api.MAX_APPROVAL_BYTES


def test_phone_can_reject_approved_draft_before_any_attempt(box):
    draft = box.stage('thread', 'no-attempt', '+15555550100', 'never sent')
    box.decide('thread', draft.id, 'approved')
    assert box.decide('thread', draft.id, 'rejected').state == 'rejected'


def test_old_active_draft_rejection_returns_exact_ack_after_projection_rolls(box):
    old = box.stage('thread', 'old-active', '+15555550100', 'old')
    box.decide('thread', old.id, 'approved')
    token = phone_api._sms_previews('thread')[0]['token']
    for index in range(11):
        draft = box.stage('thread', f'recent-{index}', '+15555550100', 'new')
        box.decide('thread', draft.id, 'rejected')
    body = phone_api._ApprovalDecision(kind='send_sms', token=token, decision='reject')
    for _ in range(2):
        result = phone_api._sms_decision('thread', body)['proposal']
        assert (result['id'], result['state']) == (old.id, 'rejected')


@pytest.mark.parametrize('terminal', ['sent', 'not_sent'])
def test_phone_terminal_outcome_dominates_older_archive_unknown(box, terminal):
    from assist.events.sms_archive import configured_archive
    record = box.stage('thread', 'receipt-order', '+15555550100', 'exact')
    box.decide('thread', record.id, 'approved')
    configured_archive().record(record.id, 'outgoing', record.number, record.text, 'unknown')
    box.decide('thread', record.id, terminal)
    preview = phone_api._sms_previews('thread')[0]
    assert preview['state'] == terminal
    replay = phone_api._sms_decision('thread', phone_api._ApprovalDecision(
        kind='send_sms', token=preview['token'], decision='approve'))
    assert replay['proposal']['state'] == terminal
    assert box.for_thread('thread')[0].state == terminal


def test_first_draft_requires_usable_archive(box, monkeypatch, tmp_path):
    from assist.events import sms_outbox
    monkeypatch.setattr(sms_outbox, 'get_config', lambda: {'configurable': {'thread_id': 'thread'}})
    tool = sms_sending_tools(box)[0]
    monkeypatch.delenv('ASSIST_SMS_ARCHIVE_DIR')
    assert tool('+15555550100', 'exact', SimpleNamespace(tool_call_id='no-config')).startswith('SMS draft status unavailable:')
    assert box.for_thread('thread') == []
    unavailable = tmp_path / 'bad-mode'
    unavailable.mkdir(mode=0o755)
    monkeypatch.setenv('ASSIST_SMS_ARCHIVE_DIR', str(unavailable))
    assert tool('+15555550100', 'exact', SimpleNamespace(tool_call_id='bad-config')).startswith('SMS draft status unavailable:')
    assert box.for_thread('thread') == []


def test_snapshot_projects_saved_drafts_without_archive_settlement(box, monkeypatch):
    from unittest.mock import Mock
    from assist.events import sms_archive
    from assist.events.sms_outbox import SmsDraft
    from manage.web import egress, geo
    records = [SmsDraft(f'outbox_{i:064x}', 'thread', '+15555550100', 'exact',
                        '2026-10-25T00:00:00+00:00', 'unknown') for i in range(1000)]
    records.extend(SmsDraft(f'outbox_{i:064x}', 'thread', '+15555550100', 'exact',
                           '2026-10-25T00:00:00+00:00') for i in range(1000, 1005))
    box._write('thread', records)
    archive = Mock()
    archive.read.return_value = None
    configured = Mock(return_value=archive)
    monkeypatch.setattr(sms_archive, 'configured_archive', configured)
    monkeypatch.setattr(egress, 'approval_previews', lambda tid: [])
    monkeypatch.setattr(geo, 'approval_previews', lambda tid: [])
    assert phone_api._approval_summary('thread', {'stage': 'ready'}) == (5, True)
    configured.assert_not_called()
    archive.read.assert_not_called()


@pytest.mark.parametrize('saved', ['sent', 'approved', 'unknown'])
def test_matching_replay_preserves_saved_outcome_when_archive_unavailable(box, monkeypatch, saved):
    from assist.events import sms_outbox
    monkeypatch.setattr(sms_outbox, 'get_config', lambda: {'configurable': {'thread_id': 'thread'}})
    tool = sms_sending_tools(box)[0]
    runtime = SimpleNamespace(tool_call_id='saved-attempt')
    tool('+15555550100', 'exact', runtime)
    draft = box.for_thread('thread')[0]
    box.decide('thread', draft.id, 'approved')
    if saved != 'approved':
        box.decide('thread', draft.id, saved)
    monkeypatch.delenv('ASSIST_SMS_ARCHIVE_DIR')
    result = tool('+15555550100', 'exact', runtime)
    assert f': {saved}.' in result
    assert 'SMS not sent' not in result
    assert len(box.for_thread('thread')) == 1
    assert box.for_thread('thread')[0].state == saved


def test_new_stage_does_not_reconcile_unrelated_unknown_history(box, monkeypatch):
    from unittest.mock import Mock
    from assist.events import sms_outbox
    from assist.events.sms_outbox import SmsDraft
    records = [SmsDraft(f'outbox_{i:064x}', 'thread', '+15555550100', 'exact',
                        '2026-10-25T00:00:00+00:00', 'unknown') for i in range(1000)]
    records.append(SmsDraft('pending', 'thread', '+15555550100', 'exact',
                            '2026-10-25T00:00:00+00:00'))
    box._write('thread', records)
    archive = Mock()
    archive.read.return_value = None
    monkeypatch.setattr(sms_outbox, 'configured_archive', lambda: archive)
    assert box.stage('thread', 'new-operation', '+15555550100', 'new').state == 'pending'
    archive.read.assert_not_called()
    assert len(box.for_thread('thread')) == 1002


@pytest.mark.parametrize('mail_body,retained,fallback', [
    ('x' * (32 * 1024), 10, False),
    ('\x01' * (64 * 1024), 0, True),
], ids=['trim-results', 'mail-fallback'])
def test_combined_approval_budget_preserves_exact_active_sms(box, monkeypatch, mail_body, retained, fallback):
    import json
    from manage.web import egress, geo, threads
    monkeypatch.setattr(threads, '_is_pi_thread', lambda tid: False)
    monkeypatch.setattr(threads, 'email_identity', lambda: ('sender@example.test', ''))
    mail = threads.email_approval_preview({
        'pending_email_token': 'synthetic-email', 'pending_email_to': 'recipient@example.test',
        'pending_email_subject': 'Synthetic', 'pending_email_body': mail_body})
    monkeypatch.setattr(phone_api, '_approval_preview', lambda tid: {'proposal': mail})
    monkeypatch.setattr(threads, 'reply_approval_preview', lambda tid: None)
    monkeypatch.setattr(egress, 'approval_previews', lambda tid: [])
    monkeypatch.setattr(geo, 'approval_previews', lambda tid: [])
    for index in range(retained):
        record = box.stage('thread', f'finished-{index}', '+15555550100', '\x01' * 4096)
        box.decide('thread', record.id, 'rejected')
    active = [box.stage('thread', f'active-{index}', '+15555550100', '\x01' * 4096)
              for index in range(10)]
    before = {'thread_id': 'thread', 'proposals': [{'state': 'pending', **mail}, *phone_api._sms_previews('thread')]}
    assert len(json.dumps(before).encode()) > phone_api.MAX_APPROVAL_BYTES
    result = phone_api._approval_collection('thread')
    assert len(json.dumps(result).encode()) <= phone_api.MAX_APPROVAL_BYTES
    drafts = [value for value in result['proposals'] if value['kind'] == 'send_sms' and value['state'] == 'pending']
    assert {value['id'] for value in drafts} == {record.id for record in active}
    assert all(value['text'] == '\x01' * 4096 and value['number'] == '+15555550100' for value in drafts)
    email = next(value for value in result['proposals'] if value['kind'] == 'send_email')
    if fallback:
        assert email['state'] == 'unavailable' and email['decisions'] == []
        assert 'Assist Web' in email['error'] and 'action' not in email
    else:
        assert email['action'] == mail['action'] and email['token'] == mail['token']
    assert len(box.for_thread('thread')) == retained + 10


def test_unknown_is_terminal_without_archive_reconciliation_or_automatic_send(box, monkeypatch):
    from assist.events import sms_outbox
    record=box.stage('thread','uncertain-attempt','+15555550100','exact')
    box.decide('thread',record.id,'approved')
    box.decide('thread',record.id,'unknown')
    monkeypatch.setattr(sms_outbox,'configured_archive',lambda: pytest.fail('terminal unknown queried archive'))
    assert box.stage('thread','uncertain-attempt','+15555550100','exact').state=='unknown'
    preview=phone_api._sms_previews('thread')[0]
    assert preview['state']=='unknown' and preview['decisions']==[]
    with pytest.raises(ValueError,match='no longer applicable'):
        box.decide('thread',record.id,'sent')
    assert box.for_thread('thread')[0].state=='unknown'


@pytest.mark.parametrize('outcome',['sent','not_sent','unknown'])
def test_write_ahead_unknown_poll_cannot_complete_inflight_draft(box, outcome):
    from assist.events.sms_archive import configured_archive
    record=box.stage('thread','inflight-attempt','+15555550100','exact')
    preview=phone_api._sms_previews('thread')[0]
    box.decide('thread',record.id,'approved')
    archive=configured_archive()
    # Actual helper commits unknown before entering ModemManager.Send.
    archive.record(record.id,'outgoing',record.number,record.text,'unknown')
    assert phone_api._sms_previews('thread')[0]['state']=='approved'
    assert box.stage('thread','inflight-attempt','+15555550100','exact').state=='approved'
    assert box.for_thread('thread')[0].state=='approved'
    result=phone_api._sms_decision('thread',phone_api._ApprovalDecision(
        kind='send_sms',token=preview['token'],decision=outcome))
    assert result['proposal']['state']==outcome
    if outcome=='unknown':
        archive.record(record.id,'outgoing',record.number,record.text,'sent')
        assert phone_api._sms_previews('thread')[0]['state']=='unknown'
    assert box.for_thread('thread')[0].state==outcome
