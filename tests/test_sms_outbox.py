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
        # Receipt handoff succeeded while the phone's lifecycle POST was lost.
        archive.record(draft.id, 'outgoing', draft.number, draft.text, 'sent')
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
