"""Durable call-event admission preserves chosen handling across retry crashes."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from assist.events.calls import admit_call
from assist.events.model import Subscription


def test_run_admission_failure_retries_exact_persisted_handling(tmp_path):
    event = {'event_id': 'call1', 'number': '+15555550100', 'outcome': 'terminated_before_active', 'reason': 4}
    sub = Subscription('sub1', 'thread1', '.*', 'Caller {sender}: {text}', event_kind='call')
    store = SimpleNamespace(route=lambda number, kind: sub)
    attempts = []
    def interrupted(tid, prompt, **kwargs):
        attempts.append((tid, prompt, kwargs))
        raise OSError('crash before Run admission')
    with pytest.raises(OSError): admit_call(tmp_path, event, store, interrupted)
    store.route = lambda *args: replace(sub, thread_id='CHANGED')
    def create(tid, prompt, **kwargs):
        attempts.append((tid, prompt, kwargs)); return SimpleNamespace(id=kwargs['run_id'], thread_id=tid)
    run = admit_call(tmp_path, event, store, create)
    assert attempts[0] == attempts[1] and run.thread_id == 'thread1'
    assert 'No voicemail transcript' in attempts[1][1]
    assert admit_call(tmp_path, event, store, create) is None
    assert len(attempts) == 2
    with pytest.raises(ValueError): admit_call(tmp_path, {**event, 'reason': 8}, store, create)


def test_unmatched_call_does_not_later_dispatch_when_subscription_changes(tmp_path):
    event = {'event_id': 'call1', 'number': '+15555550100', 'outcome': 'terminated_after_active', 'reason': 7}
    store = SimpleNamespace(route=lambda *args: None)
    create = lambda *args, **kwargs: pytest.fail('unmatched call dispatched')
    assert admit_call(tmp_path, event, store, create) is None
    store.route = lambda *args: Subscription('s', 'thread', '.*', '{text}', event_kind='call')
    assert admit_call(tmp_path, event, store, create) is None


def test_unknown_caller_and_refresh_without_reason_still_route_untrusted(tmp_path):
    event = {'event_id': 'unknown_call', 'number': None, 'outcome': 'terminated_before_active', 'reason': None}
    sub = Subscription('s', 'thread', '.*', '{sender}: {text}', event_kind='call')
    routed = []
    store = SimpleNamespace(route=lambda number, kind: routed.append((number, kind)) or sub)
    calls = []
    def create(tid, prompt, **kwargs):
        calls.append((prompt, kwargs)); return SimpleNamespace(id=kwargs['run_id'])
    assert admit_call(tmp_path, event, store, create) is not None
    assert routed == [('', 'call')]
    assert calls[0][1]['sender'] == '(unknown caller)'  # Retains triage trust boundary.
    assert 'unavailable' in calls[0][0]


def test_call_admission_metadata_never_becomes_a_visible_thread(tmp_path):
    from assist.thread_manager import ThreadManager
    event = {'event_id': 'call1', 'number': '+15555550100', 'outcome': 'terminated_before_active', 'reason': None}
    assert admit_call(tmp_path, event, SimpleNamespace(route=lambda *args: None), lambda *args, **kwargs: pytest.fail('unexpected Run')) is None
    manager = ThreadManager(str(tmp_path))
    try: assert manager.list() == []
    finally: manager.conn.close()


@pytest.mark.parametrize('caller',['(unknown caller)','Anonymous','12;drop','+1555\n123',''])
def test_approved_reply_to_unsafe_or_unavailable_raw_caller_never_posts(monkeypatch,caller):
    from assist.events import reply
    monkeypatch.setattr(reply,'get_config',lambda:{'configurable':{'sms_sender':caller}})
    monkeypatch.setattr(reply.requests,'post',lambda *args,**kwargs:pytest.fail('unsafe caller reached SMS transport'))
    outcome = reply.send_reply('approved')
    assert ('no SMS attempted' in outcome if caller else 'no inbound message' in outcome)
