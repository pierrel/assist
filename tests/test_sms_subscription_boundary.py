"""Synthetic real SQLite/Run admission proves subscription arrival cutoffs."""
import pytest
from assist.events.model import Subscription
from assist.events.store import SubscriptionStore
from assist.events.sms_archive import SmsArchive
from assist.run_service import RunService
from manage.web import threads

@pytest.fixture
def system(tmp_path, monkeypatch):
    root = tmp_path / 'threads'; (root / 'synthetic-sms-thread').mkdir(parents=True)
    archive = SmsArchive(tmp_path / 'archive', thread_root=root)
    runs = RunService(str(root))
    subscriptions = SubscriptionStore(str(root))
    subscriptions.add(Subscription('synthetic-sub', 'synthetic-sms-thread', '.*',
                                   'Handle message from {sender}: {text}',
                                   '2026-10-26T10:00:00+00:00'))
    monkeypatch.setattr(threads, 'SUBSCRIPTION_STORE', subscriptions)
    monkeypatch.setattr(threads, '_sms_archive', lambda: archive)
    monkeypatch.setattr(threads, '_runs', lambda: runs)
    monkeypatch.setattr(threads, '_require_deep_thread', lambda tid: None)
    return archive, runs

def admit(identity, arrived):
    return threads._admit_inbound_sms(threads._InboundSms(
        message_id=identity, sender='+15555550100', text='same independent text', occurred_at=arrived))

def test_predating_subscription_is_archived_without_run(system):
    archive, runs = system
    response, run = admit('old-message', '2026-10-26T09:00:00+00:00')
    assert archive.read('old-message') is not None
    assert run is None
    assert runs.for_thread('synthetic-sms-thread') == []

def test_later_message_and_same_receipt_retry_admit_one_run(system):
    archive, runs = system
    _, run = admit('later-message', '2026-10-26T10:01:00+00:00')
    assert run is not None
    response, duplicate = admit('later-message', '2026-10-26T10:01:00+00:00')
    assert response['status'] == 'duplicate' and duplicate is None
    assert len(runs.for_thread('synthetic-sms-thread')) == 1



def test_bootstrap_archive_only_cannot_admit_even_after_subscription(system):
    archive, runs = system
    response, run = threads._admit_inbound_sms(threads._InboundSms(
        message_id='bootstrap', sender='+15555550100', text='old history',
        occurred_at='2026-10-26T11:00:00+00:00', archive_only=True))
    assert archive.read('bootstrap')['handling'] == 'archive_only'
    assert run is None and runs.for_thread('synthetic-sms-thread') == []


def test_missing_arrival_archives_without_run(system):
    archive, runs = system
    _, run = admit('unknown-arrival', None)
    assert archive.read('unknown-arrival')['handling'] == 'archive_only'
    assert run is None


def test_newer_subscription_does_not_steal_old_arrival(system, monkeypatch):
    store = threads.SUBSCRIPTION_STORE
    store.add(Subscription('older', 'synthetic-sms-thread', '.*', 'older {text}',
                           '2026-10-26T08:00:00+00:00'))
    assert store.route('+15555550100', occurred_at='2026-10-26T09:00:00+00:00').id == 'older'
    assert store.route('+15555550100', occurred_at='2026-10-26T08:00:00+00:00') is None
