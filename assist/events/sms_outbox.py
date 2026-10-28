"""Exact pending phone SMS drafts; only the phone's confirmation may send."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading

from langchain.tools import ToolRuntime
from langgraph.config import get_config

from assist.events.sms_archive import configured_archive
from assist.record_store import PerThreadJsonStore

_STATES = {'pending', 'approved', 'rejected', 'sent', 'not_sent', 'unknown'}
_LOCK = threading.Lock()


@dataclass(frozen=True)
class SmsDraft:
    id: str
    thread_id: str
    number: str
    text: str
    created_at: str
    state: str = 'pending'

    def to_dict(self):
        return asdict(self)


class SmsOutbox(PerThreadJsonStore[SmsDraft]):
    FILENAME = 'sms-outbox.json'

    def __init__(self, root_dir):
        super().__init__(root_dir)
        self._lock = _LOCK

    @staticmethod
    def current(record, archive=None):
        """Project a previously attempted pending draft; terminal outcomes stay final."""
        if record.state in {'sent', 'not_sent', 'unknown', 'rejected'}:
            return record
        observed = (archive or configured_archive()).read(record.id)
        if observed is not None:
            if (observed['direction'], observed['number'], observed['text']) != (
                    'outgoing', record.number, record.text):
                raise ValueError('SMS receipt does not match the draft')
            return replace(record, state=observed['status'])
        return record

    @staticmethod
    def _from_dict(value):
        record = SmsDraft(**value)
        if record.state not in _STATES:
            raise ValueError('Invalid SMS draft state')
        return record

    def _write(self, tid, recs):
        path = self._path(tid)
        with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(path), delete=False) as stream:
            temporary = stream.name
            try:
                json.dump([record.to_dict() for record in recs], stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
                descriptor = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def stage(self, tid, operation, number, text):
        if not isinstance(operation, str) or not operation:
            raise ValueError('No durable SMS operation identity')
        number = re.sub(r'[ ()-]', '', number)
        if not re.fullmatch(r'\+?[0-9]{5,15}', number):
            raise ValueError('Use a grounded concrete phone number')
        if not isinstance(text, str) or not text or '\0' in text or len(text.encode()) > 4096:
            raise ValueError('Invalid exact SMS text')
        identity = 'outbox_' + hashlib.sha256(f'{tid}:{operation}'.encode()).hexdigest()
        with self._lock:
            records = self._read(tid)
            old = next((record for record in records if record.id == identity), None)
            if old:
                if (old.number, old.text) != (number, text):
                    raise ValueError('SMS operation content changed')
                observed = self.current(old)
                if observed != old:
                    self._write(tid, [observed if record.id == old.id else record for record in records])
                return observed
            archive = configured_archive()
            records = [self.current(record, archive)
                       if record.state in {'pending', 'approved'} else record
                       for record in records]
            if sum(record.state in {'pending', 'approved'} for record in records) >= 10:
                raise ValueError('Finish pending phone SMS drafts first')
            record = SmsDraft(identity, tid, number, text, datetime.now(timezone.utc).isoformat())
            self._write(tid, [*records, record])
            return record

    def decide(self, tid, identity, state):
        if state not in _STATES - {'pending'}:
            raise ValueError('Invalid SMS decision')

        def apply(record):
            if state in {'approved', 'rejected'}:
                record = self.current(record)
            if state == 'approved' and record.state in {'sent', 'not_sent', 'unknown'}:
                return record
            allowed = {
                'pending': {'approved', 'rejected'},
                'approved': {'sent', 'not_sent', 'unknown', 'rejected'},
            }.get(record.state, set())
            if state != record.state and state not in allowed:
                raise ValueError('SMS decision no longer applicable')
            return replace(record, state=state)

        return self.update(tid, identity, apply)


def sms_sending_tools(store: SmsOutbox):
    def send_sms(number: str, text: str, runtime: ToolRuntime) -> str:
        """Send a text after the user approves its exact recipient and text on their phone.

        Use a number supplied by the user or found in trusted contact files; ask
        if it is missing or ambiguous. Preserve the exact requested text. This
        stages a draft and returns before approval; it never sends directly.
        """
        tid = ((get_config() or {}).get('configurable') or {}).get('thread_id')
        if not tid:
            return 'SMS not sent: no active ordinary thread.'
        try:
            record = store.stage(tid, runtime.tool_call_id, number, text)
        except ValueError as error:
            return (f'SMS draft status unavailable: {error}. '
                    'Do not automatically retry under a new identity; await user direction.')
        except (OSError, sqlite3.Error):
            return ('SMS draft status unavailable: private storage is unavailable. '
                    'Do not automatically retry under a new identity; await user direction.')
        if record.state == 'pending':
            return (
                'Proposed to the user; if approved on the phone it will be sent. '
                'Open this thread on the phone and choose Review. Nothing sent. '
                f'Draft {record.id}.'
            )
        return (f'SMS draft {record.id}: {record.state}. '
                'No automatic retry. A new explicit user request needs a new phone-approved attempt.')

    return [send_sms]
