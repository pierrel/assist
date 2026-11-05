"""Durable admission receipts for phone-observed incoming-call outcomes."""
from contextlib import closing
from pathlib import Path
import os
import sqlite3
import stat
import uuid


def admit_call(root, event, subscriptions, create_run):
    """Under the caller's admission lock, fix handling once before acknowledging.

    Caller data remains untrusted triage input. No carrier voicemail or reason
    interpretation is inferred from a modem termination event.
    """
    directory = Path(root) / '.thread-call-events'
    if directory.is_symlink():
        raise ValueError('unsafe call receipt directory')
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('unsafe call receipt directory')
    path = directory / 'receipts.sqlite3'
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    info = os.fstat(descriptor)
    os.close(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('unsafe call receipt database')
    with closing(sqlite3.connect(path, timeout=2)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, number TEXT, outcome TEXT, reason INTEGER, tid TEXT, prompt TEXT, run_id TEXT, admitted INTEGER DEFAULT 0)')
        row = connection.execute('SELECT * FROM calls WHERE id=?', (event['event_id'],)).fetchone()
        if row is None:
            subscription = subscriptions.route(event['number'] or '', 'call')
            reason = event['reason'] if event['reason'] is not None else "unavailable"
            description = f"Observed incoming call: {event['outcome']}; ModemManager termination reason code {reason}. No voicemail transcript is available from this event."
            tid = subscription.thread_id if subscription else None
            prompt = subscription.render(event['number'] or '(unknown caller)', description) if subscription else None
            connection.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,0)', (event['event_id'], event['number'], event['outcome'], event['reason'], tid, prompt, uuid.uuid4().hex if subscription else None))
            connection.commit()
            row = connection.execute('SELECT * FROM calls WHERE id=?', (event['event_id'],)).fetchone()
        elif (row['number'], row['outcome'], row['reason']) != (event['number'], event['outcome'], event['reason']):
            raise ValueError('call event conflicts with its immutable observation')
        if row['tid'] is None or row['admitted']:
            return None
        run = create_run(row['tid'], row['prompt'], sender=row['number'] or '(unknown caller)', run_id=row['run_id'], dispatch_key='call:' + row['id'])
        connection.execute('UPDATE calls SET admitted=1 WHERE id=?', (row['id'],))
        connection.commit()
        return run
