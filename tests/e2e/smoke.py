#!/usr/bin/env python3
"""Exercise the standalone fixture through real HTTPS, without Emacs."""
import argparse
import json
from pathlib import Path
import ssl
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    credentials = fixture['credentials']
    token = Path(credentials['token']).read_text().strip()
    context = ssl.create_default_context(cafile=credentials['ca.crt'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                        urllib.request.HTTPSHandler(context=context))
    def request(method, path, body=None, key=None):
        headers = {'Authorization': 'Bearer ' + token}
        if key:
            headers['Idempotency-Key'] = key
        data = None if body is None else json.dumps(body).encode()
        if data:
            headers['Content-Type'] = 'application/json'
        return opener.open(urllib.request.Request(fixture['url'] + path,
            data=data, headers=headers, method=method), timeout=10)
    def api(method, path, body=None, key=None):
        with request(method, path, body, key) as response:
            return json.load(response)
    catalog = api('GET', '/api/v1/phone/threads')
    body = {'message': 'Write http-smoke.txt direct-http',
            'repo_key': catalog['repositories'][0]['repo_key'], 'harness': 'deepagents'}
    key = 'http-smoke-' + uuid.uuid4().hex
    accepted = api('POST', '/api/v1/phone/threads', body, key)
    replay = api('POST', '/api/v1/phone/threads', body, key)
    if (accepted['thread_id'], accepted['run_id']) != (replay['thread_id'], replay['run_id']):
        raise RuntimeError('Idempotent create changed the durable Run identity')
    if accepted.get('replayed') is not False or replay.get('replayed') is not True:
        raise RuntimeError('Idempotent create returned incorrect replay flags')
    path = '/api/v1/phone/threads/' + accepted['thread_id']
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        run = api('GET', path + '/runs/' + accepted['run_id'])
        if run['status'] == 'success':
            break
        if run['status'] in {'error', 'cancelled', 'interrupted'}:
            raise RuntimeError('HTTP Run failed: ' + json.dumps(run))
        time.sleep(.2)
    else:
        raise TimeoutError('HTTP Run did not complete within 90 seconds')
    snapshot = api('GET', path)
    if not any('Saved http-smoke.txt' in item['text'] for item in snapshot['messages']):
        raise RuntimeError('Persisted canonical transcript lacks deterministic result')
    with request('GET', path + '/runs/' + accepted['run_id'] + '/events') as response:
        events = response.read(1024 * 1024).decode()
    if 'event: terminal' not in events or 'success' not in events:
        raise RuntimeError('SSE did not project the durable terminal result')
    receipt = {'thread_id': accepted['thread_id'], 'run_id': accepted['run_id'],
               'status': run['status'], 'replayed': replay['replayed'],
               'https': True, 'persisted_transcript': True, 'terminal_sse': True}
    (args.fixture.parent / 'http-smoke.json').write_text(json.dumps(receipt, indent=2))
    print('PASS direct HTTPS create/replay, real Docker publication, transcript and terminal SSE')


if __name__ == '__main__':
    main()
