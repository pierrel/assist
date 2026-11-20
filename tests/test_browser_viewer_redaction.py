"""Human frames mask passwords; text retains the existing private-value boundary."""
import json
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from assist.browser import manager, runner
from assist.browser.runner import BrowserInputError, BrowserWorker
from tests.test_browser_handoff import admitted, controlled
from tests.test_browser_runner import _Page


def test_viewer_envelope_uses_same_unlocked_catalog(controlled, monkeypatch):
    *_, session, _stopped = controlled
    store = manager._secret_store(session)
    store.unlock('synthetic phrase', create=True)
    store.put('Fixture', 'https://example.com/', 'fixture-user', 'fixture-secret')
    calls = []
    def transport(_generation, payload, **kwargs):
        calls.append(json.loads(payload))
        return b'{"result":{"pages":[]}}'
    monkeypatch.setattr(manager, '_docker_exec', transport)
    session.viewer_command(session.viewer_token, session.human_client, 'tabs')
    assert set(calls[0]['private_values']) == {'fixture-user', 'fixture-secret'}
    assert calls[0]['operation'] == 'viewer'


def test_viewer_unavailable_catalog_cannot_capture(controlled, monkeypatch):
    *_, session, _stopped = controlled
    monkeypatch.setattr(manager, '_private_values', lambda _store: None)
    monkeypatch.setattr(manager, '_docker_exec', lambda *_a, **_k: pytest.fail('must not capture'))
    with pytest.raises(manager.BrowserUnavailable, match='unlock or repair'):
        session.viewer_command(session.viewer_token, session.human_client, 'frame', page_id='page')


def test_viewer_tabs_masks_current_and_saved_values_before_clipping(monkeypatch):
    monkeypatch.setenv('BROWSER_SESSION_TOKEN', 'fixture-token')
    worker = BrowserWorker()
    page = _Page()
    saved, current = 'fixture-saved /value', 'fixture-current'
    page.url = 'https://example.com/?pad=' + 'x' * 225 + quote(saved, safe='') + current
    worker.pages['opaque-page'] = page
    worker.web_pages.add(page)
    worker._password_values = lambda: [current]
    result = worker.handle({'session':'fixture-token', 'operation':'viewer',
                            'private_values':[saved], 'args':{'action':'tabs'}})
    visible = json.dumps(result)
    assert quote(saved, safe='') not in visible and current not in visible
    assert '[password hidden]' in visible
    assert result['pages'][0]['page_id'] == 'opaque-page'


@pytest.mark.parametrize('stage', ['collection', 'selection', 'capture'])
def test_viewer_frame_failure_never_falls_back_to_raw(monkeypatch, stage):
    monkeypatch.setenv('BROWSER_SESSION_TOKEN', 'fixture-token')
    worker = BrowserWorker()
    page = _Page()
    page.viewport_size = {'width':640, 'height':480}
    worker.pages['page'] = page
    calls = []
    def current():
        if stage == 'collection':
            raise BrowserInputError('password fields unavailable')
        return ['fixture-current']
    def masks(_page):
        assert worker._hidden_values() == ('fixture-current', 'fixture-saved')
        if stage == 'selection':
            raise RuntimeError('fixture selection failed')
        return ['native-mask-locator']
    def capture(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('fixture capture failed')
    worker._password_values = current
    worker._screenshot_masks = masks
    page.screenshot = capture
    with pytest.raises(BrowserInputError, match='password fields unavailable|viewer frame redaction unavailable'):
        worker.handle({'session':'fixture-token', 'operation':'viewer',
                       'private_values':['fixture-saved'], 'args':{'action':'frame', 'page_id':'page'}})
    assert len(calls) == (1 if stage == 'capture' else 0)
    assert all(call['mask'] == ['native-mask-locator'] for call in calls)
    assert worker._private_values == worker._current_before == ()


def test_human_frames_use_native_password_masks_without_collecting_dom():
    masks = [object(), object()]
    def frame(mask):
        def locator(selector):
            assert selector == 'input[type="password"]'
            return mask
        return SimpleNamespace(locator=locator)
    page = SimpleNamespace(frames=[frame(mask) for mask in masks])
    assert BrowserWorker()._screenshot_masks(page) == masks


@pytest.mark.parametrize('private_values,current', [
    ([], []), (['fixture-saved'], []), ([], ['fixture-current'])])
def test_human_frame_keeps_native_masks_with_saved_or_current_values(monkeypatch, private_values, current):
    monkeypatch.setenv('BROWSER_SESSION_TOKEN', 'fixture-token')
    worker = BrowserWorker()
    page = _Page()
    page.viewport_size = {'width':640, 'height':480}
    mask = object()
    page.frames = [SimpleNamespace(locator=lambda selector: mask
        if selector == 'input[type="password"]' else pytest.fail('must not collect DOM'))]
    worker.pages['page'] = page
    worker._password_values = lambda: current
    def capture(**kwargs):
        assert kwargs['mask'] == [mask]
        assert kwargs['mask_color'] == '#ff00ff'
        return b'synthetic-jpeg'
    page.screenshot = capture
    result = worker.handle({'session':'fixture-token', 'operation':'viewer',
                            'private_values':private_values,
                            'args':{'action':'frame', 'page_id':'page'}})
    assert result['image']
    assert worker._private_values == worker._current_before == ()
