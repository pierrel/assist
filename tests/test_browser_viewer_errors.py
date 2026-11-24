"""Actual viewer DOM coverage; opt in with an owned local Chromium executable."""
import json
import os

import pytest

from manage.web.browser_viewer import VIEWER


@pytest.mark.skipif(not os.getenv('ASSIST_BROWSER_HANDOFF_CHROMIUM'), reason='owned Chromium opt-in')
def test_frame_error_stays_visible_during_status_poll_until_capture_recovers():
    from playwright.sync_api import sync_playwright, expect
    error = 'Synthetic frame capture refused'
    recovering = False
    jpeg = None
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ['ASSIST_BROWSER_HANDOFF_CHROMIUM'],
                                             headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page()
            def route_request(route):
                request = route.request
                if request.url == 'http://viewer.example/':
                    route.fulfill(content_type='text/html', body=VIEWER.replace('__CONFIG__',
                        json.dumps({'url':'/control','token':'a'*32})))
                    return
                action = request.post_data_json['action']
                if action == 'frame':
                    if recovering:
                        route.fulfill(content_type='image/jpeg', body=jpeg,
                                      headers={'X-Page-Width':'390','X-Page-Height':'600'})
                    else:
                        route.fulfill(status=409, json={'error':error})
                else:
                    route.fulfill(json={'active':True,'owner':True,'requested':True,
                        'pages':[{'page_id':'page','url':'https://fixture.example/'}]})
            page.route('**/*', route_request)
            page.goto('http://viewer.example/')
            expect(page.get_by_text(error, exact=False)).to_be_visible()
            page.evaluate("""error => {
                window.lostFrameError = false;
                window.frameObserver = new MutationObserver(() => {
                    if (!document.body.innerText.includes(error)) window.lostFrameError = true;
                });
                frameObserver.observe(document.body, {subtree:true, childList:true, characterData:true});
            }""", error)
            page.evaluate('update()')
            assert not page.evaluate('lostFrameError'), 'status polling erased the capture failure'
            page.evaluate('frameObserver.disconnect()')
            jpeg = page.screenshot(type='jpeg')
            recovering = True
            page.evaluate('update()')
            expect(page.get_by_text(error, exact=False)).not_to_be_visible()
            expect(page.locator('#screen')).to_have_attribute('data-page','page')
        finally:
            browser.close()
