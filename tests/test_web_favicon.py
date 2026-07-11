"""Favicon routes + head links. The SVG covers browsers that render SVG favicons; the
PNG (/apple-touch-icon.png) covers iOS Safari, which does NOT — so the page heads must
carry the PNG <link> tags or Safari mobile shows no icon."""
from fastapi.testclient import TestClient

from manage import web

_client = TestClient(web.app)


def test_favicon_served_as_svg():
    resp = _client.get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert resp.text.startswith("<svg") and "</svg>" in resp.text


def test_apple_touch_icon_served_as_png():
    # iOS Safari needs a raster icon; assert a real PNG (magic bytes), not SVG.
    resp = _client.get("/apple-touch-icon.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_index_head_links_png_icon_for_safari():
    # Without these <link> tags in the head, iOS Safari can't find a renderable icon.
    html = _client.get("/").text
    assert 'rel="apple-touch-icon" href="/apple-touch-icon.png"' in html
    assert 'type="image/png"' in html and "/apple-touch-icon.png" in html
