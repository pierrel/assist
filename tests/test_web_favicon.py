"""The /favicon.ico route — every page gets a tab icon from one inline-SVG route."""
from fastapi.testclient import TestClient

from manage import web


def test_favicon_served_as_svg():
    resp = TestClient(web.app).get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert resp.text.startswith("<svg")
    assert "</svg>" in resp.text
