"""Isolated synthetic destination checks; no production endpoints or tokens."""
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import pytest
import requests

from assist import control_api, url_fetch, tools, pdf_text


@pytest.fixture
def dummy_services(monkeypatch):
    hits = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.server.server_port, self.path))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", control + "/control")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Length", "12")
                self.end_headers()
                self.wfile.write(b"dummy answer")
        def log_message(self, *_args):
            pass
    protected = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    normal = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    control = f"http://127.0.0.1:{protected.server_port}"
    allowed = f"http://127.0.0.1:{normal.server_port}"
    monkeypatch.setenv("ASSIST_PORT", str(protected.server_port))
    monkeypatch.delenv("ASSIST_BASE_URL", raising=False)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (normal, protected)]
    for t in threads:
        t.start()
    try:
        yield control, allowed, hits
    finally:
        for s in (normal, protected):
            s.shutdown()
            s.server_close()
        for t in threads:
            t.join(2)


def test_host_html_pdf_redirects_and_human_access(dummy_services, monkeypatch):
    control, allowed, hits = dummy_services
    # An environment proxy cannot replace the checked destination transport.
    monkeypatch.setenv("HTTP_PROXY", control)
    with url_fetch.get(allowed + "/normal", timeout=2) as response:
        assert response.text == "dummy answer"
    for target in (control + "/control", allowed + "/redirect"):
        with pytest.raises(control_api.ControlApiDenied):
            url_fetch.get(target, timeout=2)
    tools._host_last_call.clear()
    assert tools.read_url(control + "/control").startswith("Error fetching URL:")
    with pytest.raises(control_api.ControlApiDenied):
        pdf_text._fetch_pdf(allowed + "/redirect")
    assert all(path != "/control" for _, path in hits)
    # Human HTTP traffic uses the existing unrestricted client.
    with requests.Session() as human:
        human.trust_env = False
        assert human.get(control + "/control", timeout=2).status_code == 200
    assert hits[-1][1] == "/control"


def test_resolver_denies_alias_and_mixed_answers_before_dial(monkeypatch):
    monkeypatch.setenv("ASSIST_PORT", "5050")
    monkeypatch.delenv("ASSIST_BASE_URL", raising=False)
    answers = ["93.184.216.34", "10.1.2.3"]
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in answers])
    with pytest.raises(control_api.ControlApiDenied):
        control_api.resolve_target("alias.example.test", 5050)
    assert control_api.resolve_target("docs.example.test", 8484) == "93.184.216.34"
    monkeypatch.setenv("ASSIST_BASE_URL", "https://control.example.test")
    with pytest.raises(control_api.ControlApiDenied):
        control_api.resolve_target("alternate.example.test", 443)


def test_http_and_tls_pin_dial_address_keep_hostname(monkeypatch):
    monkeypatch.setattr(url_fetch, "resolve_target", lambda host, port: "93.184.216.34")
    dial = Mock(return_value=object())
    monkeypatch.setattr(url_fetch, "create_connection", dial)
    for cls in (url_fetch._HttpConnection, url_fetch._HttpsConnection):
        connection = cls("page.example.test", 443, timeout=3)
        connection._new_conn()
        assert connection.host == "page.example.test"
        assert dial.call_args.args[:2] == (("93.184.216.34", 443), 3)


def test_configured_public_origin_and_failed_inventory_deny(monkeypatch):
    monkeypatch.setenv("ASSIST_PORT", "5050")
    monkeypatch.setenv("ASSIST_BASE_URL", "https://control.example.test")
    def fail(*_args, **_kwargs):
        raise socket.gaierror("synthetic inventory unavailable")
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(control_api.ControlApiDenied):
        control_api.resolve_target("CONTROL.EXAMPLE.TEST.", 443)
    with pytest.raises(OSError):
        control_api.resolve_target("other.example.test", 443)


def test_docs_and_model_ports_survive_local_control_deny(monkeypatch):
    monkeypatch.setenv("ASSIST_PORT", "5050")
    monkeypatch.delenv("ASSIST_BASE_URL", raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", port))])
    for port in (8484, 9001, 8000):
        assert control_api.resolve_target("private.example.test", port) == "10.1.2.3"
    with pytest.raises(control_api.ControlApiDenied):
        control_api.resolve_target("private.example.test", 5050)


@pytest.mark.parametrize('alias', ['::ffff:93.184.216.34', '::ffff:5db8:d822'])
def test_published_ipv4_origin_denies_mapped_ipv6_alias(monkeypatch, alias):
    monkeypatch.setenv('ASSIST_PORT', '5050')
    monkeypatch.setenv('ASSIST_BASE_URL', 'https://control.example.test')
    monkeypatch.setattr(socket, 'getaddrinfo', lambda host, port, **kwargs: [
        (socket.AF_INET if host == 'control.example.test' else socket.AF_INET6,
         socket.SOCK_STREAM, 6, '',
         ('93.184.216.34' if host == 'control.example.test' else alias, port))])
    with pytest.raises(control_api.ControlApiDenied):
        control_api.resolve_target('alternate.example.test', 443)
