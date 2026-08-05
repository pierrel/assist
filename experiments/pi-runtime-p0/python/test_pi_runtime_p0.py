import pytest
import pi_runtime_p0 as p0
def request(body: bytes) -> bytes:
    headers = [
        "POST /v1/chat/completions HTTP/1.1", "host: 127.0.0.1:1234",
        "connection: keep-alive", "accept: application/json", "user-agent: OpenAI/JS 6.26.0",
        "x-stainless-retry-count: 0", "x-stainless-timeout: 300", "x-stainless-lang: js",
        "x-stainless-package-version: 6.26.0", "x-stainless-os: Linux", "x-stainless-arch: x64",
        "x-stainless-runtime: node",
        f"x-stainless-runtime-version: {p0.NODE_VERSION}",
        "authorization: Bearer local", "content-type: application/json", "accept-language: *",
        "sec-fetch-mode: cors", "accept-encoding: gzip, deflate", f"content-length: {len(body)}",
    ]
    return "\r\n".join(headers).encode() + b"\r\n\r\n" + body
@pytest.mark.parametrize("body", [b'{"model":"fixture-model","model":"other"}', b'{"value":NaN}', b'\xff'])

def test_strict_json_rejects_ambiguous_input(body: bytes) -> None:
    with pytest.raises(p0.P0Error):
        p0.strict_json(body)

def test_provider_request_rejects_nested_or_typed_drift() -> None:
    body = p0.canonical(p0.expected_payload())
    variants = [body.replace(b'"max_tokens":256', b'"max_tokens":"256"'),
                body.replace(b'"include_usage":true', b'"extra":1,"include_usage":true'),
                body.replace(b'"role":"user"', b'"extra":1,"role":"user"')]
    for variant in variants:
        with pytest.raises(p0.P0Error, match="payload changed"):
            p0.parse_provider_request(request(variant))
