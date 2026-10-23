"""Opaque review bindings for existing non-mail approval services.

The configured client credential keeps tokens stable across process restarts.
Web-only installations use a process key and require fresh previews on restart.
Tokens carry no decision, grant, or execution state.
"""
import hashlib
import hmac
import json
import secrets
import os


_SECRET = secrets.token_bytes(32)


def review_token(kind: str, values: object) -> str:
    payload = json.dumps([kind, values], sort_keys=True, separators=(",", ":"))
    secret = os.environ.get("ASSIST_PHONE_API_TOKEN")
    return hmac.new(secret.encode() if secret else _SECRET, payload.encode(), hashlib.sha256).hexdigest()
