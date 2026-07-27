"""Human-approved outbound email for the web agent."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from email.headerregistry import Address
import requests
from langchain.tools import ToolRuntime

logger = logging.getLogger(__name__)

_RESEND_EMAILS_URL = "https://api.resend.com/emails"
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_SUBJECT_BYTES = 998
_MAX_BODY_BYTES = 64 * 1024

EMAIL_INTERRUPT_ON = {
    "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
}


@dataclass(frozen=True)
class _EmailConfig:
    sender: str
    sender_name: str
    oversight_cc: str
    api_key: str


def _bare_address(value: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n"):
        raise ValueError("address is invalid")
    try:
        address = Address(addr_spec=value)
    except (TypeError, ValueError) as error:
        raise ValueError("address is invalid") from error
    if str(address) != value:
        raise ValueError("address must not include a display name")
    return value


def _private_api_key(path: str) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("email delivery is not configured") from error
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600):
            raise ValueError("email delivery is not configured")
        raw_key = os.read(descriptor, 4097)
        if len(raw_key) > 4096:
            raise ValueError("email delivery is not configured")
        key = raw_key.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("email delivery is not configured") from error
    finally:
        os.close(descriptor)
    if not re.fullmatch(r"re_[A-Za-z0-9_]{16,200}", key):
        raise ValueError("email delivery is not configured")
    return key


def _config() -> _EmailConfig:
    sender = _bare_address(os.getenv("EMAIL_FROM_ADDRESS", ""))
    oversight_cc = _bare_address(os.getenv("EMAIL_ALWAYS_CC", ""))
    sender_name = os.getenv("EMAIL_FROM_NAME", "").strip()
    if not sender_name or any(char in sender_name for char in "\r\n"):
        raise ValueError("email delivery is not configured")
    key_path = os.getenv("EMAIL_RESEND_API_KEY_FILE", "")
    if not key_path:
        raise ValueError("email delivery is not configured")
    return _EmailConfig(sender, sender_name, oversight_cc, _private_api_key(key_path))


def email_identity() -> tuple[str, str] | None:
    """Return the configured sender and fixed CC for an approval card."""
    try:
        sender = _bare_address(os.getenv("EMAIL_FROM_ADDRESS", ""))
        oversight_cc = _bare_address(os.getenv("EMAIL_ALWAYS_CC", ""))
        sender_name = os.getenv("EMAIL_FROM_NAME", "").strip()
        if not sender_name or any(char in sender_name for char in "\r\n"):
            raise ValueError
    except ValueError:
        return None
    return str(Address(display_name=sender_name, addr_spec=sender)), oversight_cc


def _response_body(response) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(8192):
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            raise ValueError("provider response is unexpectedly large")
        chunks.append(chunk)
    return b"".join(chunks)


def valid_email_content(to: str, subject: str, body: str) -> bool:
    """Check the user-editable fields before resuming a pending email action."""
    try:
        _bare_address(to)
        if (not isinstance(subject, str) or not subject.strip()
                or len(subject.encode("utf-8")) > _MAX_SUBJECT_BYTES
                or any(char in subject for char in "\r\n") or not isinstance(body, str)
                or len(body.encode("utf-8")) > _MAX_BODY_BYTES):
            raise ValueError
    except ValueError:
        return False
    return True


def send_email(to: str, subject: str, body: str, runtime: ToolRuntime) -> str:
    """Send one plain-text email after the user approves its exact contents.

    Give one recipient address, a subject, and the complete plain-text body. The sender
    and oversight CC are fixed by the web service and cannot be changed here.
    """
    try:
        if not valid_email_content(to, subject, body):
            raise ValueError("body is invalid")
        recipient = _bare_address(to)
        config = _config()
    except ValueError:
        return "Email not sent: email delivery is not configured or the message is invalid."
    if not runtime.tool_call_id:
        return "Email not sent: this email has no durable delivery identifier."

    payload = {
        "from": str(Address(display_name=config.sender_name, addr_spec=config.sender)),
        "to": [recipient],
        "cc": [config.oversight_cc],
        "subject": subject,
        "text": body,
    }
    idempotency_key = "assist-email/" + hashlib.sha256(
        runtime.tool_call_id.encode()).hexdigest()
    response = None
    try:
        response = requests.post(
            _RESEND_EMAILS_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Idempotency-Key": idempotency_key,
            },
            timeout=10,
            stream=True,
        )
        if response.status_code < 200 or response.status_code >= 300:
            logger.warning("email provider rejected delivery with HTTP %s", response.status_code)
            return "Email not sent: the delivery provider rejected it."
        result = json.loads(_response_body(response))
        if not isinstance(result, dict) or not isinstance(result.get("id"), str) or not result["id"]:
            return "Email not sent: the delivery provider returned an invalid response."
    except requests.RequestException:
        logger.warning("email delivery connection failed", exc_info=True)
        return "Email delivery status is unknown: the provider connection failed."
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        logger.warning("email provider returned an invalid response", exc_info=True)
        return "Email not sent: the delivery provider returned an invalid response."
    finally:
        if response is not None:
            response.close()
    return f"Email sent to {recipient}."


def email_tools() -> list:
    """Return the normal-web-only email tool."""
    return [send_email]
