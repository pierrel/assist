"""Operator-only Desktop OAuth enrollment. Never run this through the agent."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from assist.gmail import GMAIL_SCOPE, GmailError, private_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Connect Gmail for search, read, archive and Trash.")
    parser.add_argument("--client-file", required=True, help="0600 Google Desktop OAuth client JSON")
    parser.add_argument("--token-file", default=os.getenv("ASSIST_GMAIL_TOKEN_FILE"),
                        help="New 0600 token file outside Assist's thread directory")
    parser.add_argument("--port", type=int, default=8765, help="Loopback callback/SSH-forward port")
    parser.add_argument("--no-browser", action="store_true", help="Print consent URL for another browser")
    args = parser.parse_args(argv)
    if not args.token_file:
        parser.error("set ASSIST_GMAIL_TOKEN_FILE or pass --token-file")
    target = Path(args.token_file).expanduser().absolute()
    root = Path(os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads")).resolve()
    if target.resolve().is_relative_to(root) or target.exists() or target.is_symlink():
        parser.error("token file must be new and outside the thread directory")
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        parser.error('install the operator setup extra: pip install -e ".[gmail-setup]"')
    try:
        client = private_json(args.client_file)
        desktop = client.get("installed", {})
        # Never let a supplied client file change the credential destinations.
        if (desktop.get("auth_uri") != "https://accounts.google.com/o/oauth2/auth"
                or desktop.get("token_uri") != "https://oauth2.googleapis.com/token"):
            parser.error("use an unmodified Google Desktop OAuth client JSON")
        flow = InstalledAppFlow.from_client_config(client, scopes=[GMAIL_SCOPE])
        credentials = flow.run_local_server(
            host="127.0.0.1", port=args.port, open_browser=not args.no_browser,
            timeout_seconds=300, access_type="offline", prompt="consent",
            authorization_prompt_message="Open this Google consent link:\n{url}",
            success_message="Gmail connected. You can close this tab.")
        if not credentials.refresh_token:
            parser.error("Google returned no refresh token; repeat enrollment with consent")
        value = json.loads(credentials.to_json())
        value["scopes"] = [GMAIL_SCOPE]
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
        print("Gmail connected. Set ASSIST_GMAIL_TOKEN_FILE to this private token file.")
        print("Google external apps in Testing expire refresh tokens after 7 days.")
    except GmailError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
