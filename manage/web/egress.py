"""The /egress management view + the approve/decline/revoke actions.

Pending approval lives IN-THREAD (the card in the thread render — the geo
lesson: approval lands where the user is); this page is the kill switch —
active grants with their remaining lifetime + Revoke (a live exfil-capable
grant needs one; TTL alone leaves a regret window). The POST routes are
shared by the in-thread card (a ``redirect`` form field returns the user to
the thread). CSRF: the in-memory ``EGRESS_CSRF`` token embedded in every
form + required on the POST (the geo T1 pattern). Every handler runs its
whole chain inside threadpool callables — nothing store-locked touches the
single-worker asyncio loop. Absent the store (``ASSIST_EGRESS_APPROVALS_DIR``
unset) the view 404s.
"""
from __future__ import annotations

import html
import os
import secrets
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from assist.egress.store import REVOKED_ONLY, request_key
from manage.web.app import app
from manage.web.state import EGRESS_CSRF, EGRESS_STORE, MANAGER
from manage.web.threads import (_FAVICON_LINKS, _PULL_TO_REFRESH_SCRIPT,
                                _dispatch_egress_resolution)


def _require_egress():
    if EGRESS_STORE is None:
        raise HTTPException(status_code=404, detail="egress approvals are not configured")


def _check_csrf(token: str | None):
    if not token or not secrets.compare_digest(token, EGRESS_CSRF):
        raise HTTPException(status_code=403, detail="bad or missing form token")


def _next(form) -> str:
    """Same-site redirect target only (the geo `_next` discipline): a single
    leading '/', no '//' or backslash, no control chars — else /egress."""
    dest = form.get("redirect") or ""
    ok = (dest.startswith("/") and not dest.startswith("//") and "\\" not in dest
          and not any(ord(c) < 0x20 for c in dest))
    return dest if ok else "/egress"


def _fmt_left(rec) -> str:
    if rec.expires_at == REVOKED_ONLY:
        return "until revoked or the thread is deleted"
    try:
        mins = max(0, int((datetime.fromisoformat(rec.expires_at)
                           - datetime.now(timezone.utc)).total_seconds() // 60))
        return f"~{mins} min left"
    except Exception:
        return "expiring"


def _render(grants) -> str:
    rows = ""
    for r in sorted(grants, key=lambda r: (r.origin_tid, r.host, r.port)):
        tid = html.escape(r.origin_tid)
        rows += (
            f"<tr><td><code>{html.escape(r.host)}:{r.port}</code></td>"
            f"<td><a href='/thread/{tid}'>{tid}</a></td>"
            f"<td>{html.escape(_fmt_left(r))}</td>"
            f"<td><form method='post' action='/egress/revoke' style='display:inline'>"
            f"<input type='hidden' name='token' value='{EGRESS_CSRF}'>"
            f"<input type='hidden' name='tid' value='{tid}'>"
            f"<input type='hidden' name='host' value='{html.escape(r.host)}'>"
            f"<input type='hidden' name='port' value='{r.port}'>"
            f"<button class='btn' type='submit'>Revoke</button></form></td></tr>")
    tbl = (f"<table><thead><tr><th>Host</th><th>Thread</th><th>Lifetime</th><th></th>"
           f"</tr></thead><tbody>{rows}</tbody></table>" if rows
           else "<p>No active grants. Pending requests are approved in their thread.</p>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Egress grants</title>"
        f"{_FAVICON_LINKS}"
        "<style>body{font-family:system-ui;margin:1rem;max-width:760px}"
        "table{width:100%;border-collapse:collapse;margin:.5rem 0}"
        "td,th{text-align:left;padding:.4rem;border-bottom:1px solid #ddd}"
        ".btn{padding:.3rem .6rem;border:1px solid #ccc;border-radius:6px;"
        "cursor:pointer;background:#f6f6f6}code{color:#666}</style>"
        "</head><body><h1>Egress grants</h1>"
        "<p style='color:#6b7280'>User-approved sandbox network access, scoped "
        "to one host, port, and thread. Permanent access is managed in the "
        "committed allowlist, not here.</p>"
        f"{tbl}{_PULL_TO_REFRESH_SCRIPT}</body></html>")


@app.get("/egress", response_class=HTMLResponse)
async def egress_view():
    _require_egress()
    grants = await run_in_threadpool(EGRESS_STORE.live_grants)
    return HTMLResponse(await run_in_threadpool(_render, grants))


def _resolve_and_maybe_dispatch(tid: str, host: str, port: int, decision: str):
    """The whole approve/decline chain in ONE threadpool callable: resolve
    the STORED record (never re-derived state), then — when the thread's
    pending set has emptied — dispatch the single resolution turn."""
    if not os.path.isdir(MANAGER.thread_dir(tid)):
        # Thread deleted between request and approval: no consumer — drop the
        # record instead of granting access nothing can use.
        EGRESS_STORE.remove(request_key(tid, host, port))
        return
    rec = EGRESS_STORE.resolve(request_key(tid, host, port), decision)
    if rec is None:
        return   # unknown / already resolved (double-click) — no-op
    from assist.events.thread_log import append_event
    kind = ("egress_declined" if decision == "decline" else "egress_approved")
    append_event(MANAGER.thread_dir(tid), kind, host=host, port=port)
    if not any(r.state == "pending" for r in EGRESS_STORE.for_thread(tid)):
        _dispatch_egress_resolution(tid)


@app.post("/egress/{decision}")
async def egress_action(decision: str, request: Request):
    _require_egress()
    if decision not in ("hour", "always", "decline", "revoke"):
        raise HTTPException(status_code=404, detail="unknown action")
    form = await request.form()
    _check_csrf(form.get("token"))
    tid = str(form.get("tid") or "")
    host = str(form.get("host") or "")
    try:
        port = int(form.get("port") or 0)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad port")
    if not (tid and host and 0 < port < 65536):
        raise HTTPException(status_code=400, detail="missing fields")
    if decision == "revoke":
        from assist.events.thread_log import append_event
        removed = await run_in_threadpool(
            EGRESS_STORE.revoke, request_key(tid, host, port))
        if removed:
            await run_in_threadpool(
                append_event, MANAGER.thread_dir(tid), "egress_revoked",
                host=host, port=port)
    else:
        await run_in_threadpool(
            _resolve_and_maybe_dispatch, tid, host, port, decision)
    return RedirectResponse(_next(form), status_code=303)
