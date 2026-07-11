"""The /geo management view — loaded regions (delete / add-transit) + pending download
proposals (approve / decline).

All mutations go through the off-loop Provisioner or the ProposalStore; the routes only
enqueue (non-blocking) or do a quick store read (``run_in_threadpool``), so nothing heavy
runs on the single-worker asyncio loop. The destructive routes (delete especially — it
triggers a Nominatim rebuild) carry an in-memory CSRF token embedded in the page's forms:
a cross-origin/DNS-rebind forge can't read the page to obtain it, so it can't fire a
rebuild off one blind POST (design doc T1). Absent the geo stores (TRAVEL_INFRA_DIR unset)
the view 404s.
"""
from __future__ import annotations

import html
import secrets

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from assist.geo.model import STATE_READY
from manage.web.app import app
from manage.web.state import GEO_PROPOSALS, GEO_REGISTRY
from manage.web.threads import _PROVISIONER, _PULL_TO_REFRESH_SCRIPT

# One app-lifetime token (single-user app). Embedded in every mutating form + required on
# the POST; rotates on restart (reload the page to get the new one).
_CSRF = secrets.token_urlsafe(16)


def _require_geo():
    if GEO_REGISTRY is None or _PROVISIONER is None:
        raise HTTPException(status_code=404, detail="geo is not configured")


def _check_csrf(token: str | None):
    if not token or not secrets.compare_digest(token, _CSRF):
        raise HTTPException(status_code=403, detail="bad or missing form token")


def _fmt_size(n: int | None) -> str:
    if not n:
        return "size unknown"   # HEAD couldn't determine it — say so, don't render blank
    mb = n / 1e6
    return f"~{mb / 1000:.1f} GB" if mb >= 1000 else f"~{mb:.0f} MB"


def _btn(action: str, label: str, cls: str = "btn-secondary") -> str:
    return (f'<form method="post" action="{action}" style="display:inline">'
            f'<input type="hidden" name="token" value="{_CSRF}">'
            f'<button class="btn {cls}" type="submit">{label}</button></form>')


def _render(regions, proposals) -> str:
    prop_rows = ""
    for p in sorted(proposals, key=lambda p: p.created_at):
        s = html.escape(p.slug)
        prop_rows += (
            f"<tr><td>{html.escape(p.display_name)} <code>{s}</code></td>"
            f"<td>{_fmt_size(p.size_bytes)}</td>"
            f"<td>{_btn(f'/geo/{s}/approve', 'Approve download', 'merge-btn')} "
            f"{_btn(f'/geo/{s}/decline', 'Decline')}</td></tr>")
    prop_tbl = (f"<h2>Pending downloads</h2><p class='muted'>Approving starts a download + "
                f"map-data rebuild (address lookups are degraded until it finishes).</p>"
                f"<table><tbody>{prop_rows}</tbody></table>" if prop_rows else "")

    reg_rows = ""
    for r in sorted(regions, key=lambda r: r.display_name):
        s = html.escape(r.slug)
        transit = "transit" if r.has_transit else "no transit"
        state = "" if r.state == STATE_READY else f' <span class="paused">({html.escape(r.state)})</span>'
        add_transit = "" if r.has_transit or r.state != STATE_READY else _btn(f"/geo/{s}/transit", "Add transit")
        delete = _btn(f"/geo/{s}/delete", "Delete") if r.state == STATE_READY else ""
        reg_rows += (f"<tr><td>{html.escape(r.display_name)} <code>{s}</code>{state}</td>"
                     f"<td>{transit}</td><td>{add_transit} {delete}</td></tr>")
    reg_tbl = (f"<table><thead><tr><th>Region</th><th>Transit</th><th></th></tr></thead>"
               f"<tbody>{reg_rows}</tbody></table>" if reg_rows else "<p>No regions loaded.</p>")

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Geo regions</title>"
        "<style>body{font-family:system-ui;margin:1rem;max-width:760px}"
        "table{width:100%;border-collapse:collapse;margin:.5rem 0}"
        "td,th{text-align:left;padding:.4rem;border-bottom:1px solid #ddd}"
        ".btn{padding:.3rem .6rem;border:1px solid #ccc;border-radius:6px;cursor:pointer;background:#f6f6f6}"
        ".merge-btn{background:#e7f0ff}.muted,.paused{color:#888}code{color:#666;font-size:.85em}</style>"
        "</head><body><h1>Geo regions</h1>"
        f"{prop_tbl}<h2>Loaded regions</h2>{reg_tbl}"
        f"{_PULL_TO_REFRESH_SCRIPT}</body></html>")


@app.get("/geo", response_class=HTMLResponse)
async def geo_view():
    _require_geo()
    regions = await run_in_threadpool(GEO_REGISTRY.all)
    proposals = await run_in_threadpool(GEO_PROPOSALS.all)
    return HTMLResponse(await run_in_threadpool(_render, regions, proposals))


@app.post("/geo/{slug:path}/approve")
async def geo_approve(slug: str, request: Request):
    _require_geo()
    _check_csrf((await request.form()).get("token"))
    # fire the RECORDED slug's add (submit refuses without a pending proposal)
    await run_in_threadpool(_PROVISIONER.submit, "add", slug)
    return RedirectResponse("/geo", status_code=303)


@app.post("/geo/{slug:path}/decline")
async def geo_decline(slug: str, request: Request):
    _require_geo()
    _check_csrf((await request.form()).get("token"))
    await run_in_threadpool(GEO_PROPOSALS.remove, slug)
    return RedirectResponse("/geo", status_code=303)


@app.post("/geo/{slug:path}/delete")
async def geo_delete(slug: str, request: Request):
    _require_geo()
    _check_csrf((await request.form()).get("token"))
    await run_in_threadpool(_PROVISIONER.submit, "remove", slug)
    return RedirectResponse("/geo", status_code=303)


@app.post("/geo/{slug:path}/transit")
async def geo_transit(slug: str, request: Request):
    _require_geo()
    _check_csrf((await request.form()).get("token"))
    await run_in_threadpool(_PROVISIONER.submit, "transit", slug)
    return RedirectResponse("/geo", status_code=303)
