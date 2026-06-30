"""The /schedules management view — list ALL schedules across threads + delete.

Read+delete only (creation/edit is agent-driven, per the PRD). Both routes run OFF the
asyncio loop (``run_in_threadpool``) because the store takes a lock + reads/writes disk —
a blocking op on the single-worker loop would stall the whole server.
"""
from __future__ import annotations

import html

from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from assist.schedule import cadence
from assist.schedule.store import ScheduleNotFound
from manage.web.app import app
from manage.web.state import SCHEDULE_STORE


def _row(s) -> str:
    paused = "" if s.enabled else ' <span class="paused">(paused)</span>'
    return (
        "<tr>"
        f'<td><a class="thread-link" href="/thread/{html.escape(s.thread_id)}">'
        f'{html.escape(s.thread_id)}</a></td>'
        f"<td>{html.escape(cadence.describe(s.cadence))}{paused}</td>"
        f'<td class="prompt">{html.escape(s.prompt)}</td>'
        f"<td>{cadence.fmt_instant(s.next_fire_at, s.tz)}</td>"
        f'<td><form method="post" action="/schedules/{html.escape(s.thread_id)}/'
        f'{html.escape(s.id)}/delete">'
        f'<button class="btn btn-secondary" type="submit">Delete</button></form></td>'
        "</tr>"
    )


def _render(scheds) -> str:
    if scheds:
        body = ("<table><thead><tr><th>Thread</th><th>Schedule</th><th>Prompt</th>"
                "<th>Next run</th><th></th></tr></thead><tbody>"
                + "".join(_row(s) for s in scheds) + "</tbody></table>")
    else:
        body = "<p>No schedules yet. Ask the assistant in a thread to set one up.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Schedules</title><style>
  body {{ font-family: system-ui, sans-serif; margin: 1rem; }}
  .topbar {{ display: flex; gap: .5rem; align-items: center; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: .5rem; border-bottom: 1px solid #ddd; vertical-align: top; }}
  td.prompt {{ max-width: 28rem; }}
  .paused {{ color: #999; }}
  .btn {{ display: inline-block; padding: .4rem .7rem; border-radius: 6px; border: 1px solid #ccc;
          background: #f5f5f5; text-decoration: none; color: #222; cursor: pointer; }}
  .btn-secondary {{ background: #fff; }}
</style></head><body>
  <div class="topbar"><a href="/" class="btn">&larr; Threads</a><h2>Scheduled prompts</h2></div>
  {body}
</body></html>"""


@app.get("/schedules", response_class=HTMLResponse)
async def schedules_page():
    scheds = await run_in_threadpool(SCHEDULE_STORE.all)
    return HTMLResponse(_render(scheds))


@app.post("/schedules/{tid}/{sid}/delete")
async def delete_schedule_route(tid: str, sid: str):
    try:
        await run_in_threadpool(SCHEDULE_STORE.remove, tid, sid)
    except ScheduleNotFound:
        pass  # already gone is fine; other errors surface
    return RedirectResponse("/schedules", status_code=303)
