"""Eval result pages — index table + per-test detail."""
from __future__ import annotations

import html
import urllib.parse

from fastapi import Query
from fastapi.responses import HTMLResponse

from manage.web.app import app


def _status_cell_style(status: str | None) -> str:
    if status == "passed":
        return "background:#c6efce; color:#276221;"
    if status in ("failed", "error"):
        return "background:#ffc7ce; color:#9c0006;"
    if status == "skipped":
        return "background:#ffeb9c; color:#9c5700;"
    # never run
    return "background:#fff; color:#aaa;"


def render_evals() -> str:
    from manage.eval_history import get_runs

    runs = get_runs(limit=10)
    if not runs:
        body = "<p><em>No eval results found in edd/history/.</em></p>"
        return f"""
        <html><head><title>Eval Results</title>
        <style>body{{font-family:sans-serif;margin:0}}
        .container{{max-width:1200px;margin:0 auto;padding:1rem}}
        .nav a{{text-decoration:none;padding:.4rem .6rem;border-radius:6px}}</style></head>
        <body><div class="container">
        <div class="nav"><a href="/">← Back</a></div>
        <h1 style="font-size:1.4rem">Eval Results</h1>{body}</div></body></html>"""

    # Collect all test keys across all runs, preserving insertion order per run
    all_keys: list[str] = []
    seen: set[str] = set()
    for run in runs:
        for key in run["tests"]:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)
    all_keys.sort()

    # Header row: run IDs
    header_cells = "<th style='min-width:7rem;padding:.4rem .5rem;font-size:.75rem;text-align:center;border:1px solid #ddd;background:#f5f5f5;white-space:nowrap'>"
    header_cells += "</th><th style='min-width:7rem;padding:.4rem .5rem;font-size:.75rem;text-align:center;border:1px solid #ddd;background:#f5f5f5;white-space:nowrap'>".join(
        html.escape(r["id"]) for r in runs
    )
    header_cells += "</th>"

    # Stats row: pass/fail counts
    stat_cells = ""
    for run in runs:
        total = len(run["tests"])
        passed = sum(1 for t in run["tests"].values() if t["status"] == "passed")
        failed = sum(1 for t in run["tests"].values() if t["status"] in ("failed", "error"))
        stat_cells += (
            f"<td style='text-align:center;padding:.3rem .4rem;border:1px solid #ddd;"
            f"font-size:.75rem;background:#f9f9f9'>"
            f"<span style='color:#276221'>✓{passed}</span> "
            f"<span style='color:#9c0006'>✗{failed}</span>"
            f"</td>"
        )

    rows_html = []
    for key in all_keys:
        # Short display name: method, with just the class name as subtitle
        # (full module path lives in the tooltip — saves horizontal space).
        parts = key.split("::")
        short = html.escape(parts[-1])
        class_full = parts[0] if len(parts) > 1 else ""
        class_part = html.escape(class_full.rsplit(".", 1)[-1])
        tooltip = html.escape(key)

        row_cells = (
            f"<td style='padding:.35rem .6rem;border:1px solid #ddd;white-space:nowrap;"
            f"font-size:.8rem;position:sticky;left:0;background:#fafafa;z-index:1'>"
            f"<span title='{tooltip}'>{short}</span>"
            f"<div style='font-size:.68rem;color:#888;margin-top:.1rem'>{class_part}</div></td>"
        )

        for run in runs:
            result = run["tests"].get(key)
            status = result["status"] if result else None
            cell_style = _status_cell_style(status)
            label = status[0].upper() if status else "–"
            encoded_key = urllib.parse.quote(key, safe="")
            href = f"/evals/run/{html.escape(run['id'])}?test={encoded_key}"
            if result:
                tip = html.escape((result["message"][:120] if result["message"] else status) or "")
                row_cells += (
                    f"<td style='text-align:center;border:1px solid #ddd;padding:0;{cell_style}'>"
                    f"<a href='{href}' style='display:block;padding:.35rem .4rem;"
                    f"text-decoration:none;color:inherit;font-size:.8rem;font-weight:600' "
                    f"title='{tip}'>"
                    f"{label}</a></td>"
                )
            else:
                row_cells += (
                    f"<td style='text-align:center;border:1px solid #ddd;{cell_style}'>"
                    f"<span style='font-size:.8rem'>–</span></td>"
                )

        rows_html.append(f"<tr>{row_cells}</tr>")

    rows = "\n".join(rows_html)

    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Eval Results</title>
        <style>
          body {{ font-family: sans-serif; margin: 0; }}
          .container {{ max-width: 100%; padding: 1rem; }}
          .nav a {{ text-decoration: none; padding: .4rem .6rem; border-radius: 6px; }}
          .table-wrap {{ overflow-x: auto; margin-top: 1rem; }}
          table {{ border-collapse: collapse; font-size: .85rem; }}
          th {{ padding: .4rem .5rem; border: 1px solid #ddd; background: #f5f5f5;
                font-size: .75rem; white-space: nowrap; text-align: center; }}
          tr:hover td {{ filter: brightness(0.95); }}
          .legend {{ display: flex; gap: 1rem; margin: .5rem 0 1rem; font-size: .82rem; flex-wrap: wrap; }}
          .legend-item {{ display: flex; align-items: center; gap: .3rem; }}
          .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid #ccc; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav"><a href="/">← Back</a></div>
          <h1 style="font-size:1.4rem; margin-bottom:.5rem">Eval Results</h1>
          <div class="legend">
            <div class="legend-item"><div class="legend-swatch" style="background:#c6efce"></div> Pass</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#ffc7ce"></div> Fail / Error</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#ffeb9c"></div> Skipped</div>
            <div class="legend-item"><div class="legend-swatch" style="background:#fff; border:1px solid #ccc"></div> Not run</div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th style="min-width:220px;text-align:left;padding:.4rem .6rem;border:1px solid #ddd;background:#f5f5f5;position:sticky;left:0;z-index:2">Test</th>
                  {header_cells}
                </tr>
                <tr>
                  <td style="border:1px solid #ddd;background:#f9f9f9;padding:.3rem .6rem;font-size:.75rem;color:#666;position:sticky;left:0;z-index:2">Pass ✓ / Fail ✗</td>
                  {stat_cells}
                </tr>
              </thead>
              <tbody>
                {rows}
              </tbody>
            </table>
          </div>
        </div>
      </body>
    </html>
    """


def render_eval_detail(run_id: str, test_key: str) -> str:
    from manage.eval_history import get_runs

    runs = get_runs(limit=50)
    run = next((r for r in runs if r["id"] == run_id), None)
    if run is None:
        return f"<html><body><p>Run '{html.escape(run_id)}' not found.</p><a href='/evals'>Back</a></body></html>"

    result = run["tests"].get(test_key)
    if result is None:
        return (
            f"<html><body><p>Test not found in run {html.escape(run_id)}.</p>"
            f"<a href='/evals'>Back</a></body></html>"
        )

    status = result["status"]
    cell_style = _status_cell_style(status)
    parts = test_key.split("::")
    short_name = parts[-1]
    class_name = "::".join(parts[:-1]) if len(parts) > 1 else ""

    message_html = (
        f"<pre style='background:#f8f8f8;border:1px solid #ddd;padding:.8rem 1rem;"
        f"border-radius:6px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;"
        f"font-size:.82rem'>{html.escape(result['message'])}</pre>"
        if result["message"] else ""
    )
    details_html = (
        f"<h3 style='font-size:1rem;margin-top:1.5rem'>Traceback</h3>"
        f"<pre style='background:#f8f8f8;border:1px solid #ddd;padding:.8rem 1rem;"
        f"border-radius:6px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;"
        f"font-size:.8rem'>{html.escape(result['details'])}</pre>"
        if result["details"] else ""
    )

    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{html.escape(short_name)} — {html.escape(run_id)}</title>
        <style>
          body {{ font-family: sans-serif; margin: 0; }}
          .container {{ max-width: 900px; margin: 0 auto; padding: 1rem; }}
          .nav a {{ text-decoration: none; padding: .4rem .6rem; border-radius: 6px; }}
          .badge {{ display: inline-block; padding: .3rem .8rem; border-radius: 12px;
                    font-weight: 600; font-size: .9rem; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav"><a href="/evals">← Eval Results</a></div>
          <h1 style="font-size:1.3rem; margin:.8rem 0 .2rem">{html.escape(short_name)}</h1>
          <div style="color:#666; font-size:.85rem; margin-bottom:1rem">{html.escape(class_name)}</div>

          <table style="border-collapse:collapse; font-size:.9rem; margin-bottom:1rem">
            <tr>
              <td style="padding:.3rem .8rem .3rem 0; color:#555; font-weight:500">Run</td>
              <td style="padding:.3rem 0">{html.escape(run_id)}</td>
            </tr>
            <tr>
              <td style="padding:.3rem .8rem .3rem 0; color:#555; font-weight:500">Timestamp</td>
              <td style="padding:.3rem 0">{html.escape(run.get('timestamp', ''))}</td>
            </tr>
            <tr>
              <td style="padding:.3rem .8rem .3rem 0; color:#555; font-weight:500">Duration</td>
              <td style="padding:.3rem 0">{result['time']:.2f}s</td>
            </tr>
            <tr>
              <td style="padding:.3rem .8rem .3rem 0; color:#555; font-weight:500">Status</td>
              <td style="padding:.3rem 0">
                <span class="badge" style="{cell_style}">{html.escape(status.upper())}</span>
              </td>
            </tr>
          </table>

          {message_html}
          {details_html}
        </div>
      </body>
    </html>
    """


@app.get("/evals", response_class=HTMLResponse)
async def evals_index() -> str:
    return render_evals()


@app.get("/evals/run/{run_id}", response_class=HTMLResponse)
async def eval_run_detail(run_id: str, test: str = Query(...)) -> str:
    test_key = urllib.parse.unquote(test)
    return render_eval_detail(run_id, test_key)
