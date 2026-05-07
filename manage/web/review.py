"""Code-review page: line-clickable diff + Submit-review endpoint.

GET ``/thread/{tid}/review`` renders the full diff with inline comment
editors anchored under each clickable line, plus an overall textarea
and a Submit button.  Drafts persist in localStorage keyed per thread.

POST ``/thread/{tid}/review`` accepts the JSON payload, formats it as a
markdown message, and routes it through ``threads._process_message`` so
the submission flows through the affinity queue exactly like a regular
``/message`` post.
"""
from __future__ import annotations

import html
import json
import os

from fastapi import BackgroundTasks, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from assist.domain_manager import Change
from assist.thread import Thread

from manage.web import threads as _threads
from manage.web.app import app
from manage.web.diff import _DIFF_CSS, _rename_pair, render_file_diff
from manage.web.state import (
    BUSY_STAGES,
    MANAGER,
    _get_domain_manager,
    _get_status,
    _thread_title,
)


# Submit-message format: ``threads.render_thread`` recognises a user
# message that opens with this header as a review submission and renders
# it as markdown rather than escaped plain text.  Stable across the
# test suite — update with care.
_REVIEW_HEADER = "## Change review"
_REVIEW_OPENER = "I've reviewed the changes and have some comments. Please address them."


def _format_review_message(
    overall: str,
    comments: list[dict],
    changes: list[Change],
) -> str:
    """Build the markdown message posted to the thread on review submit.

    *comments* is the localStorage payload's ``lines`` list, one dict
    per per-line comment with keys ``file``, ``row``, ``lineText``,
    ``comment``.  *changes* is the same ``main_diff()`` snapshot the
    review page rendered, used to detect renames so the agent doesn't
    have to guess that "foo.py" is the new name of "old_foo.py".

    Raises ValueError when both *overall* and *comments* are empty —
    the caller (``POST /thread/{tid}/review``) maps this to 400.
    """
    overall = (overall or "").strip()
    cleaned: list[dict] = []
    for c in comments or []:
        text = (c.get("comment") or "").strip()
        if not text:
            continue
        cleaned.append({
            "file": c.get("file") or "",
            "row": c.get("row"),
            "lineText": c.get("lineText") or "",
            "comment": text,
        })
    if not overall and not cleaned:
        raise ValueError("Review must include an overall comment or at least one line comment.")

    rename_map: dict[str, str] = {}
    for ch in changes or []:
        pair = _rename_pair(ch.diff)
        if pair:
            old, new = pair
            rename_map[ch.path] = old if ch.path == new else (new if ch.path == old else "")

    parts: list[str] = [_REVIEW_HEADER, "", _REVIEW_OPENER, ""]
    if overall:
        parts.append(overall)
        parts.append("")
    if cleaned:
        parts.append("### Per-line comments")
        parts.append("")
        for i, c in enumerate(cleaned):
            label = f"`{c['file']}`"
            old = rename_map.get(c["file"])
            if old:
                label += f" (renamed from `{old}`)"
            row = c["row"]
            row_str = f" at diff line {row}" if isinstance(row, int) else ""
            parts.append(f"**{label}**{row_str}:")
            parts.append("")
            parts.append("```")
            parts.append(c["lineText"])
            parts.append("```")
            parts.append("")
            parts.append(c["comment"])
            if i != len(cleaned) - 1:
                parts.append("")
                parts.append("---")
                parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_review_page(tid: str, chat: Thread | None) -> str:
    """Render the full-fat diff review page at ``/thread/{tid}/review``.

    Per-file ``<details>`` blocks contain every line of every changed
    file, each clickable to attach an inline comment.  The header
    carries an overall textarea and the Submit button (disabled when
    the thread is busy).  All draft state is kept client-side in
    ``localStorage`` until the user hits Submit.
    """
    title = _thread_title(tid)
    status = _get_status(tid)
    busy = status.get("stage") in BUSY_STAGES

    diffs: list[Change] = []
    try:
        dm = _get_domain_manager(tid)
        if dm:
            diffs = dm.main_diff()
    except Exception:
        pass

    if not diffs:
        return f"""
        <html><head>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Review — {html.escape(title)}</title>
          <style>body {{ font-family: sans-serif; margin: 0; }}
                 .container {{ max-width: 900px; margin: 0 auto; padding: 1rem; }}
                 .nav a {{ text-decoration: none; padding: .4rem .6rem; border-radius: 6px; }}</style>
        </head><body>
          <div class="container">
            <div class="nav"><a href="/thread/{tid}">← Back to thread</a></div>
            <h1 style="font-size:1.3rem">Review</h1>
            <p><em>No diff to review — the working tree matches main.</em></p>
          </div>
        </body></html>
        """

    files_html = "\n".join(render_file_diff(c, i, full=True) for i, c in enumerate(diffs))
    submit_disabled = "disabled" if busy else ""
    busy_note = (
        '<div class="busy-note">Thread is busy — submit will be available once it finishes.</div>'
        if busy else ""
    )

    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Review — {html.escape(title)}</title>
        <style>
          body {{ font-family: sans-serif; margin: 0; background: #fff; -webkit-tap-highlight-color: rgba(0,0,0,0.05); }}
          .container {{ max-width: 1100px; margin: 0 auto; padding: 1rem; }}
          .nav a {{ text-decoration: none; padding: .6rem .8rem; min-height: 44px; display: inline-flex; align-items: center; border-radius: 6px; color: #0969da; touch-action: manipulation; }}
          h1 {{ font-size: 1.3rem; margin: .5rem 0; }}
          .review-header {{ position: sticky; top: 0; background: #fff; padding: .8rem 0; border-bottom: 1px solid #d0d7de; z-index: 10; margin-bottom: 1rem; }}
          /* font-size: 16px on every editable input prevents iOS zoom on focus. */
          .review-header textarea {{ width: 100%; min-height: 4.5rem; box-sizing: border-box; padding: .7rem; border: 1px solid #d0d7de; border-radius: 6px; font-family: inherit; font-size: 16px; resize: vertical; }}
          .review-header label {{ display: block; font-size: .85rem; color: #57606a; margin-bottom: .3rem; }}
          .review-actions {{ display: flex; align-items: center; gap: .6rem; margin-top: .6rem; flex-wrap: wrap; }}
          .submit-btn {{ background: #1a7f37; color: #fff; border: 1px solid #156529; padding: .7rem 1.2rem; min-height: 44px; font-size: 16px; font-weight: 600; border-radius: 6px; cursor: pointer; touch-action: manipulation; }}
          .submit-btn:hover:not(:disabled) {{ background: #156529; }}
          .submit-btn:disabled {{ background: #94d3a2; border-color: #94d3a2; color: #fff; cursor: not-allowed; }}
          .cancel-link {{ color: #57606a; text-decoration: none; font-size: .95rem; padding: .7rem .9rem; min-height: 44px; display: inline-flex; align-items: center; touch-action: manipulation; }}
          .cancel-link:hover {{ color: #24292f; }}
          .comment-count {{ font-size: .85rem; color: #57606a; }}
          .busy-note {{ background: #fff8c5; border: 1px solid #d4a72c; padding: .5rem .75rem; border-radius: 6px; color: #57606a; font-size: .85rem; margin-top: .5rem; }}
          .comment-editor {{ margin: .3rem .5rem .6rem 1rem; padding: .6rem; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; }}
          .comment-editor textarea {{ width: 100%; min-height: 4rem; box-sizing: border-box; padding: .6rem; border: 1px solid #d0d7de; border-radius: 4px; font-family: inherit; font-size: 16px; resize: vertical; }}
          .comment-editor .editor-actions {{ display: flex; justify-content: flex-end; gap: .4rem; margin-top: .5rem; }}
          .comment-editor button {{ font-size: .9rem; padding: .55rem .9rem; min-height: 36px; border-radius: 4px; cursor: pointer; border: 1px solid #d0d7de; background: #fff; color: #57606a; touch-action: manipulation; }}
          .comment-editor button:hover {{ background: #f6f8fa; }}
          {_DIFF_CSS}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav"><a href="/thread/{tid}">← Back to thread</a></div>
          <h1>Review — {html.escape(title)}</h1>
          <form id="reviewForm" action="/thread/{tid}/review" method="post">
            <input type="hidden" id="payload" name="payload" value="" />
            <div class="review-header">
              <label for="overall">Overall comment</label>
              <textarea id="overall" name="overall_display"
                        placeholder="Optional general comment about the change..."></textarea>
              <div class="review-actions">
                <button type="submit" class="submit-btn" {submit_disabled}>Submit review</button>
                <a href="/thread/{tid}" class="cancel-link">Cancel</a>
                <span id="comment-count" class="comment-count">No line comments yet</span>
              </div>
              {busy_note}
              <div style="font-size:.8rem; color:#57606a; margin-top:.5rem">
                Click any line in the diff below to leave a comment.  Drafts persist in this browser until you submit.
              </div>
            </div>
            <div class="diff-files">
              {files_html}
            </div>
          </form>
        </div>
        <script>
          (function() {{
            const tid = {json.dumps(tid)};
            const KEY = "assist:review:" + tid;
            const SCHEMA_VERSION = 1;
            let state = {{ schemaVersion: SCHEMA_VERSION, overall: "", lines: [], updatedAt: null }};

            function load() {{
              try {{
                const raw = localStorage.getItem(KEY);
                if (!raw) return;
                const parsed = JSON.parse(raw);
                if (parsed && parsed.schemaVersion === SCHEMA_VERSION) state = parsed;
              }} catch (_) {{}}
            }}
            function save() {{
              state.updatedAt = new Date().toISOString();
              try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (_) {{}}
            }}
            function clearStorage() {{
              try {{ localStorage.removeItem(KEY); }} catch (_) {{}}
            }}

            function attachEditor(row, initialText) {{
              let editor = row.nextElementSibling;
              if (editor && editor.classList && editor.classList.contains("comment-editor")) {{
                if (initialText !== undefined) editor.querySelector("textarea").value = initialText;
                editor.querySelector("textarea").focus();
                return editor;
              }}
              editor = document.createElement("div");
              editor.className = "comment-editor";
              const ta = document.createElement("textarea");
              ta.value = initialText || "";
              ta.placeholder = "Comment on this line...";
              ta.addEventListener("input", persistFromDOM);
              ta.addEventListener("blur", persistFromDOM);
              const actions = document.createElement("div");
              actions.className = "editor-actions";
              const remove = document.createElement("button");
              remove.type = "button";
              remove.textContent = "Remove";
              remove.addEventListener("click", () => {{ editor.remove(); persistFromDOM(); }});
              actions.appendChild(remove);
              editor.appendChild(ta);
              editor.appendChild(actions);
              row.parentNode.insertBefore(editor, row.nextSibling);
              ta.focus();
              return editor;
            }}

            function persistFromDOM() {{
              const overall = document.getElementById("overall").value;
              const lines = [];
              document.querySelectorAll(".comment-editor").forEach(ed => {{
                const row = ed.previousElementSibling;
                if (!row || !row.dataset || !row.dataset.key) return;
                const text = ed.querySelector("textarea").value.trim();
                if (!text) return;
                lines.push({{
                  file: row.dataset.file,
                  row: parseInt(row.dataset.row, 10),
                  lineText: row.dataset.linetext,
                  comment: text,
                }});
              }});
              state = {{ schemaVersion: SCHEMA_VERSION, overall, lines, updatedAt: new Date().toISOString() }};
              save();
              updateCount();
            }}

            function updateCount() {{
              const n = state.lines.length;
              const el = document.getElementById("comment-count");
              if (!el) return;
              el.textContent = n === 0 ? "No line comments yet" : (n + " line comment" + (n === 1 ? "" : "s"));
            }}

            function rehydrate() {{
              document.getElementById("overall").value = state.overall || "";
              const rows = document.querySelectorAll(".diff-row[data-key]");
              for (const c of state.lines) {{
                for (const row of rows) {{
                  if (row.dataset.file === c.file && parseInt(row.dataset.row, 10) === c.row) {{
                    // Auto-open the parent <details> so saved comments are visible.
                    let el = row.closest("details");
                    if (el) el.open = true;
                    attachEditor(row, c.comment);
                    break;
                  }}
                }}
              }}
              updateCount();
            }}

            document.addEventListener("click", function(ev) {{
              const row = ev.target.closest(".diff-row[data-key]");
              if (!row) return;
              if (ev.target.closest(".comment-editor")) return;
              attachEditor(row);
            }});
            document.getElementById("overall").addEventListener("input", persistFromDOM);
            document.getElementById("reviewForm").addEventListener("submit", function(ev) {{
              const overall = document.getElementById("overall").value.trim();
              const lines = [];
              document.querySelectorAll(".comment-editor").forEach(ed => {{
                const row = ed.previousElementSibling;
                if (!row || !row.dataset || !row.dataset.key) return;
                const text = ed.querySelector("textarea").value.trim();
                if (!text) return;
                lines.push({{
                  file: row.dataset.file,
                  row: parseInt(row.dataset.row, 10),
                  lineText: row.dataset.linetext,
                  comment: text,
                }});
              }});
              if (!overall && lines.length === 0) {{
                ev.preventDefault();
                alert("Add an overall comment or at least one line comment before submitting.");
                return;
              }}
              document.getElementById("payload").value = JSON.stringify({{overall, lines}});
              // Don't clear localStorage here — if the POST fails (network
              // hiccup, 4xx), the user keeps their draft.  The thread
              // page wipes the key on the ``?reviewed=1`` redirect, which
              // only fires after the server has accepted the submission.
            }});

            load();
            rehydrate();
          }})();
        </script>
      </body>
    </html>
    """


@app.get("/thread/{tid}/review", response_class=HTMLResponse)
async def get_review(tid: str) -> str:
    tdir = MANAGER.thread_dir(tid)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Thread not found")
    chat: Thread | None = None
    try:
        chat = MANAGER.get(tid, sandbox_backend=None)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    return render_review_page(tid, chat)


@app.post("/thread/{tid}/review")
async def post_review(tid: str, background_tasks: BackgroundTasks, payload: str = Form(...)):
    """Accept the localStorage payload, format it as a thread message, queue it.

    Reuses ``threads._process_message`` so the submission flows through
    ``ThreadAffinityQueue`` and surfaces the same busy/queued/error UI
    as a regular ``/message`` post.
    """
    tdir = MANAGER.thread_dir(tid)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Thread not found")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Malformed review payload")

    overall = (data.get("overall") or "").strip() if isinstance(data, dict) else ""
    comments = data.get("lines") if isinstance(data, dict) else []
    if not isinstance(comments, list):
        comments = []

    # Snapshot the current diff for rename-detection in the formatter.
    changes: list[Change] = []
    try:
        dm = _get_domain_manager(tid)
        if dm:
            changes = dm.main_diff()
    except Exception:
        pass

    try:
        message = _format_review_message(overall, comments, changes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(_threads._process_message, tid, message)
    return RedirectResponse(url=f"/thread/{tid}?reviewed=1", status_code=303)
