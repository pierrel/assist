"""Index + thread page rendering and the routes that drive them.

Owns ``_process_message`` (the synchronous worker spawned as a
``BackgroundTask`` for both ``/message`` and ``/review`` submissions),
``_initialize_thread`` (first-turn clone + sandbox boot), and
``_capture_conversation`` (capture-this-thread side-quest).
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import subprocess
import urllib.parse

import markdown
from fastapi import BackgroundTasks, Form, HTTPException
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)

from assist.domain_manager import (
    Change,
    DomainManager,
    MergeConflictError,
    OriginAdvancedError,
)
from assist.sandbox import SandboxContainerLostError
from assist.sandbox_manager import SandboxManager
from assist.thread import Thread
from assist.thread_manager import InvalidThreadId
from assist.thread_queue import THREAD_QUEUE

from manage.web.app import app
from manage.web.diff import _DIFF_CSS, _render_inline_diffs
from manage.web.state import (
    BUSY_STAGES,
    DESCRIPTION_CACHE,
    DOMAIN_MANAGERS,
    DOMAINS,
    INIT_STAGES,
    MANAGER,
    MERGE_LOCK,
    STAGE_LABELS,
    _clear_conflict,
    _domain_selector_html,
    _evict_caches,
    _get_conflict,
    _get_domain_manager,
    _get_sandbox_backend,
    _get_status,
    _has_unmerged_changes,
    _set_conflict,
    _set_status,
    _thread_domain_html,
    _thread_title,
    get_cached_description,
    set_description,
)


@app.exception_handler(InvalidThreadId)
async def _invalid_thread_id(request, exc):
    # A crafted tid (traversal/separator) reaching any tid-based route surfaces
    # here from ThreadManager.thread_dir — map it to a clean 404 everywhere.
    return HTMLResponse("Thread not found", status_code=404)


_MD_EXTENSIONS = ["fenced_code", "tables"]


def render_index(query: str = "") -> str:
    q = (query or "").strip()
    ql = q.lower()
    items = []
    for tid in MANAGER.list():
        title = _thread_title(tid)
        # Search matches the displayed title (== the thread's description),
        # case-insensitive substring — the text the user actually sees and types.
        if ql and ql not in title.lower():
            continue
        status = _get_status(tid)
        stage = status.get("stage", "ready")
        badge = ""
        if stage == "queued":
            # Distinguish "queued" visually from other busy stages so
            # the user can tell their message is held behind another
            # thread (vs. actively running).
            badge = (
                f'<span style="font-size:.7rem; color:#1e3a5f; background:#e1ecf4;'
                f' border:1px solid #b6d4ef; padding:.1rem .4rem; border-radius:10px;'
                f' margin-right:.4rem;">{html.escape(STAGE_LABELS.get(stage, stage))}</span>'
            )
        elif stage in BUSY_STAGES:
            badge = (
                f'<span style="font-size:.7rem; color:#555; background:#fff3cd;'
                f' border:1px solid #ffeeba; padding:.1rem .4rem; border-radius:10px;'
                f' margin-right:.4rem;">{html.escape(STAGE_LABELS.get(stage, stage))}</span>'
            )
        elif stage == "error":
            badge = (
                '<span style="font-size:.7rem; color:#721c24; background:#f8d7da;'
                ' border:1px solid #f5c6cb; padding:.1rem .4rem; border-radius:10px;'
                ' margin-right:.4rem;">error</span>'
            )
        elif _has_unmerged_changes(tid):
            # Soft amber, distinct from yellow (busy) and red (error).
            # Strictly secondary to the process-state badges above —
            # only shows when the thread is otherwise idle.
            badge = (
                '<span style="font-size:.7rem; color:#7c4a1d; background:#fef0e0;'
                ' border:1px solid #fbcfa0; padding:.1rem .4rem; border-radius:10px;'
                ' margin-right:.4rem;">unmerged</span>'
            )
        items.append(
            f'<li>'
            f'<a class="thread-link" href="/thread/{tid}">{badge}{html.escape(title)}</a>'
            f'<form action="/thread/{tid}/delete" method="post" style="margin:0">'
            f'<button type="submit" class="del-btn" aria-label="Delete thread" '
            f'onclick="return confirm(\'Permanently delete this thread? This cannot be undone.\')">&#x2715;</button>'
            f'</form></li>'
        )
    matched = len(items)
    if not items:
        items.append(
            f'<li><em>No threads match &ldquo;{html.escape(q)}&rdquo;.</em> '
            f'<a href="/">clear</a></li>'
            if ql else "<li><em>No threads yet</em></li>"
        )
    items_html = "\n".join(items)
    search_status = (
        f'<p style="font-size:.85rem; color:#666; margin:.2rem 0 .6rem;">'
        f'{matched} match{"" if matched == 1 else "es"} for '
        f'&ldquo;{html.escape(q)}&rdquo; &middot; <a href="/">clear</a></p>'
    ) if q else ""
    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Assist Web</title>
        <style>
          :root {{ --pad: 1rem; }}
          body {{ font-family: sans-serif; margin: 0; -webkit-tap-highlight-color: rgba(0,0,0,0.05); }}
          .container {{ max-width: 800px; margin: 0 auto; padding: var(--pad); }}
          .topbar {{ display: flex; gap: .5rem; flex-wrap: wrap; justify-content: space-between; align-items: center; }}
          ul {{ line-height: 1.4; padding-left: 0; list-style: none; }}
          /* Each row is flex so the title link expands to fill, leaving the
             delete button anchored on the right.  min-height matches Apple's
             44 pt touch-target guidance — enough to tap reliably on mobile. */
          li {{ margin: .4rem 0; display: flex; align-items: stretch; gap: .25rem; }}
          .thread-link {{ flex: 1; display: flex; align-items: center; padding: .85rem .8rem; border-radius: 6px; min-height: 44px; text-decoration: none; color: inherit; touch-action: manipulation; }}
          .thread-link:hover {{ background: #f3f6fa; }}
          .thread-link:active {{ background: #e7edf4; }}
          .del-btn {{ background: none; border: none; color: #999; cursor: pointer; font-size: 1.4rem; padding: 0 .8rem; border-radius: 6px; min-width: 44px; min-height: 44px; touch-action: manipulation; }}
          .del-btn:hover {{ color: #c00; background: #fee; }}
          .del-btn:active {{ background: #fdd; }}
          a:active, a:focus {{ outline: none; }}
          /* inline-flex so the same .btn class works on both <button>
             and <a> (e.g., the Evals link in the topbar): the 44 px
             min-height needs flex centering or the text floats up. */
          .btn {{ display: inline-flex; align-items: center; justify-content: center; padding: .7rem 1rem; min-height: 44px; border: 1px solid #333; border-radius: 8px; background: #eee; color: inherit; font-size: 16px; text-decoration: none; cursor: pointer; touch-action: manipulation; box-sizing: border-box; }}
          .new-thread-form {{ margin-bottom: 1.5rem; padding: 1rem; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd; }}
          /* font-size: 16px (not 1rem) explicitly prevents iOS Safari from
             auto-zooming on focus.  Anything below 16px triggers the zoom. */
          .new-thread-form textarea {{ width: 100%; min-height: 5rem; box-sizing: border-box; padding: .8rem; border: 1px solid #ccc; border-radius: 6px; font-family: inherit; font-size: 16px; resize: vertical; }}
          .new-thread-form textarea:focus {{ outline: 2px solid #4a90e2; border-color: #4a90e2; }}
          .new-thread-form select {{ font-size: 16px; padding: .6rem; min-height: 44px; }}
          .new-thread-btn {{ margin-top: .6rem; display: none; }}
          .new-thread-btn.visible {{ display: block; }}
          @media (max-width: 480px) {{
            .btn {{ width: 100%; }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="topbar">
            <h1 style="font-size:1.4rem; margin:0">Assist Web</h1>
            <a href="/evals" class="btn">Evals</a>
          </div>

          <div class="new-thread-form">
            <form action="/threads/with-message" method="post" id="newThreadForm">
              {_domain_selector_html()}
              <textarea
                id="initialMessage"
                name="text"
                placeholder="Type a message to start a new thread..."
                oninput="toggleNewThreadButton()"
              ></textarea>
              <button class="btn new-thread-btn" id="newThreadBtn" type="submit">New Thread</button>
            </form>
          </div>

          <h2 style="font-size:1.2rem">Threads</h2>
          <form method="get" action="/" style="margin:0 0 .6rem;">
            <input type="search" name="q" value="{html.escape(q)}"
                   placeholder="Search threads..." aria-label="Search threads"
                   style="width:100%; box-sizing:border-box; padding:.7rem .8rem; font-size:16px; border:1px solid #ccc; border-radius:6px;" />
          </form>
          {search_status}
          <ul>
            {items_html}
          </ul>
        </div>

        <script>
          function toggleNewThreadButton() {{
            const textarea = document.getElementById('initialMessage');
            const button = document.getElementById('newThreadBtn');
            if (textarea.value.trim().length > 0) {{
              button.classList.add('visible');
            }} else {{
              button.classList.remove('visible');
            }}
          }}
        </script>
      </body>
    </html>
    """


def render_thread(
    tid: str,
    chat: Thread | None,
    captured: bool = False,
    merged: bool = False,
    reviewed: bool = False,
    pushed: bool = False,
) -> str:
    # Local import to avoid circular dependency with review.py at module load.
    from manage.web.review import _REVIEW_HEADER

    status = _get_status(tid)
    stage = status.get("stage", "ready")
    busy = stage in BUSY_STAGES
    is_init = stage in INIT_STAGES
    title = _thread_title(tid)

    # Rename is only offered when idle: while busy/initializing the displayed
    # title is the pending-message snippet, not the description, so editing it
    # then would bake that snippet (with its "...") in as the permanent title.
    can_rename = not (busy or is_init)
    rename_button = (
        '<button type="button" onclick="showRename()" aria-label="Rename thread" '
        'style="background:none; border:none; color:#999; cursor:pointer; '
        'font-size:1rem; padding:.2rem .4rem; line-height:1;">&#x270e;</button>'
    ) if can_rename else ""
    # Stacked: full-width input on its own line, Save/Cancel on the line below
    # (showRename() flips display to flex; flex-direction:column does the stack).
    rename_form = (
        f'<form id="titleEdit" action="/thread/{tid}/rename" method="post" '
        f'style="display:none; flex-direction:column; gap:.5rem; margin:.3rem 0;">'
        f'<input type="text" name="description" value="{html.escape(title)}" '
        f'maxlength="120" required aria-label="Thread name" '
        f'style="width:100%; box-sizing:border-box; padding:.6rem .7rem; '
        f'font-size:16px; border:1px solid #ccc; border-radius:6px;" />'
        f'<div style="display:flex; gap:.5rem;">'
        f'<button class="btn" type="submit" style="min-height:auto; padding:.5rem 1rem;">Save</button>'
        f'<button class="btn btn-secondary" type="button" onclick="hideRename()" '
        f'style="min-height:auto; padding:.5rem 1rem;">Cancel</button>'
        f'</div>'
        f'</form>'
    ) if can_rename else ""

    # During the initial setup stages there is no agent state worth showing yet.
    msgs: list[dict] = [] if is_init or chat is None else chat.get_messages()

    # While busy, surface the pending (just-submitted) message as a user
    # bubble so it's visible right after the redirect — unless the agent has
    # already persisted an identical user message into the conversation (the
    # `not any(...)` dedup guard below), in which case it's already shown.
    # Append (not insert-at-0): get_messages() is chronological and the page
    # renders reversed (newest-at-top), so appending places the pending
    # message at the TOP — as the latest message, right below the in-progress
    # "..." placeholder — instead of stranding it at the very bottom under the
    # whole prior conversation.
    pending = (status.get("pending_message") or "").strip()
    # Compare stripped on BOTH sides: the persisted message can carry trailing
    # whitespace the stripped `pending` won't (review submissions from
    # `_format_review_message` end with a newline), and an exact `==` would
    # miss the match and render a duplicate bubble while the turn runs.
    if busy and pending and not any(
        m.get("role") == "user" and (m.get("content") or "").strip() == pending
        for m in msgs
    ):
        msgs.append({"role": "user", "content": pending})

    # Compute diff vs main (only when repo is ready) — rendered as its own
    # top-of-page block, separate from the message bubbles, so the per-file
    # collapse stack and the Merge / Review buttons sit together.
    diffs: list[Change] = []
    if not is_init:
        try:
            dm = _get_domain_manager(tid)
            if dm:
                diffs = dm.main_diff()
        except Exception:
            pass

    # Surface a persistent merge-conflict banner above the diff stack
    # whenever the most recent merge attempt aborted on a rebase
    # conflict.  The banner clears the moment the next merge call
    # succeeds (see ``merge_thread`` below), and stays put across
    # ``processing`` ↔ ``ready`` transitions so the user can ask the
    # agent to fix the conflict and the banner doesn't disappear when
    # the agent's response lands.
    conflict_state = _get_conflict(tid) if not is_init else None
    conflict_banner_html = ""
    if conflict_state:
        files = conflict_state.get("files") or []
        files_html = "".join(
            f'<li><code>{html.escape(f)}</code></li>' for f in files
        ) or "<li><em>(unmerged file list unavailable)</em></li>"
        conflict_banner_html = f"""
        <div class="conflict-banner">
          <strong>Merge conflict on <code>{html.escape(conflict_state.get("branch", "?"))}</code>.</strong>
          The rebase onto <code>origin/main</code> aborted because the
          following file(s) need manual reconciliation:
          <ul>{files_html}</ul>
          The agent can attempt to resolve this — type a message asking
          it to fix the conflict, then re-click <em>Merge to Main</em>.
        </div>
        """

    # The push-to-origin button is visible only when a previous merge
    # has put unpushed work on local ``main``.  ``has_unpushed_main``
    # is a no-fetch ref-distance check; the push endpoint does the
    # authoritative ``fetch + ancestor check`` server-side.
    show_push_button = False
    if not is_init:
        try:
            dm_for_push = _get_domain_manager(tid)
            if dm_for_push:
                show_push_button = dm_for_push.has_unpushed_main()
        except Exception:
            pass

    push_btn_html = (
        f"""<form action="/thread/{tid}/push-main" method="post" style="margin: 0;">
              <button class="btn push-btn" type="submit"
                      onclick="return confirm('Push local main to origin?');">
                Push to origin
              </button>
            </form>"""
        if show_push_button else ""
    )

    diff_block_html = ""
    if diffs:
        diff_files_html = _render_inline_diffs(tid, diffs)
        diff_block_html = f"""
        <div class="diff-container">
          <div class="diff-actions">
            <a class="btn btn-secondary review-btn" href="/thread/{tid}/review">Review</a>
            {push_btn_html}
            <form action="/thread/{tid}/merge" method="post" style="margin: 0;">
              <button class="btn merge-btn" type="submit"
                      onclick="return confirm('Merge this branch into main? This will rebase onto origin/main and squash into a single commit.');">
                Merge to Main
              </button>
            </form>
          </div>
          <div class="diff-files">
            {diff_files_html}
          </div>
        </div>
        """
    elif show_push_button:
        # Diff is empty (post-merge) but the user still needs the push
        # button — render a slimmed-down action row.
        diff_block_html = f"""
        <div class="diff-container">
          <div class="diff-actions">
            {push_btn_html}
          </div>
        </div>
        """

    rendered = []
    for m in reversed(msgs):
        if m.get("role") == "show_file":
            rendered.append(_render_show_file(tid, m.get("path", "")))
            continue
        role = html.escape(m.get("role", ""))
        raw = str(m.get("content", ""))
        if role == "assistant" or role == "tools":
            # Render assistant and tool content as Markdown to HTML
            content_html = markdown.markdown(raw, extensions=_MD_EXTENSIONS)
        elif role == "user" and raw.startswith(_REVIEW_HEADER):
            # Review submissions are markdown-formatted (headers, fenced
            # blocks).  Render them as such so the user sees the same
            # structure the agent receives, instead of escaped backticks.
            content_html = markdown.markdown(raw, extensions=_MD_EXTENSIONS)
        else:
            # Human/user content is plain text with basic escaping
            content_html = html.escape(raw).replace("\n", "<br/>")
        cls = "user" if role == "user" else ("tools" if role == "tools" else "assistant")
        bubble = f"<div class=\"msg {cls}\"><div class=\"role\">{role}</div><div class=\"content\">{content_html}</div></div>"
        rendered.append(bubble)
    if busy:
        rendered.insert(
            0,
            '<div class="msg assistant placeholder">'
            '<div class="role">assistant</div>'
            '<div class="content"><span class="dots"><span>.</span><span>.</span><span>.</span></span></div>'
            '</div>',
        )
    body = "\n".join(rendered) or "<p><em>No messages yet.</em></p>"

    # Status banner
    status_banner = ""
    if busy:
        label = STAGE_LABELS.get(stage, "Working...")
        status_banner = (
            f'<div class="status-banner">'
            f'<span class="spinner"></span>'
            f'<span>{html.escape(label)}</span>'
            f'</div>'
        )
    elif stage == "error":
        err = html.escape(status.get("error", "Unknown error"))
        # description.txt is only written after the first successful turn, so
        # its absence distinguishes a setup-time failure from a mid-conversation one.
        had_prior_turn = os.path.isfile(
            os.path.join(MANAGER.thread_dir(tid), "description.txt")
        )
        label = "Couldn't process your message:" if had_prior_turn else "Setup failed:"
        status_banner = f'<div class="error-msg"><strong>{label}</strong> {err}</div>'

    # Disable the input form during the initial setup phase
    form_disabled = "disabled" if is_init else ""
    form_note = (
        "Thread is being set up, please wait..."
        if is_init
        else "If you close or refresh, your message will still be processed."
    )
    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{html.escape(title)}</title>
        <style>
          :root {{ --pad: 1rem; }}
          body {{ font-family: sans-serif; margin: 0; -webkit-tap-highlight-color: rgba(0,0,0,0.05); }}
          .container {{ max-width: 800px; margin: 0 auto; padding: var(--pad); }}
          /* inline-flex centers the back-link text vertically inside
             the 44 px min-height; inline-block leaves the text floating
             at the top of the box. */
          .nav a {{ display: inline-flex; align-items: center; padding: .6rem .8rem; min-height: 44px; border-radius: 6px; text-decoration: none; touch-action: manipulation; }}
          .msg {{ margin: .6rem 0; padding: .6rem .8rem; border-radius: 8px; max-width: 100%; word-wrap: break-word; overflow-wrap: anywhere; }}
          .msg.user {{ background: #e6f3ff; border: 1px solid #b5dbff; }}
          .msg.assistant {{ background: #f6f6f6; border: 1px solid #ddd; }}
          .msg.show {{ background: #fff; border: 1px solid #ddd; }}
          /* Embedded shown file (org/md rendered page, or pdf viewer). */
          .show-file {{ width: 100%; height: 65vh; border: 1px solid #e3e3e3; border-radius: 6px; background: #fff; }}
          .show-cap {{ font-size: .85rem; margin-top: .3rem; }}
          .role {{ font-size: .8rem; color: #555; margin-bottom: .2rem; text-transform: uppercase; }}
          /* font-size: 16px on every editable form input — prevents iOS
             Safari from auto-zooming into the field on focus.  Anything
             below 16px (including 0.95rem) triggers the zoom. */
          form textarea {{ width: 100%; min-height: 6rem; height: 24vh; box-sizing: border-box; padding: .6rem; font-family: inherit; font-size: 16px; border: 1px solid #ccc; border-radius: 6px; }}
          form {{ margin-top: 1rem; }}
          .btn {{ display: inline-flex; align-items: center; justify-content: center; padding: .7rem 1rem; min-height: 44px; border: 1px solid #333; border-radius: 8px; background: #eee; color: inherit; font-size: 16px; text-decoration: none; cursor: pointer; touch-action: manipulation; box-sizing: border-box; }}
          .btn-secondary {{ background: #ddd; }}
          .success-msg {{ background: #d4edda; border: 1px solid #c3e6cb; padding: .8rem; margin: .5rem 0; border-radius: 6px; color: #155724; }}
          .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.4); }}
          .modal-content {{ background: #fff; margin: 10% auto; padding: 1.5rem; width: min(95%, 500px); border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); box-sizing: border-box; }}
          .modal-content h3 {{ margin-top: 0; }}
          .modal-content textarea {{ width: 100%; min-height: 100px; padding: .6rem; border: 1px solid #ccc; border-radius: 4px; font-family: inherit; font-size: 16px; box-sizing: border-box; }}
          .modal-content label {{ display: block; margin-bottom: .5rem; font-weight: 500; }}
          .button-group {{ display: flex; gap: .5rem; margin-top: 1rem; }}
          .diff-container {{ margin: 1rem 0 .5rem; }}
          .diff-actions {{ display: flex; justify-content: flex-end; gap: .5rem; margin-bottom: .6rem; flex-wrap: wrap; }}
          .merge-btn {{ background: #28a745; color: white; border: 1px solid #1e7e34; padding: .65rem .9rem; min-height: 44px; font-size: .95rem; white-space: nowrap; touch-action: manipulation; }}
          .merge-btn:hover {{ background: #218838; border-color: #1c7430; }}
          .push-btn {{ background: #0366d6; color: white; border: 1px solid #024ea4; padding: .65rem .9rem; min-height: 44px; font-size: .95rem; white-space: nowrap; touch-action: manipulation; }}
          .push-btn:hover {{ background: #024ea4; border-color: #023672; }}
          .review-btn {{ display: inline-flex; align-items: center; padding: .65rem .9rem; min-height: 44px; font-size: .95rem; white-space: nowrap; text-decoration: none; color: #24292f; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; touch-action: manipulation; }}
          .review-btn:hover {{ background: #eaeef2; }}
          .conflict-banner {{ background: #fff3f3; border: 1px solid #f5c6cb; padding: .8rem 1rem; margin: .8rem 0; border-radius: 6px; color: #721c24; font-size: .95rem; }}
          .conflict-banner ul {{ margin: .4rem 0 .4rem 1.2rem; padding: 0; }}
          .conflict-banner code {{ background: #fbe9eb; padding: 0 .25rem; border-radius: 3px; }}
          {_DIFF_CSS}
          .error-msg {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: .8rem; margin: .5rem 0; border-radius: 6px; color: #721c24; }}
          .status-banner {{ display: flex; align-items: center; gap: .6rem; background: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: .6rem .8rem; margin: .5rem 0; border-radius: 6px; font-size: .9rem; }}
          .spinner {{ width: 14px; height: 14px; border: 2px solid #d6ad00; border-top-color: transparent; border-radius: 50%; display: inline-block; animation: spin 1s linear infinite; }}
          @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
          .placeholder .dots span {{ display: inline-block; animation: blink 1.4s infinite both; opacity: .2; font-weight: bold; font-size: 1.4rem; line-height: 0; }}
          .placeholder .dots span:nth-child(2) {{ animation-delay: .2s; }}
          .placeholder .dots span:nth-child(3) {{ animation-delay: .4s; }}
          @keyframes blink {{ 0%, 80%, 100% {{ opacity: .2; }} 40% {{ opacity: 1; }} }}
          form textarea[disabled] {{ background: #f5f5f5; cursor: not-allowed; }}
          .btn[disabled] {{ background: #eee; color: #aaa; cursor: not-allowed; border-color: #ddd; }}
          @media (max-width: 480px) {{
            .msg {{ padding: .5rem .6rem; }}
            .button-group {{ flex-direction: column; }}
            .btn {{ width: 100%; }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav"><a href="/">← All threads</a></div>
          <div id="titleView" style="display:flex; align-items:center; gap:.3rem;">
            <h2 style="font-size:1.2rem; margin:0">{html.escape(title)}</h2>
            {rename_button}
          </div>
          {rename_form}
          {_thread_domain_html(tid)}
          {status_banner}
          {"<div class='success-msg'>Conversation capture started! This will complete in the background.</div>" if captured else ""}
          {"<div class='success-msg'>Branch successfully merged to main!</div>" if merged else ""}
          {"<div class='success-msg'>Review submitted. The agent will respond in this thread.</div>" if reviewed else ""}
          {"<div class='success-msg'>Pushed local main to origin/main.</div>" if pushed else ""}
          {conflict_banner_html}
          {f'<script>try {{ localStorage.removeItem("assist:review:" + {json.dumps(tid)}); }} catch (_) {{}}</script>' if reviewed else ""}
          <form action="/thread/{tid}/message" method="post">
            <label for="text">Your message</label><br/>
            <textarea id="text" name="text" required placeholder="Type your message..." {form_disabled}></textarea><br/>
            <div class="button-group">
              <button class="btn" type="submit" {form_disabled}>Send</button>
              <button class="btn btn-secondary" type="button" onclick="showCaptureModal()" {form_disabled}>Capture Conversation</button>
            </div>
            <div style="font-size:.85rem; color:#666; margin-top:.4rem;">{form_note}</div>
          </form>

          <!-- Capture Modal -->
          <div id="captureModal" class="modal">
            <div class="modal-content">
              <h3>Capture Conversation</h3>
              <p>Save this conversation for future testing and replay.</p>
              <form action="/thread/{tid}/capture" method="post">
                <label for="reason">Why are you capturing this conversation?</label>
                <textarea id="reason" name="reason" required placeholder="e.g., Good example of authentication bug handling"></textarea>
                <div class="button-group">
                  <button class="btn" type="submit">Save</button>
                  <button class="btn btn-secondary" type="button" onclick="hideCaptureModal()">Cancel</button>
                </div>
              </form>
            </div>
          </div>

          <script>
            function showRename() {{
              document.getElementById('titleView').style.display = 'none';
              const f = document.getElementById('titleEdit');
              f.style.display = 'flex';
              const inp = f.querySelector('input[name=description]');
              inp.focus(); inp.select();
            }}
            function hideRename() {{
              document.getElementById('titleEdit').style.display = 'none';
              document.getElementById('titleView').style.display = 'flex';
            }}
            function showCaptureModal() {{
              document.getElementById('captureModal').style.display = 'block';
            }}
            function hideCaptureModal() {{
              document.getElementById('captureModal').style.display = 'none';
            }}
            // Close modal when clicking outside
            window.onclick = function(event) {{
              const modal = document.getElementById('captureModal');
              if (event.target == modal) {{
                hideCaptureModal();
              }}
            }}
          </script>
          <hr/>
          {diff_block_html}
          <div>
            {body}
          </div>
        </div>
      </body>
    </html>
    """


def _initialize_thread(tid: str, text: str, domain: str | None) -> None:
    """Background task: clone the repo, start sandbox, process the first message."""
    try:
        if domain:
            _set_status(tid, "cloning", pending_message=text, domain=domain)
            try:
                dm = DomainManager(
                    MANAGER.thread_default_working_dir(tid),
                    domain,
                    branch_suffix=tid[-4:],
                )
                # Refresh cache: a previous render may have cached a no-remote DM.
                DOMAIN_MANAGERS[tid] = dm
            except Exception as e:
                logging.error("Clone failed for thread %s: %s", tid, e, exc_info=True)
                _set_status(tid, "error", error=f"Clone failed: {e}", pending_message=text)
                return
        _process_message(tid, text)
    except Exception as e:
        logging.error("Initialization failed for thread %s: %s", tid, e, exc_info=True)
        _set_status(tid, "error", error=str(e), pending_message=text)


def _process_message(tid: str, text: str) -> None:
    # Carry the pending message in the status so the thread page can show
    # it as a placeholder bubble while processing (cleared once status==ready).
    pending_kwargs = {"pending_message": text}

    def on_queue_wait(stage: str) -> None:
        # `ThreadAffinityQueue.acquire` fires the callback with "queued"
        # (when this thread has to wait) and then "running" (when it
        # acquires).  We only HANDLE "queued" here — the post-acquire
        # flow below sets the more specific "starting_sandbox" →
        # "processing" statuses itself, so the "running" callback is
        # intentionally ignored.
        if stage == "queued":
            _set_status(tid, "queued", **pending_kwargs)

    try:
        # Acquire the queue BEFORE starting the sandbox.  We create a fresh
        # container per turn and tear it down when the turn ends; creating it
        # only after acquiring the queue means it never ages against the 3h
        # backstop TTL (sleep 10800 in Dockerfile.sandbox) while waiting in
        # line (behind a holder past its hold_timeout_s, or many backlogged
        # threads).  Observed pre-defer on 2026-05-30 thread
        # 20260530160651-fee1ddc5: sandbox started at 16:06:52, sat queued
        # behind a 2-hour-wedged thread, then 404'd 1h45m later.
        #
        # `chat.message()` below tries to acquire the queue again
        # internally; the reentrant fast path (same thread_id + same
        # contextvar) makes it a no-op, so we don't double-count or
        # double-callback.
        with THREAD_QUEUE.acquire(tid, on_state_change=on_queue_wait):
            _set_status(tid, "starting_sandbox", **pending_kwargs)
            try:
                # Inside the try so the `finally` reaps even if sandbox
                # creation registers a container and then raises — cleanup
                # keys on work_dir, not on the `sandbox` handle.
                sandbox = _get_sandbox_backend(tid)
                try:
                    # on_queue_state=None: the outer acquire above already
                    # owns the callback; the inner acquire is the reentrant
                    # no-op fast path (no state callback fires from it).
                    chat = MANAGER.get(tid, sandbox_backend=sandbox,
                                       on_queue_state=None)
                except FileNotFoundError:
                    return
                _set_status(tid, "processing", **pending_kwargs)
                resp = chat.message(text)
            finally:
                # One container per turn: kill it as soon as this turn's agent
                # run finishes — success, error, or the early return above —
                # while we still hold the queue, so the next turn always starts
                # in a fresh sandbox and no container outlives its turn.  This,
                # plus the >2h backstop TTL, is what makes the mid-flight reap
                # impossible: container age == turn age, capped by the queue.
                # cleanup() SIGKILLs (the response is already committed to the
                # checkpoint here, and the sandbox has nothing to flush).
                SandboxManager.cleanup(MANAGER.thread_default_working_dir(tid))
        MANAGER.touch(tid)

        # Generate description if there is none
        try:
            DESCRIPTION_CACHE.pop(tid, None)
            get_cached_description(tid)
        except Exception as e:
            logging.warning("Description generation failed for %s: %s", tid, e)

        # After message, sync changes if any
        dm = _get_domain_manager(tid)
        if dm and dm.changes():
            last_assistant = resp if resp else "assistant update"
            dm.sync(last_assistant)
        _set_status(tid, "ready")
    except SandboxContainerLostError as e:
        # Distinct status message: a dead container is recoverable —
        # the user can simply retry — but they should know their
        # previous turn's work didn't land.  Without this branch the
        # generic except below shows a raw exception repr to the user.
        logging.error("Sandbox lost for thread %s: %s", tid, e)
        # The per-turn teardown (the `finally` above) already reaped this
        # turn's container; just drop the cached domain manager so a retry
        # re-checks cleanly instead of poking at the corpse of the old one.
        DOMAIN_MANAGERS.pop(tid, None)
        _set_status(
            tid, "error",
            error=("The sandbox container for this thread was lost mid-run. "
                   "Your last message was not completed. Send the message "
                   "again to retry in a fresh sandbox."),
            **pending_kwargs,
        )
    except Exception as e:
        logging.error("Message processing failed for thread %s: %s", tid, e, exc_info=True)
        _set_status(tid, "error", error=str(e), **pending_kwargs)


def _capture_conversation(tid: str, reason: str) -> None:
    """Background task to capture a conversation."""
    try:
        thread = MANAGER.get(tid)
    except FileNotFoundError:
        logging.error(f"Thread {tid} not found for capture")
        return

    # Get repo root (navigate up from manage/web/threads.py to repo root)
    current_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    improvements_dir = os.path.join(repo_root, "improvements")

    from edd.capture import capture_conversation
    try:
        capture_path = capture_conversation(thread, reason, improvements_dir)
        logging.info(f"Conversation captured successfully to {capture_path}")
    except Exception as e:
        logging.error(f"Failed to capture conversation for thread {tid}: {e}", exc_info=True)


# --- Routes -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(q: str = "") -> str:
    return render_index(q)


@app.post("/threads")
async def create_thread(domain: str | None = Form(None)):
    chat = MANAGER.new()
    tid = chat.thread_id
    selected = domain or (DOMAINS[0] if DOMAINS else None)
    if selected:
        DomainManager(
            MANAGER.thread_default_working_dir(tid),
            selected,
            branch_suffix=tid[-4:],
        )
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


@app.post("/threads/with-message")
async def create_thread_with_message(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    domain: str | None = Form(None),
):
    # Reserve the thread directory synchronously so the redirect target is valid,
    # but defer everything slow (clone, sandbox, agent, description) to the background.
    chat = MANAGER.new()
    tid = chat.thread_id
    selected = domain or (DOMAINS[0] if DOMAINS else None)
    _set_status(tid, "initializing", pending_message=text, domain=selected or "")
    background_tasks.add_task(_initialize_thread, tid, text, selected)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


@app.get("/thread/{tid}", response_class=HTMLResponse)
async def get_thread(
    tid: str,
    captured: int = 0,
    merged: int = 0,
    reviewed: int = 0,
    pushed: int = 0,
) -> str:
    tdir = MANAGER.thread_dir(tid)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Thread not found")

    stage = _get_status(tid).get("stage", "ready")
    # During the initial setup stages there is no point constructing a Thread
    # (which would also race with the background task starting the sandbox).
    chat: Thread | None = None
    if stage not in INIT_STAGES:
        # Skip the sandbox during plain renders; it gets started by the
        # background task when a message is being processed.
        try:
            chat = MANAGER.get(tid, sandbox_backend=None)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Thread not found")
    return render_thread(
        tid, chat,
        captured=bool(captured),
        merged=bool(merged),
        reviewed=bool(reviewed),
        pushed=bool(pushed),
    )


@app.get("/thread/{tid}/status")
async def thread_status(tid: str):
    if not os.path.isdir(MANAGER.thread_dir(tid)):
        raise HTTPException(status_code=404, detail="Thread not found")
    return JSONResponse(_get_status(tid))


def _mark_pending(tid: str, text: str) -> None:
    """Record an inbound message as busy+pending *synchronously*, before the
    POST handler returns its redirect.

    The thread page has no client-side polling: all in-flight feedback (the
    pending-message bubble and the status banner) is gated on the thread's
    status being a ``BUSY_STAGES`` value, and the page renders only once — on
    the redirect-GET that follows the POST.  If the first status write is left
    to the background ``_process_message`` task, that write races the redirect
    render; under load (every queued thread parks a threadpool worker in
    ``cond.wait``) the task loses the race, the page renders the stale
    ``ready`` status, and the message silently vanishes from the UI with no
    feedback.

    Writing the status here — the way ``create_thread_with_message`` already
    does for new threads — closes the race.  ``_process_message`` then refines
    the stage as it runs.

    The initial stage is "queued" when another thread currently holds the LLM
    slot, else "processing".  Both are deliberately NON-``INIT_STAGES`` values:
    an INIT stage (e.g. "starting_sandbox") would make ``get_thread`` render
    this existing thread as ``is_init`` — hiding its history and disabling the
    input — which is wrong for a thread that's just received a follow-up
    message.

    No-op when this thread is already busy, so a second message to a mid-turn
    thread doesn't clobber the in-flight turn's status.

    Runs on the asyncio event-loop thread, so it must never block: it uses
    ``THREAD_QUEUE.peek_holder()`` (a lock-free read), NOT ``current_handle()``
    — taking the queue's condition lock here would couple the event loop to
    the queue and freeze the whole server whenever that lock is held by a
    long-running turn.
    """
    if _get_status(tid).get("stage") in BUSY_STAGES:
        return
    holder_tid = THREAD_QUEUE.peek_holder()
    stage = "queued" if (holder_tid is not None and holder_tid != tid) else "processing"
    _set_status(tid, stage, pending_message=text)


@app.post("/thread/{tid}/message")
async def post_message(tid: str, background_tasks: BackgroundTasks, text: str = Form(...)):
    _existing_thread_dir(tid)  # validates tid (404 on traversal/NUL) + existence
    _mark_pending(tid, text)
    background_tasks.add_task(_process_message, tid, text)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


def _existing_thread_dir(tid: str) -> str:
    """Return the thread's dir, or 404 if it doesn't exist. ``MANAGER.thread_dir``
    validates the tid (a traversal/separator id raises InvalidThreadId, mapped to
    404 by the handler above), so this only adds the existence check."""
    tdir = MANAGER.thread_dir(tid)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Thread not found")
    return tdir


# ---- show_file: render a workspace file (org/md/pdf) in the web UI ----

_SHOW_PAGE_CSS = (
    "body{font-family:sans-serif;max-width:780px;margin:1rem auto;padding:0 1rem;"
    "line-height:1.55;color:#222}pre,code{background:#f5f5f5;border-radius:4px}"
    "pre{padding:.6rem;overflow:auto}table{border-collapse:collapse}"
    "td,th{border:1px solid #ccc;padding:.3rem .5rem}img{max-width:100%}"
)


def _safe_workspace_file(tid: str, path: str) -> str | None:
    """Resolve PATH against the thread's agent working dir, traversal-safe.
    Returns the absolute host path, or None if it would escape the workspace
    (or is malformed, e.g. an embedded NUL).  Same realpath-child check as the
    tid guard — a crafted ``../`` can't read outside the agent's own files."""
    base = os.path.realpath(MANAGER.thread_default_working_dir(tid))
    try:
        target = os.path.realpath(os.path.join(base, path))
    except ValueError:  # embedded NUL etc. -> treat as not-found, not a 500
        return None
    if target != base and not target.startswith(base + os.sep):
        return None
    return target


# Org files are AGENT-generated (possibly from fetched web content), so they are
# rendered by this pure, escape-first converter — NEVER by emacs/org-export,
# which executes elisp during export (babel, AND #+MACRO: (eval ...), #+CALL:,
# table formulas) and is therefore a host-RCE vector on untrusted org.  Here
# every character of the file is html-escaped first; the only HTML emitted is
# this function's own tags, so no markup (or eval) in the file can take effect.
# Covers the common constructs (headings, lists, tables, src/example blocks,
# inline emphasis, links); richer org degrades to readable text.
_ORG_LIST_RE = re.compile(r"\s*([-+]|\d+[.)])\s+(.*)")
_ORG_HEADING_RE = re.compile(r"(\*+)\s+(.*)")
# One combined inline pattern, applied in a SINGLE left-to-right pass so a
# substitution's output (e.g. the "/" in an inserted "</b>") is never re-scanned
# by a later rule.  Order in the alternation = match priority.
_ORG_INLINE_RE = re.compile(
    r"\[\[(?P<lt>[^\]]+?)\](?:\[(?P<ll>[^\]]*?)\])?\]"      # [[link]] / [[link][label]]
    r"|(?<![\w*])\*(?P<b>\S(?:.*?\S)?)\*(?![\w*])"           # *bold*
    r"|(?<![\w/])/(?P<i>\S(?:.*?\S)?)/(?![\w/])"             # /italic/
    r"|(?<![\w=])=(?P<c>\S(?:.*?\S)?)=(?![\w=])"             # =code=
    r"|(?<![\w~])~(?P<v>\S(?:.*?\S)?)~(?![\w~])"             # ~verbatim~
)


def _org_inline_sub(m: re.Match) -> str:
    if m.group("lt") is not None:
        target = m.group("lt")
        label = m.group("ll") if m.group("ll") else target
        scheme_ok = re.match(r"(https?:|mailto:|/|\.|#)", html.unescape(target))
        href = target if scheme_ok else "#"
        return f'<a href="{href}" target="_blank" rel="noopener">{label}</a>'
    if m.group("b") is not None:
        return f"<b>{m.group('b')}</b>"
    if m.group("i") is not None:
        return f"<i>{m.group('i')}</i>"
    code = m.group("c") if m.group("c") is not None else m.group("v")
    return f"<code>{code}</code>"


def _org_inline(text: str) -> str:
    """Escape TEXT (so file content can't inject HTML), then apply org inline
    markup in one pass over the escaped string."""
    return _ORG_INLINE_RE.sub(_org_inline_sub, html.escape(text))


def _org_table(rows: list[str]) -> str:
    out = ["<table>"]
    for r in rows:
        if set(r) <= set("|-+ "):   # separator row
            continue
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        out.append("<tr>" + "".join(f"<td>{_org_inline(c)}</td>" for c in cells) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _org_to_html(src: str) -> str:
    """Render an org SOURCE STRING to body HTML, safely (see the note above)."""
    lines = src.splitlines()
    parts: list[str] = []
    para: list[str] = []
    open_list: str | None = None

    def flush_para():
        if para:
            parts.append("<p>" + _org_inline(" ".join(para)) + "</p>")
            para.clear()

    def close_list():
        nonlocal open_list
        if open_list:
            parts.append(f"</{open_list}>")
            open_list = None

    i = 0
    while i < len(lines):
        line, stripped = lines[i], lines[i].strip()
        if re.match(r"#\+BEGIN_(SRC|EXAMPLE)", stripped, re.I):
            flush_para(); close_list()
            block, i = [], i + 1
            while i < len(lines) and not re.match(r"#\+END_", lines[i].strip(), re.I):
                block.append(lines[i]); i += 1
            i += 1  # skip the END line
            parts.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            continue
        if stripped.startswith("#+") or stripped.startswith("# "):
            flush_para(); i += 1; continue  # keyword/comment line: drop (never eval)
        hm = _ORG_HEADING_RE.match(line)
        if hm:
            flush_para(); close_list()
            lvl = min(len(hm.group(1)), 6)
            parts.append(f"<h{lvl}>{_org_inline(hm.group(2).strip())}</h{lvl}>")
            i += 1; continue
        if stripped.startswith("|"):
            flush_para(); close_list()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            parts.append(_org_table(rows))
            continue
        lm = _ORG_LIST_RE.fullmatch(line)
        if lm:
            flush_para()
            tag = "ol" if lm.group(1)[0].isdigit() else "ul"
            if open_list != tag:
                close_list(); parts.append(f"<{tag}>"); open_list = tag
            parts.append(f"<li>{_org_inline(lm.group(2).strip())}</li>")
            i += 1; continue
        if not stripped:
            flush_para(); close_list(); i += 1; continue
        para.append(stripped); i += 1
    flush_para(); close_list()
    return "\n".join(parts)


def _render_show_file(tid: str, path: str) -> str:
    """The transcript bubble for a show_file call: the file embedded inline
    (pdf in the browser viewer, md/org as a rendered HTML page in an iframe),
    plus a caption link that opens it in its own tab."""
    if not path:
        return ""
    src = f"/thread/{tid}/show?path={urllib.parse.quote(path)}"
    ext = os.path.splitext(path)[1].lower()
    label = html.escape(path)
    if ext == ".pdf":
        viewer = f'<embed class="show-file" type="application/pdf" src="{src}" />'
    else:
        # sandbox WITHOUT allow-scripts: the embedded md/org page is static, so
        # any <script>/onerror in agent-generated content can't execute (defence
        # in depth over the org renderer's escaping — and it covers the md path,
        # whose markdown lib passes raw HTML through).  allow-popups keeps
        # target=_blank links in the content working.
        viewer = (f'<iframe class="show-file" src="{src}" loading="lazy" '
                  f'sandbox="allow-popups"></iframe>')
    return (
        '<div class="msg show"><div class="role">shown</div>'
        f'<div class="content">{viewer}'
        f'<div class="show-cap"><a href="{src}" target="_blank" rel="noopener">'
        f'{label} ↗</a></div></div></div>'
    )


@app.get("/thread/{tid}/show")
async def show_file_view(tid: str, path: str):
    """Render a file from the thread's agent workspace for embedding: pdf as
    bytes (browser viewer), md/org as a styled HTML page.  Pure renderers (no
    subprocess), so safe to run inline on the event loop."""
    _existing_thread_dir(tid)  # 404 on a bad/missing tid (traversal-safe)
    fpath = _safe_workspace_file(tid, path)
    if fpath is None or not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail="file not found")
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".pdf":
        return FileResponse(fpath, media_type="application/pdf")
    with open(fpath, encoding="utf-8", errors="replace") as f:
        src = f.read()
    if ext == ".md":
        body = markdown.markdown(src, extensions=_MD_EXTENSIONS)
    elif ext == ".org":
        body = _org_to_html(src)
    else:
        raise HTTPException(status_code=415,
                            detail="show supports .org, .md, .pdf")
    return HTMLResponse(
        f"<!doctype html><html><head><meta charset=utf-8>"
        f'<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<style>{_SHOW_PAGE_CSS}</style></head><body>{body}</body></html>")


@app.post("/thread/{tid}/delete")
async def delete_thread(tid: str):
    _existing_thread_dir(tid)
    MANAGER.hard_delete(tid, on_delete=[_evict_caches])
    return RedirectResponse(url="/", status_code=303)


@app.post("/thread/{tid}/rename")
async def rename_thread(tid: str, description: str = Form("")):
    _existing_thread_dir(tid)
    new = description.strip()[:120]
    if new:  # ignore an empty rename — keep the existing title
        set_description(tid, new)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


@app.post("/thread/{tid}/capture")
async def capture_thread(tid: str, background_tasks: BackgroundTasks, reason: str = Form(...)):
    try:
        thread = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Validate thread has messages before queuing
    try:
        messages = thread.get_messages()
        if not messages:
            raise HTTPException(status_code=400, detail="Cannot capture empty conversation")
    except Exception:
        pass  # Let the background task handle it

    # Queue the capture as a background task
    background_tasks.add_task(_capture_conversation, tid, reason)

    # Return immediately
    return RedirectResponse(
        url=f"/thread/{tid}?captured=1",
        status_code=303
    )


@app.post("/thread/{tid}/merge")
async def merge_thread(tid: str):
    """Rebase the thread branch onto origin/main and squash into local main.

    Holds ``MERGE_LOCK`` for the duration so two web requests merging or
    pushing at the same instant don't race the host's git operations.
    Persists a ``merge_conflict.json`` marker on rebase conflict so the
    UI can render a banner across subsequent renders; clears the marker
    on a clean merge.

    Refuses with 409 when the thread is mid-turn — the agent inside
    the sandbox is concurrently writing into the same working tree,
    and the lock doesn't extend across the host/sandbox boundary.
    """
    try:
        thread = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")

    if _get_status(tid).get("stage") in BUSY_STAGES:
        raise HTTPException(
            status_code=409,
            detail="Thread is busy. Wait for the current turn to finish before merging.",
        )

    dm = _get_domain_manager(tid)
    if not dm or not dm.repo:
        raise HTTPException(status_code=400, detail="No git repository configured for this thread")

    # Get a model for summarizing
    from assist.model_manager import select_assistant_model
    try:
        summary_model = select_assistant_model(temperature=0.1)
    except Exception:
        # If model fails to load, pass None and use fallback summary
        summary_model = None

    with MERGE_LOCK:
        try:
            dm.merge_to_main(summary_model)
            _clear_conflict(tid)
            return RedirectResponse(
                url=f"/thread/{tid}?merged=1",
                status_code=303,
            )
        except MergeConflictError as e:
            _set_conflict(tid, e.branch, e.files)
            return RedirectResponse(
                url=f"/thread/{tid}?conflict=1",
                status_code=303,
            )
        except ValueError as e:
            # User-friendly error (already on main, no changes, unpushed local main).
            raise HTTPException(status_code=400, detail=str(e))
        except subprocess.CalledProcessError as e:
            # Git command failed
            raise HTTPException(status_code=500, detail=f"Git operation failed: {e}")
        except Exception as e:
            # Unexpected error
            logging.error(f"Merge failed for thread {tid}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")


@app.post("/thread/{tid}/push-main")
async def push_main(tid: str):
    """Fast-forward push local ``main`` to ``origin/main``.

    User-initiated only — the agent has no way to reach this endpoint.
    Holds ``MERGE_LOCK`` so a concurrent merge can't slip in between
    the fetch and the push.  Returns 409 when ``origin/main`` has
    advanced past local ``main`` so the user knows to re-merge.
    """
    try:
        MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")

    if _get_status(tid).get("stage") in BUSY_STAGES:
        raise HTTPException(
            status_code=409,
            detail="Thread is busy. Wait for the current turn to finish before pushing.",
        )

    dm = _get_domain_manager(tid)
    if not dm or not dm.repo:
        raise HTTPException(status_code=400, detail="No git repository configured for this thread")

    with MERGE_LOCK:
        try:
            dm.push_main()
            return RedirectResponse(
                url=f"/thread/{tid}?pushed=1",
                status_code=303,
            )
        except OriginAdvancedError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Git operation failed: {e}")
        except Exception as e:
            logging.error(f"Push failed for thread {tid}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Push failed: {str(e)}")
