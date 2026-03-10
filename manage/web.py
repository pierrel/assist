import logging, sys
import html
import os
import subprocess
import urllib.parse
from typing import Dict

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks, Query
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, RedirectResponse

from assist.env import load_dev_env
from assist.thread import Thread, ThreadManager
import markdown
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import HtmlFormatter
from .custom_diff_formatter import CustomHtmlFormatter
from assist.domain_manager import DomainManager
from assist.sandbox_manager import SandboxManager

# debug logging by default
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("assist.model").setLevel(logging.DEBUG)

load_dev_env()

ROOT = os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads")
MANAGER = ThreadManager(ROOT)
_raw = os.getenv("ASSIST_DOMAINS", "")
DOMAINS: list[str] = [d.strip() for d in _raw.split(",") if d.strip()]
DESCRIPTION_CACHE: Dict[str, str] = {}
DOMAIN_MANAGERS: Dict[str, DomainManager] = {}  # tid -> DomainManager


def _domain_label(url: str) -> str:
    """'user@host:/path/to/life.git' -> 'life'"""
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def _domain_selector_html() -> str:
    """Return HTML for the domain selector in the new-thread form."""
    if len(DOMAINS) > 1:
        opts = "\n".join(
            f'<option value="{html.escape(d)}">{html.escape(_domain_label(d))}</option>'
            for d in DOMAINS
        )
        return (
            '<select name="domain" style="margin-bottom:.5rem; padding:.4rem; '
            'border:1px solid #ccc; border-radius:6px; font-size:1rem; width:100%;">'
            f"{opts}</select>"
        )
    if len(DOMAINS) == 1:
        return f'<input type="hidden" name="domain" value="{html.escape(DOMAINS[0])}" />'
    return ""


def _thread_domain_html(tid: str) -> str:
    """Return a small badge showing the domain name for a thread, if any."""
    dm = _get_domain_manager(tid)
    if dm and dm.repo:
        label = html.escape(_domain_label(dm.repo))
        return (
            f'<span style="display:inline-block; font-size:.8rem; color:#555; '
            f'background:#f0f0f0; padding:.2rem .5rem; border-radius:4px; '
            f'margin-bottom:.5rem;">{label}</span>'
        )
    return ""


def _get_domain_manager(tid: str, domain: str | None = None) -> DomainManager | None:
    """Get or create a DomainManager for a thread, caching by tid.

    For new threads pass *domain* (a git URL to clone).
    For existing threads pass None — DomainManager auto-detects the remote.
    """
    if tid in DOMAIN_MANAGERS:
        return DOMAIN_MANAGERS[tid]
    twdir = MANAGER.thread_default_working_dir(tid)
    try:
        dm = DomainManager(twdir, domain)
        DOMAIN_MANAGERS[tid] = dm
        return dm
    except Exception:
        return None


def _get_sandbox_backend(tid: str):
    """Get sandbox backend for a thread, or None if Docker is unavailable."""
    work_dir = MANAGER.thread_default_working_dir(tid)
    return SandboxManager.get_sandbox_backend(work_dir)

def get_cached_description(tid: str) -> str:
    """Get thread description from cache, or read from FS and cache if miss."""
    if tid in DESCRIPTION_CACHE:
        return DESCRIPTION_CACHE[tid]

    # Cache miss - read from FS or thread and cache
    try:
        chat = MANAGER.get(tid)
        thread_dir = MANAGER.thread_dir(tid)
        description_file = os.path.join(thread_dir,
                                        "description.txt")
        if os.path.isfile(description_file):
            with open(description_file, 'r') as f:
                description = f.read()
        else:
            description = chat.description()
            os.makedirs(os.path.dirname(description_file), exist_ok=True)
            with open(description_file, 'w') as f:
                f.write(description)
        DESCRIPTION_CACHE[tid] = description
        return description
    except Exception:
        return tid

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure thread root exists at startup
    os.makedirs(ROOT, exist_ok=True)

    # Populate description cache at startup
    for tid in MANAGER.list():
        get_cached_description(tid)
    try:
        yield
    finally:
        # Clean up Docker sandbox containers
        try:
            SandboxManager.cleanup_all()
        except Exception:
            pass
        # Close shared resources (e.g., sqlite connection) to avoid leaks
        try:
            MANAGER.close()
        except Exception:
            pass

app = FastAPI(title="Assist Web", lifespan=lifespan)

def render_diff(text: str) -> str:
    # Use Pygments to render unified diffs with HTML formatting
    formatter = CustomHtmlFormatter(nowrap=False)
return highlight(text, DiffLexer(), formatter)

def render_index() -> str:
    items = []
    tids = MANAGER.list()
    if not tids:
        items.append("<li><em>No threads yet</em></li>")
    else:
        for tid in tids:
            title = get_cached_description(tid)
            items.append(
                f'<li style="display:flex; align-items:center; gap:.5rem;">'
                f'<a href="/thread/{tid}" style="flex:1">{html.escape(title)}</a>'
                f'<form action="/thread/{tid}/delete" method="post" style="margin:0">'
                f'<button type="submit" class="del-btn" '
                f'onclick="return confirm(\'Delete this thread?\')">&#x2715;</button>'
                f'</form></li>'
            )
    items_html = "\n".join(items)
    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Assist Web</title>
        <style>
          :root {{ --pad: 1rem; }}
          body {{ font-family: sans-serif; margin: 0; }}
          .container {{ max-width: 800px; margin: 0 auto; padding: var(--pad); }}
          .topbar {{ display: flex; gap: .5rem; flex-wrap: wrap; justify-content: space-between; align-items: center; }}
          ul {{ line-height: 1.8; padding-left: 1rem; list-style: none; }}
          li {{ margin: .2rem 0; }}
          a {{ text-decoration: none; display: block; padding: .5rem .6rem; border-radius: 6px; }}
          .del-btn {{ background: none; border: none; color: #999; cursor: pointer; font-size: 1.1rem; padding: .2rem .4rem; border-radius: 4px; }}
          .del-btn:hover {{ color: #c00; background: #fee; }}
          a:active, a:focus {{ outline: none; }}
          .btn {{ padding: .6rem 1rem; border: 1px solid #333; border-radius: 8px; background: #eee; font-size: 1rem; cursor: pointer; }}
          .new-thread-form {{ margin-bottom: 1.5rem; padding: 1rem; background: #f9f9f9; border-radius: 8px; border: 1px solid #ddd; }}
          .new-thread-form textarea {{ width: 100%; min-height: 4rem; box-sizing: border-box; padding: .6rem; border: 1px solid #ccc; border-radius: 6px; font-family: inherit; font-size: 1rem; resize: vertical; }}
          .new-thread-form textarea:focus {{ outline: 2px solid #4a90e2; border-color: #4a90e2; }}
          .new-thread-btn {{ margin-top: .5rem; display: none; }}
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
            <a href="/evals" class="btn" style="font-size:.9rem; padding:.4rem .8rem">Evals</a>
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


def render_thread(tid: str, chat: Thread, captured: bool = False, merged: bool = False) -> str:
    title = get_cached_description(tid)
    msgs = chat.get_messages()

    # Append diffs from domain repo (computed at render time)
    try:
        dm = _get_domain_manager(tid)
        if dm:
            diffs = dm.main_diff()
            if diffs:
                diff_content = "\n".join([f"{c.path}\n{c.diff}\n" for c in diffs])
                msgs.append({"role": "diff", "content": diff_content})
    except Exception:
        pass

    rendered = []
    diff_counter = 0
    for m in reversed(msgs):
        role = html.escape(m.get("role", ""))
        raw = str(m.get("content", ""))
        if role == "diff":
            # Render diffs using Pygments for proper coloring/formatting
            # But wrap in collapsible container that's hidden by default
            diff_counter += 1
            diff_id = f"diff-{diff_counter}"
            diff_content = render_diff(raw)
            content_html = f"""
            <div class="diff-container">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                    <button class="diff-toggle" onclick="toggleDiff('{diff_id}')" style="flex: 1; min-width: 200px;">
                        <span class="toggle-icon">▶</span> Show diff
                    </button>
                    <form action="/thread/{tid}/merge" method="post" style="margin: 0;">
                        <button class="btn merge-btn" type="submit" onclick="return confirm('Merge this branch into main? This will squash all commits.');">
                            Merge to Main
                        </button>
                    </form>
                </div>
                <div id="{diff_id}" class="diff-content" style="display: none;">
                    {diff_content}
                </div>
            </div>
            """
        elif role == "assistant" or role == "tools":
            # Render assistant and tool content as Markdown to HTML
            content_html = markdown.markdown(raw, extensions=["fenced_code", "tables"])
        else:
            # Human/user content is plain text with basic escaping
            content_html = html.escape(raw).replace("\n", "<br/>")
        cls = "user" if role == "user" else ("tools" if role == "tools" else "assistant")
        bubble = f"<div class=\"msg {cls}\"><div class=\"role\">{role}</div><div class=\"content\">{content_html}</div></div>"
        rendered.append(bubble)
    body = "\n".join(rendered) or "<p><em>No messages yet.</em></p>"
    return f"""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>{html.escape(title)}</title>
        <style>
          :root {{ --pad: 1rem; }}
          body {{ font-family: sans-serif; margin: 0; }}
          .container {{ max-width: 800px; margin: 0 auto; padding: var(--pad); }}
          .nav a {{ display: inline-block; padding: .4rem .6rem; border-radius: 6px; text-decoration: none; }}
          .msg {{ margin: .6rem 0; padding: .6rem .8rem; border-radius: 8px; max-width: 100%; word-wrap: break-word; overflow-wrap: anywhere; }}
          .msg.user {{ background: #e6f3ff; border: 1px solid #b5dbff; }}
          .msg.assistant {{ background: #f6f6f6; border: 1px solid #ddd; }}
          .role {{ font-size: .8rem; color: #555; margin-bottom: .2rem; text-transform: uppercase; }}
          form textarea {{ width: 100%; min-height: 6rem; height: 24vh; box-sizing: border-box; }}
          form {{ margin-top: 1rem; }}
          .btn {{ padding: .6rem 1rem; border: 1px solid #333; border-radius: 8px; background: #eee; font-size: 1rem; cursor: pointer; }}
          .btn-secondary {{ background: #ddd; }}
          .success-msg {{ background: #d4edda; border: 1px solid #c3e6cb; padding: .8rem; margin: .5rem 0; border-radius: 6px; color: #155724; }}
          .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.4); }}
          .modal-content {{ background: #fff; margin: 10% auto; padding: 1.5rem; width: 90%; max-width: 500px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
          .modal-content h3 {{ margin-top: 0; }}
          .modal-content textarea {{ width: 100%; min-height: 100px; padding: .5rem; border: 1px solid #ccc; border-radius: 4px; font-family: inherit; }}
          .modal-content label {{ display: block; margin-bottom: .5rem; font-weight: 500; }}
          .button-group {{ display: flex; gap: .5rem; margin-top: 1rem; }}
          .diff-container {{ margin: .5rem 0; }}
          .diff-toggle {{ background: #f8f9fa; border: 1px solid #dee2e6; padding: .5rem .75rem; border-radius: 6px; cursor: pointer; font-size: .9rem; width: 100%; text-align: left; display: flex; align-items: center; gap: .5rem; transition: background .2s; }}
          .diff-toggle:hover {{ background: #e9ecef; }}
          .toggle-icon {{ display: inline-block; transition: transform .2s; font-size: .8rem; }}
          .toggle-icon.expanded {{ transform: rotate(90deg); }}
          .diff-content {{ margin-top: .5rem; overflow-x: auto; }}
          .merge-btn {{ background: #28a745; color: white; border: 1px solid #1e7e34; padding: .5rem .75rem; font-size: .9rem; white-space: nowrap; }}
          .merge-btn:hover {{ background: #218838; border-color: #1c7430; }}
          .error-msg {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: .8rem; margin: .5rem 0; border-radius: 6px; color: #721c24; }}
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
          <h2 style="font-size:1.2rem">{html.escape(title)}</h2>
          {_thread_domain_html(tid)}
          {"<div class='success-msg'>Conversation capture started! This will complete in the background.</div>" if captured else ""}
          {"<div class='success-msg'>Branch successfully merged to main!</div>" if merged else ""}
          <form action="/thread/{tid}/message" method="post">
            <label for="text">Your message</label><br/>
            <textarea id="text" name="text" required placeholder="Type your message..."></textarea><br/>
            <div class="button-group">
              <button class="btn" type="submit">Send</button>
              <button class="btn btn-secondary" type="button" onclick="showCaptureModal()">Capture Conversation</button>
            </div>
            <div style="font-size:.85rem; color:#666; margin-top:.4rem;">If you close or refresh, your message will still be processed.</div>
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
            function showCaptureModal() {{
              document.getElementById('captureModal').style.display = 'block';
            }}
            function hideCaptureModal() {{
              document.getElementById('captureModal').style.display = 'none';
            }}
            function toggleDiff(diffId) {{
              const diffContent = document.getElementById(diffId);
              const toggleButton = event.currentTarget;
              const toggleIcon = toggleButton.querySelector('.toggle-icon');

              if (diffContent.style.display === 'none') {{
                diffContent.style.display = 'block';
                toggleIcon.classList.add('expanded');
                toggleButton.innerHTML = toggleButton.innerHTML.replace('Show diff', 'Hide diff');
              }} else {{
                diffContent.style.display = 'none';
                toggleIcon.classList.remove('expanded');
                toggleButton.innerHTML = toggleButton.innerHTML.replace('Hide diff', 'Show diff');
              }}
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
          <div>
            {body}
          </div>
        </div>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return render_index()


@app.post("/threads")
async def create_thread(domain: str | None = Form(None)):
    chat = MANAGER.new()
    tid = chat.thread_id
    selected = domain or (DOMAINS[0] if DOMAINS else None)
    if selected:
        DomainManager(MANAGER.thread_default_working_dir(tid), selected)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


@app.post("/threads/with-message")
async def create_thread_with_message(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    domain: str | None = Form(None),
):
    chat = MANAGER.new()
    tid = chat.thread_id
    selected = domain or (DOMAINS[0] if DOMAINS else None)
    if selected:
        DomainManager(MANAGER.thread_default_working_dir(tid), selected)
    background_tasks.add_task(_process_message, tid, text)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


def _process_message(tid: str, text: str) -> None:
    sandbox = _get_sandbox_backend(tid)
    try:
        chat = MANAGER.get(tid, sandbox_backend=sandbox)
    except FileNotFoundError:
        return
    resp = chat.message(text)
    MANAGER.touch(tid)

    # Generate description if there is none
    get_cached_description(tid)

    # After message, sync changes if any
    dm = _get_domain_manager(tid)
    if dm and dm.changes():
        last_assistant = resp if resp else "assistant update"
        dm.sync(last_assistant)


@app.get("/thread/{tid}", response_class=HTMLResponse)
async def get_thread(tid: str, captured: int = 0, merged: int = 0) -> str:
    sandbox = _get_sandbox_backend(tid)
    try:
        chat = MANAGER.get(tid, sandbox_backend=sandbox)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    return render_thread(tid, chat, captured=bool(captured), merged=bool(merged))


@app.post("/thread/{tid}/message")
async def post_message(tid: str, background_tasks: BackgroundTasks, text: str = Form(...)):
    tdir = os.path.join(MANAGER.root_dir, tid)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Thread not found")
    background_tasks.add_task(_process_message, tid, text)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


@app.post("/thread/{tid}/delete")
async def delete_thread(tid: str):
    tdir = os.path.join(MANAGER.root_dir, tid)
    if not os.path.isdir(tdir):
        raise HTTPException(status_code=404, detail="Thread not found")
    MANAGER.soft_delete(tid)
    DESCRIPTION_CACHE.pop(tid, None)
    DOMAIN_MANAGERS.pop(tid, None)
    return RedirectResponse(url="/", status_code=303)


def _capture_conversation(tid: str, reason: str) -> None:
    """Background task to capture a conversation."""
    try:
        thread = MANAGER.get(tid)
    except FileNotFoundError:
        logging.error(f"Thread {tid} not found for capture")
        return

    # Get repo root (navigate up from manage/web.py to repo root)
    current_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(current_file))
    improvements_dir = os.path.join(repo_root, "improvements")

    from edd.capture import capture_conversation
    try:
        capture_path = capture_conversation(thread, reason, improvements_dir)
        logging.info(f"Conversation captured successfully to {capture_path}")
    except Exception as e:
        logging.error(f"Failed to capture conversation for thread {tid}: {e}", exc_info=True)


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
    """Merge the current branch into main with AI-generated summary."""
    try:
        thread = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")

    dm = _get_domain_manager(tid)
    if not dm or not dm.repo:
        raise HTTPException(status_code=400, detail="No git repository configured for this thread")

    # Get a model for summarizing
    from assist.model_manager import select_chat_model
    try:
        summary_model = select_chat_model("gpt-oss-20b", temperature=0.1)
    except Exception:
        # If model fails to load, pass None and use fallback summary
        summary_model = None

    try:
        summary = dm.merge_to_main(summary_model)
        # Redirect with success message (could use query param)
        return RedirectResponse(
            url=f"/thread/{tid}?merged=1",
            status_code=303
        )
    except ValueError as e:
        # User-friendly error (merge conflicts, no changes, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except subprocess.CalledProcessError as e:
        # Git command failed
        raise HTTPException(status_code=500, detail=f"Git operation failed: {e}")
    except Exception as e:
        # Unexpected error
        logging.error(f"Merge failed for thread {tid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")


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
        # Short display name: just the method part
        parts = key.split("::")
        short = html.escape(parts[-1])
        class_part = html.escape("::".join(parts[:-1])) if len(parts) > 1 else ""
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


if __name__ == "__main__":
    import uvicorn
    os.makedirs(ROOT, exist_ok=True)
    port = int(os.getenv("ASSIST_PORT", "8000"))
    uvicorn.run("manage.web:app", host="0.0.0.0", port=port, log_level="info", reload=False)
