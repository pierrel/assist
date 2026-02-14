import logging, sys
import html
import os
from typing import Dict

from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse, RedirectResponse

from assist.thread import Thread, ThreadManager
import markdown
from pygments import highlight
from pygments.lexers import DiffLexer
from pygments.formatters import HtmlFormatter
from assist.config_manager import get_domain
from assist.domain_manager import DomainManager

# debug logging by default
logging.basicConfig(stream=sys.stdout)
logging.getLogger("assist.model").setLevel(logging.DEBUG)

ROOT = os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads")
MANAGER = ThreadManager(ROOT)
DEFAULT_DOMAIN = get_domain()
DESCRIPTION_CACHE: Dict[str, str] = {}

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
        # Close shared resources (e.g., sqlite connection) to avoid leaks
        try:
            MANAGER.close()
        except Exception:
            pass

app = FastAPI(title="Assist Web", lifespan=lifespan)

def render_diff(text: str) -> str:
    # Use Pygments to render unified diffs with HTML formatting
    return highlight(text, DiffLexer(), HtmlFormatter(nowrap=False))

def render_index() -> str:
    items = []
    tids = MANAGER.list()
    if not tids:
        items.append("<li><em>No threads yet</em></li>")
    else:
        for tid in tids:
            title = get_cached_description(tid)
            items.append(f'<li><a href="/thread/{tid}">{html.escape(title)}</a></li>')
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
          ul {{ line-height: 1.8; padding-left: 1rem; }}
          a {{ text-decoration: none; display: block; padding: .5rem .6rem; border-radius: 6px; }}
          a:active, a:focus {{ outline: none; }}
          .btn {{ padding: .6rem 1rem; border: 1px solid #333; border-radius: 8px; background: #eee; font-size: 1rem; }}
          @media (max-width: 480px) {{
            .btn {{ width: 100%; }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="topbar">
            <h1 style="font-size:1.4rem; margin:0">Assist Web</h1>
            <form action="/threads" method="post" style="margin:0">
              <button class="btn" type="submit">New thread</button>
            </form>
          </div>
          <h2 style="font-size:1.2rem">Threads</h2>
          <ul>
            {items_html}
          </ul>
        </div>
      </body>
    </html>
    """


def render_thread(tid: str, chat: Thread, captured: bool = False) -> str:
    title = get_cached_description(tid)
    msgs = chat.get_messages()

    # Append diffs from domain repo (computed at render time)
    try:
        twdir = MANAGER.thread_default_working_dir(tid)
        dm = DomainManager(twdir)
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
                <button class="diff-toggle" onclick="toggleDiff('{diff_id}')">
                    <span class="toggle-icon">▶</span> Show diff
                </button>
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
          {"<div class='success-msg'>Conversation capture started! This will complete in the background.</div>" if captured else ""}
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
async def create_thread():
    chat = MANAGER.new()
    tid = chat.thread_id
    DomainManager(MANAGER.thread_default_working_dir(tid),
                  DEFAULT_DOMAIN)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


def _process_message(tid: str, text: str) -> None:
    try:
        chat = MANAGER.get(tid)
    except FileNotFoundError:
        return
    resp = chat.message(text)

    # Generate description is there is none
    get_cached_description(tid)

    # After message, sync changes if any
    twdir = MANAGER.thread_default_working_dir(tid)
    dm = DomainManager(twdir)
    if dm.changes():
        # resp is now a string (the assistant's response), use it as commit message
        last_assistant = resp if resp else "assistant update"
        dm.sync(last_assistant)        


@app.get("/thread/{tid}", response_class=HTMLResponse)
async def get_thread(tid: str, captured: int = 0) -> str:
    try:
        chat = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    return render_thread(tid, chat, captured=bool(captured))


@app.post("/thread/{tid}/message")
async def post_message(tid: str, background_tasks: BackgroundTasks, text: str = Form(...)):
    try:
        chat = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    background_tasks.add_task(_process_message, tid, text)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


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


if __name__ == "__main__":
    import uvicorn
    os.makedirs(ROOT, exist_ok=True)
    uvicorn.run("manage.web:app", host="0.0.0.0", port=5050, log_level="info", reload=False)
