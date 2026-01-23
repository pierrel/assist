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

# debug logging by default
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("deepagents").setLevel(logging.DEBUG)

ROOT = os.getenv("ASSIST_THREADS_DIR", "/tmp/assist_threads")
MANAGER = ThreadManager(ROOT)
DEFAULT_DOMAIN = get_domain()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure thread root exists at startup
    os.makedirs(ROOT, exist_ok=True)
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
            try:
                chat = MANAGER.get(tid)
                title = chat.description()
            except Exception:
                title = tid
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


def render_thread(tid: str, chat: Thread) -> str:
    try:
        title = MANAGER.get(tid).description()
    except Exception:
        title = tid
    msgs = chat.get_messages()
    rendered = []
    for m in reversed(msgs):
        role = html.escape(m.get("role", ""))
        raw = str(m.get("content", ""))
        if role == "diff":
            # Render diffs using Pygments for proper coloring/formatting
            content_html = render_diff(raw)
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
          .btn {{ padding: .6rem 1rem; border: 1px solid #333; border-radius: 8px; background: #eee; font-size: 1rem; }}
          @media (max-width: 480px) {{
            .msg {{ padding: .5rem .6rem; }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="nav"><a href="/">← All threads</a></div>
          <h2 style="font-size:1.2rem">{html.escape(title)}</h2>
          <form action="/thread/{tid}/message" method="post">
            <label for="text">Your message</label><br/>
            <textarea id="text" name="text" required placeholder="Type your message..."></textarea><br/>
            <button class="btn" type="submit">Send</button>
            <div style="font-size:.85rem; color:#666; margin-top:.4rem;">If you close or refresh, your message will still be processed.</div>
          </form>
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
    chat = MANAGER.new(DEFAULT_DOMAIN)
    return RedirectResponse(url=f"/thread/{chat.thread_id}", status_code=303)


def _process_message(tid: str, text: str) -> None:
    try:
        chat = MANAGER.get(tid)
    except FileNotFoundError:
        return
    chat.message(text)


@app.get("/thread/{tid}", response_class=HTMLResponse)
async def get_thread(tid: str) -> str:
    try:
        chat = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    return render_thread(tid, chat)


@app.post("/thread/{tid}/message")
async def post_message(tid: str, background_tasks: BackgroundTasks, text: str = Form(...)):
    try:
        chat = MANAGER.get(tid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    background_tasks.add_task(_process_message, tid, text)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


if __name__ == "__main__":
    import uvicorn
    os.makedirs(ROOT, exist_ok=True)
    uvicorn.run("manage.web:app", host="0.0.0.0", port=5050, log_level="info", reload=False)
