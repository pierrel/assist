import html
import uuid
from typing import Dict

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from assist.deepagents_agent import DeepAgentsChat

app = FastAPI(title="Assist Web")

# In-memory thread store: tid -> DeepAgentsChat
THREADS: Dict[str, DeepAgentsChat] = {}


def render_index() -> str:
    items = []
    if not THREADS:
        items.append("<li><em>No threads yet</em></li>")
    else:
        for tid in THREADS:
            items.append(f'<li><a href="/thread/{tid}">{tid}</a></li>')
    items_html = "\n".join(items)
    return f"""
    <html>
      <head>
        <title>Assist Web</title>
        <style>
          body {{ font-family: sans-serif; margin: 2rem; }}
          .topbar {{ display: flex; justify-content: space-between; align-items: center; }}
          ul {{ line-height: 1.8; }}
          a {{ text-decoration: none; }}
          .btn {{ padding: .4rem .8rem; border: 1px solid #333; border-radius: 4px; background: #eee; }}
        </style>
      </head>
      <body>
        <div class="topbar">
          <h1>Assist Web</h1>
          <form action="/threads" method="post">
            <button class="btn" type="submit">New thread</button>
          </form>
        </div>
        <h2>Threads</h2>
        <ul>
          {items_html}
        </ul>
      </body>
    </html>
    """


def render_thread(tid: str, chat: DeepAgentsChat) -> str:
    msgs = chat.get_messages()
    rendered = []
    for m in msgs:
        role = html.escape(m.get("role", ""))
        content = html.escape(str(m.get("content", "")))
        cls = "user" if role == "user" else "assistant"
        bubble = f"<div class=\"msg {cls}\"><div class=\"role\">{role}</div><div class=\"content\">{content}</div></div>"
        rendered.append(bubble)
    body = "\n".join(rendered) or "<p><em>No messages yet.</em></p>"
    return f"""
    <html>
      <head>
        <title>Thread {html.escape(tid)}</title>
        <style>
          body {{ font-family: sans-serif; margin: 2rem; }}
          .nav {{ margin-bottom: 1rem; }}
          .msg {{ margin: .6rem 0; padding: .6rem .8rem; border-radius: 8px; max-width: 70ch; }}
          .msg.user {{ background: #e6f3ff; border: 1px solid #b5dbff; }}
          .msg.assistant {{ background: #f6f6f6; border: 1px solid #ddd; }}
          .role {{ font-size: .8rem; color: #555; margin-bottom: .2rem; text-transform: uppercase; }}
          form textarea {{ width: 100%; height: 8rem; }}
          .btn {{ padding: .4rem .8rem; border: 1px solid #333; border-radius: 4px; background: #eee; }}
        </style>
      </head>
      <body>
        <div class="nav"><a href="/">← All threads</a></div>
        <h2>Thread {html.escape(tid)}</h2>
        <div>
          {body}
        </div>
        <hr/>
        <form action="/thread/{tid}/message" method="post">
          <label for="text">Your message</label><br/>
          <textarea id="text" name="text" required></textarea><br/>
          <button class="btn" type="submit">Send</button>
        </form>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return render_index()


@app.post("/threads")
async def create_thread():
    tid = uuid.uuid4().hex
    THREADS[tid] = DeepAgentsChat("/")
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


@app.get("/thread/{tid}", response_class=HTMLResponse)
async def get_thread(tid: str) -> str:
    chat = THREADS.get(tid)
    if not chat:
        raise HTTPException(status_code=404, detail="Thread not found")
    return render_thread(tid, chat)


@app.post("/thread/{tid}/message")
async def post_message(tid: str, text: str = Form(...)):
    chat = THREADS.get(tid)
    if not chat:
        raise HTTPException(status_code=404, detail="Thread not found")
    chat.message(text)
    return RedirectResponse(url=f"/thread/{tid}", status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
