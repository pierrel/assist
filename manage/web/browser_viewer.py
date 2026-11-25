"""Thread browser controls on the same trusted-client boundary as the web UI."""
import base64
import html
import json
import re

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from assist.browser.manager import BrowserManager, BrowserUnavailable
from manage.web.app import app
from manage.web import threads

HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
           "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"}


def _session(tid):
    threads._require_deep_thread(tid)
    session = BrowserManager.current_session(tid)
    if session is None or session.closed or session.identity is None:
        raise BrowserUnavailable("No live browser. Open a website in this thread first.")
    return session


def _control(tid, token, body):
    session = _session(tid)
    client, action = body.get("client"), body.get("action")
    if not isinstance(action, str):
        raise HTTPException(422, "Invalid viewer action")
    if not isinstance(client, str) or not re.fullmatch(r"[0-9a-f]{32}", client):
        raise HTTPException(422, "Invalid viewer identity")
    if action == "takeover":
        session.request_human(token, client)
        holder = threads.THREAD_QUEUE.peek_holder()
        if holder == tid or (holder and any(
                run.mode == "child" and run.parent_thread_id == tid and run.status == "running"
                for run in threads._runs().list(holder))):
            threads.THREAD_QUEUE.request_pause(holder)
    elif action == "return":
        session.return_control(token, client)
        return {"returned": True}
    elif action != "status":
        allowed = {"tabs": set(), "frame": {"page_id"}, "viewport": {"page_id", "width", "height"},
                   "click": {"page_id", "x", "y"}, "text": {"page_id", "text"},
                   "key": {"page_id", "key"}, "scroll": {"page_id", "delta"}}
        if action not in allowed or set(body) - {"action", "client"} - allowed[action]:
            raise HTTPException(422, "Unsupported viewer input")
        result = session.viewer_command(token, client, action,
            **{key: value for key, value in body.items() if key not in {"action", "client"}})
        if "error" in result:
            raise BrowserUnavailable(result["error"])
        return result["result"]
    with BrowserManager.bounded_thread_gate(tid), session._lock:
        session._viewer_check(token)
        return {"requested": session.human_requested, "active": session.human_active,
                "owner": session.human_client == client,
                "unclaimed": bool(session.handoff_url and session.human_client is None)}


@app.post("/thread/{tid}/browser/control")
async def browser_control(tid: str, request: Request):
    # A custom token header is unavailable to cross-origin forms/images. Require
    # the browser's exact Origin as well; no CORS access is enabled here.
    origin = str(request.base_url).rstrip("/")
    if request.headers.get("origin") != origin:
        raise HTTPException(403, "Same-origin browser control required")
    token = request.headers.get("x-browser-token", "")
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise HTTPException(403, "Missing browser control token")
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > 12000:
            raise HTTPException(413, "Viewer input exceeds limit")
    try:
        body = json.loads(payload)
        if not isinstance(body, dict):
            raise ValueError
    except (ValueError, UnicodeError):
        raise HTTPException(422, "Invalid viewer input") from None
    try:
        result = await run_in_threadpool(_control, tid, token, body)
    except BrowserUnavailable as error:
        return JSONResponse({"error": str(error)}, status_code=409, headers=HEADERS)
    if body.get("action") == "frame":
        return Response(base64.b64decode(result["image"], validate=True), media_type="image/jpeg",
            headers={**HEADERS, "X-Page-Width": str(result["width"]), "X-Page-Height": str(result["height"])})
    return JSONResponse(result, headers=HEADERS)


@app.get("/thread/{tid}/browser", response_class=HTMLResponse)
async def browser_viewer(tid: str, request: Request):
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Open browser control from its thread")
    try:
        session = await run_in_threadpool(_session, tid)
    except BrowserUnavailable as error:
        return HTMLResponse(html.escape(str(error)), status_code=409, headers=HEADERS)
    config = json.dumps({"url": f"/thread/{tid}/browser/control", "token": session.viewer_token})
    content = VIEWER.replace("__CONFIG__", config.replace("<", "\\u003c"))
    return HTMLResponse(content, headers={**HEADERS, "Content-Security-Policy":
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})


VIEWER = '''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Assist browser</title>
<style>body{margin:0;font:16px system-ui;background:#fafafa}header{padding:.5rem;position:sticky;top:0;background:#fafafa;z-index:1}
button,select,input{font:inherit;min-height:44px;max-width:100%}button{touch-action:manipulation}#screen{display:block;width:100%;touch-action:none}
#typing{width:60%}#status{display:block;padding:.4rem}nav{display:flex;flex-wrap:wrap;gap:.3rem}</style></head><body>
<header id="toolbar"><nav><button id="take">Take over</button><button id="return" disabled>Return to Assist</button>
<select id="tabs" aria-label="Browser tab"></select><button id="refresh">Refresh view</button></nav>
<span id="status" role="status">The website stays in Assist's browser.</span>
<p id="view-error" role="alert" hidden></p>
<input id="typing" aria-label="Text for selected website field" autocomplete="off" maxlength="2000" type="password" placeholder="Type into selected field">
<button id="type">Type</button><button id="select">Select all</button><button id="backspace">⌫</button><button id="enter">Enter</button>
<button id="up">Scroll up</button><button id="down">Scroll down</button></header>
<img id="screen" alt="Live website in Assist's browser"><script>
const config=__CONFIG__,key='assist-viewer:'+location.pathname;
let client=sessionStorage.getItem(key);if(!client){client=[...crypto.getRandomValues(new Uint8Array(16))].map(v=>v.toString(16).padStart(2,'0')).join('');sessionStorage.setItem(key,client)}
const $=id=>document.getElementById(id),status=$('status'),screen=$('screen'),tabs=$('tabs'),frameProblem=$('view-error');
let active=false,busy=false,pageWidth=390,pageHeight=600,ended=false,chain=Promise.resolve(),renderedPage=null,viewRevision=0;
async function request(action,args={}){const r=await fetch(config.url,{method:'POST',headers:{'Content-Type':'application/json','X-Browser-Token':config.token},body:JSON.stringify({action,client,...args}),cache:'no-store'});
if(!r.ok){let body=await r.json();throw Error(body.error||body.detail||'Browser unavailable')}return r}
function send(action,args={}){let result=chain.then(async()=>{try{return await request(action,args)}finally{await new Promise(r=>setTimeout(r,65))}});chain=result.catch(()=>{});return result}
function problem(e){status.textContent=e.message}
function canInput(){return active&&renderedPage!==null&&renderedPage===tabs.value}
function retireFrame(clearError=false){viewRevision++;renderedPage=null;if(clearError){frameProblem.hidden=true;frameProblem.textContent='';}let old=screen.src;screen.removeAttribute('src');delete screen.dataset.page;if(old)URL.revokeObjectURL(old)}
async function frame(){if(!active||busy||!tabs.value)return;busy=true;const page=tabs.value,revision=viewRevision;try{let r=await send('frame',{page_id:page}),blob=await r.blob();if(revision!==viewRevision||tabs.value!==page||ended)return;let old=screen.src,url=URL.createObjectURL(blob);screen.src=url;await screen.decode();if(revision===viewRevision&&tabs.value===page&&!ended){pageWidth=Number(r.headers.get('X-Page-Width'));pageHeight=Number(r.headers.get('X-Page-Height'));renderedPage=page;screen.dataset.page=page;frameProblem.hidden=true;frameProblem.textContent=''}if(old)URL.revokeObjectURL(old)}catch(e){if(!ended&&revision===viewRevision&&tabs.value===page){frameProblem.textContent='Live view unavailable: '+e.message;frameProblem.hidden=false}}finally{busy=false}}
async function loadTabs(){let data=await(await send('tabs')).json(),old=tabs.value||sessionStorage.getItem(key+':tab');tabs.replaceChildren();for(let p of data.pages){let o=document.createElement('option');o.value=p.page_id;o.textContent=p.url;tabs.append(o)}if([...tabs.options].some(o=>o.value===old))tabs.value=old;if(renderedPage!==tabs.value)retireFrame(old!==tabs.value)}
async function fit(){if(tabs.value)await send('viewport',{page_id:tabs.value,width:Math.max(320,Math.min(1024,innerWidth)),height:Math.max(240,Math.min(1024,innerHeight-$('toolbar').offsetHeight))})}
async function update(){if(ended)return;try{let state=await(await send('status')).json();let was=active;active=state.active&&state.owner;$('take').disabled=state.requested&&!state.unclaimed;$('return').disabled=!active;
status.textContent=active?'You control this browser.':state.requested?'Waiting for the current operation to finish.':'Assist controls this browser.';
if(active){await loadTabs();if(!was)await fit();await frame()}}catch(e){problem(e)}finally{if(!ended)setTimeout(update,750)}}
$('take').onclick=()=>send('takeover').catch(problem);
$('return').onclick=async()=>{try{await send('return');ended=true;active=false;retireFrame(true);status.textContent='Returned to Assist. You can close this page.';$('return').disabled=true;$('take').disabled=true}catch(e){problem(e)}};
$('type').onclick=async()=>{if(!canInput())return;let text=$('typing').value,page=renderedPage;$('typing').value='';try{await send('text',{page_id:page,text});await frame()}catch(e){problem(e)}};
for(let [id,key] of [['select','Control+a'],['backspace','Backspace'],['enter','Enter']])$(id).onclick=()=>canInput()&&send('key',{page_id:renderedPage,key}).then(frame).catch(problem);
for(let [id,delta] of [['up',-450],['down',450]])$(id).onclick=()=>canInput()&&send('scroll',{page_id:renderedPage,delta}).then(frame).catch(problem);
tabs.onchange=async()=>{retireFrame(true);sessionStorage.setItem(key+':tab',tabs.value);try{await fit();await frame()}catch(e){problem(e)}};$('refresh').onclick=frame;
let start=null;screen.onpointerdown=e=>{if(canInput()){start={x:e.clientX,y:e.clientY,page:renderedPage,revision:viewRevision};screen.setPointerCapture(e.pointerId)}};
screen.onpointerup=e=>{if(!start)return;if(!canInput()||start.page!==renderedPage||start.revision!==viewRevision){start=null;return}let dy=start.y-e.clientY,r=screen.getBoundingClientRect(),page=start.page;start=null;let action=Math.abs(dy)>20?'scroll':'click',args=action==='scroll'?{delta:Math.max(-1500,Math.min(1500,dy*pageHeight/r.height))}:{x:(e.clientX-r.left)*pageWidth/r.width,y:(e.clientY-r.top)*pageHeight/r.height};send(action,{page_id:page,...args}).then(frame).catch(problem)};
screen.onpointercancel=()=>{start=null};update();
</script></body></html>'''
