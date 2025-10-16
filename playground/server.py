import sys
import time
from pathlib import Path
import threading

import requests
import uvicorn

from assist import server
from assist.server import app

HOST = "0.0.0.0"
PORT = 5001

server.INDEX_DB_ROOT = "~/.cache/assist/dbs/"


def run_server() -> tuple[uvicorn.Server, threading.Thread]:
    """Run the FastAPI server in a separate thread.

    Returns
    -------
    tuple[uvicorn.Server, threading.Thread]
        The running server instance and the thread executing it.
    """
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="debug")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def stop_server(server: uvicorn.Server, thread: threading.Thread):
    server.should_exit = True
    server_thread.join()

server, server_thread = run_server()

# Wait for the server to start before sending the request
while not server.started:
    time.sleep(0.1)

response = requests.post(
    f"http://{HOST}:{PORT}/chat/completions",
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello. What is the capital of France?"},
            {"role": "assistant",
             "content": "The capital of France is Paris."},
            {"role": "user",
             "content": "And what's it's population?"}
        ],
        "stream": True,
    },
    stream=True
)

saved = []

for counter, line in enumerate(response.iter_lines()):
    if line:
        print(f"{counter}: {line}")
        saved.append(line)

print("FULL RESPONSE")
print("".join(saved))

stop_server(server, server_thread)
