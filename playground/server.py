import sys
import time
from pathlib import Path
import threading

import requests
import uvicorn

# Ensure the package can be imported when running this script directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from assist.server import app


HOST = "0.0.0.0"
PORT = 5000


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
            {"role": "user", "content": "Hello from the client."},
        ],
    },
)
print(response.text)

# Shut down the server gracefully
server.should_exit = True
server_thread.join()
