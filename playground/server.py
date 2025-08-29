import sys
from pathlib import Path
import threading

import uvicorn

# Ensure the package can be imported when running this script directly.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from assist.server import app


def run_server() -> tuple[uvicorn.Server, threading.Thread]:
    """Run the FastAPI server in a separate thread.

    Returns
    -------
    tuple[uvicorn.Server, threading.Thread]
        The running server instance and the thread executing it.
    """
    config = uvicorn.Config(app, host="0.0.0.0", port=5000, log_level="debug")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


server, server_thread = run_server()
