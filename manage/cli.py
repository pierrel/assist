import sys
import os
import tempfile
import logging
from datetime import datetime

from langchain_core.messages import AIMessage, AIMessageChunk

from assist.thread import ThreadManager, Thread, render_tool_calls

# Setup logging to file
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
log_filename = datetime.now().strftime("%Y-%m-%d_%H.log")
log_path = os.path.join(logs_dir, log_filename)

logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("assist.model").setLevel(logging.DEBUG)

logger = logging.getLogger("assist.model")

def print_update(chunk) -> None:
    model_call = chunk.get('model', None)
    if model_call:
        last_message = model_call.get("messages", [])[-1]
        if last_message and isinstance(last_message, AIMessage):
            print(render_tool_calls(last_message))


def print_message(chunk) -> None:
    if isinstance(chunk, AIMessageChunk):
        print(chunk.content, end='')


def stream_message(thread: Thread, message: str):
    for ch_type, chunk in thread.stream_message(message):
        if ch_type == 'updates':
            print_update(chunk)
        elif ch_type == 'messages':
            [print_message(c) for c in chunk]


def main():
    working_dir = os.getcwd()
    thread_dir = tempfile.mkdtemp()
    print(f"Working directory: {working_dir}")
    tm = ThreadManager(thread_dir)
    chat = tm.new(working_dir)
    logger.info(f"Starting cli in {working_dir} with thread {chat.thread_id} and thread dir of {thread_dir}")
    try:
        while True:
            try:
                user_input = input("> ")
            except EOFError:
                print()
                break
            if user_input.strip() == "":
                continue
            if user_input.strip().lower() == "/quit":
                break
            stream_message(chat, user_input)
    finally:
        pass

if __name__ == "__main__":
    sys.exit(main() or 0)
