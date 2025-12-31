import sys
import os
from datetime import datetime

from assist.deepagents_agent import deepagents_agent, DeepAgentsChat

def main():
    # Initialize reusable chat interface with working dir
    working_dir = os.getcwd()
    print(f"Working directory: {working_dir}")

    chat = DeepAgentsChat(working_dir)

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

            # Send user message and print response
            reply = chat.message(user_input)
            print(reply)
    finally:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)
