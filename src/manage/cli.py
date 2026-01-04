import sys
import os
from assist.deepagents_agent import DeepAgentsThread

def main():
    working_dir = os.getcwd()
    print(f"Working directory: {working_dir}")
    chat = DeepAgentsThread(working_dir)
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
            reply = chat.message(user_input)
            print(reply)
    finally:
        pass

if __name__ == "__main__":
    sys.exit(main() or 0)
