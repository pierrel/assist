import sys
import os
from datetime import datetime

from assist.deepagents_agent import deepagents_agent


def main():
    agent = deepagents_agent()

    # Save working directory and derive a thread id from dir + timestamp
    working_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    thread_id = f"{working_dir}:{timestamp}"

    state = {"messages": []}
    print(f"Working directory: {working_dir}")
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

            # Append user message
            state["messages"].append({"role": "user", "content": user_input})

            # Invoke once, echo response, append it
            res = agent.invoke(state, {"configurable": {"thread_id": thread_id}})
            ai_msg = res['messages'][-1]
            print(ai_msg.content)
            state["messages"].append({"role": "assistant", "content": ai_msg.content})
    finally:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)
