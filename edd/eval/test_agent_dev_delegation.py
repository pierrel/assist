"""
Evals that verify the general agent delegates software development tasks
to the dev-agent subagent.

These tests only check that the general agent initiates a call to dev-agent;
they do not verify whether the dev-agent successfully completes the work.
"""
import tempfile
from textwrap import dedent
from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness

from .utils import create_filesystem, AgentTestMixin


class TestAgentDevDelegation(AgentTestMixin, TestCase):
    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def _send(self, agent, text: str) -> None:
        """Send a message, ignoring recursion-limit errors from subagents.

        We only care whether the general agent *initiated* a dev-agent call,
        not whether the dev-agent itself finished successfully.
        """
        try:
            agent.message(text)
        except Exception:
            pass  # subagent may crash; delegation check still meaningful

    def test_delegates_button_color_change(self):
        """Changing a UI element's color is a code change — should call dev-agent."""
        agent, root = self.create_agent({
            "README.md": dedent("""\
                # Web App
                A FastAPI-based web application.
                The UI is in web.py.
                """),
            "web.py": dedent("""\
                from fastapi import FastAPI
                from fastapi.responses import HTMLResponse

                app = FastAPI()

                @app.get("/")
                def index():
                    return HTMLResponse(\"\"\"
                    <html>
                    <body>
                        <button style="background-color: blue; color: white;">Submit</button>
                    </body>
                    </html>
                    \"\"\")
                """),
        })

        self._send(
            agent,
            "Change the color of the Submit button in web.py from blue to green.",
        )

        self.assertSubAgentCall(
            agent,
            "dev-agent",
            "Changing button color is a code change — should delegate to dev-agent",
        )

    def test_delegates_thread_ordering_change(self):
        """Changing how threads are sorted is a code change — should call dev-agent."""
        agent, root = self.create_agent({
            "README.md": dedent("""\
                # Thread Manager
                Manages conversation threads stored on disk.
                ThreadManager is in thread.py.
                """),
            "thread.py": dedent("""\
                import os

                class ThreadManager:
                    def __init__(self, root_dir: str):
                        self.root_dir = root_dir

                    def list(self) -> list[str]:
                        \"\"\"Return all thread directory names.\"\"\"
                        return [
                            d for d in os.listdir(self.root_dir)
                            if d != "__pycache__"
                        ]
                """),
        })

        self._send(
            agent,
            "Change the ThreadManager.list() method so that threads are returned "
            "sorted by most-recently-modified first instead of filesystem order.",
        )

        self.assertSubAgentCall(
            agent,
            "dev-agent",
            "Changing thread sort order is a code change — should delegate to dev-agent",
        )

    def test_delegates_message_ordering_change(self):
        """Changing message ordering in Thread is a code change — should call dev-agent."""
        agent, root = self.create_agent({
            "README.md": dedent("""\
                # Conversation Threads
                Each Thread holds a list of messages.
                Thread is implemented in thread.py.
                """),
            "thread.py": dedent("""\
                class Thread:
                    def __init__(self, messages: list):
                        self._messages = messages

                    def get_messages(self) -> list:
                        \"\"\"Return messages newest-first (reversed).\"\"\"
                        return list(reversed(self._messages))
                """),
        })

        self._send(
            agent,
            "Change Thread.get_messages() so messages are returned in chronological "
            "order (oldest first) instead of being reversed.",
        )

        self.assertSubAgentCall(
            agent,
            "dev-agent",
            "Changing message ordering is a code change — should delegate to dev-agent",
        )
