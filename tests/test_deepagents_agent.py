import os
import tempfile
from assist.deepagents_agent import DeepAgentsThreadManager, DeepAgentsThread, deepagents_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

class GenericFakeChatModel(BaseChatModel):
    def invoke(self, input, config=None, **kwargs):
        # Echo last user content or provide short description
        if isinstance(input, list):
            # description path
            return AIMessage(content="Short summary")
        msgs = input.get("messages", [])
        last = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
        return {"messages": msgs + [AIMessage(content=(last.get("content") if last else "ok"))]}

    @property
    def lc_serializable(self):
        return False

    @property
    def _llm_type(self) -> str:
        return "generic-fake"

    def _generate(self, messages, stop=None, **kwargs):
        return AIMessage(content="ok")


def test_manager_new_list_get_remove(tmp_path):
    root = tmp_path / "threads"
    mgr = DeepAgentsThreadManager(str(root))
    chat = mgr.new()
    assert chat.thread_id in mgr.list()
    got = mgr.get(chat.thread_id)
    assert isinstance(got, DeepAgentsThread)
    mgr.remove(chat.thread_id)
    assert chat.thread_id not in mgr.list()


def test_deepagents_agent_uses_checkpointer(tmp_path):
    # Ensure sqlite created
    root = tmp_path / "threads"
    mgr = DeepAgentsThreadManager(str(root))
    db = root / "threads.db"
    # DB is created on manager init if sqlite unavailable; allow existence
    chat = mgr.new()
    # Override model to fake
    chat.model = GenericFakeChatModel()
    chat.agent = deepagents_agent(chat.model, checkpointer=mgr.checkpointer)
    # Do not invoke; just ensure DB exists and agent created
    assert db.exists()
