"""Checkpoint-shaped fake for narrow web tests; restart coverage uses real SQLite."""
from types import SimpleNamespace


def checkpoint_chat(chat, configurable=None):
    """Give a fake chat observable interrupt/END checkpoints and host metadata."""
    if getattr(chat, "agent", None) is not None:
        if getattr(chat, "_test_checkpoint", False):
            chat.runconfig["configurable"].update(configurable or {})
        return chat
    chat._test_checkpoint = True
    chat.runconfig = {"configurable": {"thread_id": getattr(chat, "thread_id", "mail-thread"),
                                        **(configurable or {})}}

    def snapshot(config):
        if hasattr(chat, "pending_actions"):
            requests = chat.pending_actions() if getattr(chat, "pending", True) else []
        else:
            email = chat.pending_email() if hasattr(chat, "pending_email") else None
            requests = [{"name":"send_email", "args":email}] if email else []
        interrupt_id = getattr(chat, "interrupt_id", "gmail-interrupt")
        if requests and hasattr(chat, "pending_action_interrupt_id"):
            interrupt_id = chat.pending_action_interrupt_id(requests[0]["name"]) or interrupt_id
        return SimpleNamespace(
            config={"configurable": {"checkpoint_id": "proposal" if requests else "end", "checkpoint_ns":"", "thread_id":chat.runconfig["configurable"]["thread_id"]}},
            metadata=dict(chat.runconfig["configurable"]), parent_config=None,
            interrupts=(SimpleNamespace(id=interrupt_id, value={"action_requests":requests}),) if requests else (),
            next=("gate",) if requests else (), values={"messages": []})
    chat.agent = SimpleNamespace(get_state=snapshot)
    return chat
