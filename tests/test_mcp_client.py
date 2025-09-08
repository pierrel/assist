from __future__ import annotations

from mcp import types
from assist import mcp_client


def test_load_mcp_tools_handles_error(monkeypatch):
    async def boom(url: str):  # pragma: no cover - patched
        raise RuntimeError("fail")

    monkeypatch.setattr(mcp_client, "_fetch_tools_async", boom)
    tools = mcp_client.load_mcp_tools("http://example.com")
    assert tools == []


def test_load_mcp_tools_namespaces(monkeypatch):
    async def fake(url: str):
        return [types.Tool(name="ping", description="", inputSchema={})]

    monkeypatch.setattr(mcp_client, "_fetch_tools_async", fake)
    tools = mcp_client.load_mcp_tools("http://example.com")
    assert len(tools) == 1
    assert tools[0].name == "example.com:ping"
