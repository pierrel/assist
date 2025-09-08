"""Utilities for loading tools from external MCP servers."""
from __future__ import annotations

from typing import Any, List
from urllib.parse import urlparse

import anyio
from loguru import logger
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp import types


class _ToolArgs(BaseModel):
    """Accept any parameters for an MCP tool call."""

    model_config = ConfigDict(extra="allow")


class MCPRemoteTool(BaseTool):
    """A LangChain tool that proxies calls to an MCP server."""

    args_schema: type[BaseModel] = _ToolArgs

    def __init__(self, url: str, tool: types.Tool, namespace: str) -> None:
        super().__init__(
            name=f"{namespace}:{tool.name}",
            description=tool.description or "",
        )
        self._url = url
        self._tool_name = tool.name

    def _result_to_text(self, result: types.CallToolResult) -> str:
        return "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )

    def _run(self, **kwargs: Any) -> str:  # pragma: no cover - network call
        async def _call() -> str:
            async with sse_client(self._url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(self._tool_name, kwargs or None)
                    return self._result_to_text(result)

        return anyio.run(_call)

    async def _arun(self, **kwargs: Any) -> str:
        async with sse_client(self._url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(self._tool_name, kwargs or None)
                return self._result_to_text(result)


def _namespace_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or parsed.path or "mcp"


async def _fetch_tools_async(url: str) -> List[types.Tool]:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


def load_mcp_tools(url: str) -> List[BaseTool]:
    """Return LangChain tools provided by the MCP server at ``url``.

    Any connection errors are logged and result in no tools being returned.
    """

    namespace = _namespace_from_url(url)
    try:
        tools = anyio.run(_fetch_tools_async, url)
    except Exception as exc:  # pragma: no cover - depends on network
        logger.warning(f"Failed to load MCP server {url}: {exc}")
        return []

    return [MCPRemoteTool(url, tool, namespace) for tool in tools]


__all__ = ["load_mcp_tools", "MCPRemoteTool"]
