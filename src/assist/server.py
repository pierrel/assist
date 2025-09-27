import time
import os
import argparse
from pathlib import Path
import tempfile
from datetime import datetime
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import json
from loguru import logger
from typing import Any, Awaitable, Callable, Iterator, List, Optional, Union, Mapping, Sequence
from itertools import takewhile

from pydantic import BaseModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall,
)
from langchain_core.runnables import Runnable
from assist.reflexion_agent import build_reflexion_graph
from assist.agent_types import AgentInvokeResult
from assist.model_manager import get_model_pair

# ---------------------------------------------------------------------------
# Safeguard: record server startup file and project root so tools can avoid
# accessing the Assist codebase when the server is running.
# ---------------------------------------------------------------------------

def _find_project_root(start: Path) -> Path:
    """Locate the project root by searching for ``pyproject.toml``."""
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


_SERVER_STARTUP_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_SERVER_STARTUP_FILE)
os.environ.setdefault("ASSIST_SERVER_STARTUP_FILE", str(_SERVER_STARTUP_FILE))
os.environ.setdefault("ASSIST_SERVER_PROJECT_ROOT", str(_PROJECT_ROOT))

from assist.tools.base import base_tools


AnyMessage = Union[SystemMessage, HumanMessage, AIMessage]

INDEX_DB_ROOT = Path(tempfile.gettempdir())

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-gpt-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: datetime
    model: str
    choices: List[ChatCompletionChoice]


app = FastAPI(title="OpenAI-compatible API")


def openai_to_lanchain_message(message: ChatMessage) -> AnyMessage:
    match message.role:
        case "system":
            return SystemMessage(content=message.content)
        case "user":
            return HumanMessage(content=message.content)
        case _:
            return AIMessage(content=message.content)

def extract_content(message: BaseMessage) -> str:
    """Return a human-readable output based on the agent message"""
    if isinstance(message, AIMessage) and isinstance(message.content, str):
        return message.content
    return ""

def openai_to_langchain(messages: List[ChatMessage]) -> List[AnyMessage]:
    return list(map(openai_to_lanchain_message, messages))


@app.middleware("http")
async def log_middle(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    logger.debug(f"{request.method} {request.url}")
    routes = request.app.router.routes
    logger.debug("Params:")
    logger.debug(request.query_params)
    logger.debug(request.path_params)
    for route in routes:
        match, scope = route.matches(request)
    body = (await request.body()).decode()
    logger.debug(f"Request body: {body}")
    response = await call_next(request)
    return response


@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Response:
    start = time.time()
    agent = get_agent(request.model, request.temperature or 0.1)
    langchain_messages = openai_to_langchain(request.messages)
    user_request = langchain_messages[-1].content

    if request.stream:
        def event_gen() -> Iterator[str]:
            created = int(time.time())
            first = {
                "id": "1337",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            skip_idx = 0
            for ch, metadata in agent.stream({"messages": langchain_messages}, stream_mode="messages"):
                if skip_idx < len(langchain_messages) and ch == langchain_messages[skip_idx]:
                    skip_idx += 1
                    continue
                content = extract_content(ch)
                chunk = {
                    "id": "1337",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"delta": {"content": content}, "index": 0}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            finish = {
                "id": "1337",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"delta": {}, "finish_reason": "stop", "index": 0}
                ],
            }
            yield f"data: {json.dumps(finish)}\n\n"
            yield "data: [DONE]\n\n"
            logger.debug(f"Reponse took {time.time() - start}s")

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    resp_raw = agent.invoke({"messages": langchain_messages})
    resp = AgentInvokeResult.model_validate(resp_raw)
    message = resp.messages[-1]
    created = int(time.time())
    cm = ChatMessage(role="assistant", content=message.content)
    ccr = {
        'id': "1337",
        'object': "chat.completion",
        'created': created,
        'model': request.model,
        'choices': [
            ChatCompletionChoice(
                message=cm
            ).model_dump()
        ],
    }
    return JSONResponse(content=ccr)


def not_human_message(message: BaseMessage) -> bool:
    return not isinstance(message, HumanMessage)


def render_tool_call(tc: Mapping[str, Any]) -> str:
    return f"{tc['name']}: {tc['args']}"


def render_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(map(render_tool_call, tool_calls))

def render_ai_message(message: AIMessage) -> str:
    if message.tool_calls:
        tcs = render_tool_calls(message.tool_calls)
        # Prepend each line in `tcs` with "- "
        tcs = "\n".join([f"{idx+1}. {line}" for idx, line in enumerate(tcs.split("\n"))])
        return f"AI Tool Calls:\n{tcs}"
    else:
        return f"AIMessage: {message.content}"


def render_tool_message(message: ToolMessage) -> str:
    return f"ToolMessage: {message.content}"


def work_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return messages between the last human input and the final response."""
    with_result = list(
        reversed(list(takewhile(not_human_message, reversed(messages))))
    )
    return with_result[:-1]


def check_tavily_api_key() -> None:
    if not os.getenv('TAVILY_API_KEY'):
        raise RuntimeError('Please define the environment variable TAVILY_API_KEY')


def get_agent(model: str, temperature: float) -> Runnable[Any, Any]:
    check_tavily_api_key()
    plan_llm, exec_llm = get_model_pair(model, temperature)
    tools = base_tools(INDEX_DB_ROOT)
    return build_reflexion_graph(
        plan_llm,
        tools,
        execution_llm=exec_llm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-db", type=Path, default=INDEX_DB_ROOT)
    args = parser.parse_args()
    path = args.index_db.expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - can't simulate
        raise SystemExit(f"Cannot create index directory {path}: {exc}") from exc
    if not os.access(path, os.R_OK | os.W_OK):  # pragma: no cover - simple
        raise SystemExit(f"Cannot access index directory {path}")
    INDEX_DB_ROOT = path
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")
