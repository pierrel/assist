import time
import os
import argparse
from pathlib import Path
import tempfile
from datetime import datetime
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import json
from loguru import logger
from typing import Any, Iterator, List, Optional, Union
from itertools import takewhile

from pydantic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from assist.tools import filesystem, project_index
from assist.tools.system_info import SystemInfoIndex
from assist.reflexion_agent import build_reflexion_graph
from assist.agent_types import AgentInvokeResult
from assist.model_manager import get_model_pair


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
    if isinstance(message, AIMessage):
        return message.content
    else:
        return ''

def openai_to_langchain(messages: List[ChatMessage]) -> List[AnyMessage]:
    return list(map(openai_to_lanchain_message, messages))


@app.middleware("http")
async def log_middle(request: Request, call_next) -> Response:
    logger.debug(f"{request.method} {request.url}")
    routes = request.app.router.routes
    logger.debug("Params:")
    logger.debug(request.query_params)
    logger.debug(request.path_params)
    for route in routes:
        match, scope = route.matches(request)
    logger.debug("Headers:")
    for name, value in request.headers.items():
        logger.debug(f"\t{name}: {value}")

    body = await request.body()
    logger.debug(f"Body: {body}")
    response = await call_next(request)
    return response


@app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Response:
    start = time.time()
    agent = get_agent(request.model, request.temperature)
    langchain_messages = openai_to_langchain(request.messages)
    user_request = langchain_messages[-1].content

    logger.debug(f"Request: {user_request}")

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
            for ch, metadata in agent.stream({"messages": langchain_messages}, stream_mode="messages"):
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
    debug_tool_use(resp)
    message = resp.messages[-1]
    logger.debug(f"Got response {message}")
    created = datetime.fromtimestamp(time.time())
    logger.debug(f"Reponse tool {time.time() - start}s")
    return ChatCompletionResponse(
        id="1337",
        object="chat.completion",
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=message.content)
            )
        ],
    )


def not_human_message(message: AnyMessage) -> bool:
    return not isinstance(message, HumanMessage)


def render_tool_call(tc: dict[str, Any]) -> str:
    return f"{tc['name']}: {tc['args']}"


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
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


def work_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Takes a list of messages and returns a list of all the messages
    between the last HumanMessage and the next to last element

    """
    with_result = list(reversed(list(takewhile(not_human_message,
                                               reversed(messages)))))
    return with_result[:-1]


def debug_tool_use(response: AgentInvokeResult) -> None:
    messages = work_messages(response.messages)
    for message in messages:
        if isinstance(message, ToolMessage):
            logger.debug(render_tool_message(message))
        elif isinstance(message, AIMessage):
            logger.debug(render_ai_message(message))

def check_tavily_api_key() -> None:
    if not os.getenv('TAVILY_API_KEY'):
        raise RuntimeError('Please define the environment variable TAVILY_API_KEY')

def get_agent(model: str, temperature: float) -> Runnable:
    check_tavily_api_key()
    search = TavilySearchResults(max_results=10)

    pi = project_index.ProjectIndex(base_dir=INDEX_DB_ROOT)
    proj_tool = pi.search_tool()

    sys_index = SystemInfoIndex(base_dir=INDEX_DB_ROOT)
    sys_search = sys_index.search_tool()
    sys_list = sys_index.list_tool()

    plan_llm, exec_llm = get_model_pair(model, temperature)
    return build_reflexion_graph(
        plan_llm,
        [
            filesystem.file_contents,
            filesystem.list_files,
            filesystem.project_context,
            proj_tool,
            sys_search,
            sys_list,
            search,
        ],
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
