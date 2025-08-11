import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
from loguru import logger
from typing import List, Optional, Union
from itertools import takewhile

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
import assist.tools as tools
from assist.reflexion_agent import reflexion_agent


AnyMessage = Union[SystemMessage, HumanMessage, AIMessage]


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
async def log_middle(request: Request, call_next):
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
def chat_completions(request: ChatCompletionRequest):
    agent = get_agent(request.model, request.temperature)
    langchain_messages = openai_to_langchain(request.messages)
    user_request = langchain_messages[-1].content

    logger.debug(f"Request: {user_request}")

    if request.stream:
        def event_gen():
            created = int(time.time())
            first = {
                "id": "1337",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            for ch, metadata in agent.invoke({"messages": langchain_messages}):
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

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    resp = agent.invoke({"messages": langchain_messages})
    debug_tool_use(resp)
    message = resp["messages"][-1]
    logger.debug(f"Got response {message}")
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [
            {"message": ChatMessage(role="assistant", content=message.content)}
        ],
    }


def not_human_message(message: AnyMessage) -> bool:
    return not isinstance(message, HumanMessage)


def render_tool_call(tc: dict) -> str:
    return f"{tc['name']}: {tc['args']}"


def render_tool_calls(tool_calls: list[dict]) -> str:
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


def debug_tool_use(response):
    messages = work_messages(response["messages"])
    for message in messages:
        if isinstance(message, ToolMessage):
            logger.debug(render_tool_message(message))
        elif isinstance(message, AIMessage):
            logger.debug(render_ai_message(message))


def get_agent(model: str, temperature: float) -> Runnable:
    llm = ChatOllama(model=model, temperature=temperature)
    pi = tools.project_index.ProjectIndex()
    proj_tool = pi.search_tool()
    return reflexion_agent(llm,
                           [tools.filesystem.file_contents,
                            tools.filesystem.list_files,
                            proj_tool])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")
