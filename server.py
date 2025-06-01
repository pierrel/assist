import time
from datetime import datetime
from fastapi import FastAPI, Request
from loguru import logger
from typing import List, Optional, Union

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from general_agent import general_agent

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
llm = ChatOllama(model="llama3.2", temperature=0)
agent = general_agent(llm)


def openai_to_lanchain_message(message: ChatMessage) -> AnyMessage:
    match message.role:
        case "system":
            return SystemMessage(content=message.content)
        case "user":
            return HumanMessage(content=message.content)
        case _:
            return AIMessage(content=message.content)


def openai_to_langchain(messages: List[ChatMessage]) -> List[AnyMessage]:
    return list(map(openai_to_lanchain_message,
                    messages))


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
    langchain_messages = openai_to_langchain(request.messages)

    logger.debug("not streaming response")
    logger.debug("messages sending:")
    for message in langchain_messages:
        logger.debug(message)
    resp = agent.invoke({"messages": langchain_messages})
    message = resp['messages'][-1]
    logger.debug(f'Got response {message}')
    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{
            "message": ChatMessage(role="assistant",
                                   content=message.content)
        }]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")
