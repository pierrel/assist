import json
import httpx

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def log_request(request: httpx.Request):
    print("\n----- OpenAI HTTP Request -----")
    print(request.method, request.url)
    if request.content:
        try:
            print(json.dumps(json.loads(request.content.decode()), indent=2))
        except Exception:
            print("Exception during request")
            print(request.content)
    print("--------------------------------")

def log_response(response: httpx.Response):
    print("----- OpenAI HTTP Response -----")
    print("Status:", response.status_code, response.reason_phrase)
    ctype = response.headers.get("content-type","")
    print("Content-Type:", ctype)
    if ctype.startswith("text/event-stream"):
        print("(Streaming SSE; body arrives token by token)")
    else:
        try:
            print(json.dumps(response.json(), indent=2))
        except Exception:
            print("Exception during response")
            print(response.text[:1000])
    print("--------------------------------\n")

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

http_client = httpx.Client(event_hooks={
    "request": [log_request],
    "reqponse": [log_response]
})

graph = create_react_agent(
    ChatOpenAI(model="models/mistral",
               temperature=0.2,
               base_url="http://my-llm-run.westus2.cloudapp.azure.com:8000/v1",
               http_client=http_client,
               api_key="sk-local"),
    [check_weather]
)
inputs = {"messages": [{"role": "user", "content": "How many liters in a cup?"}]}

def streaming():
    idx = 0
    for chunk in graph.stream(inputs,
                              stream_mode=["messages"],
                              subgraphs=True):
        idx += 1
        print(chunk, flush=True)
    print(idx)

def straight():
    print(graph.invoke(inputs))


    

