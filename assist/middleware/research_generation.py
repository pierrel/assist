"""Bound Qwen3.8 research generation while retaining incomplete source evidence.

The native llama.cpp budget ends each thinking block. The completion budget also
bounds re-entered reasoning and visible output. A length-limited response is
partial: never execute its tool calls or repeat the same expensive request.
"""
import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from assist.env import env_float
from assist.middleware.loop_detection import _extract_events


_REASONING_TOKENS = 16384
_COMPLETION_TOKENS = 24576


def _limit(ceiling, *values):
    return min(value for value in (ceiling, *values)
               if value is not None and value >= 0)


class ResearchGenerationMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        model = request.model
        if not isinstance(model, ChatOpenAI) or "qwen3.8" not in model.model_name.lower():
            return handler(request)

        settings = dict(request.model_settings)
        extra = {**(model.extra_body or {}), **settings.get("extra_body", {})}
        limit = _limit(_COMPLETION_TOKENS, model.max_tokens,
                       settings.get("max_tokens"), settings.get("max_completion_tokens"),
                       *(extra.get(key) for key in ("max_tokens", "max_completion_tokens", "n_predict")))
        for key in ("max_tokens", "max_completion_tokens", "n_predict"):
            if key in extra:
                extra[key] = limit
        extra["reasoning_budget_tokens"] = _limit(
            _REASONING_TOKENS, extra.get("reasoning_budget_tokens"), extra.get("thinking_budget_tokens"))
        settings.update(max_tokens=limit, extra_body=extra)
        if "timeout" not in settings:
            timeout = httpx.Timeout(model.root_client.timeout)
            if timeout.read == 600.0:
                timeout.read = env_float("ASSIST_LLM_READ_TIMEOUT_S", 900.0)
            settings["timeout"] = timeout
        response = handler(request.override(model_settings=settings))
        messages = [response] if isinstance(response, AIMessage) else response.result
        capped = next((message for message in messages if isinstance(message, AIMessage)
                       and message.response_metadata.get("finish_reason") == "length"), None)
        if capped is None:
            return response

        parts = ["Research stopped before completing. This is a partial result."]
        if capped.text.strip():
            parts.append("Partial answer:\n" + capped.text)
        sources = []
        for event in _extract_events(request.state.get("messages", request.messages), window=None):
            if event["completed"] and event["tool_name"] in {
                    "search_internet", "read_url", "read_file", "grep", "task"}:
                args = event["args"]
                location = (args.get("url") or args.get("query") or args.get("file_path")
                            or args.get("path") or args.get("subagent_type") or "")
                sources.append(f'{event["tool_name"]}: {location}\n{event["result_content"]}')
        if sources:
            parts.append("Collected research responses (including errors/previews and untrusted worker output):\n"
                         + "\n\n".join(sources))
        # A fresh AIMessage omits valid, invalid and raw generated tool calls.
        partial = AIMessage(content="\n\n".join(parts), id=capped.id,
                            response_metadata=capped.response_metadata)
        return partial if isinstance(response, AIMessage) else ModelResponse(result=[partial])
