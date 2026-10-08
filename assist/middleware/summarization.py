"""Conversation compaction with bounded summary requests and truthful failures.

Deep Agents retains the raw message log and offloads compacted history. Its
untrimmed summary call can itself overflow; its one-shot retry can also leave
an oversized suffix. Split overflowing summary inputs and keep reducing rejected
full requests. Commit the summary event only after the complete request succeeds.
"""
from deepagents.middleware.summarization import (
    SummarizationMiddleware, compute_summarization_defaults,
)
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage, ToolMessage, get_buffer_string
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from openai import BadRequestError


def _overflow(exc):
    return (isinstance(exc, ContextOverflowError)
            or isinstance(exc, BadRequestError)
            and exc.type == "exceed_context_size_error")


class BoundedSummarizationMiddleware(SummarizationMiddleware):
    def __init__(self, model, backend):
        defaults = compute_summarization_defaults(model)
        super().__init__(model=model, backend=backend, **defaults)
        limit = self._get_profile_limits() or 32768
        output_limit = min(2048, max(1, limit // 16))
        settings = {"max_tokens": output_limit}
        if isinstance(model, ChatOpenAI):
            # LangChain sends max_completion_tokens, while the serving llama.cpp
            # reads max_tokens. Preserve chat-template settings and send both.
            settings["extra_body"] = {**(model.extra_body or {}), "max_tokens": output_limit}
        self._summary_model = model.bind(**settings)

    def _summary_prompt(self, text):
        return self._lc_helper.summary_prompt.format(messages=text).rstrip()

    @staticmethod
    def _split(text):
        if len(text) < 2:
            raise ValueError("The summary prompt cannot fit the model context")
        middle = len(text) // 2
        return text[:middle], text[middle:]

    @staticmethod
    def _combine(text, left, right):
        combined = left + "\n\n" + right
        if len(combined) >= len(text):
            raise ValueError("Summarization did not reduce the oversized history")
        return combined

    def _summarize_text(self, text):
        try:
            result = self._summary_model.invoke(
                self._summary_prompt(text),
                config={"metadata": {"lc_source": "summarization"}})
        except (BadRequestError, ContextOverflowError) as exc:
            if not _overflow(exc):
                raise
            left, right = self._split(text)
            combined = self._combine(text, self._summarize_text(left),
                                     self._summarize_text(right))
            return self._summarize_text(combined)
        summary = result.text.strip()
        if not summary:
            raise ValueError("The model returned an empty conversation summary")
        return summary

    async def _asummarize_text(self, text):
        try:
            result = await self._summary_model.ainvoke(
                self._summary_prompt(text),
                config={"metadata": {"lc_source": "summarization"}})
        except (BadRequestError, ContextOverflowError) as exc:
            if not _overflow(exc):
                raise
            left, right = self._split(text)
            combined = self._combine(text, await self._asummarize_text(left),
                                     await self._asummarize_text(right))
            return await self._asummarize_text(combined)
        summary = result.text.strip()
        if not summary:
            raise ValueError("The model returned an empty conversation summary")
        return summary

    def _create_summary(self, messages):
        return self._summarize_text(get_buffer_string(messages))

    async def _acreate_summary(self, messages):
        return await self._asummarize_text(get_buffer_string(messages))

    def _effective_request(self, request):
        messages = self._get_effective_messages(request)
        messages, _ = self._truncate_args(messages, request.system_message, request.tools)
        counted = [request.system_message, *messages] if request.system_message else messages
        try:
            total = self.token_counter(counted, tools=request.tools)
        except TypeError:
            total = self.token_counter(counted)
        return messages, self._should_summarize(messages, total)

    def _cutoff(self, messages, force):
        if not force:
            return self._determine_cutoff_index(messages)
        # Retain complete AI/tool groups when advancing through raw history.
        for candidate in range(max(2, len(messages) // 2), len(messages)):
            cutoff = self._lc_helper._find_safe_cutoff_point(messages, candidate)
            if 1 < cutoff < len(messages):
                return cutoff
        # Preserve the latest user turn while shrinking a remaining summary.
        if (len(messages) == 2 and self._is_summary_message(messages[0])
                and isinstance(messages[1], HumanMessage)):
            return 1
        # A complete oversized final tool group can be summarized in full.
        # A single summary has no further raw suffix to reclaim.
        return len(messages) if len(messages) > 1 and isinstance(messages[-1], ToolMessage) else 0

    def _event(self, previous, cutoff, summary, path):
        new = self._build_new_messages_with_path(summary, path)[0]
        if previous and cutoff == 1 and len(new.content) >= len(previous["summary_message"].content):
            raise ValueError("Summary-only compaction did not reduce the oversized request")
        return {"cutoff_index": self._compute_state_cutoff(previous, cutoff),
                "summary_message": new, "file_path": path}

    def wrap_model_call(self, request, handler):
        messages, summarize = self._effective_request(request)
        error = None
        if not summarize:
            try:
                return handler(request.override(messages=messages))
            except (BadRequestError, ContextOverflowError) as exc:
                if not _overflow(exc):
                    raise
                error = exc
        previous = request.state.get("_summarization_event")
        force = False
        while True:
            cutoff = self._cutoff(messages, force)
            if cutoff <= 0 and error is not None and not force:
                cutoff = self._cutoff(messages, True)
            if cutoff <= 0:
                if error is not None:
                    raise error
                return handler(request.override(messages=messages))
            older, recent = self._partition_messages(messages, cutoff)
            backend = self._get_backend(request.state, request.runtime)
            path = (previous.get("file_path") if previous and cutoff == 1
                    else self._offload_to_backend(backend, older))
            summary = self._create_summary(older)
            event = self._event(previous, cutoff, summary, path)
            messages = [event["summary_message"], *recent]
            try:
                response = handler(request.override(messages=messages))
            except (BadRequestError, ContextOverflowError) as exc:
                if not _overflow(exc):
                    raise
                error, previous, force = exc, event, True
                continue
            return ExtendedModelResponse(model_response=response,
                                         command=Command(update={"_summarization_event": event}))

    async def awrap_model_call(self, request, handler):
        messages, summarize = self._effective_request(request)
        error = None
        if not summarize:
            try:
                return await handler(request.override(messages=messages))
            except (BadRequestError, ContextOverflowError) as exc:
                if not _overflow(exc):
                    raise
                error = exc
        previous = request.state.get("_summarization_event")
        force = False
        while True:
            cutoff = self._cutoff(messages, force)
            if cutoff <= 0 and error is not None and not force:
                cutoff = self._cutoff(messages, True)
            if cutoff <= 0:
                if error is not None:
                    raise error
                return await handler(request.override(messages=messages))
            older, recent = self._partition_messages(messages, cutoff)
            backend = self._get_backend(request.state, request.runtime)
            path = (previous.get("file_path") if previous and cutoff == 1
                    else await self._aoffload_to_backend(backend, older))
            summary = await self._acreate_summary(older)
            event = self._event(previous, cutoff, summary, path)
            messages = [event["summary_message"], *recent]
            try:
                response = await handler(request.override(messages=messages))
            except (BadRequestError, ContextOverflowError) as exc:
                if not _overflow(exc):
                    raise
                error, previous, force = exc, event, True
                continue
            return ExtendedModelResponse(model_response=response,
                                         command=Command(update={"_summarization_event": event}))
