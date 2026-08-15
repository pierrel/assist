"""Wiring tests for the AgentSpec embedder contract — the canonical
surface pins (docs/2026-06-11-embedder-contract.org): spec fields
reaching `create_deep_agent`, checkpointer/sandbox_backend forwarding,
agent-directory and memory-source wiring, and `Thread`-level `spec=` /
`configurable=` wiring.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain.tools.tool_node import ToolCallRequest

from assist.spec import AgentSpec
from tests.skill_test_utils import load_skill


def _tool_a(x: str) -> str:
    return x


def _tool_b(y: int) -> int:
    return y


def _async_task(description: str = "", subagent_type: str = "") -> str:
    return f"{subagent_type}: {description}"


_async_task_tools = tuple(
    StructuredTool.from_function(
        name=name, func=_async_task,
        description=f"{name.replace('_', ' ')}.")
    for name in (
        "start_async_task", "check_async_task", "update_async_task",
        "cancel_async_task", "list_async_tasks")
)


class _CreateAgentHarness:
    """Patch the heavy bits of create_agent; return create_deep_agent kwargs."""

    def _build(self, **kwargs):
        from assist.agent import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res:
            fake.return_value = MagicMock()
            fake_ctx.return_value = MagicMock()
            fake_res.return_value = MagicMock()
            self._fake_res = fake_res
            with tempfile.TemporaryDirectory() as wd:
                kwargs.setdefault("checkpointer", InMemorySaver())
                create_agent(MagicMock(), wd, **kwargs)
                return fake.call_args.kwargs

class TestSpecWiring(_CreateAgentHarness):
    """The spec's fields reach create_deep_agent."""

    def test_default_spec_has_only_builtin_tools(self):
        # travel/directions/map_data (real-world lookups) + read_url
        # (site navigation → download; guarded by the reread-breaker) are the
        # main agent's built-ins; a default spec adds nothing else.
        from assist.tools import directions, map_data, read_url, travel
        assert self._build()["tools"] == [travel, directions, map_data, read_url]
        assert self._build(spec=AgentSpec())["tools"] == [travel, directions, map_data, read_url]

    def test_spec_tools_reach_create_deep_agent(self):
        from assist.tools import directions, map_data, read_url, travel
        kwargs = self._build(spec=AgentSpec(tools=(_tool_a, _tool_b)))
        assert kwargs["tools"] == [_tool_a, _tool_b, travel, directions, map_data, read_url]

    def test_nested_provider_schema_tool_reaches_create_deep_agent(self):
        provider_tool = {
            "type": "function",
            "function": {
                "name": "custom_tool",
                "description": "A custom tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        kwargs = self._build(spec=AgentSpec(tools=(provider_tool,)))

        assert provider_tool in kwargs["tools"]

    def test_provider_schema_name_dedupes_builtin_tool(self):
        provider_tool = {
            "type": "function",
            "function": {
                "name": "travel",
                "description": "Provider travel tool.",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        kwargs = self._build(spec=AgentSpec(tools=(provider_tool,)))

        assert kwargs["tools"].count(provider_tool) == 1
        assert [tool for tool in kwargs["tools"]
                if getattr(tool, "name", None) == "travel"] == []

    def test_hitl_precedes_skills_and_is_not_appended_by_deepagents(self):
        from langchain.agents.middleware import HumanInTheLoopMiddleware
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        kwargs = self._build(spec=AgentSpec(
            interrupt_on={"send_email": True}))
        middleware = kwargs["middleware"]
        hitl_index = next(
            index for index, item in enumerate(middleware)
            if isinstance(item, HumanInTheLoopMiddleware))
        skills_index = next(
            index for index, item in enumerate(middleware)
            if isinstance(item, SmallModelSkillsMiddleware))

        assert hitl_index < skills_index
        assert "interrupt_on" not in kwargs

    def test_critique_subagent_keeps_hitl_for_inherited_effect_tools(self):
        interrupt_on = {"send_email": True}
        kwargs = self._build(spec=AgentSpec(interrupt_on=interrupt_on))
        critique = next(
            subagent for subagent in kwargs["subagents"]
            if isinstance(subagent, dict)
            and subagent.get("name") == "critique-agent")

        assert critique["interrupt_on"] == interrupt_on

    def test_async_subagent_tools_replace_blocking_subagents(self):
        from assist.agent import create_agent
        from assist.tools import directions, map_data, read_url, travel
        from langgraph.checkpoint.memory import InMemorySaver

        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res:
            fake.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                create_agent(
                    MagicMock(), wd, checkpointer=InMemorySaver(),
                    spec=AgentSpec(async_subagent_tools=_async_task_tools))

        kwargs = fake.call_args.kwargs
        assert kwargs["subagents"] == []
        assert kwargs["tools"] == [
            *_async_task_tools, travel, directions, map_data, read_url]
        fake_ctx.assert_not_called()
        fake_res.assert_not_called()

    def test_web_main_is_an_explicit_prompt_composition_identity(self):
        """Async lifecycle tools alone do not select the web prompt contract."""
        from assist.middleware.memory_middleware import SmallModelMemoryMiddleware
        from assist.middleware.prompt_composition import PromptCompositionMiddleware
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        legacy = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools))
        web_main = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools, web_main=True))

        assert not any(isinstance(item, PromptCompositionMiddleware)
                       for item in legacy["middleware"])
        assert any(isinstance(item, PromptCompositionMiddleware)
                   for item in web_main["middleware"])
        legacy_skills = next(item for item in legacy["middleware"]
                             if isinstance(item, SmallModelSkillsMiddleware))
        composed_skills = next(item for item in web_main["middleware"]
                               if isinstance(item, SmallModelSkillsMiddleware))
        legacy_memory = next(item for item in legacy["middleware"]
                             if isinstance(item, SmallModelMemoryMiddleware))
        web_main_memory = next(item for item in web_main["middleware"]
                               if isinstance(item, SmallModelMemoryMiddleware))
        assert legacy_skills.system_prompt_template == composed_skills.system_prompt_template
        assert legacy_memory.sources == web_main_memory.sources

    def test_web_main_requires_the_main_lifecycle_profile(self):
        with pytest.raises(ValueError, match="main role with async lifecycle tools"):
            AgentSpec(web_main=True)
        with pytest.raises(ValueError, match="main role with async lifecycle tools"):
            AgentSpec(role="delegate", async_subagent_tools=_async_task_tools,
                      web_main=True)

    def test_main_guidance_skills_are_closed_to_that_identity(self):
        from assist.backends import MAIN_GUIDANCE_SKILLS_ROUTE, BundledSkillsBackend
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        ordinary = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools, web_main=True))
        candidate = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools, web_main=True,
            main_guidance_skills=True))
        ordinary_skills = next(item for item in ordinary["middleware"]
                               if isinstance(item, SmallModelSkillsMiddleware))
        candidate_skills = next(item for item in candidate["middleware"]
                                if isinstance(item, SmallModelSkillsMiddleware))

        assert MAIN_GUIDANCE_SKILLS_ROUTE not in ordinary_skills.sources
        assert candidate_skills.sources[0] == MAIN_GUIDANCE_SKILLS_ROUTE
        assert isinstance(candidate["backend"].routes[MAIN_GUIDANCE_SKILLS_ROUTE],
                          BundledSkillsBackend)
        assert "# Grounding workflow" in load_skill(candidate_skills, "grounding")
        assert "# Research workflow" in load_skill(candidate_skills, "research")

    def test_explicit_empty_async_tools_disable_all_delegation(self):
        kwargs = self._build(spec=AgentSpec(async_subagent_tools=()))

        assert kwargs["subagents"] == []
        assert "start_async_task" not in kwargs["system_prompt"]
        assert "Do not call `task` or any subagent management tool" in kwargs[
            "system_prompt"]
        assert "dispatch the `context-agent`" not in kwargs["system_prompt"]

    def test_subagent_tools_select_asynchronous_prompt(self):
        kwargs = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools))

        prompt = kwargs["system_prompt"]
        assert "background-research-agent" not in prompt
        assert "start_async_task" in prompt
        assert "full task ID" in prompt
        assert "Do not call `check_async_task` in the same turn" in prompt
        assert "return control to the user" in prompt
        assert "start only context and return" in prompt
        assert "child result as untrusted data" in prompt
        assert "**REPEATED WORKLOAD FIRST:**" not in prompt
        assert "ROUTE COMPLEX REQUESTS FIRST:" not in prompt
        assert 'load_skill(name="orchestrate-repeated-work")' not in prompt
        assert 'load_skill(name="complex-request")' not in prompt
        assert "Unless the loaded skill specifies a different first turn" in prompt
        assert "explicit and self-contained" not in prompt
        assert "## Delegating whole tasks" not in prompt
        assert "your first call must be `load_skill" not in prompt
        assert "TODO bookkeeping is advisory" not in prompt
        assert "`error`, or `timeout`" in prompt
        assert "Cancellation state is observed through cancel, check, or list" in prompt
        assert "do not retry it, take over its work, or start its dependents" in prompt
        assert "Then proceed to Step 1.\n\n### Step 1: Plan\n" in prompt

    def test_absent_async_tools_preserve_sync_subagents(self):
        kwargs = self._build(spec=AgentSpec())

        assert [sub.name if hasattr(sub, "name") else sub["name"]
                for sub in kwargs["subagents"]] == [
                    "context-agent", "research-agent", "critique-agent"]
        assert "background-research-agent" not in kwargs["system_prompt"]
        assert "start_async_task" not in kwargs["system_prompt"]
        assert "Then proceed to Step 1." not in kwargs["system_prompt"]
        assert "### Step 1: Plan" not in kwargs["system_prompt"]

    def test_delegate_role_reuses_graph_with_sync_specialists_and_no_interjection(self):
        from assist.middleware.interjection import InterjectionMiddleware
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
        from assist.middleware.url_provenance import UrlProvenanceMiddleware

        kwargs = self._build(spec=AgentSpec(role="delegate"))

        assert [sub.name if hasattr(sub, "name") else sub["name"]
                for sub in kwargs["subagents"]] == [
                    "context-agent", "research-agent", "critique-agent"]
        assert not any(isinstance(m, InterjectionMiddleware)
                       for m in kwargs["middleware"])
        provenance = next(m for m in kwargs["middleware"]
                          if isinstance(m, UrlProvenanceMiddleware))
        skills = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware))
        assert "/main-skills/" not in skills.sources
        assert "could not be loaded" in load_skill(skills, "complex-request")
        assert "could not be loaded" in load_skill(
            skills, "orchestrate-repeated-work")
        assert provenance._trust_human_messages is False
        assert provenance._trust_task_results is False
        from assist.middleware.tool_result_to_file import ToolResultToFileMiddleware
        task_offload = next(
            middleware for middleware in kwargs["middleware"]
            if isinstance(middleware, ToolResultToFileMiddleware)
            and middleware.name == "ToolResultToFileMiddleware_child_results"
        )
        assert task_offload._untrusted is True
        assert task_offload._tools == {"task"}
        assert self._fake_res.call_args.kwargs["trust_human_messages"] is False
        assert "You own one complete task handed to you by the main agent" in kwargs[
            "system_prompt"]
        assert "start_async_task" not in kwargs["system_prompt"]
        assert "Then proceed to Step 1." not in kwargs["system_prompt"]
        assert "### Step 1: Plan" not in kwargs["system_prompt"]

    def test_delegate_config_reaches_research_url_guard(self):
        from assist.agent import create_agent
        from assist.middleware.url_provenance import (
            DELEGATE_USER_URLS_KEY,
            UrlProvenanceMiddleware,
        )

        url = "https://owner.example/provided"
        fetched = []

        def research(_state, config):
            assert config["configurable"][DELEGATE_USER_URLS_KEY] == (url,)
            middleware = UrlProvenanceMiddleware(trust_human_messages=False)

            def fetch(request):
                fetched.append(request.tool_call["args"]["url"])
                return ToolMessage(content="FETCHED", tool_call_id="read-1")

            middleware.wrap_tool_call(
                ToolCallRequest(
                    tool_call={"name": "read_url", "args": {"url": url},
                               "id": "read-1"},
                    tool=None,
                    state={"messages": [HumanMessage(content=f"Research {url}")]},
                    runtime=None,
                ),
                fetch,
            )
            return {"messages": [AIMessage(content="research complete")]}

        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(self, _tools, **_kwargs):
                return self

        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "task",
                "args": {"description": f"Research {url}",
                         "subagent_type": "research-agent"},
                "id": "task-1",
            }]),
            AIMessage(content="done"),
        ])
        compiled_research = RunnableLambda(research)
        compiled_research.name = "research-agent"
        compiled_context = RunnableLambda(
            lambda _state: {"messages": [AIMessage(content="context complete")]})
        compiled_context.name = "context-agent"

        with patch("assist.agent.create_research_agent",
                   return_value=compiled_research), \
             patch("assist.agent.create_context_agent",
                   return_value=compiled_context), \
             tempfile.TemporaryDirectory() as wd:
            agent = create_agent(model, wd, spec=AgentSpec(role="delegate"))
            agent.invoke(
                {"messages": [HumanMessage(content=f"Research {url}")]},
                {"configurable": {"thread_id": "delegate-test",
                                  DELEGATE_USER_URLS_KEY: (url,)}},
            )

        assert fetched == [url]

    def test_main_role_keeps_interjection_and_trusts_user_messages(self):
        from assist.middleware.interjection import InterjectionMiddleware
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
        from assist.middleware.url_provenance import UrlProvenanceMiddleware

        kwargs = self._build(spec=AgentSpec())

        assert any(isinstance(m, InterjectionMiddleware) for m in kwargs["middleware"])
        provenance = next(m for m in kwargs["middleware"]
                          if isinstance(m, UrlProvenanceMiddleware))
        skills = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware))
        assert "/main-skills/" not in skills.sources
        assert "could not be loaded" in load_skill(skills, "complex-request")
        assert "could not be loaded" in load_skill(
            skills, "orchestrate-repeated-work")
        assert provenance._trust_human_messages is True

    def test_async_main_can_load_supervisor_skill(self):
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
        from assist.middleware.tool_result_to_file import ToolResultToFileMiddleware

        kwargs = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools))
        skills = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware))

        assert "/main-skills/" in skills.sources
        assert skills.sources[0] == "/main-skills/"
        loaded = load_skill(skills, "orchestrate-repeated-work")
        assert "Start exactly one evidence-only delegate per group." in loaded
        complex_loaded = load_skill(skills, "complex-request")
        assert "start one `delegate-agent` per outcome" in complex_loaded
        task_offload = next(
            middleware for middleware in kwargs["middleware"]
            if isinstance(middleware, ToolResultToFileMiddleware)
            and middleware.name == "ToolResultToFileMiddleware_child_results"
        )
        assert task_offload._untrusted is True
        assert task_offload._tools == {tool.name for tool in _async_task_tools}

    def test_async_main_does_not_duplicate_overridden_supervisor_source(self):
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware

        backend = MagicMock()
        kwargs = self._build(spec=AgentSpec(
            async_subagent_tools=_async_task_tools,
            skill_sources={"/main-skills/": backend},
        ))
        skills = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelSkillsMiddleware))

        assert skills.sources.count("/main-skills/") == 1
        assert kwargs["backend"].routes["/main-skills/"] is backend

    def test_spec_skill_sources_reach_middleware(self):
        from assist.middleware.skills_middleware import SmallModelSkillsMiddleware
        backend = MagicMock()
        kwargs = self._build(
            spec=AgentSpec(skill_sources={"/client-skills/": backend}))
        mw = next(m for m in kwargs["middleware"]
                  if isinstance(m, SmallModelSkillsMiddleware))
        assert "/client-skills/" in mw.sources

    def test_loop_detection_present_by_default(self):
        """The hardened stack ships with the plain A/B loop detector and
        no per-tool exploration knob (the rollback contract; moved here
        from the deleted legacy-kwarg test file)."""
        from assist.middleware.loop_detection import LoopDetectionMiddleware
        mws = self._build()["middleware"]
        mw = next(m for m in mws if isinstance(m, LoopDetectionMiddleware))
        assert not hasattr(mw, "exploration_tools")

    def test_spec_default_backend_excludes_sandbox_backend(self):
        with pytest.raises(ValueError, match="not both"):
            self._build(spec=AgentSpec(default_backend=MagicMock()),
                        sandbox_backend=MagicMock())

    def test_thread_memory_source_reaches_main_middleware(self):
        from assist.middleware.memory_middleware import SmallModelMemoryMiddleware

        with tempfile.TemporaryDirectory() as agent_dir:
            kwargs = self._build(
                agent_dir=agent_dir,
                spec=AgentSpec(async_subagent_tools=()))
        memory = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelMemoryMiddleware))
        assert memory.sources == ["/AGENTS.md", "/agent/memory.md"]

    def test_thread_memory_omits_legacy_in_process_subagents(self):
        with tempfile.TemporaryDirectory() as agent_dir:
            kwargs = self._build(agent_dir=agent_dir, spec=AgentSpec())
        assert kwargs["subagents"] == []

    def test_native_sandbox_capability_enables_thread_memory(self):
        from assist.middleware.memory_middleware import SmallModelMemoryMiddleware

        backend = MagicMock()
        backend.native_agent_dir = True
        backend.work_dir = "/workspace"
        kwargs = self._build(
            sandbox_backend=backend,
            spec=AgentSpec(async_subagent_tools=()))
        memory = next(m for m in kwargs["middleware"]
                      if isinstance(m, SmallModelMemoryMiddleware))
        assert memory.sources == ["/workspace/AGENTS.md", "/agent/memory.md"]

    def test_local_agent_route_writes_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            agent_dir = os.path.join(temp_dir, "missing", "agent")
            kwargs = self._build(
                agent_dir=agent_dir,
                spec=AgentSpec(async_subagent_tools=()))
            assert not os.path.exists(agent_dir)
            backend = kwargs["backend"]
            result = backend.write("/agent/memory.md", "private state")
            assert result.error is None
            with open(f"{agent_dir}/memory.md") as stream:
                assert stream.read() == "private state"

    def test_local_specialized_child_cannot_read_parent_agent_dir(self):
        from assist.agent import create_context_agent

        with tempfile.TemporaryDirectory() as thread_dir:
            working_dir = os.path.join(thread_dir, "domain")
            agent_dir = os.path.join(thread_dir, "agent")
            os.makedirs(working_dir)
            os.makedirs(agent_dir)
            with open(os.path.join(agent_dir, "memory.md"), "w") as stream:
                stream.write("private-parent-canary")

            parent_kwargs = self._build(
                agent_dir=agent_dir,
                spec=AgentSpec(async_subagent_tools=()))
            parent_read = parent_kwargs["backend"].read("/agent/memory.md")
            assert "private-parent-canary" in parent_read.file_data["content"]

            with patch("assist.agent.create_deep_agent") as child_factory:
                child_factory.return_value = MagicMock()
                create_context_agent(MagicMock(), working_dir)
            child_backend = child_factory.call_args.kwargs["backend"]
            child_read = child_backend.read("/agent/memory.md")
            assert child_read.error is not None
            assert "private-parent-canary" not in str(child_read)


class TestForwardingGaps(_CreateAgentHarness):
    """create_agent-level forwarding that was previously unpinned:
    checkpointer to create_deep_agent, sandbox_backend into the
    subagent factories.  (Thread-level forwarding of both is pinned in
    TestThreadSpecForwarding.)"""

    def test_checkpointer_forwarded_to_create_deep_agent(self):
        from langgraph.checkpoint.memory import InMemorySaver
        saver = InMemorySaver()
        kwargs = self._build(checkpointer=saver)
        assert kwargs["checkpointer"] is saver

    def test_sandbox_backend_forwarded_to_subagent_factories(self):
        from assist.agent import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        sandbox = MagicMock()
        sandbox.work_dir = "/workspace"
        with patch("assist.agent.create_deep_agent") as fake, \
             patch("assist.agent.create_context_agent") as fake_ctx, \
             patch("assist.agent.create_research_agent") as fake_res, \
             patch("assist.agent.create_sandbox_composite_backend"):
            fake.return_value = MagicMock()
            fake_ctx.return_value = MagicMock()
            fake_res.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                create_agent(MagicMock(), wd, checkpointer=InMemorySaver(),
                             sandbox_backend=sandbox)
        assert fake_ctx.call_args.kwargs["sandbox_backend"] is sandbox
        assert fake_res.call_args.kwargs["sandbox_backend"] is sandbox


class _ThreadHarness:
    def _build(self, **kwargs):
        from assist.thread import Thread
        with patch("assist.thread.create_agent") as fake_ca, \
             patch("assist.thread.select_assistant_model") as fake_model:
            fake_ca.return_value = MagicMock()
            fake_model.return_value = MagicMock()
            with tempfile.TemporaryDirectory() as wd:
                t = Thread(working_dir=wd, **kwargs)
                return t, fake_ca.call_args.kwargs


class TestThreadSpecForwarding(_ThreadHarness):
    def test_agent_dir_forwarded(self):
        _, ca_kwargs = self._build(agent_dir="/host/thread/agent")
        assert ca_kwargs["agent_dir"] == "/host/thread/agent"

    def test_spec_forwarded_to_create_agent(self):
        spec = AgentSpec(tools=(_tool_a,))
        _, ca_kwargs = self._build(spec=spec)
        assert ca_kwargs["spec"] is spec

    def test_default_spec_none_forwarded(self):
        _, ca_kwargs = self._build()
        assert ca_kwargs["spec"] is None

    def test_sandbox_backend_forwarded(self):
        sandbox = MagicMock()
        _, ca_kwargs = self._build(sandbox_backend=sandbox)
        assert ca_kwargs["sandbox_backend"] is sandbox

    def test_checkpointer_forwarded(self):
        saver = MagicMock()
        _, ca_kwargs = self._build(checkpointer=saver)
        assert ca_kwargs["checkpointer"] is saver


class TestThreadConfigurable(_ThreadHarness):
    """The narrowed replacement for extra_config."""

    def test_merges_into_runconfig_configurable(self):
        t, _ = self._build(configurable={"phone_context": "ctx"})
        assert t.runconfig["configurable"]["phone_context"] == "ctx"
        assert "thread_id" in t.runconfig["configurable"]  # built-in survives

    def test_reserved_keys_raise(self):
        for key in ("thread_id", "checkpoint_ns", "checkpoint_id"):
            with pytest.raises(ValueError, match="reserved langgraph keys"):
                self._build(configurable={key: "x"})

    def test_non_mapping_raises(self):
        with pytest.raises(TypeError, match="configurable must be a mapping"):
            self._build(configurable=["not", "a", "mapping"])

    def test_embedder_mutation_after_construction_is_isolated(self):
        shared = {"phone_context": "original"}
        t, _ = self._build(configurable=shared)
        shared["phone_context"] = "MUTATED"
        shared["new_key"] = "added"
        assert t.runconfig["configurable"]["phone_context"] == "original"
        assert "new_key" not in t.runconfig["configurable"]


class TestSpecTypeValidation(_CreateAgentHarness):
    def test_non_spec_raises_clear_typeerror(self):
        with pytest.raises(TypeError, match="spec must be an AgentSpec, got dict"):
            self._build(spec={"tools": ()})

    def test_unknown_role_is_rejected(self):
        with pytest.raises(ValueError, match="role must be 'main' or 'delegate'"):
            AgentSpec(role="invented")
