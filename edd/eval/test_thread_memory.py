"""Natural behavior eval for repository versus thread memory scope."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from unittest import TestCase, mock

from assist.agent import AgentHarness, create_agent
from assist.model_manager import select_assistant_model
from assist.spec import AgentSpec

from .test_async_subagents import reset_task_fixture
from .utils import (agent_tool_calls, complete_web_main_tasks, create_filesystem,
                    prompt_rewrite_web_main_spec, read_file,
                    stub_research_subagent)


logger = logging.getLogger(__name__)


class TestThreadMemory(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_infers_repo_and_thread_scope(self):
        root = tempfile.mkdtemp()
        agent_dir = tempfile.mkdtemp()
        create_filesystem(root, {"AGENTS.md": ""})
        graph = create_agent(
            self.model,
            root,
            agent_dir=agent_dir,
            spec=AgentSpec(async_subagent_tools=()),
        )
        agent = AgentHarness(graph, thread_id="thread-memory-eval")

        agent.message(
            "Across all conversations, use my label atlas-grid for compact "
            "tables. For this "
            "thread's application workflow, use the stage name "
            "candidate-replied after I answer.",
        )

        repo = read_file(os.path.join(root, "AGENTS.md"))
        memory_path = os.path.join(agent_dir, "memory.md")
        thread = read_file(memory_path) if os.path.exists(memory_path) else ""
        self.assertIn("atlas-grid", repo.lower(), agent.all_messages())
        self.assertNotIn("candidate-replied", repo)
        self.assertIn("candidate-replied", thread)
        self.assertNotIn("atlas-grid", thread.lower())

    def test_natural_process_prompt_writes_thread_memory(self):
        """Aspirational thread-memory-write target for prompt rearchitecture.

        The current small model is inconsistent here: it may create a todo
        artifact instead of recording the process. This eval asserts only that
        the message produces a non-empty thread-memory write; it does not verify
        that the process itself was captured or gate the thread-storage feature.
        """
        root = tempfile.mkdtemp()
        agent_dir = tempfile.mkdtemp()
        create_filesystem(root, {"AGENTS.md": ""})
        graph = create_agent(
            self.model,
            root,
            agent_dir=agent_dir,
            spec=AgentSpec(async_subagent_tools=()),
        )
        agent = AgentHarness(graph, thread_id="thread-process-eval")

        agent.message(
            "We'll use this thread to manage my todo list. The process is: new "
            "items start in inbox, move to doing when I begin, and move to done "
            "only after I confirm completion.",
        )

        memory_path = os.path.join(agent_dir, "memory.md")
        thread = read_file(memory_path) if os.path.exists(memory_path) else ""
        self.assertTrue(thread.strip(), agent.all_messages())


class TestPromptRewriteThreadMemoryScopes(TestCase):
    """Natural web-main comparison for repository versus private memory scope."""

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def test_persists_each_fact_in_its_requested_scope(self):
        with tempfile.TemporaryDirectory(prefix="thread_memory_web_main_") as tmp:
            root = os.path.join(tmp, "workspace")
            agent_dir = os.path.join(tmp, "agent")
            os.makedirs(root)
            os.makedirs(agent_dir)
            create_filesystem(root, {"AGENTS.md": ""})
            with mock.patch("assist.tools.requests.get",
                            side_effect=AssertionError("thread-memory eval must not fetch URLs")) as get, \
                 stub_research_subagent():
                agent = AgentHarness(create_agent(
                    self.model, root, agent_dir=agent_dir,
                    spec=prompt_rewrite_web_main_spec()),
                    thread_id="thread-memory-web-main-eval")
                agent.message(
                    "Across all conversations, use my label atlas-grid for compact "
                    "tables. For this thread's application workflow, use the stage name "
                    "candidate-replied after I answer.")
            get.assert_not_called()
            repo = read_file(os.path.join(root, "AGENTS.md"))
            memory_path = os.path.join(agent_dir, "memory.md")
            thread = read_file(memory_path) if os.path.exists(memory_path) else ""

        self.assertIn("atlas-grid", repo.lower(), agent.all_messages())
        self.assertNotIn("candidate-replied", repo)
        self.assertIn("candidate-replied", thread)
        self.assertNotIn("atlas-grid", thread.lower())


class TestAgentWorkspaceGuidance(TestCase):
    """Natural web-main probes for private Markdown workspace continuity.

    These rows deliberately use the production-shaped web-main prompt profile,
    local lifecycle fakes, and a hard URL-fetch rejection. They exercise
    current private-state behavior without live search or external variance.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = select_assistant_model(0.1)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agent_workspace_eval_")
        self.root = os.path.join(self.tmp, "workspace")
        self.agent_dir = os.path.join(self.tmp, "agent")
        os.makedirs(self.root)
        os.makedirs(self.agent_dir)
        create_filesystem(self.root, {"AGENTS.md": ""})
        reset_task_fixture()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _agent(self) -> AgentHarness:
        return AgentHarness(
            create_agent(
                self.model,
                self.root,
                agent_dir=self.agent_dir,
                spec=prompt_rewrite_web_main_spec(),
            ),
            thread_id="agent-workspace-eval",
        )

    @staticmethod
    def _agent_path_calls(calls: list[dict], operation: str | None = None) -> list[dict]:
        """Return filesystem calls that explicitly target the private mount."""
        return [
            call for call in calls
            if (operation is None or call.get("name") == operation)
            and (isinstance(path := ((call.get("args") or {}).get("file_path")
                                     or (call.get("args") or {}).get("path")), str)
                 and path.startswith("/agent/"))
        ]

    def _message(self, agent: AgentHarness, text: str) -> str:
        """Send one natural turn and deliver any local lifecycle completions."""
        reply = agent.message(text)
        return complete_web_main_tasks(agent) or reply

    def _run(self, build):
        """Run an isolated production-shaped web-main conversation with no fetches."""
        with mock.patch.dict(os.environ, {
                "ASSIST_PROMPT_REWRITE_GUIDANCE_SKILLS": "1",
        }, clear=False), \
             mock.patch("assist.tools.requests.get",
                        side_effect=AssertionError("agent-workspace eval must not fetch URLs")) as get, \
             stub_research_subagent():
            result = build()
        get.assert_not_called()
        return result

    def _workspace_files(self) -> dict[str, bytes]:
        """Return the complete user-workspace fixture as relative file bytes."""
        files = {}
        for directory, _, names in os.walk(self.root):
            for name in names:
                path = os.path.join(directory, name)
                with open(path, "rb") as stream:
                    files[os.path.relpath(path, self.root)] = stream.read()
        return files

    def _private_markdown(self) -> str:
        """Read the final private Markdown state, not proposed tool arguments."""
        contents = []
        for directory, _, names in os.walk(self.agent_dir):
            for name in names:
                if name.endswith(".md"):
                    contents.append(read_file(os.path.join(directory, name)))
        return "\n".join(contents)

    def test_reads_private_status_for_a_natural_continuation(self):
        """A continuation question must use state stored outside conversation history."""
        create_filesystem(self.agent_dir, {
            "memory.md": "# Thread memory\nCommunity garden permit work is active.\n",
            "status.md": (
                "# Community garden permit\n\n"
                "- Blocker: the vendor must send the insurance certificate.\n"
                "- Next: submit the permit package once the certificate arrives.\n"
            ),
        })

        def run():
            agent = self._agent()
            reply = self._message(
                agent,
                "Where did we leave off on the community-garden permit?",
            )
            return agent, reply

        agent, reply = self._run(run)
        reads = self._agent_path_calls(agent_tool_calls(agent), "read_file")
        self.assertTrue(
            any("status.md" in str(call.get("args")) for call in reads),
            agent.all_messages(),
        )
        self.assertIn("insurance certificate", reply.lower(), agent.all_messages())

    def test_writes_a_private_checkpoint_after_material_progress(self):
        """A project update leaves reusable Markdown state without touching user files."""
        create_filesystem(self.agent_dir, {
            "memory.md": "# Thread memory\nCommunity garden permit work is active.\n",
        })
        before_workspace = self._workspace_files()

        def run():
            agent = self._agent()
            reply = self._message(
                agent,
                "The vendor sent the insurance certificate, so the community-garden "
                "permit package is ready. We will submit it Monday morning. Keep track "
                "of this project as we continue.",
            )
            return agent, reply

        agent, _reply = self._run(run)
        private_text = self._private_markdown()
        self.assertIn("permit", private_text.lower(), agent.all_messages())
        self.assertIn("monday", private_text.lower(), agent.all_messages())
        self.assertEqual(before_workspace, self._workspace_files(),
                         "private checkpoint changed the user workspace")

    def test_compaction_probe_reads_and_writes_private_state_across_summary(self):
        """Observe private-state work immediately before and after forced compaction.

        The model cannot act *inside* framework compaction. This diagnostic
        records private reads and writes before and after the summary event so
        the result tells us whether the model relies on the automatically loaded
        thread memory, makes an explicit private-file check, or does both.
        """
        from deepagents.middleware.summarization import SummarizationMiddleware

        create_filesystem(self.agent_dir, {
            "memory.md": "# Thread memory\nCommunity garden permit work is active.\n",
        })

        def low_threshold_summary(model, backend):
            return SummarizationMiddleware(
                model,
                backend=backend,
                trigger=("messages", 5),
                keep=("messages", 2),
            )

        def run():
            with mock.patch("deepagents.graph.create_summarization_middleware",
                            side_effect=low_threshold_summary):
                agent = self._agent()
            self._message(
                agent,
                "The insurance certificate arrived. The community-garden permit is now "
                "ready to submit Monday morning. Keep track of this project.",
            )
            before = len(agent_tool_calls(agent))
            reply = self._message(
                agent,
                "What is the next step for the community-garden permit?",
            )
            state = agent.agent.get_state({
                "configurable": {"thread_id": agent.thread_id},
            }).values
            return agent, reply, before, state

        agent, reply, before, state = self._run(run)
        initial_calls = agent_tool_calls(agent)[:before]
        post_compaction_calls = agent_tool_calls(agent)[before:]
        def operation_counts(calls):
            private = self._agent_path_calls(calls)
            return {
                "reads": sum(call["name"] == "read_file" for call in private),
                "writes": sum(call["name"] in {"write_file", "edit_file"}
                              for call in private),
            }

        initial_counts = operation_counts(initial_calls)
        post_compaction_counts = operation_counts(post_compaction_calls)
        metrics = (
            f"pre_summary_private={initial_counts}; "
            f"post_summary_private={post_compaction_counts}; "
            f"summarization_event={state.get('_summarization_event')!r}; "
            f"reply={reply!r}"
        )
        logger.info("agent-workspace compaction probe: %s", metrics)
        self.assertIsNotNone(state.get("_summarization_event"), metrics)
        self.assertIn("monday", reply.lower(), metrics)

    def test_preserves_a_user_requested_check_in_commitment_across_compaction(self):
        """A thread-specific check-in commitment survives a summary boundary."""
        from deepagents.middleware.summarization import SummarizationMiddleware
        from langchain_core.messages import HumanMessage

        report_prompt = "I haven't meditated in four days."

        class NeutralSummary(SummarizationMiddleware):
            """Remove the commitment from compacted history without another LLM call."""

            def _create_summary(self, _messages):
                return "The user is discussing an ongoing personal routine."

            async def _acreate_summary(self, _messages):
                return self._create_summary(_messages)

            def _should_summarize(self, messages, _total_tokens):
                return bool(messages and isinstance(messages[-1], HumanMessage)
                            and messages[-1].content == report_prompt)

        def compact_after_first_turn(model, backend):
            return NeutralSummary(model, backend=backend, keep=("messages", 1))

        before_workspace = self._workspace_files()

        def run():
            with mock.patch("deepagents.graph.create_summarization_middleware",
                            side_effect=compact_after_first_turn):
                agent = self._agent()
            first_reply = self._message(
                agent,
                "When I tell you I've missed meditation for four days, please "
                "encourage me to start the evening check-in again.",
            )
            memory_path = os.path.join(self.agent_dir, "memory.md")
            memory_before_compaction = (
                read_file(memory_path) if os.path.exists(memory_path) else "")
            reply = self._message(agent, report_prompt)
            state = agent.agent.get_state({
                "configurable": {"thread_id": agent.thread_id},
            }).values
            return agent, first_reply, memory_before_compaction, reply, state

        agent, first_reply, memory_before_compaction, reply, state = self._run(run)
        summary_event = state.get("_summarization_event")
        summary = "" if summary_event is None else str(
            summary_event["summary_message"].content).lower()
        memory = memory_before_compaction.lower()
        durable_commitment = (
            "meditat" in memory
            and any(term in memory for term in ("four", "4"))
            and "day" in memory
            and any(term in memory for term in ("evening", "check-in", "check in"))
            and "encourag" in memory
            and any(term in memory for term in ("start", "restart", "resume"))
        )
        follow_up_honors_commitment = (
            any(term in reply.lower() for term in ("evening", "check-in", "check in"))
            and any(term in reply.lower() for term in (
                "encourag", "restart", "resume", "nudge", "back on track",
                "should start", "can help",
            ))
        )
        visible_replies = f"{first_reply}\n{reply}".lower()
        unsupported_autonomy = any(term in visible_replies for term in (
            "i'll monitor", "i will monitor", "i'll notice", "i will notice",
            "automatically remind", "proactively remind", "on my own",
        ))
        history_path = (summary_event or {}).get("file_path")
        history_reads = [
            call for call in agent_tool_calls(agent)
            if call.get("name") == "read_file"
            and (isinstance(path := ((call.get("args") or {}).get("file_path")
                                    or (call.get("args") or {}).get("path")), str)
                 and (path == history_path
                      or path.startswith("/conversation_history/")))
        ]
        metrics = (
            f"summary={summary!r}; memory_before_compaction={memory_before_compaction!r}; "
            f"first_reply={first_reply!r}; reply={reply!r}; "
            f"unsupported_autonomy={unsupported_autonomy}; "
            f"history_reads={history_reads!r}; messages={agent.all_messages()}"
        )
        self.assertIsNotNone(summary_event, metrics)
        self.assertNotIn("meditat", summary, metrics)
        self.assertNotIn("evening", summary, metrics)
        self.assertFalse(history_reads, metrics)
        self.assertFalse(unsupported_autonomy, metrics)
        self.assertTrue(durable_commitment, metrics)
        self.assertTrue(follow_up_honors_commitment, metrics)
        self.assertEqual(before_workspace, self._workspace_files(), metrics)
