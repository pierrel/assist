"""Unit tests for SmallModelMemoryMiddleware.

The middleware no longer registers a `save_memory` tool — memory is
written via the model's existing `write_file` / `edit_file` tools
against the configured memory file.  These tests guard against
regression (re-introducing the tool), verify the system-prompt body
formats loaded memory correctly, and lock in two reliability fixes:
re-reading from disk on every turn, and rendering the absolute memory
path into the prompt.
"""
from unittest import TestCase
from unittest.mock import MagicMock, NonCallableMagicMock

from assist.middleware.memory_middleware import (
    SMALL_MODEL_MEMORY_PROMPT,
    SmallModelMemoryMiddleware,
)


class TestNoSaveMemoryTool(TestCase):
    """Guard against regression: the middleware must not contribute a tool."""

    def test_middleware_registers_no_tools(self):
        backend = MagicMock()
        mw = SmallModelMemoryMiddleware(backend=backend, memories_path="/AGENTS.md")
        # Upstream MemoryMiddleware does not assign self.tools at all
        # (only declares it as a class annotation), so unset attribute
        # is the expected state.  ``getattr(..., [])`` collapses that
        # to an empty list; if upstream ever changes to assign a
        # non-empty list, this test fails — which is the desired
        # regression signal.
        tools = getattr(mw, "tools", [])
        self.assertEqual(
            list(tools), [],
            "SmallModelMemoryMiddleware must not register any tools — "
            "memory writes go through the model's existing filesystem tools."
        )

    def test_no_save_memory_symbol_in_module(self):
        # Catches accidental re-introduction of the tool factory.
        import assist.middleware.memory_middleware as mod
        self.assertFalse(
            hasattr(mod, "_make_save_memory_tool"),
            "_make_save_memory_tool factory should be deleted, not kept as dead code"
        )


class TestMemoryPromptFormatting(TestCase):
    """The injected system-prompt block must reflect what the memory
    file currently contains so the model can use it as the `edit_file`
    anchor."""

    def setUp(self):
        backend = MagicMock()
        self.mw = SmallModelMemoryMiddleware(
            backend=backend, memories_path="/workspace/AGENTS.md"
        )

    def test_empty_memory_renders_placeholder(self):
        out = self.mw._format_agent_memory({})
        self.assertIn("(No memory loaded)", out)
        self.assertIn("<agent_memory>", out)

    def test_only_blank_content_renders_placeholder(self):
        out = self.mw._format_agent_memory({"/workspace/AGENTS.md": ""})
        self.assertIn("(No memory loaded)", out)

    def test_only_whitespace_content_renders_placeholder(self):
        # A stale ``"\n"`` in the file would render as a near-empty
        # <agent_memory> block, telling the model to use edit_file
        # with an empty-string anchor.  Treat as empty.
        out = self.mw._format_agent_memory({"/workspace/AGENTS.md": "\n"})
        self.assertIn("(No memory loaded)", out)
        out2 = self.mw._format_agent_memory({"/workspace/AGENTS.md": "   \n  "})
        self.assertIn("(No memory loaded)", out2)

    def test_loaded_content_renders_inside_tags(self):
        out = self.mw._format_agent_memory(
            {"/workspace/AGENTS.md": "User has 3 cats."}
        )
        self.assertIn("User has 3 cats.", out)
        # Must be inside <agent_memory> framing — the prompt body tells
        # the model to use the framed content as the edit_file anchor.
        body_start = out.index("<agent_memory>")
        body_end = out.index("</agent_memory>")
        self.assertIn("User has 3 cats.", out[body_start:body_end])

    def test_absolute_memory_path_rendered_in_prompt(self):
        # The deepagents write_file/edit_file schemas require absolute
        # paths.  Verify the configured path appears in the rendered
        # prompt so the worked examples tell the model the right file.
        out = self.mw._format_agent_memory({})
        self.assertIn("/workspace/AGENTS.md", out)

    def test_prompt_directs_to_filesystem_tools(self):
        # Sanity-check the prompt body itself: the new contract names
        # write_file and edit_file (not save_memory).
        self.assertIn("write_file", SMALL_MODEL_MEMORY_PROMPT)
        self.assertIn("edit_file", SMALL_MODEL_MEMORY_PROMPT)
        self.assertNotIn("save_memory", SMALL_MODEL_MEMORY_PROMPT)


class TestStaleMemoryReread(TestCase):
    """``before_agent`` must re-read on every turn, not short-circuit
    when ``memory_contents`` is already cached in state — otherwise a
    write made via ``edit_file`` on turn N is invisible to turn N+1's
    rendered ``<agent_memory>`` block."""

    def _make_response(self, content_bytes):
        resp = MagicMock()
        resp.error = None
        resp.content = content_bytes
        return resp

    def test_before_agent_rereads_when_state_already_cached(self):
        # NonCallableMagicMock so ``_get_backend``'s ``callable(self._backend)``
        # check resolves to False — otherwise the upstream code treats the
        # mock as a factory and calls it, returning a different mock whose
        # download_files isn't configured.
        backend = NonCallableMagicMock()
        # Stable response — we care about the call count, not the content.
        backend.download_files.return_value = [self._make_response(b"contents\n")]
        mw = SmallModelMemoryMiddleware(
            backend=backend, memories_path="/workspace/AGENTS.md"
        )

        runtime = MagicMock()
        config = MagicMock()

        # Turn 1: empty state — backend should be hit.
        first = mw.before_agent({}, runtime, config)
        self.assertIsNotNone(first)
        self.assertIn("memory_contents", first)

        # Turn 2: state already has memory_contents (the cache).
        # Upstream would short-circuit; our override must re-read.
        cached_state = {"memory_contents": first["memory_contents"]}
        second = mw.before_agent(cached_state, runtime, config)
        self.assertIsNotNone(
            second,
            "before_agent must not short-circuit when memory_contents is "
            "already in state — that would render stale content on "
            "subsequent turns",
        )
        self.assertEqual(
            backend.download_files.call_count, 2,
            "Backend must be read once per turn — not just on the first turn",
        )
