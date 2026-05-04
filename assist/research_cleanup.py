"""Confinement + cleanup wrapper for the research sub-agent.

The research sub-agent is given a backend rooted at
``<working_dir>/references/`` so it cannot accidentally read/write
outside that directory in normal use.  Local mode
(``FilesystemBackend(virtual_mode=True)``) blocks ``..`` traversal;
sandbox mode prepends the references prefix but does not normalize, so
a ``..`` in the path could escape — adversarial paths from a friendly
agent are not a real threat in this product, but the comment is here
so a future hardening pass knows where to start.

This module's ``ReferencesCleanupRunnable`` wraps the research agent's
runnable so that around each invocation:

1. The references directory is created if missing (deterministic — no
   reliance on the agent doing it).
2. A snapshot of existing filenames is taken before the agent runs.
   Pre-existing files are never deleted.
3. After the agent returns, the wrapper extracts the *final report's*
   filename from the agent's response and deletes any **newly-created**
   files in the directory that aren't the final report.
4. If the final filename can't be confidently extracted, *nothing* is
   deleted — leaving everything is preferable to deleting the wrong
   file.  We do not fall back to "most recently modified" because the
   critique sub-sub-agent often writes files and would be most-recent.

Filename extraction prefers structured signals over text parsing:
  (a) Walk ``result["messages"]`` backwards for an AIMessage whose
      ``tool_calls`` include ``write_file`` or ``edit_file``.  Take the
      most-recent such call's ``file_path`` argument; verify the file
      still exists in the references directory; return its basename.
  (b) If no such tool call is found, regex the last AIMessage's text for
      a literal ``FINAL_REPORT: <name>`` line that the prompt instructs
      the agent to emit as a backup signal.

On agent error: cleanup is skipped entirely (the partial state may be
useful for debugging; the next research call's snapshot will protect
those files).

Sandbox vs. local-fs parity is handled by branching on whether a
``DockerSandboxBackend`` was passed: sandbox mode shells out via
``sandbox.execute()``; local mode uses ``os.*`` directly.
"""

import logging
import os
import re
import shlex
from typing import Any

logger = logging.getLogger(__name__)


_FINAL_REPORT_LINE = re.compile(r"^FINAL_REPORT:\s*(\S+)\s*$", re.MULTILINE)


class ReferencesCleanupRunnable:
    """Wraps a Runnable and prunes intermediate files in ``references/``.

    Args:
        inner: The wrapped runnable (typically a ``RollbackRunnable`` from
            ``create_research_agent``).
        references_path: Absolute path to the references directory.
            For local-fs mode this is the host path
            (e.g. ``/tmp/working_dir/references``).
            For sandbox mode this is the path inside the container
            (e.g. ``/workspace/references``).
        sandbox_backend: Optional ``DockerSandboxBackend`` (the *parent*
            sandbox, not a sibling).  When provided, file operations are
            performed via ``sandbox.execute()`` against absolute paths;
            otherwise plain ``os.*`` calls are used on the host path.
    """

    def __init__(self, inner, references_path: str, sandbox_backend=None):
        self._inner = inner
        self._references_path = references_path
        self._sandbox = sandbox_backend

    def invoke(self, input_data, config=None, **kwargs):
        self._ensure_dir()
        before = self._snapshot()

        result = self._inner.invoke(input_data, config, **kwargs)

        self._post_invoke_cleanup(result, before)
        return result

    async def ainvoke(self, input_data, config=None, **kwargs):
        # Cleanup is filesystem I/O — run sync.  The wrapped inner is
        # the runnable being awaited; everything around it stays sync.
        # Without this method, deepagents' async ``task`` tool path
        # (subagents.py:379) would call ``self._inner.ainvoke`` via
        # ``__getattr__`` and silently bypass cleanup.
        self._ensure_dir()
        before = self._snapshot()

        result = await self._inner.ainvoke(input_data, config, **kwargs)

        self._post_invoke_cleanup(result, before)
        return result

    def _post_invoke_cleanup(self, result, before):
        keep = self._extract_final_filename(result)
        if keep is not None:
            self._prune(keep=keep, before=before)
        else:
            logger.info(
                "ReferencesCleanup: could not identify final report filename — "
                "leaving all files in %s untouched",
                self._references_path,
            )

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # --- filesystem helpers -------------------------------------------

    def _ensure_dir(self) -> None:
        if self._sandbox is not None:
            self._sandbox.execute(
                f"mkdir -p {shlex.quote(self._references_path)}"
            )
        else:
            os.makedirs(self._references_path, exist_ok=True)

    def _snapshot(self) -> set[str]:
        if self._sandbox is not None:
            resp = self._sandbox.execute(
                f"ls -1 {shlex.quote(self._references_path)} 2>/dev/null"
            )
            output = getattr(resp, "output", "") or ""
            return {line for line in output.splitlines() if line}
        try:
            return set(os.listdir(self._references_path))
        except FileNotFoundError:
            return set()

    def _exists(self, basename: str) -> bool:
        if self._sandbox is not None:
            full = os.path.join(self._references_path, basename)
            resp = self._sandbox.execute(
                f"test -e {shlex.quote(full)} && echo yes || echo no"
            )
            return (getattr(resp, "output", "") or "").strip() == "yes"
        return os.path.exists(os.path.join(self._references_path, basename))

    def _prune(self, keep: str, before: set[str]) -> None:
        """Delete files in references/ that are NEW (not in ``before``)
        and not equal to ``keep``.  Best-effort — log and continue on
        per-file errors.
        """
        after = self._snapshot()
        new_files = after - before
        for name in new_files:
            if name == keep:
                continue
            full = os.path.join(self._references_path, name)
            try:
                if self._sandbox is not None:
                    self._sandbox.execute(f"rm -f -- {shlex.quote(full)}")
                else:
                    if os.path.isdir(full):
                        # Don't recursively delete directories — the
                        # research agent shouldn't create them, but if
                        # something did, leave it for a human to inspect.
                        logger.warning(
                            "ReferencesCleanup: skipping directory %s", full,
                        )
                        continue
                    os.remove(full)
            except Exception as exc:  # best-effort
                logger.warning(
                    "ReferencesCleanup: failed to remove %s: %s", full, exc,
                )

    # --- final-filename extraction ------------------------------------

    def _extract_final_filename(self, result: Any) -> str | None:
        """Return the basename of the final report, or ``None`` if it
        can't be confidently identified.
        """
        messages = (result or {}).get("messages") or []

        # (a) Walk backwards for last write_file / edit_file tool call.
        for msg in reversed(messages):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in reversed(tool_calls):
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name not in ("write_file", "edit_file"):
                    continue
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                file_path = (args or {}).get("file_path") if args else None
                if not file_path:
                    continue
                basename = os.path.basename(file_path)
                if basename and self._exists(basename):
                    return basename

        # (b) Regex backup signal: FINAL_REPORT: <name> on a line by itself.
        if messages:
            last = messages[-1]
            text = getattr(last, "text", None) or getattr(last, "content", None) or ""
            if isinstance(text, list):
                # langchain content blocks: join text parts
                text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in text
                )
            # Use the LAST match — if the agent emitted multiple
            # FINAL_REPORT: lines (a self-correction or an earlier
            # mid-message draft), the most-recent one is the intent.
            matches = _FINAL_REPORT_LINE.findall(text)
            if matches:
                basename = os.path.basename(matches[-1])
                if basename and self._exists(basename):
                    return basename

        return None
