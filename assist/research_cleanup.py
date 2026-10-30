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
2. A snapshot of existing filenames and file stamps is taken before the agent runs.
   Pre-existing files are never deleted.
3. After the agent returns, the wrapper extracts the *final report's*
   filename from the agent's response and deletes any **newly-created**
   files in the directory that aren't the final report.
4. When a thread datastore is configured, source references in a new or
   modified final report are resolved. An unchanged prior report is untouched.
5. If the final filename can't be confidently extracted, *nothing* is
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

In sandbox mode, listing, reading, replacement and pruning run as the sandbox
user through a verified directory handle. Only bounded report bytes and file
metadata cross to the host for reference resolution.
"""

import logging
import os
import re
import json
import hashlib
import secrets
import shlex
from typing import Any

from assist.research_sandbox_files import SANDBOX_FILES
from assist.thread_datastore import (ThreadDataStore, _current_store,
                                     resolve_host_report)

logger = logging.getLogger(__name__)


_FINAL_REPORT_LINE = re.compile(r"^FINAL_REPORT:\s*(\S+)\s*$", re.MULTILINE)


class ReferencesCleanupRunnable:
    """Bind final-report references and prune intermediate files in ``references/``.

    Args:
        inner: The wrapped runnable (typically a ``RollbackRunnable`` from
            ``create_research_agent``), or ``None`` for report-only binding.
        references_path: Host path in local mode, container path in sandbox mode.
        sandbox_backend: Parent sandbox used for sandbox-owned file operations.
    """

    def __init__(self, inner, references_path: str, sandbox_backend=None):
        self._inner = inner
        self._references_path = references_path
        self._sandbox = sandbox_backend
        self._dir_identity = None

    def invoke(self, input_data, config=None, **kwargs):
        self._dir_identity = None
        self._ensure_dir()
        before = self._snapshot()

        result = self._inner.invoke(input_data, config, **kwargs)

        self._post_invoke_cleanup(result, before, config)
        return result

    async def ainvoke(self, input_data, config=None, **kwargs):
        # Cleanup is filesystem I/O — run sync.  The wrapped inner is
        # the runnable being awaited; everything around it stays sync.
        # Without this method, deepagents' async ``task`` tool path
        # (subagents.py:379) would call ``self._inner.ainvoke`` via
        # ``__getattr__`` and silently bypass cleanup.
        self._dir_identity = None
        self._ensure_dir()
        before = self._snapshot()

        result = await self._inner.ainvoke(input_data, config, **kwargs)

        self._post_invoke_cleanup(result, before, config)
        return result

    def _post_invoke_cleanup(self, result, before, config=None):
        keep = self._extract_final_filename(result)
        if keep is not None:
            path = ((config or {}).get("configurable") or {}).get("thread_datastore_path")
            store = ThreadDataStore(path) if isinstance(path, str) else _current_store()
            after = self._snapshot()
            if keep not in before or before[keep] != after.get(keep):
                self._resolve_report_references(keep, store)
            self._prune(keep=keep, before=before, after=after)
        else:
            logger.info(
                "ReferencesCleanup: could not identify final report filename — "
                "leaving all files in %s untouched",
                self._references_path,
            )

    def _resolve_report_references(self, basename: str, store: ThreadDataStore | None) -> None:
        """Replace source handles in the selected final report atomically."""
        if store is None:
            return
        if self._sandbox is not None:
            data, stamp = self._sandbox_read(basename)
            original = data.decode("utf-8")
            from assist.thread_datastore import resolve_references
            resolved = resolve_references(original, store,
                                          max_output_bytes=8 * 1024 * 1024)
            if resolved != original:
                self._sandbox_replace(basename, stamp, resolved.encode("utf-8"))
            return
        resolve_host_report(os.path.join(self._references_path, basename), store)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # --- filesystem helpers -------------------------------------------

    def _ensure_dir(self) -> None:
        if self._sandbox is not None:
            self._dir_identity = json.loads(self._sandbox_op("ensure"))
            return
        os.makedirs(self._references_path, exist_ok=True)

    def _snapshot(self) -> dict[str, object]:
        if self._sandbox is not None:
            data, _ = self._sandbox_transfer("snapshot")
            return json.loads(data)
        try:
            result = {}
            for name in os.listdir(self._references_path):
                try:
                    stat = os.lstat(os.path.join(self._references_path, name))
                except FileNotFoundError:
                    continue
                result[name] = [stat.st_mtime_ns, stat.st_size, stat.st_ino]
            return result
        except FileNotFoundError:
            return {}

    def _exists(self, basename: str) -> bool:
        if self._sandbox is not None:
            return self._sandbox_op("exists", basename) == "YES"
        return os.path.exists(os.path.join(self._references_path, basename))

    def _prune(self, keep: str, before: dict[str, object],
               after: dict[str, object]) -> None:
        """Delete files in references/ that are NEW (not in ``before``)
        and not equal to ``keep``.  Best-effort — log and continue on
        per-file errors.
        """
        if self._sandbox is not None:
            new_files = sorted(after.keys() - before.keys() - {keep})
            if new_files:
                transfer = "/tmp/assist-research-" + secrets.token_hex(16)
                try:
                    names_data = json.dumps(new_files).encode("utf-8")
                    response = self._sandbox.upload_files([(transfer, names_data)])[0]
                    if response.error:
                        raise RuntimeError("research cleanup list upload failed")
                    result = self._sandbox_op(
                        "prune", hashlib.sha256(names_data).hexdigest(), transfer)
                    if result != "OK":
                        logger.warning("ReferencesCleanup: %s", result)
                finally:
                    self._sandbox.execute("rm -f -- " + shlex.quote(transfer))
            return
        new_files = after.keys() - before.keys()
        for name in new_files:
            if name == keep:
                continue
            full = os.path.join(self._references_path, name)
            try:
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

    def _sandbox_op(self, op: str, payload=None, transfer: str = "") -> str:
        args = (op, self._references_path, json.dumps(self._dir_identity or []),
                json.dumps(payload), transfer)
        command = "python3 -c " + shlex.quote(SANDBOX_FILES)
        command += " " + " ".join(shlex.quote(arg) for arg in args)
        result = self._sandbox.execute(command)
        if result.exit_code != 0:
            raise RuntimeError(f"research sandbox {op} failed: {result.output}")
        return result.output.strip()

    def _sandbox_transfer(self, op: str, payload=None) -> tuple[bytes, str]:
        transfer = "/tmp/assist-research-" + secrets.token_hex(16)
        limit = (4 if op == "snapshot" else 8) * 1024 * 1024
        try:
            stamp = self._sandbox_op(op, payload, transfer)
            response = self._sandbox.download_files(
                [transfer], max_bytes=limit + 256 * 1024)[0]
            if response.error or response.content is None:
                raise RuntimeError(f"research sandbox {op} transfer failed")
            if len(response.content) > limit:
                raise RuntimeError(f"research sandbox {op} transfer exceeded byte limit")
            return response.content, stamp
        finally:
            self._sandbox.execute("rm -f -- " + shlex.quote(transfer))

    def _sandbox_read(self, basename: str) -> tuple[bytes, list[int]]:
        data, stamp = self._sandbox_transfer("read", basename)
        return data, json.loads(stamp)

    def _sandbox_replace(self, basename: str, stamp: list[int], data: bytes) -> None:
        transfer = "/tmp/assist-research-" + secrets.token_hex(16)
        try:
            response = self._sandbox.upload_files([(transfer, data)])[0]
            if response.error:
                raise RuntimeError("resolved research report upload failed")
            self._sandbox_op("replace", [basename, stamp,
                                         hashlib.sha256(data).hexdigest()], transfer)
        finally:
            self._sandbox.execute("rm -f -- " + shlex.quote(transfer))

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
