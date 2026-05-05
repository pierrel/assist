"""Tool-result eviction wrapper.

Layer 3 of the threads.db growth plan
(docs/2026-05-04-threads-db-layer-3-tool-result-eviction.org).
Wraps langgraph's SqliteSaver to cap the size of any single
checkpoint by moving large tool-result content out of the DB and
onto the filesystem at write time, and rehydrating it on read.

Two channels are evicted:

1. ``channel_values["messages"]`` — any ``ToolMessage`` whose content
   exceeds ``ASSIST_EVICT_THRESHOLD_KB`` is replaced with a stub
   message; the original content is sha256-hashed and written to
   ``<eviction_root>/<thread_id>/large_tool_results/<sha256_16>``.

2. ``channel_values["files"]`` — any entry under ``/large_tool_results/``
   that exceeds the threshold is replaced with a marker dict; the
   content goes to the same disk path.  This is the channel the
   existing ``ContextAwareToolEvictionMiddleware`` writes into via
   ``StateBackend``; without draining it, the middleware merely moves
   bytes from one channel to another inside the same checkpoint.

The eviction directory lives *under* the per-thread directory, so
Layer 0's ``hard_delete`` (``shutil.rmtree`` on the thread dir)
cleans up eviction files for free.

On read (``get_tuple``, ``list``), stubbed entries are rehydrated
back to their original content by reading the on-disk file.  A
missing eviction file raises ``EvictionFileMissingError`` rather
than silently returning a half-restored checkpoint.

Pinned against ``langgraph-checkpoint-sqlite==3.0.3``.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Iterator, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

DEFAULT_EVICT_THRESHOLD_KB = 20

# Sentinel keys used inside stub message ``additional_kwargs`` and
# stub ``files`` channel entries.  Underscore-prefixed so they don't
# collide with anything langchain or deepagents stores there.
EVICT_PATH_KEY = "_evicted_to"
EVICT_SIZE_KEY = "_evicted_size"
EVICT_HASH_KEY = "_evicted_sha256"
EVICT_ENCODING_KEY = "_evicted_encoding"
EVICT_ALL_KEYS = (
    EVICT_PATH_KEY,
    EVICT_SIZE_KEY,
    EVICT_HASH_KEY,
    EVICT_ENCODING_KEY,
)

# Files-channel paths under this prefix are eligible for eviction.
# Matches ``STATEFUL_PATHS`` in ``assist/backends.py`` and the path
# the existing middleware writes to.
LARGE_TOOL_RESULTS_PREFIX = "/large_tool_results/"


class EvictionFileMissingError(RuntimeError):
    """Raised when a rehydration target file is missing on disk.

    The caller (rollback path, web UI, eval) decides how to surface
    this — partial rehydration is never silently substituted.
    """


def _resolve_evict_threshold_kb() -> int:
    """Read ``ASSIST_EVICT_THRESHOLD_KB``.

    - unset → ``DEFAULT_EVICT_THRESHOLD_KB``
    - "0" / "" / "disabled" / "off" / "false" → 0 (kill switch)
    - positive int → use it
    - anything else → default with a loud warning (typo never
      silently disables)
    """
    raw = os.environ.get("ASSIST_EVICT_THRESHOLD_KB")
    if raw is None:
        return DEFAULT_EVICT_THRESHOLD_KB
    norm = raw.strip().lower()
    if norm in ("", "disabled", "off", "false"):
        return 0
    try:
        n = int(norm)
    except ValueError:
        logger.warning(
            "ASSIST_EVICT_THRESHOLD_KB=%r is not an integer; defaulting to %d",
            raw, DEFAULT_EVICT_THRESHOLD_KB,
        )
        return DEFAULT_EVICT_THRESHOLD_KB
    if n < 0:
        logger.warning(
            "ASSIST_EVICT_THRESHOLD_KB=%d is negative; defaulting to %d",
            n, DEFAULT_EVICT_THRESHOLD_KB,
        )
        return DEFAULT_EVICT_THRESHOLD_KB
    return n


def _content_to_bytes(content: Any) -> bytes:
    """Best-effort UTF-8 encoding of arbitrary message content.

    ``ToolMessage.content`` may be a string or a list of content-part
    dicts (``{"type": "text", "text": ...}``).  The list form is rare
    for tool results in this repo but is still well-defined; collapse
    it to a JSON-ish string so the size check is predictable.
    """
    if isinstance(content, str):
        return content.encode("utf-8", "surrogatepass")
    return str(content).encode("utf-8", "surrogatepass")


def _file_content_to_bytes(fd: dict) -> bytes:
    """Encode a state-backend ``FileData`` to UTF-8 bytes for sizing.

    Handles both the modern ``content: str`` and the legacy
    ``content: list[str]`` (joined on newlines) forms.
    """
    content = fd.get("content", "")
    if isinstance(content, list):
        content_str = "\n".join(str(x) for x in content)
    else:
        content_str = str(content)
    return content_str.encode("utf-8", "surrogatepass")


class EvictionSaver(SqliteSaver):
    """SqliteSaver that evicts large tool-result content to disk on put().

    Threshold is read from ``ASSIST_EVICT_THRESHOLD_KB`` at construction
    time (unless overridden via ``evict_threshold_kb=`` for tests).
    A threshold of 0 makes the wrapper a no-op pass-through —
    operational kill switch.

    ``eviction_root`` is the *parent* directory; the saver computes
    ``<eviction_root>/<thread_id>/large_tool_results/<sha256_16>`` per
    eviction file.  When ``eviction_root == ThreadManager.root_dir``,
    the eviction directory lives inside the per-thread dir and Layer 0's
    ``hard_delete`` wipes it for free.

    Async methods (``aput`` etc.) are not overridden; upstream
    ``SqliteSaver`` raises ``NotImplementedError`` from them and we
    inherit that posture.
    """

    def __init__(
        self,
        *args,
        evict_threshold_kb: int | None = None,
        eviction_root: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evict_threshold_kb = (
            evict_threshold_kb if evict_threshold_kb is not None
            else _resolve_evict_threshold_kb()
        )
        self.eviction_root = eviction_root
        if self.evict_threshold_kb <= 0:
            logger.info("EvictionSaver: eviction disabled")
        elif self.eviction_root is None:
            logger.warning(
                "EvictionSaver: evict_threshold_kb=%d but no eviction_root; "
                "disabling eviction (no place to write).",
                self.evict_threshold_kb,
            )
            self.evict_threshold_kb = 0
        else:
            logger.info(
                "EvictionSaver: evicting tool results > %d KB to %s/<tid>/%s",
                self.evict_threshold_kb,
                self.eviction_root,
                "large_tool_results/<sha256_16>",
            )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _eviction_dir(self, thread_id: str) -> str:
        assert self.eviction_root is not None
        return os.path.join(
            self.eviction_root, str(thread_id), "large_tool_results"
        )

    def _eviction_path(self, thread_id: str, digest: str) -> str:
        return os.path.join(self._eviction_dir(thread_id), digest)

    # ------------------------------------------------------------------
    # Eviction (write side)
    # ------------------------------------------------------------------

    def _evict_checkpoint(
        self, checkpoint: Checkpoint, thread_id: str
    ) -> tuple[Checkpoint, list[tuple[str, bytes]]]:
        """Return a (possibly new) checkpoint with stubs in place of large
        content, plus a list of ``(path, bytes)`` to write to disk.

        Pure: never mutates the input checkpoint.  Lazily copies only
        the parts that change — most checkpoints have no eligible
        eviction targets, so we usually return ``checkpoint`` unchanged
        with an empty side-effect list.
        """
        if self.evict_threshold_kb <= 0 or self.eviction_root is None:
            return checkpoint, []

        threshold_bytes = self.evict_threshold_kb * 1024
        cv = checkpoint.get("channel_values") or {}
        side_effects: list[tuple[str, bytes]] = []

        new_messages = self._evict_messages(
            cv.get("messages"), thread_id, threshold_bytes, side_effects
        )
        new_files = self._evict_files(
            cv.get("files"), thread_id, threshold_bytes, side_effects
        )

        if new_messages is None and new_files is None:
            return checkpoint, []

        new_cv = dict(cv)
        if new_messages is not None:
            new_cv["messages"] = new_messages
        if new_files is not None:
            new_cv["files"] = new_files
        new_checkpoint = dict(checkpoint)
        new_checkpoint["channel_values"] = new_cv
        return new_checkpoint, side_effects

    def _evict_one_message(
        self,
        msg: Any,
        thread_id: str,
        threshold_bytes: int,
    ) -> tuple[Any, tuple[str, bytes] | None]:
        """Evict a single ToolMessage if oversized.

        Returns ``(new_msg, side_effect)`` where ``side_effect`` is the
        ``(path, bytes)`` to flush, or ``None`` if the message is not
        eligible (wrong type, too small, already evicted, copy failed).
        """
        if not isinstance(msg, ToolMessage):
            return msg, None
        ak = getattr(msg, "additional_kwargs", None) or {}
        if EVICT_PATH_KEY in ak:
            # Already evicted (idempotent re-put).  Catch the case
            # where a stub somehow re-enters the write path without
            # going through rehydration first.
            return msg, None
        content_bytes = _content_to_bytes(getattr(msg, "content", ""))
        if len(content_bytes) <= threshold_bytes:
            return msg, None
        digest = hashlib.sha256(content_bytes).hexdigest()[:16]
        path = self._eviction_path(thread_id, digest)
        new_ak = {
            **ak,
            EVICT_PATH_KEY: path,
            EVICT_SIZE_KEY: len(content_bytes),
            EVICT_HASH_KEY: digest,
        }
        stub = (
            f"[evicted: {len(content_bytes)} bytes tool result, "
            f"see {LARGE_TOOL_RESULTS_PREFIX}{digest}]"
        )
        try:
            new_msg = msg.model_copy(
                update={"content": stub, "additional_kwargs": new_ak}
            )
        except Exception:
            logger.exception(
                "EvictionSaver: failed to copy ToolMessage; "
                "skipping eviction for this message."
            )
            return msg, None
        return new_msg, (path, content_bytes)

    def _evict_messages(
        self,
        messages: list | None,
        thread_id: str,
        threshold_bytes: int,
        side_effects: list[tuple[str, bytes]],
    ) -> list | None:
        if not messages:
            return None
        new_messages: list | None = None
        for i, msg in enumerate(messages):
            new_msg, se = self._evict_one_message(msg, thread_id, threshold_bytes)
            if se is None:
                continue
            if new_messages is None:
                new_messages = list(messages)
            new_messages[i] = new_msg
            side_effects.append(se)
        return new_messages

    def _evict_one_file(
        self,
        path: str,
        fd: Any,
        thread_id: str,
        threshold_bytes: int,
    ) -> tuple[Any, tuple[str, bytes] | None]:
        """Evict a single files-channel entry if eligible.

        Returns ``(new_fd, side_effect)`` or ``(fd, None)`` if not
        eligible (path not under ``/large_tool_results/``, value not a
        dict, too small, already evicted).
        """
        if not isinstance(path, str):
            return fd, None
        if not path.startswith(LARGE_TOOL_RESULTS_PREFIX):
            return fd, None
        if not isinstance(fd, dict):
            return fd, None
        if EVICT_PATH_KEY in fd:
            return fd, None
        content_bytes = _file_content_to_bytes(fd)
        if len(content_bytes) <= threshold_bytes:
            return fd, None
        digest = hashlib.sha256(content_bytes).hexdigest()[:16]
        evict_path = self._eviction_path(thread_id, digest)
        stub: dict[str, Any] = {
            EVICT_PATH_KEY: evict_path,
            EVICT_SIZE_KEY: len(content_bytes),
            EVICT_HASH_KEY: digest,
            EVICT_ENCODING_KEY: fd.get("encoding", "utf-8"),
        }
        for k in ("created_at", "modified_at"):
            if k in fd:
                stub[k] = fd[k]
        return stub, (evict_path, content_bytes)

    def _evict_files(
        self,
        files: dict | None,
        thread_id: str,
        threshold_bytes: int,
        side_effects: list[tuple[str, bytes]],
    ) -> dict | None:
        if not files or not isinstance(files, dict):
            return None
        new_files: dict | None = None
        for path, fd in files.items():
            new_fd, se = self._evict_one_file(
                path, fd, thread_id, threshold_bytes
            )
            if se is None:
                continue
            if new_files is None:
                new_files = dict(files)
            new_files[path] = new_fd
            side_effects.append(se)
        return new_files

    def _evict_write_value(
        self,
        channel: str,
        value: Any,
        thread_id: str,
        threshold_bytes: int,
        side_effects: list[tuple[str, bytes]],
    ) -> Any | None:
        """Evict large content from a single ``put_writes`` value.

        Returns the new value if mutation happened, ``None`` if the
        write is unchanged.  Handles three shapes per channel:

        - ``messages``: a list of messages, or a single message
          (langgraph's pregel emits both depending on the reducer).
        - ``files``: a dict of ``path → FileData``.
        - everything else: passes through.
        """
        if channel == "messages":
            if isinstance(value, list):
                return self._evict_messages(
                    value, thread_id, threshold_bytes, side_effects
                )
            new_msg, se = self._evict_one_message(
                value, thread_id, threshold_bytes
            )
            if se is None:
                return None
            side_effects.append(se)
            return new_msg
        if channel == "files":
            if isinstance(value, dict):
                return self._evict_files(
                    value, thread_id, threshold_bytes, side_effects
                )
        return None

    def _flush_evictions(self, side_effects: list[tuple[str, bytes]]) -> None:
        """Write eviction blobs to disk.  Best-effort, content-hash dedup.

        Uses ``O_CREAT | O_EXCL`` so a duplicate hash is a fast no-op
        (treated as success).  Other errors are logged and propagated
        so the caller can fall back to writing the un-evicted
        checkpoint.
        """
        for path, content in side_effects:
            d = os.path.dirname(path)
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                logger.exception(
                    "EvictionSaver: failed to create %s; aborting eviction.",
                    d,
                )
                raise
            try:
                fd = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            except FileExistsError:
                # Content-hash dedup: another (tid, checkpoint) already
                # wrote this exact bytes.  Treat as success.
                continue
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
            except OSError:
                # Tear down the partial file so the next put retries
                # cleanly rather than seeing a truncated dedup hit.
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise

    # ------------------------------------------------------------------
    # Rehydration (read side)
    # ------------------------------------------------------------------

    def _rehydrate_checkpoint(
        self, checkpoint: Checkpoint | None
    ) -> Checkpoint | None:
        """Replace stub messages and files-channel entries with their
        original content read from disk.

        Pure: returns the input untouched if nothing needs rehydration.
        Raises ``EvictionFileMissingError`` if a stub points at a
        missing file — partial rehydration is never silently
        substituted.
        """
        if checkpoint is None:
            return checkpoint
        cv = checkpoint.get("channel_values") or {}
        new_messages = self._rehydrate_messages(cv.get("messages"))
        new_files = self._rehydrate_files(cv.get("files"))
        if new_messages is None and new_files is None:
            return checkpoint
        new_cv = dict(cv)
        if new_messages is not None:
            new_cv["messages"] = new_messages
        if new_files is not None:
            new_cv["files"] = new_files
        new_checkpoint = dict(checkpoint)
        new_checkpoint["channel_values"] = new_cv
        return new_checkpoint

    def _rehydrate_one_message(self, msg: Any) -> Any | None:
        """Return a rehydrated message, or ``None`` if ``msg`` is not a
        stub (so the caller leaves it untouched)."""
        ak = getattr(msg, "additional_kwargs", None)
        if not isinstance(ak, dict) or EVICT_PATH_KEY not in ak:
            return None
        content_bytes = self._read_eviction(ak[EVICT_PATH_KEY])
        cleaned_ak = {k: v for k, v in ak.items() if k not in EVICT_ALL_KEYS}
        try:
            return msg.model_copy(
                update={
                    "content": content_bytes.decode("utf-8", "replace"),
                    "additional_kwargs": cleaned_ak,
                }
            )
        except Exception:
            logger.exception(
                "EvictionSaver: failed to rehydrate ToolMessage; "
                "leaving stub in place."
            )
            return None

    def _rehydrate_messages(self, messages: list | None) -> list | None:
        if not messages:
            return None
        new_messages: list | None = None
        for i, msg in enumerate(messages):
            new_msg = self._rehydrate_one_message(msg)
            if new_msg is None:
                continue
            if new_messages is None:
                new_messages = list(messages)
            new_messages[i] = new_msg
        return new_messages

    def _rehydrate_one_file(self, fd: Any) -> dict | None:
        """Return a rehydrated FileData, or ``None`` if ``fd`` is not a stub.

        The on-disk blob carries no format metadata, so rehydration
        always restores to the v2 (modern) FileData shape: ``{"content":
        str, "encoding": str, ...}``.  This is a one-way upgrade for
        any v1 (``content: list[str]``) entries that were evicted —
        ``StateBackend._normalize_content`` smooths the difference at
        read time, so callers don't notice.
        """
        if not isinstance(fd, dict) or EVICT_PATH_KEY not in fd:
            return None
        content_bytes = self._read_eviction(fd[EVICT_PATH_KEY])
        encoding = fd.get(EVICT_ENCODING_KEY, "utf-8")
        content_str = content_bytes.decode(encoding, "replace")
        restored: dict[str, Any] = {
            "content": content_str,
            "encoding": encoding,
        }
        for k in ("created_at", "modified_at"):
            if k in fd:
                restored[k] = fd[k]
        return restored

    def _rehydrate_files(self, files: dict | None) -> dict | None:
        if not files or not isinstance(files, dict):
            return None
        new_files: dict | None = None
        for path, fd in files.items():
            new_fd = self._rehydrate_one_file(fd)
            if new_fd is None:
                continue
            if new_files is None:
                new_files = dict(files)
            new_files[path] = new_fd
        return new_files

    def _rehydrate_write_value(self, channel: str, value: Any) -> Any | None:
        """Rehydrate a single ``put_writes`` value.  Returns ``None`` if
        the value is unchanged."""
        if channel == "messages":
            if isinstance(value, list):
                return self._rehydrate_messages(value)
            return self._rehydrate_one_message(value)
        if channel == "files":
            if isinstance(value, dict):
                return self._rehydrate_files(value)
        return None

    def _rehydrate_pending_writes(
        self, pending_writes: Sequence | None
    ) -> list | None:
        """Walk pending-writes (``[(task_id, channel, value), ...]``) and
        rehydrate any stub values.  Returns ``None`` if nothing changed."""
        if not pending_writes:
            return None
        new_writes: list | None = None
        for i, entry in enumerate(pending_writes):
            if not isinstance(entry, tuple) or len(entry) != 3:
                continue
            task_id, channel, value = entry
            new_value = self._rehydrate_write_value(channel, value)
            if new_value is None:
                continue
            if new_writes is None:
                new_writes = list(pending_writes)
            new_writes[i] = (task_id, channel, new_value)
        return new_writes

    def _read_eviction(self, path: str) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError as e:
            raise EvictionFileMissingError(
                f"evicted content missing on disk: {path}"
            ) from e

    # ------------------------------------------------------------------
    # SqliteSaver overrides
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Pre-process the checkpoint to evict large content, then delegate."""
        thread_id = str(config["configurable"]["thread_id"])
        try:
            evicted, side_effects = self._evict_checkpoint(checkpoint, thread_id)
            if side_effects:
                self._flush_evictions(side_effects)
            checkpoint_to_save = evicted
        except Exception:
            # Eviction is best-effort; on any failure, write the
            # un-evicted checkpoint so the put never blocks.  Disk is
            # already strictly larger than DB, so this degrades to
            # pre-Layer-3 behavior, never worse.
            logger.exception(
                "EvictionSaver: eviction failed for thread_id=%r; "
                "falling back to un-evicted put.",
                thread_id,
            )
            checkpoint_to_save = checkpoint
        return super().put(config, checkpoint_to_save, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Pre-process per-task writes before they land in the ``writes``
        table.

        Each tool node's output flows through ``put_writes`` first;
        without eviction here, the ``writes`` table accumulates the
        full ToolMessage content per super-step until Layer 2's
        retention sweep removes the parent checkpoint.  Within that
        window, the per-thread DB cost would be roughly checkpoint-size
        × number-of-writes-per-checkpoint without this hook.
        """
        if self.evict_threshold_kb <= 0 or self.eviction_root is None:
            return super().put_writes(config, writes, task_id, task_path)
        thread_id = str(config["configurable"]["thread_id"])
        threshold_bytes = self.evict_threshold_kb * 1024
        side_effects: list[tuple[str, bytes]] = []
        new_writes: list | None = None
        try:
            for i, entry in enumerate(writes):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                channel, value = entry
                new_value = self._evict_write_value(
                    channel, value, thread_id, threshold_bytes, side_effects
                )
                if new_value is None:
                    continue
                if new_writes is None:
                    new_writes = list(writes)
                new_writes[i] = (channel, new_value)
            if side_effects:
                self._flush_evictions(side_effects)
        except Exception:
            logger.exception(
                "EvictionSaver: put_writes eviction failed for "
                "thread_id=%r task_id=%r; falling back to un-evicted writes.",
                thread_id, task_id,
            )
            return super().put_writes(config, writes, task_id, task_path)
        return super().put_writes(
            config,
            new_writes if new_writes is not None else writes,
            task_id,
            task_path,
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        tup = super().get_tuple(config)
        if tup is None:
            return None
        rehydrated = self._rehydrate_checkpoint(tup.checkpoint)
        new_pending = self._rehydrate_pending_writes(tup.pending_writes)
        if rehydrated is tup.checkpoint and new_pending is None:
            return tup
        return CheckpointTuple(
            tup.config, rehydrated, tup.metadata, tup.parent_config,
            new_pending if new_pending is not None else tup.pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        for tup in super().list(
            config, filter=filter, before=before, limit=limit
        ):
            rehydrated = self._rehydrate_checkpoint(tup.checkpoint)
            new_pending = self._rehydrate_pending_writes(tup.pending_writes)
            if rehydrated is tup.checkpoint and new_pending is None:
                yield tup
            else:
                yield CheckpointTuple(
                    tup.config, rehydrated, tup.metadata, tup.parent_config,
                    new_pending if new_pending is not None
                    else tup.pending_writes,
                )
