"""Credentialed Git operations isolated from an agent-writable checkout.

The private repository owns its configuration and writable objects. The thread
checkout's object directory is a read-only alternate, admitted only while its
workspace and sandbox-generation fences exclude managed writers.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time

from assist.git_sync import GitSyncError, _OID, _directory, identity


_OBJECT_DIR = re.compile(r"[0-9a-f]{2}\Z")
_OBJECT_FILE = re.compile(r"[0-9a-f]{38}\Z")
_PACK_FILE = re.compile(
    r"(?:pack-[0-9a-f]{40}\.(?:pack|idx|rev|bitmap|keep|mtimes)|multi-pack-index)\Z"
)
_INFO_FILE = {"packs", "commit-graph"}
_SPLIT_GRAPH_FILE = re.compile(r"graph-[0-9a-f]{40}\.graph\Z")
_OBJECT_LIMIT = 2 * 1024 * 1024 * 1024
_FILE_LIMIT = 20_000
_GIT_TIMEOUT = 30
_OUTPUT_LIMIT = 1024 * 1024
_BUNDLE_LIMIT = 128 * 1024 * 1024


def _validated_objects(worktree: str) -> str:
    """Admit standard, independent Git objects; never follow alternates or links."""
    total = count = 0
    deadline = time.monotonic() + _GIT_TIMEOUT

    def bounded_entry() -> None:
        nonlocal count
        count += 1
        if count > _FILE_LIMIT or time.monotonic() > deadline:
            raise GitSyncError("Git object verification exceeded its bound")

    def regular_file(name: str, directory: int) -> None:
        nonlocal total
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise GitSyncError("Git objects need independent regular storage")
        total += info.st_size
        if total > _OBJECT_LIMIT:
            raise GitSyncError("Git objects exceed the host operation limit")

    with _directory(worktree) as root, _directory(".git", parent=root) as metadata, \
            _directory("objects", parent=metadata) as objects:
        info = os.fstat(objects)
        if info.st_nlink < 2:
            raise GitSyncError("Git object directory is unavailable")
        for entry in os.scandir(objects):
            group = entry.name
            if group not in {"pack", "info"} and not _OBJECT_DIR.fullmatch(group):
                raise GitSyncError("Unsupported Git object directory")
            with _directory(group, parent=objects) as directory:
                for child in os.scandir(directory):
                    bounded_entry()
                    name = child.name
                    if group == "info" and name == "commit-graphs":
                        try:
                            with _directory(name, parent=directory) as graphs:
                                for graph in os.scandir(graphs):
                                    bounded_entry()
                                    if (graph.name != "commit-graph-chain"
                                            and not _SPLIT_GRAPH_FILE.fullmatch(graph.name)):
                                        raise GitSyncError("Unsupported Git object entry")
                                    regular_file(graph.name, graphs)
                        except OSError as error:
                            raise GitSyncError("Unsupported Git object entry") from error
                        continue
                    if (group == "info" and name not in _INFO_FILE
                            or group == "pack" and not _PACK_FILE.fullmatch(name)
                            or group not in {"pack", "info"} and not _OBJECT_FILE.fullmatch(name)):
                        raise GitSyncError("Unsupported Git object entry")
                    regular_file(name, directory)
    return os.path.join(worktree, ".git", "objects")


class LocalGit:
    """One disposable, config-clean Git repository for exact-ref host work."""

    def __init__(self, source: str, worktree: str | None = None):
        self.source = source
        self.worktree = worktree
        self._temporary = tempfile.TemporaryDirectory(prefix="assist-git-")
        self.directory = self._temporary.name
        self._last_exit_code = 0
        self._environment = {key: value for key, value in os.environ.items()
                             if not key.startswith("GIT_")}
        self._environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null",
                                 GIT_NO_REPLACE_OBJECTS="1", GIT_TERMINAL_PROMPT="0")
        try:
            self.git("init", "--bare", self.directory, repository=False)
            if worktree is not None:
                self._environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = _validated_objects(worktree)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> LocalGit:
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._temporary.cleanup()

    def git(self, *arguments: str, repository: bool = True,
            allowed: tuple[int, ...] = (0,), timeout: int = _GIT_TIMEOUT) -> str:
        command = ["prlimit", "--as=2147483648", "--cpu=30",
                   f"--fsize={_BUNDLE_LIMIT}", "--", "git"]
        if repository:
            command += ["--git-dir", self.directory]
        command += ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                    "-c", "core.commitGraph=false", "-c", "core.multiPackIndex=false",
                    "-c", "pack.useBitmaps=false", "-c", "push.useBitmaps=false",
                    "-c", "gc.auto=0",
                    "-c", "maintenance.auto=false", "-c", "protocol.ext.allow=never",
                    "-c", "push.followTags=false", "-c", "fetch.fsckObjects=true",
                    *arguments]
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, cwd=self.directory,
                                       env=self._environment, stdout=output, stderr=errors,
                                       start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise GitSyncError("Git operation timed out") from error
            output.seek(0)
            data = output.read(_OUTPUT_LIMIT + 1)
            self._last_exit_code = process.returncode
            if process.returncode not in allowed or len(data) > _OUTPUT_LIMIT:
                raise GitSyncError("Git operation failed; committed work is preserved")
            return data.decode("utf-8", "strict").strip()

    def same_tip(self) -> tuple[str, str]:
        if self.worktree is None:
            raise GitSyncError("Thread checkout is unavailable")
        branch, revision = identity(self.worktree)
        self.git("fsck", "--strict", "--no-reflogs", "--no-dangling", revision)
        if identity(self.worktree) != (branch, revision):
            raise GitSyncError("Thread Git identity changed during verification")
        return branch, revision

    def ancestor(self, older: str, newer: str) -> bool:
        self.git("merge-base", "--is-ancestor", older, newer, allowed=(0, 1))
        return self._last_exit_code == 0

    def has(self, revision: str) -> bool:
        self.git("cat-file", "-e", revision + "^{commit}", allowed=(0, 1))
        return self._last_exit_code == 0

    def remote_ref(self, branch: str) -> str | None:
        ref = "refs/heads/" + branch
        listing = self.git("ls-remote", "--refs", self.source, ref)
        if not listing:
            return None
        fields = listing.split()
        if len(fields) != 2 or fields[1] != ref or not _OID.fullmatch(fields[0]):
            raise GitSyncError("Remote Git branch identity is invalid")
        return fields[0]

    def fetch(self, branch: str) -> tuple[str, str | None]:
        main = self.remote_ref("main")
        if main is None:
            raise GitSyncError("Configured Git main is unavailable")
        thread = self.remote_ref(branch)
        refspecs = ["refs/heads/main:refs/remotes/origin/main"]
        if thread is not None:
            refspecs.append("refs/heads/" + branch + ":refs/remotes/origin/thread")
        self.git("fetch", "--no-tags", "--no-auto-maintenance", self.source, *refspecs)
        if (self.git("rev-parse", "refs/remotes/origin/main") != main
                or thread is not None
                and self.git("rev-parse", "refs/remotes/origin/thread") != thread):
            raise GitSyncError("Remote Git branch moved during fetch")
        return main, thread

    def incoming_bundle(self, local: str, thread: str | None) -> bytes | None:
        refs = ["refs/remotes/origin/main"]
        if thread is not None:
            refs.append("refs/remotes/origin/thread")
        if self.git("rev-list", "--count", *refs, "--not", local) == "0":
            return None
        path = os.path.join(self.directory, "incoming.bundle")
        self.git("bundle", "create", path, *refs, "--not", local)
        if os.path.getsize(path) > _BUNDLE_LIMIT:
            raise GitSyncError("Incoming Git bundle exceeds its bound")
        return Path(path).read_bytes()

    def bundle_commit(self, revision: str, local: str | None) -> bytes:
        """Export one private commit's missing objects for restricted checkout import."""
        self.git("update-ref", "refs/heads/assist-review-merge", revision)
        path = os.path.join(self.directory, "review.bundle")
        arguments = ["bundle", "create", path, "refs/heads/assist-review-merge"]
        if local is not None:
            arguments.extend(("--not", local))
        self.git(*arguments)
        if os.path.getsize(path) > _BUNDLE_LIMIT:
            raise GitSyncError("Merged Git bundle exceeds its bound")
        return Path(path).read_bytes()

    def publish(self, branch: str, desired: str, *, create_only: bool = False) -> str:
        """Publish without rewriting a remote ref; verify desired ancestry."""
        self.git("fsck", "--strict", "--no-reflogs", "--no-dangling", desired)
        ref = "refs/heads/" + branch
        previous = self.remote_ref(branch)
        if previous is not None:
            self.git("fetch", "--no-tags", self.source,
                     ref + ":refs/remotes/origin/published")
            if self.git("rev-parse", "refs/remotes/origin/published") != previous:
                raise GitSyncError("Remote Git branch moved during publication")
            if self.ancestor(desired, previous):
                if self.remote_ref(branch) != previous:
                    raise GitSyncError("Remote Git branch moved during publication")
                return previous
            if create_only:
                raise GitSyncError("Result branch already names different work")
            if not self.ancestor(previous, desired):
                raise GitSyncError("Remote Git branch diverged; local commit is preserved")
        if previous != desired:
            lease = (["--force-with-lease=" + ref + ":"]
                     if previous is None else [])
            try:
                self.git("push", "--porcelain", "--no-verify", *lease,
                         self.source, desired + ":" + ref)
            except GitSyncError:
                # A timeout or lost response is not success; the fresh remote
                # read below alone can prove that our commit was incorporated.
                pass
        latest = self.remote_ref(branch)
        if latest is None:
            raise GitSyncError("Git publication is not verified")
        if latest != desired:
            self.git("fetch", "--no-tags", self.source,
                     ref + ":refs/remotes/origin/verified")
            if not self.ancestor(desired, latest):
                raise GitSyncError("Git publication is not verified")
        return latest
