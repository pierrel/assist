"""Bounded thread-branch Git sync without trusting workspace Git configuration.

The host uses a disposable private bare repository and a server-owned source
binding. Worktree mutations are performed by the restricted sandbox, never by
host Git with agent-writable hooks, helpers, filters or alternates.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import tempfile
import time
from urllib.parse import urlsplit


MAX_BYTES = 128 * 1024 * 1024
MAX_OBJECT_FILES = 10_000
GIT_TIMEOUT = 30
_OID = re.compile(r"[0-9a-f]{40}\Z")
_OBJECT = re.compile(r"(?:[0-9a-f]{38}|pack-[0-9a-f]{40}\.(?:pack|idx|rev))\Z")
_BINDING = "git-sync.json"
_WORKTREE_GIT = ("git -c core.fsmonitor=false -c core.ignoreStat=false "
                 "--git-dir=/workspace/.git --work-tree=/workspace")


class GitSyncError(RuntimeError):
    """Git sync is unavailable; preserve the worktree and show a fixed reason."""


def _read_at(directory: int, name: str, limit: int = 65_536, *, independent=False) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise GitSyncError("Git metadata is not a bounded regular file")
        if independent and info.st_nlink != 1:
            raise GitSyncError("Git objects share hardlinks; migrate to an independent clone")
        value = os.read(fd, limit + 1)
        if len(value) > limit:
            raise GitSyncError("Git metadata is too large")
        return value
    finally:
        os.close(fd)


@contextmanager
def _directory(path: str | Path, *, parent: int | None = None):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


def _branch(value: str) -> str:
    # Ref text is passed only as a single exact refs/heads argument, never a shell fragment.
    if (not value or value in {"main", "HEAD"} or len(value) > 240
            or any(c in value for c in " ~^:?*[\\") or ".." in value
            or "@{" in value or "//" in value or value.startswith(("/", "-"))
            or value.endswith(("/", ".", ".lock"))
            or any(part.startswith(".") or part.endswith(".lock") for part in value.split("/"))
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise GitSyncError("A non-main thread branch is unavailable")
    return value


def identity(worktree: str) -> tuple[str, str]:
    """Read a stable loose/packed SHA-1 branch identity without invoking Git."""
    with _directory(worktree) as root, _directory(".git", parent=root) as git:
        try:
            os.stat("commondir", dir_fd=git, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise GitSyncError("Thread Git directory must be an independent clone")
        head = _read_at(git, "HEAD", 512).decode("ascii").strip()
        if not head.startswith("ref: refs/heads/"):
            raise GitSyncError("A non-main thread branch is unavailable")
        branch = _branch(head.removeprefix("ref: refs/heads/"))
        parts = ("refs", "heads", *branch.split("/"))
        fds = []
        try:
            current = git
            for component in parts[:-1]:
                current = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                  dir_fd=current)
                fds.append(current)
            revision = _read_at(current, parts[-1], 128).decode("ascii").strip()
        except FileNotFoundError:
            packed = _read_at(git, "packed-refs", 512 * 1024).decode("ascii")
            revision = next((line.split(" ", 1)[0] for line in packed.splitlines()
                             if line.endswith(" " + "refs/heads/" + branch)), "")
        finally:
            for fd in reversed(fds):
                os.close(fd)
        if not _OID.fullmatch(revision) or _read_at(git, "HEAD", 512).decode("ascii").strip() != head:
            raise GitSyncError("Git branch identity changed or is invalid")
        return branch, revision


def read_state(thread_dir: str) -> dict | None:
    try:
        with _directory(thread_dir) as directory:
            value = json.loads(_read_at(directory, _BINDING))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError) as error:
        raise GitSyncError("Git source binding is unavailable") from error
    if (not isinstance(value, dict) or value.get("version") != 1
            or not isinstance(value.get("source"), str)
            or not isinstance(value.get("published"), dict)
            or not isinstance(value.get("preflights"), dict)):
        raise GitSyncError("Git source binding is unavailable")
    error = value.get("error")
    if error is not None and (not isinstance(error, str) or len(error) > 256):
        raise GitSyncError("Git source binding is unavailable")
    branch = value.get("branch")
    if branch is not None:
        if not isinstance(branch, str):
            raise GitSyncError("Git source binding is unavailable")
        _branch(branch)
    local_revision = value.get("local_revision")
    if ((branch is not None and local_revision is None)
            or (local_revision is not None and (not isinstance(local_revision, str)
                                                or not _OID.fullmatch(local_revision)))):
        raise GitSyncError("Git source binding is unavailable")
    for name, revision in value["published"].items():
        if not isinstance(name, str) or not isinstance(revision, str) or not _OID.fullmatch(revision):
            raise GitSyncError("Git source binding is unavailable")
        _branch(name)
    published_branch, published_revision = value.get("published_branch"), value.get("published_revision")
    if (published_branch is None) != (published_revision is None):
        raise GitSyncError("Git source binding is unavailable")
    if published_branch is not None and (not isinstance(published_branch, str)
                                         or not isinstance(published_revision, str)):
        raise GitSyncError("Git source binding is unavailable")
    if published_branch is not None and value["published"].get(published_branch) != published_revision:
        raise GitSyncError("Git source binding is unavailable")
    records = [("preflight", entry) for entry in value["preflights"].values()]
    if value.get("intent") is not None:
        records.append(("intent", value["intent"]))
    for field, entry in records:
        if not isinstance(entry, dict) or entry.get("branch") != branch:
            raise GitSyncError("Git source binding is unavailable")
        expected = entry.get("expected")
        if expected is not None and (not isinstance(expected, str) or not _OID.fullmatch(expected)):
            raise GitSyncError("Git source binding is unavailable")
        if field == "preflight" and (not isinstance(entry.get("base"), str)
                                    or not _OID.fullmatch(entry["base"])):
            raise GitSyncError("Git source binding is unavailable")
        if field == "intent" and (not isinstance(entry.get("desired"), str)
                                   or not _OID.fullmatch(entry["desired"])):
            raise GitSyncError("Git source binding is unavailable")
    return value


def _write_state(thread_dir: str, value: dict, filename: str = _BINDING) -> None:
    data = json.dumps(value, sort_keys=True).encode()
    if len(data) > 65_536:
        raise GitSyncError("Git publication metadata is full")
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=thread_dir, prefix=".git-sync-", delete=False) as stream:
            name = stream.name
            os.chmod(name, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, os.path.join(thread_dir, filename))
        with _directory(thread_dir) as directory:
            os.fsync(directory)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def bind(thread_dir: str, source: str) -> None:
    """Bind an explicitly selected source before the first agent can edit it."""
    if (not source or len(source) > 2048 or source.startswith("-") or "::" in source
            or "\n" in source or "\0" in source):
        raise GitSyncError("Unsupported Git source")
    parsed = urlsplit(source)
    if "://" in source and parsed.scheme not in {"http", "https", "ssh", "file"}:
        raise GitSyncError("Unsupported Git source")
    if (parsed.password or parsed.query or parsed.fragment
            or (parsed.scheme in {"http", "https"} and parsed.username)):
        raise GitSyncError("Use Git credentials rather than secrets in the source URL")
    current = read_state(thread_dir)
    if current is not None:
        if current["source"] != source:
            raise GitSyncError("Git source binding changed")
        return
    _write_state(thread_dir, {"version": 1, "source": source, "published": {},
                              "published_branch": None, "published_revision": None,
                              "branch": None, "local_revision": None,
                              "intent": None, "preflights": {}, "error": None})


def authorize_branch(thread_dir: str, worktree: str) -> None:
    """Authorize an independent clone without rebinding an existing thread branch."""
    state = read_state(thread_dir)
    if state is None:
        raise GitSyncError("Git source binding is unavailable")
    _independent_roots((worktree, os.path.join(os.path.dirname(worktree), "tmp"),
                        os.path.join(thread_dir, "agent")))
    with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
        branch, revision = _Store(path).snapshot(worktree)
    if state["branch"] is not None and state["branch"] != branch:
        raise GitSyncError("Thread branch changed; explicit branch reconciliation is required")
    state.update(branch=branch, local_revision=revision, preflights={})
    _write_state(thread_dir, state)


def _independent_roots(roots) -> None:
    """Bounded no-follow inode proof for all writable mounts before legacy admission."""
    count = 0
    deadline = time.monotonic() + GIT_TIMEOUT

    def visit(directory, depth=0):
        nonlocal count
        if depth > 64:
            raise GitSyncError("Independent storage verification exceeds its bound")
        for entry in os.scandir(directory):
            count += 1
            if count > MAX_OBJECT_FILES or time.monotonic() > deadline:
                raise GitSyncError("Independent storage verification exceeds its bound")
            info = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise GitSyncError("Thread files share hardlinks; migrate all writable mounts to independent storage")
            if stat.S_ISDIR(info.st_mode):
                with _directory(entry.name, parent=directory) as child:
                    visit(child, depth + 1)

    try:
        for root in roots:
            if not os.path.lexists(root):
                continue
            with _directory(root) as directory:
                visit(directory)
    except OSError as error:
        raise GitSyncError("Independent thread storage could not be verified") from error


@contextmanager
def ownership(thread_dir: str, worktree: str):
    """Nonblocking workspace fence, including hidden writers and queue-expired turns."""
    state = read_state(thread_dir)
    if state is None:
        if os.path.lexists(os.path.join(worktree, ".git")):
            raise GitSyncError("Existing Git repository needs an operator-verified source binding")
        yield None
        return
    try:
        with _directory(thread_dir) as directory:
            fd = os.open("git-sync.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
    except OSError as error:
        raise GitSyncError("Git workspace ownership is unavailable") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise GitSyncError("Git workspace ownership is unavailable")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GitSyncError("A previous turn still owns this Git workspace") from error
        except OSError as error:
            raise GitSyncError("Git workspace ownership is unavailable") from error
        yield GitSync(thread_dir, worktree)
    finally:
        os.close(fd)


class _Store:
    """One private packed-object store; no workspace config is copied or parsed."""

    def __init__(self, path: str):
        self.path = path
        self.git("init", "--bare", path, repository=False)

    def git(self, *arguments: str, repository: bool = True, allowed=(0,)) -> str:
        environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        environment.update(GIT_TERMINAL_PROMPT="0", GIT_NO_REPLACE_OBJECTS="1")
        command = ["prlimit", "--as=2147483648", "--cpu=30", f"--fsize={MAX_BYTES}",
                   "--", "git"]
        if repository:
            command += ["--git-dir", self.path]
        command += ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                    "-c", "gc.auto=0", "-c", "maintenance.auto=false",
                    "-c", "fetch.unpackLimit=0", "-c", "transfer.unpackLimit=0",
                    "-c", "protocol.ext.allow=never", "-c", "push.followTags=false",
                    "-c", "fetch.fsckObjects=true", *arguments]
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(command, cwd=self.path, env=environment,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                process.wait(timeout=GIT_TIMEOUT)
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise GitSyncError("Git operation timed out") from error
            out.seek(0)
            output = out.read(1024 * 1024 + 1)
            if len(output) > 1024 * 1024 or process.returncode not in allowed:
                raise GitSyncError("Git operation failed; branch sync is pending")
            return output.decode("utf-8", "strict").strip()

    def snapshot(self, worktree: str) -> tuple[str, str]:
        before = identity(worktree)
        copied = count = 0
        deadline = time.monotonic() + GIT_TIMEOUT
        with _directory(worktree) as root, _directory(".git", parent=root) as git, \
                _directory("objects", parent=git) as objects:
            for entry in os.scandir(objects):
                count += 1
                if count > MAX_OBJECT_FILES or time.monotonic() > deadline:
                    raise GitSyncError("Git object snapshot exceeds its bound")
                directory = entry.name
                if directory != "pack" and not re.fullmatch(r"[0-9a-f]{2}", directory):
                    continue  # Never import objects/info/alternates or other execution metadata.
                with _directory(directory, parent=objects) as entries:
                    target = Path(self.path, "objects", directory)
                    target.mkdir(exist_ok=True)
                    for entry in os.scandir(entries):
                        name = entry.name
                        count += 1
                        if count > MAX_OBJECT_FILES or time.monotonic() > deadline:
                            raise GitSyncError("Git object snapshot exceeds its bound")
                        if directory == "pack" and re.fullmatch(r"pack-[0-9a-f]{40}\.(?:bitmap|keep)", name):
                            continue  # Inert local repack accelerators/pins are never imported.
                        if not _OBJECT.fullmatch(name):
                            raise GitSyncError("Unsupported Git object entry")
                        data = _read_at(entries, name, MAX_BYTES - copied, independent=True)
                        copied += len(data)
                        if copied > MAX_BYTES:
                            raise GitSyncError("Git object snapshot is too large")
                        (target / name).write_bytes(data)
        if identity(worktree) != before:
            raise GitSyncError("Git branch changed during snapshot")
        branch, revision = before
        self.git("cat-file", "-e", revision + "^{commit}")
        # Filenames alone do not authenticate agent-writable ancestry. Verify the
        # copied objects' hashes before a force-with-lease can rely on their graph.
        self.git("fsck", "--strict", "--no-reflogs", "--no-dangling", revision)
        self.git("update-ref", "refs/heads/" + branch, revision)
        return before

    def fetch(self, source: str, branch: str) -> str | None:
        # Wildcard-free branch selector; absent branch is distinct from failed auth/fetch.
        listing = self.git("ls-remote", "--refs", source, "refs/heads/" + branch)
        remote = None
        if listing:
            fields = listing.split()
            if len(fields) != 2 or fields[1] != "refs/heads/" + branch or not _OID.fullmatch(fields[0]):
                raise GitSyncError("Remote branch identity is invalid")
            remote = fields[0]
        refs = ["refs/heads/main:refs/remotes/origin/main"]
        if remote is not None:
            refs.append("refs/heads/" + branch + ":refs/remotes/origin/thread")
        self.git("fetch", "--no-tags", "--no-auto-maintenance", source, *refs)
        if remote is not None:
            fetched = self.git("rev-parse", "refs/remotes/origin/thread")
            if fetched != remote:
                raise GitSyncError("Remote branch moved during fetch; retry before the turn")
        return remote

    def ancestor(self, older: str, newer: str) -> bool:
        try:
            self.git("merge-base", "--is-ancestor", older, newer)
            return True
        except GitSyncError:
            return False


def _sandbox_git(sandbox, command: str) -> None:
    if sandbox is None:
        raise GitSyncError("Git sync requires the restricted sandbox")
    # Hooks execute only in the restricted container, and cannot stream their
    # output through Docker's unbounded exec_run materialization on the host.
    result = sandbox.execute("timeout -k 2s 25s sh -c " + shlex.quote(command)
                             + " >/dev/null 2>&1")
    if result.exit_code != 0 or getattr(result, "truncated", False):
        raise GitSyncError("Git worktree needs reconciliation in the restricted sandbox")


def require_clean(sandbox) -> None:
    # Inspect the full index in-container. The backend truncates large stdout,
    # so only exit status crosses that boundary, never a path listing.
    probe = '''import os, subprocess, sys
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
args = sys.argv[1:]
with subprocess.Popen(args + ["ls-files", "-v", "-z"], stdout=subprocess.PIPE,
                      stderr=subprocess.DEVNULL) as process:
    total = 0
    pending = b""
    while chunk := process.stdout.read(65536):
        total += len(chunk)
        if total > 134217728:
            process.kill()
            sys.exit(1)
        entries = (pending + chunk).split(b"\\0")
        pending = entries.pop()
        if len(pending) > 8192 or any(not item.startswith(b"H ") for item in entries):
            process.kill()
            sys.exit(1)
    if pending or process.wait():
        sys.exit(1)
with subprocess.Popen(args + ["status", "--porcelain", "--untracked-files=normal",
                              "--ignore-submodules=none"], stdout=subprocess.PIPE,
                      stderr=subprocess.DEVNULL) as process:
    if process.stdout.read(1):
        process.kill()
        sys.exit(1)
    if process.wait():
        sys.exit(1)
'''
    if sandbox is None:
        raise GitSyncError("Git sync requires the restricted sandbox")
    try:
        _sandbox_git(sandbox, "python -I -c " + shlex.quote(probe) + " " + _WORKTREE_GIT)
    except GitSyncError as error:
        raise GitSyncError("Thread worktree has hidden or uncommitted changes; reconcile before the next turn") from error


class GitSync:
    """Preflight and publication for one bound thread, called under workspace ownership."""

    def __init__(self, thread_dir: str, worktree: str):
        self.thread_dir, self.worktree = thread_dir, worktree
        self.work_id = None
        self.state = read_state(thread_dir)
        if self.state is None:
            raise GitSyncError("Existing Git repository needs an operator-verified source binding")
        if self.state.get("quarantine") or self.state.get("sandbox_in_flight"):
            raise GitSyncError("Previous Git teardown or commit finalization needs operator verification")

    def sandbox_started(self) -> None:
        self.state["sandbox_in_flight"] = True
        _write_state(self.thread_dir, self.state)

    def select_work(self, work_id: str) -> None:
        """Select only this durable work's preflight, including resumed slices."""
        if not isinstance(work_id, str) or not work_id or len(work_id) > 256:
            raise GitSyncError("Git work identity is unavailable")
        self.work_id = work_id

    def receipt(self, child_thread_dir: str) -> None:
        """Durable proof of local child commit/teardown before its terminal handoff."""
        branch, revision = self._branch()
        _write_state(child_thread_dir, {"work_id": self.work_id, "branch": branch,
                                       "revision": revision}, "git-commit-receipt.json")

    def admit(self) -> tuple[str, str]:
        """Authenticate retained local work before model admission, including resumes."""
        if self.state.get("intent"):
            raise GitSyncError("Previous publication outcome is unknown; reconcile before resuming")
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, revision = store.snapshot(self.worktree)
            pending = self.state["preflights"].get(self.work_id)
            floor = self.state.get("local_revision")
            if (branch != self.state.get("branch") or not pending
                    or not store.ancestor(pending["base"], revision)
                    or (floor and not store.ancestor(floor, revision))):
                raise GitSyncError("Retained local Git history changed; reconcile before resuming")
            return branch, revision

    def verified_receipt(self, child_thread_dir: str) -> bool:
        """A child's exact receipt commit must remain in authenticated local history."""
        receipt = read_commit_receipt(child_thread_dir, self.work_id)
        if receipt is None:
            return False
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, revision = store.snapshot(self.worktree)
            floor = self.state.get("local_revision")
            if (branch != self.state.get("branch") or branch != receipt["branch"]
                    or not store.ancestor(receipt["revision"], revision)
                    or (floor and not store.ancestor(floor, revision))):
                raise GitSyncError("Child commit left the thread history; retain result and reconcile")
        return True

    def forget_work(self) -> None:
        """Drop only an already terminal child's retained preflight."""
        if self.state["preflights"].pop(self.work_id, None) is not None:
            _write_state(self.thread_dir, self.state)

    def sandbox_stopped(self) -> None:
        self.state["sandbox_in_flight"] = False
        _write_state(self.thread_dir, self.state)

    def _branch(self) -> tuple[str, str]:
        branch, revision = identity(self.worktree)
        if branch != self.state.get("branch"):
            raise GitSyncError("Thread branch changed; explicit branch reconciliation is required")
        return branch, revision

    def quarantine(self) -> None:
        self.state["quarantine"] = True
        _write_state(self.thread_dir, self.state)

    def prepare(self, sandbox, work_id: str) -> None:
        """Fetch and reconcile before a fresh Git-backed writer, never mid-resume."""
        self.select_work(work_id)
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, local = store.snapshot(self.worktree)
            if (branch, local) != self._branch():
                raise GitSyncError("Thread branch changed during preflight")
            if self.state.get("local_revision") and not store.ancestor(self.state["local_revision"], local):
                raise GitSyncError("Retained local Git history changed; reconcile explicitly")
            remote = store.fetch(self.state["source"], branch)
            intent = self.state.get("intent")
            if intent:
                if (intent["branch"] != branch or not remote
                        or not store.ancestor(intent["desired"], remote)):
                    raise GitSyncError("Previous publication outcome is unknown; reconcile explicitly")
                self.state["published"][branch] = intent["desired"]
                for record in self.state["preflights"].values():
                    if record["branch"] == branch and record["expected"] == intent["expected"]:
                        record["expected"] = intent["desired"]
                self.state.update(published_branch=branch, published_revision=intent["desired"], intent=None)
                _write_state(self.thread_dir, self.state)
            floor = self.state["published"].get(branch)
            if floor and (remote is None or not store.ancestor(floor, remote)):
                raise GitSyncError("Remote thread branch was deleted or rewritten; reconcile explicitly")
            require_clean(sandbox)
            if remote and local != remote and not (store.ancestor(local, remote) or store.ancestor(remote, local)):
                raise GitSyncError("Local and remote thread branches diverged; reconcile explicitly")
            bundle = os.path.join(path, "incoming.bundle")
            store.git("bundle", "create", bundle, "refs/remotes/origin/main",
                      *( ["refs/remotes/origin/thread"] if remote else []))
            data = Path(bundle).read_bytes()
            if len(data) > MAX_BYTES:
                raise GitSyncError("Incoming Git bundle is too large")
            transfer = "/tmp/assist-git-" + os.path.basename(path) + ".bundle"
            response = sandbox.upload_files([(transfer, data)])[0]
            if response.error:
                raise GitSyncError("Could not import remote Git objects")
            main = store.git("rev-parse", "refs/remotes/origin/main")
            commands = [_WORKTREE_GIT + " bundle unbundle " + shlex.quote(transfer),
                        _WORKTREE_GIT + " update-ref refs/remotes/origin/main " + main]
            if remote:
                commands.append(_WORKTREE_GIT + " update-ref "
                                + shlex.quote("refs/remotes/origin/" + branch) + " " + remote)
                if local != remote and store.ancestor(local, remote):
                    commands.append(_WORKTREE_GIT + " merge --ff-only --no-overwrite-ignore --no-autostash " + remote)
            commands.append("rm -- " + shlex.quote(transfer))
            _sandbox_git(sandbox, " && ".join(commands))
            expected_local = remote if remote and store.ancestor(local, remote) else local
            if self._branch() != (branch, expected_local):
                raise GitSyncError("Git preflight did not leave the expected clean thread checkout")
            require_clean(sandbox)
            self.state["preflights"][work_id] = {"branch": branch,
                                                "expected": remote, "base": self._branch()[1]}
            self.state["error"] = None
            self.state["local_revision"] = self._branch()[1]
            _write_state(self.thread_dir, self.state)

    def commit(self, sandbox, message: str) -> None:
        """Commit inside a fresh restricted Git-only generation after model teardown."""
        branch, _ = self._branch()
        pending = self.state["preflights"].get(self.work_id)
        if not pending or pending["branch"] != branch:
            raise GitSyncError("Thread branch changed; reconcile before publication")
        _sandbox_git(sandbox, _WORKTREE_GIT + " add -A && { " + _WORKTREE_GIT
                     + " diff --cached --quiet; code=$?; if [ \"$code\" = 1 ]; then "
                     + "GIT_AUTHOR_NAME=Assist GIT_AUTHOR_EMAIL=assist@localhost "
                     + "GIT_COMMITTER_NAME=Assist GIT_COMMITTER_EMAIL=assist@localhost "
                     + _WORKTREE_GIT + " -c commit.gpgSign=false commit -m "
                     + shlex.quote(message[:4096] or "assistant update")
                     + "; else exit \"$code\"; fi; }")
        require_clean(sandbox)

    def publish(self, *, on_committed=None) -> None:
        """Publish exact FF branch only; callers have already verified sandbox teardown."""
        pending = self.state["preflights"].get(self.work_id)
        if not pending:
            raise GitSyncError("Git turn has no verified remote preflight")
        if self.state.get("intent"):
            raise GitSyncError("Previous publication outcome is unknown; reconcile before publication")
        with tempfile.TemporaryDirectory(prefix="assist-git-") as path:
            store = _Store(path)
            branch, desired = store.snapshot(self.worktree)
            if (branch, desired) != self._branch():
                raise GitSyncError("Thread branch changed during publication")
            if branch != pending["branch"]:
                raise GitSyncError("Thread branch changed; reconcile before publication")
            floor = self.state.get("local_revision")
            if (not store.ancestor(pending["base"], desired)
                    or (floor and not store.ancestor(floor, desired))):
                raise GitSyncError("Local thread history changed; sync is pending")
            self.state["local_revision"] = desired
            _write_state(self.thread_dir, self.state)
            if on_committed is not None:
                on_committed()
            remote = store.fetch(self.state["source"], branch)
            expected = pending["expected"]
            if remote != expected or (remote and not store.ancestor(remote, desired)):
                raise GitSyncError("Remote thread branch advanced or local history changed; sync is pending")
            if desired != remote:
                self.state["intent"] = {"branch": branch, "expected": expected, "desired": desired}
                _write_state(self.thread_dir, self.state)
                receipt = store.git("push", "--porcelain", "--no-verify",
                          "--force-with-lease=refs/heads/" + branch + ":" + (expected or ""),
                          self.state["source"], desired + ":refs/heads/" + branch, allowed=(0, 1))
                if any(line.startswith("!\t") for line in receipt.splitlines()):
                    self.state["intent"] = None
                    _write_state(self.thread_dir, self.state)
                    raise GitSyncError("Remote rejected publication; retry on a later successful turn")
            listing = store.git("ls-remote", "--refs", self.state["source"], "refs/heads/" + branch)
            if listing.split() != [desired, "refs/heads/" + branch]:
                raise GitSyncError("Remote publication is not yet verified; sync is pending")
            self.state["published"][branch] = desired
            del self.state["preflights"][self.work_id]
            # Other suspended work shares this thread branch. Advance its exact
            # expectation only across our own verified FF publication, never a
            # merely observed user move. Keep its original ancestry floor.
            for record in self.state["preflights"].values():
                if record["branch"] == branch and record["expected"] == expected:
                    record["expected"] = desired
            self.state.update(published_branch=branch, published_revision=desired,
                              intent=None, error=None)
            _write_state(self.thread_dir, self.state)

    def failed(self, error: Exception) -> None:
        self.state["error"] = str(error)[:256]
        _write_state(self.thread_dir, self.state)


def source_label(source: str) -> str:
    """Only a repository basename, including scp-style SSH sources."""
    path = urlsplit(source).path if "://" in source else source.split(":", 1)[-1]
    return path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")[:200]


def read_commit_receipt(child_thread_dir: str, work_id: str) -> dict | None:
    """Read a bounded exact-work receipt; callers must authenticate its ancestry."""
    try:
        with _directory(child_thread_dir) as directory:
            value = json.loads(_read_at(directory, "git-commit-receipt.json", 1024))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError) as error:
        raise GitSyncError("Child Git completion receipt is unavailable") from error
    if (not isinstance(value, dict) or not isinstance(value.get("branch"), str)
            or not isinstance(value.get("revision"), str)
            or not _OID.fullmatch(value["revision"])):
        raise GitSyncError("Child Git completion receipt is unavailable")
    _branch(value["branch"])
    return value if value.get("work_id") == work_id else None


def workspace(thread_dir: str, worktree: str) -> dict:
    """Bound phone metadata; no remote URL, dirty probe or host Git invocation."""
    value = {"repo_key": None, "repo_label": "No repository", "branch": None,
             "revision": None, "published_branch": None, "published_revision": None,
             "sync_error": None}
    try:
        state = read_state(thread_dir)
        if state is None:
            if os.path.lexists(os.path.join(worktree, ".git")):
                value["repo_label"] = "Repository needs operator binding"
                value["sync_error"] = "Existing Git repository needs an operator-verified source binding"
            return value
        source = state["source"]
        value.update(repo_key=hashlib.sha256(source.encode()).hexdigest()[:20],
                     repo_label=source_label(source),
                     published_branch=state.get("published_branch"),
                     published_revision=state.get("published_revision"),
                     sync_error=state.get("error"))
        branch, revision = identity(worktree)
        value.update(branch=branch, revision=revision)
    except (GitSyncError, OSError, UnicodeError):
        value["repo_label"] = "Repository unavailable"
        value["sync_error"] = "Git workspace metadata is unavailable"
    return value
