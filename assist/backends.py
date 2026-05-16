import os
import tempfile

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol


class _ReferencesNormalizingBackend(FilesystemBackend):
    """FilesystemBackend that strips a leading ``references/`` (or
    ``/references/``) from incoming paths before resolution.

    The research agent's backend is rooted at ``<working_dir>/references``;
    if the agent then writes to ``references/foo.org`` (because its prompt
    or the caller's task description still mentions the directory name),
    ``virtual_mode=True`` resolution would produce a NESTED
    ``<working_dir>/references/references/foo.org`` instead of the
    intended ``<working_dir>/references/foo.org``.

    Stripping the prefix here keeps the file in the right place even when
    Qwen3.6 slips and prefixes its own paths.  This is purely a
    convenience layer — the actual confinement still comes from
    ``virtual_mode=True``'s ``..`` blocking, not from this normalization.
    """

    def _normalize(self, path):
        if not path:
            return path
        for prefix in ("/references/", "references/"):
            if path.startswith(prefix):
                stripped = path[len(prefix):]
                return "/" + stripped if path.startswith("/") else stripped
        return path

    def read(self, file_path, *args, **kwargs):
        return super().read(self._normalize(file_path), *args, **kwargs)

    def write(self, file_path, *args, **kwargs):
        return super().write(self._normalize(file_path), *args, **kwargs)

    def edit(self, file_path, *args, **kwargs):
        return super().edit(self._normalize(file_path), *args, **kwargs)

    # Override ls / grep / glob (the protocol's current API) — NOT
    # ls_info / grep_raw / glob_info, which deepagents marks deprecated
    # for v0.7 removal.  The framework calls the new names; overriding
    # only the deprecated ones leaves ``_normalize`` as dead code (see
    # the parallel comment in DockerSandboxBackend).

    def ls(self, path, *args, **kwargs):
        return super().ls(self._normalize(path), *args, **kwargs)

    def grep(self, pattern, path=None, glob=None):
        return super().grep(pattern, self._normalize(path), glob)

    def glob(self, pattern, path="/", *args, **kwargs):
        return super().glob(pattern, self._normalize(path), *args, **kwargs)

STATEFUL_PATHS = [
    "/question.txt",
    "question.txt",
    "/dev_notes.txt",
    "dev_notes.txt",
    "/large_tool_results/",
    "large_tool_results/",
    "large_tool_results",
    # deepagents 0.6.1's SummarizationMiddleware offloads the pre-
    # summarization conversation history here as a per-thread file so
    # the agent can read_file it back if a summary is too thin.  Without
    # this routing the offload lands in the user's git-tracked workspace
    # (the default backend root) — wrong layer.  Mirror large_tool_results/
    # routing so it's thread-local + ephemeral.
    "/conversation_history/",
    "conversation_history/",
    "conversation_history",
]

SKILLS_ROUTE = "/skills/"
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


def routes(stateful_paths: list[str]) -> dict:
    routes = {path: StateBackend() for path in stateful_paths}
    routes[SKILLS_ROUTE] = FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True)
    return routes


def create_composite_backend(fs_root: str = None,
                             stateful_paths: list[str] = []) -> CompositeBackend:

    if not fs_root:
        fs_root = tempfile.mkdtemp()
    return CompositeBackend(
        default=FilesystemBackend(root_dir=fs_root,
                                  virtual_mode=True),
        routes=routes(stateful_paths)
    )


def create_sandbox_composite_backend(sandbox_backend: BackendProtocol,
                                     stateful_paths: list[str] | None = None) -> CompositeBackend:
    """Create a composite backend that routes to a sandbox for default operations.

    Ephemeral paths (question.txt, large_tool_results/) go to StateBackend,
    everything else goes to the sandbox backend.
    """
    if stateful_paths is None:
        stateful_paths = STATEFUL_PATHS

    return CompositeBackend(
        default=sandbox_backend,
        routes=routes(stateful_paths)
    )


def create_references_backend(working_dir: str) -> CompositeBackend:
    """Create a composite backend rooted at ``<working_dir>/references/``.

    Used by the research sub-agent so its file operations are confined to
    a single directory inside the user's workspace.  The same
    ``STATEFUL_PATHS`` routing applies — ``question.txt`` etc. still go
    to ``StateBackend`` (ephemeral, never hits disk).

    Uses ``_ReferencesNormalizingBackend`` so paths like
    ``references/foo.org`` (which would otherwise nest under the already-
    references-rooted backend) get the leading ``references/`` stripped.

    Construction must NOT create ``references/`` on disk.
    ``ThreadManager.new()`` instantiates a ``Thread`` with
    ``sandbox_backend=None`` synchronously inside ``POST /threads*``, and
    ``Thread`` builds the research sub-agent eagerly via
    ``create_research_agent`` → ``create_references_backend``.  If we
    eagerly mkdir ``references/`` here, the workspace is no longer empty
    when the background ``_initialize_thread`` later constructs
    ``DomainManager`` — its ``is_empty`` check then short-circuits the
    git clone, leaving the thread without a ``.git/`` and breaking the
    post-run sync (prod incident 2026-05-04, thread 20260504091127-
    183e8f35).  ``ReferencesCleanupRunnable._ensure_dir`` mkdirs at
    invoke time; that is the right moment.
    """
    references_root = os.path.join(working_dir, "references")
    return CompositeBackend(
        default=_ReferencesNormalizingBackend(
            root_dir=references_root, virtual_mode=True,
        ),
        routes=routes(STATEFUL_PATHS),
    )

