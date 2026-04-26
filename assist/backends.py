import os
import tempfile

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol

STATEFUL_PATHS = [
    "/question.txt",
    "question.txt",
    "/dev_notes.txt",
    "dev_notes.txt",
    "/large_tool_results/",
    "large_tool_results/",
    "large_tool_results",
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

