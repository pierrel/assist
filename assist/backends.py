import tempfile
from typing import Callable

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


def routes(rt, stateful_paths: list[str]) -> dict:
    the_routes = {}
    for stateful_path in stateful_paths:
        the_routes[stateful_path] = StateBackend(rt)

    return the_routes

def create_composite_backend(fs_root: str = None,
                             stateful_paths: list[str] = []) -> Callable:

    if not fs_root:
        fs_root = tempfile.mkdtemp()
    return lambda rt: CompositeBackend(
        default=FilesystemBackend(root_dir=fs_root,
                                  virtual_mode=True),
        routes=routes(rt, stateful_paths)
    )


def create_sandbox_composite_backend(sandbox_backend: BackendProtocol,
                                     stateful_paths: list[str] | None = None) -> Callable:
    """Create a composite backend that routes to a sandbox for default operations.

    Ephemeral paths (question.txt, large_tool_results/) go to StateBackend,
    everything else goes to the sandbox backend.
    """
    if stateful_paths is None:
        stateful_paths = STATEFUL_PATHS

    return lambda rt: CompositeBackend(
        default=sandbox_backend,
        routes=routes(rt, stateful_paths)
    )

