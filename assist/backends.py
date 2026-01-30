import tempfile
from typing import Callable

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

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
    
