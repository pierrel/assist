from langchain_core.tools import tool
import os
# Tools for working with the filesystem


@tool
def list_files(root: str) -> list[str]:
    """List all files recursively in the given ``root`` directory.

    Use this to learn more about the structure of the project files.

    Returns:
    list[str]: A list of the absolute paths of all files under ``root``."""
    return [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]


@tool
def file_contents(path: str) -> str:
    """Returns the contents of the file at `path`.

    Args:
        path (str): The path to the desired file.

    Returns:
        str: The content of the specified file as a string.
    """
    with open(path, 'r') as f:
        return f.read()
