import os

def read_file(path: str):
    """Returns the full contents of file at path"""
    with open(path, 'r') as f:
        return f.read()

def create_filesystem(root_dir: str,
                      structure: dict):
    """Creates a directory structure and files according to `structure`. For example:
    {"README.org": "This is the readme file",
    "gtd": {"inbox.org": "This is the inbox file"},
           {"projects": {"project1.org": "This is a project file"}}}

    Creates:
    a README.org file with content "This is the readme file"
    a gtd directory
    a gtd/inbox.org file with content "This is the inbox file"
    ..."""
    for name, content in structure.items():
        path = os.path.join(root_dir, name)

        if isinstance(content, str):
            # Create a file with the given content
            with open(path, 'w') as f:
                f.write(content)
        elif isinstance(content, dict):
            # Create a directory and recursively process its contents
            os.makedirs(path, exist_ok=True)
            create_filesystem(path, content)
