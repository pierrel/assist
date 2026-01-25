"""
Conversation capture module for eval-driven development.

This module uses an intelligent agent to analyze conversations and generate
focused, reproducible test cases with only the necessary files.
"""
import os
import json
import re
import shutil
import logging
from datetime import datetime
from typing import Dict, List

from assist.thread import Thread
from assist.model_manager import select_chat_model
from edd.agent import create_capture_agent

logger = logging.getLogger(__name__)


def sanitize_dirname(description: str) -> str:
    """
    Create a sanitized directory name from a description.

    Takes first 5 words, lowercases, replaces non-alphanumeric with hyphens.
    Returns format: {timestamp}-{slug}
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Take first 5 words
    words = description.split()[:5]
    slug = " ".join(words)

    # Convert to lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug.lower())

    # Remove leading/trailing hyphens and limit length
    slug = slug.strip('-')[:50]

    return f"{timestamp}-{slug}"


def capture_conversation(thread: Thread, reason: str, improvements_dir: str) -> str:
    """
    Capture a conversation using an intelligent agent.

    The agent will:
    - Analyze the conversation
    - Identify relevant files
    - Generate a focused test case
    - Create minimal supporting files

    Args:
        thread: The Thread object with the conversation
        reason: User's reason for capturing
        improvements_dir: Base directory for improvements

    Returns:
        Path to the created capture directory

    Raises:
        ValueError: If the conversation is empty
    """
    # Validate thread has messages
    messages = thread.get_messages()
    if not messages:
        raise ValueError("Cannot capture empty conversation")

    # Create improvements directory if it doesn't exist
    os.makedirs(improvements_dir, exist_ok=True)

    # Generate directory name
    try:
        description = thread.description()
    except Exception:
        description = "captured conversation"

    dirname = sanitize_dirname(description)

    # Handle directory name collisions
    capture_dir = os.path.join(improvements_dir, dirname)
    counter = 2
    while os.path.exists(capture_dir):
        capture_dir = os.path.join(improvements_dir, f"{dirname}-{counter}")
        counter += 1

    os.makedirs(capture_dir)

    # Copy the domain directory from the original thread to the capture directory
    # This gives the agent safe, sandboxed access to read and analyze files
    source_dir = thread.working_dir
    source_domain_dir = os.path.join(source_dir, "domain")
    capture_domain_dir = os.path.join(capture_dir, "domain")

    if os.path.exists(source_domain_dir):
        logger.info(f"Copying domain directory from {source_domain_dir} to {capture_domain_dir}")
        shutil.copytree(source_domain_dir, capture_domain_dir,
                       ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.pytest_cache'))
        logger.debug(f"Domain directory copied successfully")
    else:
        logger.debug(f"No domain directory found at {source_domain_dir}, creating empty domain dir")
        os.makedirs(capture_domain_dir, exist_ok=True)

    # Prepare conversation data for the agent
    conversation_summary = _format_conversation_for_agent(messages, thread, reason)

    # Create the capture agent with the capture directory as working dir
    # The agent can now safely access domain/ within its sandboxed working directory
    model = select_chat_model("gpt-oss-20b", 0.1)
    agent = create_capture_agent(model, capture_dir)

    # Get first user message for context
    first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

    # Construct the agent's task
    task = f"""You are an expert test case generator. Your task is to analyze a conversation and create a focused, reproducible test case.

## CRITICAL INSTRUCTIONS

**Your working directory structure**:
```
. (your current directory - the capture output)
├── domain/           # Copy of the original conversation's files
│   └── (source files used in the conversation)
├── test_case.py      # You will create this
├── conversation.json # You will create this
└── README.md         # You will create this
```

**What to do**:
1. Use `ls domain` to see what files exist from the original conversation
2. Use `read_file("domain/filename")` to read relevant files
3. Analyze which files are NECESSARY for the test (not all of them!)
4. Write test_case.py with ONLY the necessary files included
5. Write conversation.json with metadata (include list of files you chose)
6. Write README.md with documentation

## Conversation Context

**Thread ID**: {thread.thread_id}
**Reason for Capture**: {reason}
**First User Message**: {first_user_msg[:200]}...

## Full Conversation

{conversation_summary}

## Example Test Case Structure

You should write a test_case.py that looks like this:

```python
import os
import tempfile
from textwrap import dedent
from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness
from tests.integration.validation.utils import read_file, create_filesystem

class TestCaptured_{thread.thread_id.replace('-', '_')}(TestCase):
    \"\"\"
    {reason}
    \"\"\"

    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def test_scenario(self):
        # Setup filesystem with ONLY necessary files
        agent, root = self.create_agent({{
            # Add minimal files here
        }})

        # Send key message
        res = agent.message({repr(first_user_msg)})

        # Add assertions
        self.assertIsNotNone(res)
```

## START NOW

Step 1: Run `ls domain` to see available files
Step 2: Read only the relevant files with `read_file("domain/filename")`
Step 3: Write test_case.py with proper assertions and MINIMAL filesystem setup
Step 4: Write conversation.json and README.md

Begin!
"""

    # Invoke the agent to generate the test case
    logger.info(f"Invoking capture agent for thread {thread.thread_id}")
    logger.debug(f"Capture directory: {capture_dir}")
    logger.debug(f"Domain directory copied to: {capture_domain_dir}")

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            {"configurable": {"thread_id": f"capture-{thread.thread_id}"}}
        )
        logger.debug(f"Agent completed successfully")
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        _write_fallback_files(capture_dir, thread, messages, reason)
        return capture_dir

    # The agent will have written the files directly to capture_dir
    # We just need to verify they exist and return the path

    required_files = ["test_case.py", "conversation.json", "README.md"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(capture_dir, f))]

    if missing_files:
        # Agent didn't complete the task properly
        logger.warning(f"Agent did not create all required files. Missing: {missing_files}")
        logger.info("Using fallback file generation")
        _write_fallback_files(capture_dir, thread, messages, reason)
    else:
        logger.info(f"All files created successfully in {capture_dir}")

    return capture_dir


def _format_conversation_for_agent(messages: List[dict], thread: Thread, reason: str) -> str:
    """Format the conversation history for the agent to analyze."""
    formatted = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000] + "\n... (truncated)"

        formatted.append(f"### Message {i+1} ({role})")
        formatted.append(content)
        formatted.append("")

    return "\n".join(formatted)


def _write_fallback_files(capture_dir: str, thread: Thread, messages: List[dict], reason: str):
    """Write basic fallback files if the agent fails."""

    # Basic conversation.json
    metadata = {
        "thread_id": thread.thread_id,
        "captured_at": datetime.now().isoformat(),
        "reason": reason,
        "message_count": len(messages),
        "first_message": messages[0]["content"] if messages else "No messages"
    }

    with open(os.path.join(capture_dir, "conversation.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Basic README
    readme = f"""# Captured Conversation

## Reason
{reason}

## Details
- Thread ID: {thread.thread_id}
- Messages: {len(messages)}
- Captured: {datetime.now().isoformat()}

## Note
This capture was created with fallback mode. The agent did not complete generation.
Review the conversation history and create the test case manually.
"""

    with open(os.path.join(capture_dir, "README.md"), "w") as f:
        f.write(readme)

    # Basic test_case.py template
    test_template = '''import os
import tempfile
from unittest import TestCase

from assist.model_manager import select_chat_model
from assist.agent import create_agent, AgentHarness
from tests.integration.validation.utils import create_filesystem

class TestCapturedConversation(TestCase):
    """
    TODO: Customize this test case based on the conversation.

    Reason: {reason}
    """

    def setUp(self):
        self.model = select_chat_model("gpt-oss-20b", 0.1)

    def create_agent(self, filesystem: dict):
        root = tempfile.mkdtemp()
        create_filesystem(root, filesystem)
        return AgentHarness(create_agent(self.model, root)), root

    def test_conversation(self):
        # TODO: Set up the filesystem with relevant files
        agent, root = self.create_agent({{}})

        # TODO: Send the key message
        res = agent.message("REPLACE WITH FIRST MESSAGE")

        # TODO: Add appropriate assertions
        self.assertIsNotNone(res)
'''

    with open(os.path.join(capture_dir, "test_case.py"), "w") as f:
        f.write(test_template.format(reason=reason))
