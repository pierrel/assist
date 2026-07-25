"""Shared model and transport contract for subagent tasks."""

MAX_TASK_DESCRIPTION_CHARS = 64_000

GENERAL_PURPOSE_DESCRIPTION = (
    "Use this instead of direct work tools for each non-trivial, specific, well-defined "
    "non-research todo using supplied inputs and the local workspace, including the one "
    "unblocked step in a dependent chain. When two or more independent non-research "
    "todos are briefable, you MUST launch one general-purpose task per todo together "
    "before using filesystem or action tools. Use it to "
    "extract or transform a page at a URL literally supplied by the user. Never "
    "use it for a research todo or a preparatory slice of research, including "
    "date or place interpretation and query preparation, or for investigation, "
    "verification, current facts, or external fact-finding. If every requested "
    "outcome is research, use no general-purpose task."
)
