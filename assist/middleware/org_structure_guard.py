"""Stop the small model from inserting a new org heading mid-section.

Failure mode (2026-06-03 prod threads, reproduced in
``edd/eval/test_org_insertion.py``): asked to add a section to a large
``.org`` file, the model anchors its ``edit_file`` ``old_string`` on a
body line — often an org *bold* line like ``*Direction.*`` that it
mistakes for a heading — so the new heading lands in the MIDDLE of an
existing section and splits it (orphaning the rest of that section).  Six
skill/prompt variants failed to fix this (the model ignores the guidance
on a file this size), so this is a deterministic backstop — the PR-#120
pattern: the prompt holds the easy cases, a deterministic check holds the
hard one.

Detection (purely from the edit args — no file read needed): the model is
*adding a heading* when ``new_string`` introduces a heading line (``*``…
followed by a SPACE) that isn't in ``old_string``.  A correct insertion
anchors that on a heading line (``old_string`` contains the heading the
new section goes next to — verified across the passing simple/inbox
cases).  A mid-section mis-anchor has ``old_string`` made of pure body
text (no heading line at all — verified on the failing case, where
``old_string`` was the bold line ``*What landed first as a stopgap*``).
So: new heading added + ``old_string`` has no heading line -> reject the
edit BEFORE it applies and return a recipe-style redirect telling the
model to anchor on a real heading.  The file is never left broken.

``status="error"`` is preserved so ``LoopDetectionMiddleware`` still sees
an error event (same rationale as ``WriteCollisionMiddleware``).  Place
this middleware *before* ``LoopDetectionMiddleware``.
"""
import logging
import re
from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

# A heading is asterisks followed by a SPACE: "* Foo", "** Bar".  An org
# *bold* line ("*Direction.*", "*Note.* text") has no space after the "*"
# and is body text, NOT a heading — this distinction is the whole bug.
_HEADING_RE = re.compile(r"^\*+ ")


def _level(line: str) -> int | None:
    """Heading depth (``* `` -> 1, ``** `` -> 2), or None if not a heading."""
    st = line.lstrip()
    if _HEADING_RE.match(st):
        return len(st) - len(st.lstrip("*"))
    return None


def _heading_levels(text: str) -> list[int]:
    return [lvl for line in text.splitlines() if (lvl := _level(line)) is not None]


def _has_heading(text: str) -> bool:
    return bool(_heading_levels(text))


def _new_heading_lines(old_string: str, new_string: str) -> list[str]:
    """Heading lines introduced by new_string that aren't in old_string."""
    old_heads = {l for l in old_string.splitlines() if _level(l) is not None}
    return [l for l in new_string.splitlines()
            if _level(l) is not None and l not in old_heads]


def _would_split(args: dict) -> bool:
    """The edit inserts a NEW heading anchored such that it lands mid-section.

    A correct insertion places the new heading immediately before an
    existing heading at the SAME level or shallower (the heading it should
    precede), or at end of file anchored on the last same-level heading.  It
    splits a section when the new heading is SHALLOWER (more top-level) than
    every heading in ``old_string`` — including the case where old_string has
    NO heading at all (anchored on pure body text).  Examples seen in evals:
    ``* Moonshot`` anchored on a body line (no heading) -> split; ``* Moonshot``
    anchored on a ``** TODO`` sub-heading (level 2) -> splits the parent
    ``*`` section; ``* Reading`` anchored on ``* Errands`` (same level) -> OK.
    """
    old = args.get("old_string")
    new = args.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    new_heads = _new_heading_lines(old, new)
    if not new_heads:
        return False  # not adding a heading — none of our concern
    new_min = min(_level(l) for l in new_heads)        # shallowest new heading
    old_levels = _heading_levels(old)
    old_min = min(old_levels) if old_levels else 10 ** 6  # no heading -> "infinitely deep"
    # New heading is shallower than the anchor's shallowest heading -> it
    # breaks out of / splits the section the anchor sits inside.
    return new_min < old_min


def _redirect_message() -> str:
    # Deliberately does NOT start with an error prefix ("error:", "cannot",
    # "failed to", …) and is returned with status="success", so
    # LoopDetectionMiddleware does NOT count it as a loop error.  The model
    # was observed cycling through *different* anchors (body line, then a
    # sub-heading) — each a step toward a valid one — so we want it to keep
    # correcting rather than be killed after 2 identical rejections.  A
    # genuinely stuck model that repeats the *same* args still trips
    # LoopDetection Pattern B (same-tool-same-args), and recursion_limit
    # bounds the worst case.
    return (
        "Adjust the anchor before this edit can apply.  Your `old_string` "
        "anchors the new heading on body text (or a deeper sub-heading), so "
        "the new heading would land in the MIDDLE of an existing section and "
        "split it.\n\n"
        "To add the section, anchor on a HEADING line at the same level or "
        "shallower:\n"
        "Step 1: pick the existing heading your new section should come "
        "immediately BEFORE.\n"
        "Step 2: set `old_string` to that exact heading line.\n"
        "Step 3: set `new_string` to your new heading and its body, then that "
        "same heading line.\n\n"
        "A heading is asterisks then a SPACE (`* Foo`, `** Bar`).  A line like "
        "`*Word.*` (no space after the `*`) is BOLD text, NOT a heading — do "
        "not anchor on it.\n\n"
        "To put the section at the very end, anchor on the LAST top-level "
        "heading and its body, and add yours after."
    )


def _is_org_edit(request: ToolCallRequest) -> bool:
    tool = getattr(request, "tool", None)
    name = getattr(tool, "name", None) or request.tool_call.get("name", "")
    if name != "edit_file":
        return False
    args = request.tool_call.get("args") or {}
    path = args.get("file_path") or args.get("path") or ""
    return isinstance(path, str) and path.endswith(".org")


def _reject(request: ToolCallRequest) -> ToolMessage:
    args = request.tool_call.get("args") or {}
    path = args.get("file_path") or args.get("path") or "<file>"
    logger.warning(
        "OrgStructureGuard: blocked a mid-section heading insert on %s "
        "(new heading anchored shallower than the anchor's headings)", path)
    # status="success" + a non-error-prefixed message so the corrective
    # redirect does NOT register as a loop error in LoopDetectionMiddleware —
    # the model should keep re-anchoring until it finds a valid heading.
    return ToolMessage(
        content=_redirect_message(),
        name="edit_file",
        tool_call_id=request.tool_call.get("id", ""),
        status="success",
    )


class OrgStructureGuardMiddleware(AgentMiddleware):
    """Reject a ``.org`` ``edit_file`` that would drop a new heading
    mid-section (new heading anchored on body text), with a redirect to
    anchor on a real heading.  The bad edit never applies, so the file is
    not left broken.  Place before ``LoopDetectionMiddleware``.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if _is_org_edit(request) and _would_split(request.tool_call.get("args") or {}):
            return _reject(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        if _is_org_edit(request) and _would_split(request.tool_call.get("args") or {}):
            return _reject(request)
        return await handler(request)
