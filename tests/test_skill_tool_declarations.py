"""Bundled Agent Skills declarations for non-kernel Assist tools."""
from pathlib import Path

from deepagents.middleware.skills import _parse_skill_metadata

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "assist/skills/travel/SKILL.md": (
        "travel", "directions", "list_regions", "find_regions",
        "propose_region_download"),
    "assist/skills/explore-website/SKILL.md": ("read_url",),
    "assist/skills/schedule/SKILL.md": (
        "create_schedule", "list_schedules", "modify_schedule",
        "pause_schedule", "resume_schedule", "delete_schedule"),
    "assist/skills/subscribe-events/SKILL.md": (
        "create_subscription", "list_subscriptions",
        "modify_subscription", "delete_subscription"),
    "assist/skills/egress/SKILL.md": (
        "request_egress", "list_allowed_hosts", "remove_allowed_host"),
    "assist/web_skills/send-email/SKILL.md": ("send_email",),
    "assist/web_skills/render/SKILL.md": ("map_data",),
}


def _metadata(path):
    text = path.read_text(encoding="utf-8")
    return _parse_skill_metadata(
        text, f"/{path.parent.name}/SKILL.md", path.parent.name)


def test_bundled_non_kernel_owners_use_official_allowed_tools_metadata():
    actual = {}
    for root in ("assist/skills", "assist/main_skills", "assist/web_skills"):
        for path in sorted((ROOT / root).glob("*/SKILL.md")):
            metadata = _metadata(path)
            assert metadata is not None, path
            if metadata["allowed_tools"]:
                actual[str(path.relative_to(ROOT))] = tuple(
                    metadata["allowed_tools"])

    assert actual == EXPECTED
