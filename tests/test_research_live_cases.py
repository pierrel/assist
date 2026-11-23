"""Freeze natural bundled/component comparisons before any model trial."""

import json
from collections import defaultdict
from pathlib import Path


_MANIFEST = (Path(__file__).resolve().parents[1] / "edd" / "eval" /
             "research_live_cases.json")


def test_bundled_components_share_notes_sources_and_deadline():
    cases = json.loads(_MANIFEST.read_text())["cases"]
    sources = json.loads(_MANIFEST.with_name("research_live_sources.json").read_text())[
        "fixtures"]
    assert len({case["id"] for case in cases}) == len(cases)
    groups = defaultdict(list)
    for case in cases:
        groups[case["pair_group"]].append(case)
        assert case["expected_updated_files"]
        assert set(case["expected_updated_files"]) <= set(case["workspace_files"])
        original_notes = "\n".join(case["workspace_files"].values()).casefold()
        assert all(marker.casefold() not in original_notes
                   for marker in case["diagnostic_source_markers"])
        lowered = case["prompt"].casefold()
        assert not any(word in lowered for word in (
            "search_internet", "read_url", "subagent", "delegate", "tool call"))
        fixture = sources[case["fixture"]]
        urls = {row["url"] for row in fixture}
        assert len(urls) == len(fixture)
        assert set(case.get("primary_reference_urls", ())) <= urls
        assert all(row["tags"] and isinstance(row["status"], int)
                   for row in fixture)
        available_text = "\n".join(str(row[key]) for row in fixture
                                   for key in ("snippet", "body"))
        assert all(marker in available_text
                   for marker in case["diagnostic_source_markers"])
    for name, expected in (("music-bundled-components", 3),
                           ("ballot-bundled-components", 4),
                           ("helio-three-components", 4)):
        group = groups[name]
        assert len(group) == expected
        assert len({json.dumps(case["workspace_files"], sort_keys=True)
                    for case in group}) == 1
        assert len({case["fixture"] for case in group}) == 1
        assert len({case["deadline_seconds"] for case in group}) == 1
    assert groups["helio-three-partial-source"][0]["fixture"] == "helio-partial"
