"""Unit tests for Proposal.from_dict — shape-tolerant parsing of proposals.json records
(defensive against a corrupted/forward-versioned file so the /geo page never 500s)."""
from assist.geo.proposals import Proposal

_WA = [-124.8, 45.5, -116.9, 49.1]


def _rec(**over):
    d = {"slug": "us/washington", "display_name": "Washington", "bbox": list(_WA),
         "size_bytes": 700_000_000, "origin_tid": "t1", "user_request": "x",
         "created_at": "2026-07-11T00:00:00+00:00"}
    d.update(over)
    return d


def test_from_dict_valid_roundtrips():
    p = Proposal.from_dict(_rec())
    assert p.slug == "us/washington" and p.created_at == "2026-07-11T00:00:00+00:00"


def test_from_dict_non_string_created_at_falls_back():
    # a corrupted/forward-versioned record with a non-string created_at must not slip an
    # int through — geo.py sorts proposals by created_at and a mixed str/int comparison
    # would raise TypeError and 500 the /geo page.
    p = Proposal.from_dict(_rec(created_at=12345))
    assert isinstance(p.created_at, str) and p.created_at   # coerced to a real ISO string
    p2 = Proposal.from_dict(_rec(created_at=None))
    assert isinstance(p2.created_at, str) and p2.created_at


def test_from_dict_malformed_required_fields_skipped():
    assert Proposal.from_dict(_rec(slug=None)) is None
    assert Proposal.from_dict(_rec(origin_tid=123)) is None
