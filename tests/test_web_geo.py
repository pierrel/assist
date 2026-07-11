"""Unit tests for the /geo view helpers (rendering + CSRF).

The route handlers are thin (call the Provisioner/ProposalStore, both unit-tested
elsewhere); here we pin the model-facing HTML + the CSRF gate. Importing manage.web.geo
pulls the web stack; the pure helpers don't touch the (possibly-None) geo stores.
"""
import pytest
from fastapi import HTTPException

from assist.geo.model import Region, STATE_IMPORTING, STATE_READY
from assist.geo.proposals import Proposal
from manage.web import geo


def _region(slug, name, transit=False, state=STATE_READY):
    return Region(slug=slug, display_name=name, bbox=(-124.0, 32.0, -114.0, 42.0),
                  has_transit=transit, state=state)


def _proposal(slug="us/oregon", name="Oregon", size=700_000_000):
    return Proposal(slug=slug, display_name=name, bbox=(-124.6, 41.9, -116.4, 46.3),
                    size_bytes=size, origin_tid="t1", user_request="directions in Eugene",
                    created_at="2026-07-11T00:00:00+00:00")


def test_fmt_size():
    assert geo._fmt_size(None) == "size unknown" and geo._fmt_size(0) == "size unknown"
    assert geo._fmt_size(700_000_000) == "~700 MB"
    assert geo._fmt_size(3_500_000_000) == "~3.5 GB"


def test_render_lists_regions_with_actions():
    html = geo._render(
        [_region("norcal", "Northern California", transit=True),
         _region("us/oregon", "Oregon", transit=False)],
        [])
    assert "Northern California" in html and "Oregon" in html
    assert "transit" in html and "no transit" in html
    # a ready region without transit offers add-transit + delete; the CSRF token is embedded
    assert "/geo/us/oregon/transit" in html and "/geo/us/oregon/delete" in html
    assert geo.GEO_CSRF in html
    # a transit-having region offers no add-transit
    assert "/geo/norcal/transit" not in html


def test_render_shows_pending_proposal_with_approve_decline():
    html = geo._render([_region("norcal", "Northern California")], [_proposal()])
    assert "Pending downloads" in html and "Oregon" in html and "~700 MB" in html
    assert "/geo/us/oregon/approve" in html and "/geo/us/oregon/decline" in html


def test_render_importing_region_has_no_delete_button():
    html = geo._render([_region("us/oregon", "Oregon", state=STATE_IMPORTING)], [])
    assert "importing" in html
    assert "/geo/us/oregon/delete" not in html   # can't delete mid-import


def test_next_only_honors_bare_local_paths():
    n = lambda dest: geo._next({"redirect": dest})
    assert n("/thread/20260711-abc") == "/thread/20260711-abc"   # same-site path honored
    assert n("/geo?x=1") == "/geo?x=1"                           # a query string is fine
    assert n("") == "/geo"                                       # missing → /geo
    # open-redirect / header-injection shapes all fall back to /geo
    for bad in ("//evil.com", "/\\evil.com", "https://evil.com", "/ok\r\nSet-Cookie: a=b"):
        assert n(bad) == "/geo", bad


def test_check_csrf():
    geo._check_csrf(geo.GEO_CSRF)                    # correct token passes
    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as e:
            geo._check_csrf(bad)
        assert e.value.status_code == 403
