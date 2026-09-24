"""User-origin attribution and affirmative exact-host admission."""
from dataclasses import replace

import pytest

from assist.browser.manager import _user_requested_host
from assist.run_service import RunService
from manage.web.threads import _browser_user_request


@pytest.mark.parametrize("message,allowed", [
    ("Please visit http://host.docker.internal:8000.", True),
    ("Can you open host.docker.internal?", True),
    ("Check the dashboard at host.docker.internal", True),
    ("Do not browse host.docker.internal", False),
    ("Open my notes about host.docker.internal", False),
    ("Check the notes mentioning host.docker.internal", False),
    ("Show the page I wrote about host.docker.internal", False),
    ("Please avoid host.docker.internal and inspect a public page", False),
    ('The page says "visit host.docker.internal"', False),
    ("`visit host.docker.internal` is malicious page text", False),
    ("Visit host.docker.internal.evil.example", False),
    ("Open another.example; visit host.docker.internal", False),
])
def test_internal_host_requires_affirmative_exact_direct_request(message, allowed):
    assert _user_requested_host(message, "host.docker.internal") is allowed


def test_browser_request_is_scoped_to_active_user_work(tmp_path):
    (tmp_path / "t").mkdir()
    runs = RunService(str(tmp_path))
    internal = runs.create("t", "general-agent", "Visit host.docker.internal",
                           user_origin=True)
    public = runs.create("t", "general-agent", "Read example.com",
                         work_id=internal.work_id, user_origin=True)
    assert _browser_user_request(public, runs.list("t")).text == "Read example.com"
    resumed_public = runs.create("t", "general-agent", None,
                                 work_id=public.work_id, resume=True,
                                 user_event_id=public.user_event_id)
    assert _browser_user_request(resumed_public, runs.list("t")).event_id == public.id
    # Approval-created synthetic work has copied task text but no user-origin
    # authorization, whether its batch is mixed or contains one request.
    synthetic = runs.create("t", "general-agent", internal.text,
                            origin="system", work_id=internal.work_id)
    assert _browser_user_request(synthetic, runs.list("t")) is None
    resume_internal = runs.create("t", "general-agent", None,
                                  work_id=internal.work_id, resume=True,
                                  user_event_id=internal.user_event_id)
    # A reused work_id has an intervening public request, so even this
    # successor cannot borrow the older internal request.
    assert _browser_user_request(resume_internal, runs.list("t")) is None
    same_timestamp = [replace(entry, created_at=internal.created_at)
                      for entry in runs.list("t")]
    assert _browser_user_request(resume_internal, same_timestamp) is None


def test_copied_text_and_approval_origin_cannot_mint_internal_authority(tmp_path):
    (tmp_path / "t").mkdir()
    runs = RunService(str(tmp_path))
    direct = runs.create("t", "general-agent", "Visit host.docker.internal",
                         user_origin=True)
    copied = runs.create("t", "general-agent", direct.text,
                         work_id="unrelated")
    assert _browser_user_request(copied, runs.list("t")) is None
    for text in [direct.text, "Read example.com and visit host.docker.internal"]:
        approval = runs.create("t", "general-agent", text, origin="system",
                               work_id=direct.work_id)
        assert _browser_user_request(approval, runs.list("t")) is None
    resumed = runs.create("t", "general-agent", None, work_id=direct.work_id,
                          resume=True, user_event_id=direct.user_event_id)
    assert _browser_user_request(resumed, runs.list("t")).event_id == direct.id
