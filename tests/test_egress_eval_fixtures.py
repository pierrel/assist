"""The existing real-model panel cannot escape its offline fixture transports."""
from unittest import mock

import pytest
import requests

from assist.egress.store import EgressStore
from assist import tools
from edd.eval import test_egress_approval as panel


def test_panel_host_and_shell_transports_share_the_exact_synthetic_grant(tmp_path):
    case = panel.TestEgressApproval("runTest")
    case.model = object()  # Agent construction is replaced; no provider is loaded.
    case.store = EgressStore(str(tmp_path / "approvals"))
    case.tid = "offline-fixture"
    url = "https://api.github.com/repos/acme/widget/releases"
    try:
        with mock.patch.object(panel, "create_agent", return_value=object()):
            _agent, backend = case._agent()
        assert backend.execute("curl " + url).exit_code == 56
        with pytest.raises(AssertionError):
            tools.url_fetch.get(url)
        with pytest.raises(AssertionError):
            requests.get("https://unrelated.example.test")
        assert "Error fetching URL" in tools.read_url("https://pdf.example.test/report.pdf")
        backend.allow = True
        assert "9.3.1" in backend.execute("curl " + url).output
        assert "9.3.1" in tools.url_fetch.get(url).text
        assert backend.execute("curl https://unrelated.example.test").exit_code == 56
        with pytest.raises(AssertionError):
            tools.url_fetch.get("https://unrelated.example.test")
    finally:
        case.doCleanups()
