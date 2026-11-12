"""Offline browser observations must expose the retained form's real state."""
from edd.eval.test_browser_agent import _BrowserSite


def test_form_observation_keeps_values_targets_and_page_boundary(tmp_path):
    site = _BrowserSite("v1-form", str(tmp_path))
    site.command("open", url="https://v1-form.fern.example/")
    site.command("act", page_id="page-1", snapshot_id="form-1", action="fill",
                 target={"ref": "destination", "text": "Portland"})
    observed = site.command("observe", page_id="page-1")["result"]
    assert observed["page_id"] == "page-1" and observed["snapshot_id"] == "form-1"
    assert "Portland" in observed["snapshot"]
    assert any(target["ref"] == "review" for target in observed["targets"])
    reviewed = site.command("act", page_id="page-1", snapshot_id="form-1",
                            action="click", target={"ref": "review"})
    assert site.command("observe", page_id="page-1") == reviewed
    site.end_turn()
    assert "error" in site.command("observe", page_id="page-1")
