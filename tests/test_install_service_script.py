"""Regression tests for the service-install preconditions."""
import os
import subprocess


def test_install_service_refuses_a_missing_thread_directory(tmp_path):
    result = subprocess.run(
        ["bash", "scripts/install-service.sh"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=os.environ | {"ASSIST_THREADS_DIR": str(tmp_path / "missing")},
        capture_output=True, text=True)

    assert result.returncode == 1
    assert "must already exist and be writable" in result.stderr
