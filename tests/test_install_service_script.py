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


def test_install_service_renders_egress_host_paths(tmp_path):
    threads = tmp_path / "threads"
    threads.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "assist-web.service.template").write_text("{{ENVIRONMENT_VARS}}\n")
    capture = tmp_path / "unit"
    sudo = tmp_path / "sudo"
    sudo.write_text("#!/bin/sh\nif [ \"$1\" = tee ]; then cat > \"$CAPTURE\"; fi\n")
    sudo.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/install-service.sh"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=os.environ | {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CAPTURE": str(capture), "DEPLOY_PATH": str(tmp_path),
            "ASSIST_THREADS_DIR": str(threads),
            "ASSIST_EGRESS_CLIENT_MAP_DIR": "/var/lib/assist/proxy-map",
            "ASSIST_EGRESS_RUNTIME_DIR": "/var/lib/assist/egress-runtime",
        }, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    unit = capture.read_text()
    assert 'Environment="ASSIST_EGRESS_CLIENT_MAP_DIR=/var/lib/assist/proxy-map"' in unit
    assert 'Environment="ASSIST_EGRESS_RUNTIME_DIR=/var/lib/assist/egress-runtime"' in unit
