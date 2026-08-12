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


def test_install_service_writes_pi_health_environment(tmp_path):
    service = tmp_path / "assist-web.service"
    fake_sudo = tmp_path / "sudo"
    fake_sudo.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        f"tee) cat > {service} ;;\n"
        "systemctl) exit 0 ;;\n"
        "*) exit 1 ;;\n"
        "esac\n")
    fake_sudo.chmod(0o755)
    repository = os.path.dirname(os.path.dirname(__file__))
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "DEPLOY_PATH": repository,
        "SERVICE_NAME": "assist-web-test",
        "ASSIST_THREADS_DIR": str(tmp_path),
        "ASSIST_PI_HEALTH_DIR": "/var/lib/assist/pi-provider-health",
        "ASSIST_PI_PROVIDER_SERVICE": "llamacpp.service",
    }

    subprocess.run(["bash", "scripts/install-service.sh"], check=True,
                   cwd=repository, env=environment)

    unit = service.read_text()
    assert 'Environment="ASSIST_PI_HEALTH_DIR=/var/lib/assist/pi-provider-health"' in unit
    assert 'Environment="ASSIST_PI_PROVIDER_SERVICE=llamacpp.service"' in unit
