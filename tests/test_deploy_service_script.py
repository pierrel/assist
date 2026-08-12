"""Regression tests for safe service-install configuration transport."""
import os
import subprocess


def test_deploy_service_transports_display_name_with_apostrophe(tmp_path):
    captured = tmp_path / "payload"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\ncat > {captured}\n")
    fake_ssh.chmod(0o755)
    installer = tmp_path / "installer"
    installer.write_text("# end configuration\n")
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "INSTALL_SERVICE_SCRIPT": str(installer),
        "EMAIL_FROM_NAME": "Pierre's assistant",
    }

    subprocess.run(["bash", "scripts/deploy-service.sh", "test-host"],
                   check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
                   env=environment)

    configuration = captured.read_text().partition("# end configuration")[0]
    assert configuration.startswith("set -euo pipefail\n")
    result = subprocess.run(
        ["bash", "-c", configuration + 'printf %s "$EMAIL_FROM_NAME"'],
        check=True, capture_output=True, text=True)
    assert result.stdout == "Pierre's assistant"


def test_deploy_service_transports_urgent_sms_configuration(tmp_path):
    captured = tmp_path / "payload"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\ncat > {captured}\n")
    fake_ssh.chmod(0o755)
    installer = tmp_path / "installer"
    installer.write_text("# end configuration\n")
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "INSTALL_SERVICE_SCRIPT": str(installer),
        "URGENT_SMS_RECIPIENT": "+15555550100",
        "URGENT_SMS_THREAD_URL_BASE": "https://web.example.test:5050",
    }

    subprocess.run(["bash", "scripts/deploy-service.sh", "test-host"],
                   check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
                   env=environment)

    configuration = captured.read_text().partition("# end configuration")[0]
    result = subprocess.run(
        ["bash", "-c", configuration +
         'printf "%s|%s" "$URGENT_SMS_RECIPIENT" "$URGENT_SMS_THREAD_URL_BASE"'],
        check=True, capture_output=True, text=True)
    assert result.stdout == "+15555550100|https://web.example.test:5050"


def test_deploy_service_transports_pi_health_configuration(tmp_path):
    captured = tmp_path / "payload"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\ncat > {captured}\n")
    fake_ssh.chmod(0o755)
    installer = tmp_path / "installer"
    installer.write_text("# end configuration\n")
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "INSTALL_SERVICE_SCRIPT": str(installer),
        "ASSIST_PI_HEALTH_DIR": "/var/lib/assist/pi-provider-health",
        "ASSIST_PI_PROVIDER_SERVICE": "llamacpp.service",
    }

    subprocess.run(["bash", "scripts/deploy-service.sh", "test-host"],
                   check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
                   env=environment)

    configuration = captured.read_text().partition("# end configuration")[0]
    result = subprocess.run(
        ["bash", "-c", configuration +
         'printf "%s|%s" "$ASSIST_PI_HEALTH_DIR" "$ASSIST_PI_PROVIDER_SERVICE"'],
        check=True, capture_output=True, text=True)
    assert result.stdout == "/var/lib/assist/pi-provider-health|llamacpp.service"
