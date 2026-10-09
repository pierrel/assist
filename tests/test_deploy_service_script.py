"""Regression tests for safe service-install configuration transport."""
import os
import shutil
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


def test_deploy_service_transports_egress_host_paths(tmp_path):
    captured = tmp_path / "payload"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\ncat > {captured}\n")
    fake_ssh.chmod(0o755)
    installer = tmp_path / "installer"
    installer.write_text("# end configuration\n")
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "INSTALL_SERVICE_SCRIPT": str(installer),
        "ASSIST_EGRESS_CLIENT_MAP_DIR": "/var/lib/assist/proxy-map",
        "ASSIST_EGRESS_RUNTIME_DIR": "/var/lib/assist/egress-runtime",
    }
    subprocess.run(["bash", "scripts/deploy-service.sh", "test-host"],
                   check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
                   env=environment)
    configuration = captured.read_text().partition("# end configuration")[0]
    result = subprocess.run(
        ["bash", "-c", configuration +
         'printf "%s|%s" "$ASSIST_EGRESS_CLIENT_MAP_DIR" "$ASSIST_EGRESS_RUNTIME_DIR"'],
        check=True, capture_output=True, text=True)
    assert result.stdout == "/var/lib/assist/proxy-map|/var/lib/assist/egress-runtime"


def test_make_exports_egress_host_paths_to_deploy_recipe():
    result = subprocess.run(
        ["make", "-s", "--eval",
         'show-egress-paths:;@printf "%s|%s" "$$ASSIST_EGRESS_CLIENT_MAP_DIR" "$$ASSIST_EGRESS_RUNTIME_DIR"',
         "show-egress-paths",
         "ASSIST_EGRESS_CLIENT_MAP_DIR=/var/lib/assist/proxy-map",
         "ASSIST_EGRESS_RUNTIME_DIR=/var/lib/assist/egress-runtime"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env={key: value for key, value in os.environ.items()
             if key not in {"ASSIST_EGRESS_CLIENT_MAP_DIR", "ASSIST_EGRESS_RUNTIME_DIR"}},
        capture_output=True, text=True, check=True)
    assert result.stdout == "/var/lib/assist/proxy-map|/var/lib/assist/egress-runtime"


def test_make_leaves_unconfigured_runtime_directory_unset(tmp_path):
    repo = os.path.dirname(os.path.dirname(__file__))
    shutil.copyfile(os.path.join(repo, "Makefile"), tmp_path / "Makefile")
    environment = {key: value for key, value in os.environ.items()
                   if key != "ASSIST_EGRESS_RUNTIME_DIR"}
    result = subprocess.run(
        ["make", "-s", "--eval",
         'show-egress-runtime:;@python -c \'import os; print("ASSIST_EGRESS_RUNTIME_DIR" in os.environ)\'',
         "show-egress-runtime"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    assert result.stdout == "False\n"

def test_make_deployment_preserves_gmail_token_path_in_service_unit(tmp_path):
    deploy = tmp_path / "deployment"
    (deploy / "scripts").mkdir(parents=True)
    repo = os.path.dirname(os.path.dirname(__file__))
    from pathlib import Path
    (deploy / "scripts" / "assist-web.service.template").write_text(
        (Path(repo) / "scripts" / "assist-web.service.template").read_text())
    data = tmp_path / "threads"
    data.mkdir()
    captured = tmp_path / "service-unit"
    ssh = tmp_path / "ssh"
    ssh.write_text("#!/bin/sh\nexec bash -s\n")
    ssh.chmod(0o755)
    sudo = tmp_path / "sudo"
    sudo.write_text('#!/bin/sh\nif [ "$1" = tee ]; then cat > "$GMAIL_TEST_UNIT"; fi\n')
    sudo.chmod(0o755)
    token_path = str(tmp_path / "private" / "gmail-token.json")
    configuration = tmp_path / "gmail.mk"
    configuration.write_text(f"ASSIST_GMAIL_TOKEN_FILE := {token_path}\nDEPLOY_PATH := {deploy}\nASSIST_THREADS_DIR := {data}\nDEPLOY_HOST := synthetic-host\nSERVICE_NAME := synthetic-assist\n")
    subprocess.run(["make", "-f", "Makefile", "-f", str(configuration), "deploy-service"],
                   check=True, cwd=repo,
                   env=os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}",
                                     "GMAIL_TEST_UNIT": str(captured)})
    assert f'Environment="ASSIST_GMAIL_TOKEN_FILE={token_path}"' in captured.read_text()
