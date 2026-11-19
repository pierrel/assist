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


def test_deploy_service_preserves_approved_sms_configuration(tmp_path):
    captured = tmp_path / "payload"
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(f"#!/bin/sh\ncat > {captured}\n")
    fake_ssh.chmod(0o755)
    installer = tmp_path / "installer"
    installer.write_text("# end configuration\n")
    environment = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "INSTALL_SERVICE_SCRIPT": str(installer),
        "ASSIST_SMS_ARCHIVE_DIR": "/var/lib/assist-sms-private",
        "ASSIST_SMS_SECRET": "synthetic-shared-secret",
        "ASSIST_SMS_OUTBOUND_URL": "https://phone.example.test/outbound/sms/v1",
    }

    subprocess.run(["bash", "scripts/deploy-service.sh", "test-host"],
                   check=True, cwd=os.path.dirname(os.path.dirname(__file__)),
                   env=environment)

    configuration = captured.read_text().partition("# end configuration")[0]
    result = subprocess.run(
        ["bash", "-c", configuration +
         'printf "%s|%s" "$ASSIST_SMS_SECRET" "$ASSIST_SMS_OUTBOUND_URL"'],
        check=True, capture_output=True, text=True)
    assert result.stdout == "synthetic-shared-secret|https://phone.example.test/outbound/sms/v1"
    archive = subprocess.run(["bash", "-c", configuration + 'printf %s "$ASSIST_SMS_ARCHIVE_DIR"'],
                             check=True, capture_output=True, text=True)
    assert archive.stdout == "/var/lib/assist-sms-private"


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
        "ASSIST_BASE_URL": "https://control.example.test:5050",
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
    origin = subprocess.run(["bash", "-c", configuration + 'printf %s "$ASSIST_BASE_URL"'],
                            check=True, capture_output=True, text=True)
    assert origin.stdout == "https://control.example.test:5050"


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
    configuration.write_text(f"ASSIST_GMAIL_TOKEN_FILE := {token_path}\nASSIST_BASE_URL := https://control.example.test:5050\nDEPLOY_PATH := {deploy}\nASSIST_THREADS_DIR := {data}\nDEPLOY_HOST := synthetic-host\nSERVICE_NAME := synthetic-assist\n")
    subprocess.run(["make", "-f", "Makefile", "-f", str(configuration), "deploy-service"],
                   check=True, cwd=repo,
                   env=os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}",
                                     "GMAIL_TEST_UNIT": str(captured)})
    assert f'Environment="ASSIST_GMAIL_TOKEN_FILE={token_path}"' in captured.read_text()
    assert 'Environment="ASSIST_BASE_URL=https://control.example.test:5050"' in captured.read_text()


def test_make_service_export_reaches_archive_relay(tmp_path):
    capture=tmp_path/'payload'
    ssh=tmp_path/'ssh';ssh.write_text('#!/bin/sh\ncat > "$CAPTURE"\n');ssh.chmod(0o755)
    installer=tmp_path/'installer';installer.write_text('# end configuration\n')
    fixture=tmp_path/'archive.mk';fixture.write_text('ASSIST_SMS_ARCHIVE_DIR := /var/lib/synthetic-sms-private\n')
    environment=os.environ|{'PATH':f"{tmp_path}:{os.environ['PATH']}",'CAPTURE':str(capture),'INSTALL_SERVICE_SCRIPT':str(installer)}
    subprocess.run(['make','-s','-f','Makefile','-f',str(fixture),'deploy-service','DEPLOY_HOST=synthetic-host'],
                   cwd=os.path.dirname(os.path.dirname(__file__)),env=environment,check=True,capture_output=True)
    configuration=capture.read_text().partition('# end configuration')[0]
    result=subprocess.run(['bash','-c',configuration+'printf %s "$ASSIST_SMS_ARCHIVE_DIR"'],check=True,capture_output=True,text=True)
    assert result.stdout=='/var/lib/synthetic-sms-private'


def test_make_service_preserves_native_guidance_through_systemd_quoting(tmp_path):
    import json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    deploy = tmp_path / "deploy"
    (deploy / "scripts").mkdir(parents=True)
    (deploy / "scripts" / "assist-web.service.template").write_text("[Service]\n{{ENVIRONMENT_VARS}}\n")
    data = tmp_path / "threads"
    data.mkdir()
    captured = tmp_path / "unit"
    ssh = tmp_path / "ssh"
    ssh.write_text("#!/bin/sh\nexec bash -s\n")
    ssh.chmod(0o755)
    sudo = tmp_path / "sudo"
    sudo.write_text('#!/bin/sh\nif [ "$1" = tee ]; then cat > "$CAPTURE"; fi\n')
    sudo.chmod(0o755)
    guidance = 'Keep "quoted" words and owner\'s text & pipes | literal $HOME \\ path 100%.\nSecond line.'
    fixture = tmp_path / "guidance.mk"
    fixture.write_text(f"DEPLOY_PATH := {deploy}\nASSIST_THREADS_DIR := {data}\n"
                       + "define ASSIST_HERMES_SYSTEM_MESSAGE\n"
                       + guidance.replace("$", "$$") + "\nendef\n")
    subprocess.run(["make", "-s", "-f", "Makefile", "-f", str(fixture),
                    "deploy-service", "DEPLOY_HOST=synthetic-host"], cwd=repo,
                   env=os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}",
                                     "CAPTURE": str(captured)}, check=True)
    line = next(line for line in captured.read_text().splitlines()
                if line.startswith('Environment="ASSIST_HERMES_SYSTEM_MESSAGE='))
    assert "100%%" in line
    assert json.loads(line.split("=", 1)[1]).replace("%%", "%") == "ASSIST_HERMES_SYSTEM_MESSAGE=" + guidance
