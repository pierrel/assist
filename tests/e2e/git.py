"""Disposable Git/SSH repository for the standalone Assist fixture."""
import os
from pathlib import Path
import subprocess


def git_service():
    subprocess.run(["ssh-keygen", "-A"], check=True)
    subprocess.run(["usermod", "-p", "e2e-public-key-only", "test"], check=True)
    subprocess.run(["git", "init", "--bare", "--initial-branch=main", "/repos/domain.git"], check=True)
    subprocess.run(["git", "init", "--initial-branch=main", "/tmp/seed"], check=True)
    Path("/tmp/seed/README.txt").write_text("Disposable E2E repository\n")
    for arguments in (["add", "."], ["-c", "user.name=E2E", "-c", "user.email=e2e@localhost",
                                      "commit", "-m", "Initial"],
                      ["push", "/repos/domain.git", "HEAD:main"]):
        subprocess.run(["git", "-C", "/tmp/seed", *arguments], check=True)
    subprocess.run(["chown", "-R", "1000:1000", "/repos", "/home/test/.ssh"], check=True)
    Path("/run/sshd").mkdir(exist_ok=True)
    os.execv("/usr/sbin/sshd", ["/usr/sbin/sshd", "-D", "-e", "-p", "2222", "-h", "/secrets/ssh_host",
        "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
        "-o", "PermitRootLogin=no", "-o", "UsePAM=no", "-o", "AllowUsers=test"])


if __name__ == '__main__':
    git_service()
