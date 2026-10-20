"""The Git sync skill must preserve a published thread branch's ancestry."""

import subprocess
import os
from pathlib import Path


SKILL = Path(__file__).parents[1] / "assist/skills/git-sync/SKILL.md"


def git(directory: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
    for key in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        environment.pop(key, None)
    return subprocess.run(
        ["git", *args], cwd=directory, env=environment, text=True, capture_output=True,
        check=check,
    )


def test_git_sync_skill_merges_without_rebasing_published_history(tmp_path: Path) -> None:
    skill = SKILL.read_text()
    assert "-c user.name=Assist -c user.email=assist@localhost" in skill
    assert "merge --no-edit --no-gpg-sign origin/main" in skill
    assert "-c core.editor=true merge --continue" in skill
    assert "If a merge is already in progress, stop" in skill
    assert "Never abort a merge that was already in progress" in skill
    assert "git -C /workspace rebase" not in skill

    source = tmp_path / "source.git"
    source.mkdir()
    git(source, "init", "--bare", "--initial-branch=main")
    thread = tmp_path / "thread"
    git(tmp_path, "clone", str(source), str(thread))
    git(thread, "config", "user.name", "Test")
    git(thread, "config", "user.email", "test@example.invalid")
    document = thread / "document.txt"
    document.write_text("base\n")
    git(thread, "add", "document.txt")
    git(thread, "commit", "-m", "base")
    git(thread, "push", "origin", "main")
    git(thread, "switch", "-c", "assist/thread")
    document.write_text("thread change\n")
    git(thread, "commit", "-am", "thread")
    git(thread, "push", "origin", "assist/thread")
    published = git(thread, "rev-parse", "HEAD").stdout.strip()

    main = tmp_path / "main"
    git(tmp_path, "clone", str(source), str(main))
    git(main, "config", "user.name", "Test")
    git(main, "config", "user.email", "test@example.invalid")
    (main / "document.txt").write_text("main change\n")
    git(main, "commit", "-am", "main")
    git(main, "push", "origin", "main")
    git(thread, "fetch", "origin", "main")  # Host-side preparation, not the skill.
    main_tip = git(thread, "rev-parse", "origin/main").stdout.strip()
    git(thread, "config", "--unset", "user.name")
    git(thread, "config", "--unset", "user.email")
    assert git(thread, "var", "GIT_COMMITTER_IDENT", check=False).returncode != 0

    identity = ("-c", "user.name=Assist", "-c", "user.email=assist@localhost", "-c", "commit.gpgsign=false")
    conflict = git(thread, *identity, "merge", "--no-edit", "--no-gpg-sign", "origin/main", check=False)
    assert conflict.returncode != 0
    assert "CONFLICT" in conflict.stdout + conflict.stderr
    document.write_text("thread and main change\n")
    git(thread, "add", "document.txt")
    git(thread, *identity, "-c", "core.editor=true", "merge", "--continue")

    assert git(thread, "merge-base", "--is-ancestor", published, "HEAD").returncode == 0
    assert git(thread, "merge-base", "--is-ancestor", main_tip, "HEAD").returncode == 0
    assert git(source, "rev-parse", "refs/heads/assist/thread").stdout.strip() == published
    assert not git(thread, "status", "--porcelain").stdout
