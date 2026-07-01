"""sync() pushes the thread branch to origin each turn (host-side, not the agent)."""
import subprocess

from assist.domain_manager import DomainManager, current_branch


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _origin_with_main(tmp_path):
    origin = str(tmp_path / "origin.git")
    _git("init", "--bare", "-b", "main", origin)
    seed = str(tmp_path / "seed")
    _git("clone", origin, seed)
    _git("config", "user.email", "t@example.com", cwd=seed)
    _git("config", "user.name", "Test", cwd=seed)
    (tmp_path / "seed" / "README").write_text("seed")
    _git("add", ".", cwd=seed)
    _git("commit", "-m", "seed", cwd=seed)
    _git("push", "origin", "main", cwd=seed)
    return origin


def _has_ref(repo, ref):
    return subprocess.run(["git", "-C", repo, "rev-parse", "--verify", ref],
                          capture_output=True, text=True).returncode == 0


def test_sync_pushes_thread_branch_to_origin(tmp_path):
    origin = _origin_with_main(tmp_path)
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="ab12")
    _git("config", "user.email", "t@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    branch = current_branch(clone)
    assert branch != "main" and branch.endswith("-ab12")

    (tmp_path / "clone" / "work.txt").write_text("agent work")
    dm.sync("did some work")

    # The thread branch is now on origin (recoverable from another machine); main is NOT
    # advanced by the per-turn push.
    assert _has_ref(origin, branch)
    before = subprocess.run(["git", "-C", clone, "rev-parse", branch],
                            capture_output=True, text=True).stdout.strip()
    assert subprocess.run(["git", "-C", origin, "rev-parse", branch],
                          capture_output=True, text=True).stdout.strip() == before


def test_sync_force_pushes_after_a_rebase_rewrite(tmp_path):
    # The agent may rebase the thread branch (rewriting history); the next turn's push
    # must still succeed (--force-with-lease), not fail as non-fast-forward.
    origin = _origin_with_main(tmp_path)
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="cd34")
    _git("config", "user.email", "t@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    branch = current_branch(clone)
    (tmp_path / "clone" / "a.txt").write_text("a")
    dm.sync("turn 1")                       # pushes branch@v1
    # Rewrite history (simulate the agent's rebase) + a new turn.
    _git("commit", "--amend", "-m", "rewritten", "--allow-empty", cwd=clone)
    (tmp_path / "clone" / "b.txt").write_text("b")
    dm.sync("turn 2")                       # must force-with-lease over the rewrite
    assert subprocess.run(["git", "-C", origin, "rev-parse", branch],
                          capture_output=True, text=True).stdout.strip() == \
        subprocess.run(["git", "-C", clone, "rev-parse", branch],
                       capture_output=True, text=True).stdout.strip()
