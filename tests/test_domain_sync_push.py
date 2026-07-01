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


def test_push_preview_shows_local_main_ahead_of_origin(tmp_path):
    # push_preview = what a push would send (local main vs origin/main).
    origin = _origin_with_main(tmp_path)
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="ef56")
    _git("config", "user.email", "t@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    _git("checkout", "main", cwd=clone)
    (tmp_path / "clone" / "landed.txt").write_text("merged, not pushed")
    _git("add", ".", cwd=clone)
    _git("commit", "-m", "landed a merge", cwd=clone)
    diffs = dm.push_preview()
    assert any("landed.txt" in c.path for c in diffs)


def test_push_preview_empty_when_in_sync(tmp_path):
    origin = _origin_with_main(tmp_path)
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="ef56")
    assert dm.push_preview() == []   # nothing unpushed


def test_sync_aborts_agent_left_inprogress_rebase(tmp_path):
    # If the agent's turn ends mid-rebase (HEAD detached, rebase in progress), sync() must
    # abort it and return to the thread branch — not commit an orphan onto detached HEAD.
    import os
    origin = _origin_with_main(tmp_path)   # seeds README
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="gh78")
    _git("config", "user.email", "t@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    branch = current_branch(clone)
    (tmp_path / "clone" / "README").write_text("thread edit\n")
    _git("commit", "-am", "thread edits README", cwd=clone)
    # advance origin main with a conflicting README edit, then host-fetch
    _git("checkout", "main", cwd=clone)
    (tmp_path / "clone" / "README").write_text("main edit\n")
    _git("commit", "-am", "main edits README", cwd=clone)
    _git("push", "origin", "main", cwd=clone)
    _git("checkout", branch, cwd=clone)
    _git("fetch", "origin", cwd=clone)
    # start a rebase that conflicts -> leaves rebase-merge + detached HEAD
    r = subprocess.run(["git", "-C", clone, "rebase", "origin/main"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and os.path.isdir(os.path.join(clone, ".git", "rebase-merge"))

    dm.sync("end of turn")

    assert current_branch(clone) == branch          # reattached, not detached "HEAD"
    assert not os.path.isdir(os.path.join(clone, ".git", "rebase-merge"))  # rebase aborted


def test_sync_reattaches_plain_detached_head(tmp_path):
    # Agent detached HEAD WITHOUT a rebase (e.g. git checkout <sha>); sync() must
    # re-branch at the current commit and commit on a branch — not orphan (Copilot rd2).
    import os
    origin = _origin_with_main(tmp_path)
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="ij90")
    _git("config", "user.email", "t@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    sha = subprocess.run(["git", "-C", clone, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    _git("checkout", sha, cwd=clone)                     # detached HEAD, no rebase
    assert current_branch(clone) == "HEAD"
    (tmp_path / "clone" / "detached_work.txt").write_text("work\n")

    dm.sync("work while detached")

    b = current_branch(clone)
    assert b.startswith("assist/") and b != "HEAD"       # re-attached to a thread branch
    shown = subprocess.run(["git", "-C", clone, "show", "--name-only", "--oneline", "HEAD"],
                           capture_output=True, text=True).stdout
    assert "detached_work.txt" in shown   # work committed, not orphaned


def test_sync_skips_commit_when_rebase_abort_fails(tmp_path, monkeypatch):
    # If aborting an in-progress rebase fails (corrupt state), sync() must NOT commit/push
    # into the broken repo — it bails out for manual attention (Copilot rd4).
    origin = _origin_with_main(tmp_path)
    clone = str(tmp_path / "clone")
    dm = DomainManager(repo_path=clone, repo=origin, branch_suffix="kl12")
    _git("config", "user.email", "t@example.com", cwd=clone)
    _git("config", "user.name", "Test", cwd=clone)
    monkeypatch.setattr(dm, "_abort_inprogress_rebase", lambda: False)
    (tmp_path / "clone" / "uncommitted.txt").write_text("work\n")
    dm.sync("should not commit")
    status = subprocess.run(["git", "-C", clone, "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    assert "uncommitted.txt" in status   # sync bailed before git_commit; still uncommitted
