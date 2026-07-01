"""DomainMirror — host-side bare mirror refresh, against a real git origin."""
import os
import subprocess
import threading

from assist.domain_mirror import DomainMirror, _safe_label


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _make_origin(tmp_path):
    """A bare origin with one commit on main + a working clone to push more."""
    origin = str(tmp_path / "origin.git")
    _git("init", "--bare", "-b", "main", origin)
    work = str(tmp_path / "work")
    _git("clone", origin, work)
    _git("config", "user.email", "t@example.com", cwd=work)
    _git("config", "user.name", "Test", cwd=work)
    (tmp_path / "work" / "f.txt").write_text("v1")
    _git("add", ".", cwd=work)
    _git("commit", "-m", "init", cwd=work)
    _git("push", "origin", "main", cwd=work)
    return origin, work


def _mirror_head(mirror_path):
    r = subprocess.run(["git", "-C", mirror_path, "rev-parse", "main"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _origin_head(work):
    return subprocess.run(["git", "-C", work, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def test_refresh_creates_mirror_with_main(tmp_path):
    origin, work = _make_origin(tmp_path)
    m = DomainMirror(str(tmp_path), origin, "life")
    m.refresh()
    assert os.path.isdir(m.path)
    assert m.path.endswith(os.path.join(".mirrors", "life.git"))
    assert _mirror_head(m.path) == _origin_head(work)


def test_refresh_picks_up_new_origin_commits(tmp_path):
    origin, work = _make_origin(tmp_path)
    m = DomainMirror(str(tmp_path), origin, "life")
    m.refresh()
    (tmp_path / "work" / "f.txt").write_text("v2")
    _git("commit", "-am", "second", cwd=work)
    _git("push", "origin", "main", cwd=work)
    m.refresh()
    assert _mirror_head(m.path) == _origin_head(work)


def test_container_clone_can_fetch_mirror(tmp_path):
    # Simulate the container: a clone whose `mirror` remote is the bare mirror; it must
    # fetch mirror/main (the whole point — the agent rebases onto this).
    origin, work = _make_origin(tmp_path)
    m = DomainMirror(str(tmp_path), origin, "life")
    m.refresh()
    clone = str(tmp_path / "clone")
    _git("clone", origin, clone)
    _git("remote", "add", "mirror", f"file://{m.path}", cwd=clone)
    _git("fetch", "mirror", cwd=clone)
    got = subprocess.run(["git", "-C", clone, "rev-parse", "mirror/main"],
                         capture_output=True, text=True)
    assert got.returncode == 0 and got.stdout.strip() == _origin_head(work)


def test_concurrent_refresh_no_corruption(tmp_path):
    origin, _ = _make_origin(tmp_path)
    m = DomainMirror(str(tmp_path), origin, "life")
    m.refresh()  # create first, so all threads take the fetch path
    errors = []

    def refresh():
        try:
            m.refresh()
        except Exception as e:  # a race would surface as a git failure
            errors.append(e)

    threads = [threading.Thread(target=refresh) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert _mirror_head(m.path) is not None


def test_safe_label_sanitizes():
    assert _safe_label("life") == "life"
    assert _safe_label("user@host:/path/to/life.git").endswith("life.git") or "/" not in _safe_label("user@host:/path/to/life.git")
    assert "/" not in _safe_label("a/b/c")
    assert _safe_label("..") == "domain"
