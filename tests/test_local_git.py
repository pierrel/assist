"""Real Git fixtures for the private host credential boundary."""

from pathlib import Path
import subprocess

import pytest

from assist.git_sync import GitSyncError, identity
from assist.local_git import LocalGit


def git(*arguments, cwd=None):
    result = subprocess.run(["git", *map(str, arguments)], cwd=cwd, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture
def repositories(tmp_path):
    source = tmp_path / "source.git"
    seed = tmp_path / "seed"
    server = tmp_path / "server"
    phone = tmp_path / "phone"
    git("init", "--bare", source)
    git("init", "-b", "main", seed)
    (seed / "README").write_text("initial\n")
    git("add", "README", cwd=seed)
    git("-c", "user.name=Seed", "-c", "user.email=seed@localhost",
        "commit", "-m", "initial", cwd=seed)
    git("remote", "add", "origin", source, cwd=seed)
    git("push", "origin", "main", cwd=seed)
    git("clone", "--no-hardlinks", "-b", "main", source, server)
    git("switch", "-c", "assist/thread", cwd=server)
    git("push", "origin", "assist/thread", cwd=server)
    git("clone", "--no-hardlinks", "-b", "assist/thread", source, phone)
    return source, server, phone


def commit(directory, name, content):
    (directory / name).write_text(content)
    git("add", name, cwd=directory)
    git("-c", "user.name=Test", "-c", "user.email=test@localhost",
        "commit", "-m", name, cwd=directory)
    return git("rev-parse", "HEAD", cwd=directory)


def test_phone_publish_incoming_bundle_and_server_push(repositories, tmp_path):
    source, server, phone = repositories
    phone_tip = commit(phone, "phone.txt", "phone edit\n")
    git("push", "origin", "assist/thread", cwd=phone)

    with LocalGit(str(source)) as host:
        main, remote = host.fetch("assist/thread")
        assert remote == phone_tip
        assert host.ancestor(identity(str(server))[1], remote)
        bundle = host.incoming_bundle(identity(str(server))[1], remote)
    assert bundle is not None
    incoming = tmp_path / "incoming.bundle"
    incoming.write_bytes(bundle)
    git("bundle", "unbundle", incoming, cwd=server)
    git("update-ref", "refs/remotes/origin/main", main, cwd=server)
    git("update-ref", "refs/remotes/origin/assist/thread", remote, cwd=server)
    git("merge", "--ff-only", remote, cwd=server)
    assert (server / "phone.txt").read_text() == "phone edit\n"

    desired = commit(server, "server.txt", "server edit\n")
    with LocalGit(str(source), str(server)) as host:
        assert host.same_tip() == ("assist/thread", desired)
        assert host.publish("assist/thread", desired) == desired
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == desired


def test_concurrent_phone_push_preserves_server_result_branch(repositories):
    source, server, phone = repositories
    desired = commit(server, "server.txt", "server edit\n")
    commit(phone, "phone.txt", "phone edit\n")
    git("push", "origin", "assist/thread", cwd=phone)

    with LocalGit(str(source), str(server)) as host:
        _, remote = host.fetch("assist/thread")
        assert not host.ancestor(remote, desired)
        result = "assist-result/" + desired
        assert host.publish(result, desired, create_only=True) == desired
        assert host.publish(result, desired, create_only=True) == desired
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) != desired
    assert git("rev-parse", "refs/heads/" + result, cwd=source) == desired


def test_result_branch_descendant_proves_incorporation(repositories):
    source, server, _ = repositories
    desired = commit(server, "server.txt", "first result\n")
    descendant = commit(server, "later.txt", "later work\n")
    result = "assist-result/" + desired
    git("push", "origin", "HEAD:" + result, cwd=server)
    with LocalGit(str(source), str(server)) as host:
        assert host.publish(result, desired, create_only=True) == descendant
    assert git("rev-parse", "refs/heads/" + result, cwd=source) == descendant


def test_result_branch_ancestor_collision_cannot_be_advanced(repositories):
    source, server, _ = repositories
    base = git("rev-parse", "HEAD", cwd=server)
    desired = commit(server, "server.txt", "result\n")
    result = "assist-result/" + desired
    git("push", "origin", base + ":refs/heads/" + result, cwd=server)
    with LocalGit(str(source), str(server)) as host:
        with pytest.raises(GitSyncError, match="different work"):
            host.publish(result, desired, create_only=True)
    assert git("rev-parse", "refs/heads/" + result, cwd=source) == base


def test_existing_thread_ref_rejects_non_fast_forward_push(repositories, monkeypatch):
    source, server, phone = repositories
    desired = commit(server, "server.txt", "server work\n")
    remote = commit(phone, "phone.txt", "phone work\n")
    git("push", "origin", "assist/thread", cwd=phone)
    with LocalGit(str(source), str(server)) as host:
        genuine_ancestor = host.ancestor
        def mistaken_ancestor(older, newer):
            if older == remote and newer == desired:
                return True
            return genuine_ancestor(older, newer)
        monkeypatch.setattr(host, "ancestor", mistaken_ancestor)
        with pytest.raises(GitSyncError, match="not verified"):
            host.publish("assist/thread", desired)
    assert git("rev-parse", "refs/heads/assist/thread", cwd=source) == remote


def test_workspace_config_hooks_and_alternates_never_cross_host_boundary(
        repositories, tmp_path):
    source, server, _ = repositories
    marker = tmp_path / "host-command-ran"
    git("config", "credential.helper", "!touch " + str(marker), cwd=server)
    git("config", "url.ext::sh -c 'touch " + str(marker) + "'.insteadOf",
        str(source), cwd=server)
    hooks = server / ".git" / "hooks"
    hook = hooks / "pre-push"
    hook.write_text("#!/bin/sh\ntouch " + str(marker) + "\n")
    hook.chmod(0o700)
    with LocalGit(str(source), str(server)) as host:
        _, tip = host.same_tip()
        assert host.publish("assist/thread", tip) == tip
    assert not marker.exists()

    alternate = server / ".git" / "objects" / "info" / "alternates"
    alternate.write_text(str(source / "objects") + "\n")
    with pytest.raises(GitSyncError, match="Unsupported Git object entry"):
        LocalGit(str(source), str(server))


def test_standard_split_commit_graph_is_bounded_not_rejected(repositories):
    source, server, _ = repositories
    git("commit-graph", "write", "--split", "--reachable", cwd=server)
    graphs = server / ".git" / "objects" / "info" / "commit-graphs"
    assert (graphs / "commit-graph-chain").is_file()
    assert len(list(graphs.glob("graph-*.graph"))) == 1
    with LocalGit(str(source), str(server)) as host:
        assert host.same_tip() == identity(str(server))

    graph = next(graphs.glob("graph-*.graph"))
    graph.unlink()
    graph.symlink_to(source / "objects" / "info" / "commit-graph")
    with pytest.raises(GitSyncError, match="Git object"):
        LocalGit(str(source), str(server))
