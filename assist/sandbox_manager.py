import hashlib
import logging
import os
import re
import threading
import time

logger = logging.getLogger(__name__)
_ANY_CONTAINER = object()


def _rewrite_localhost(value: str) -> str:
    """Replace localhost/127.0.0.1 references with host.docker.internal."""
    return re.sub(r'localhost|127\.0\.0\.1', 'host.docker.internal', value)


def _sandbox_timezone(override: str | None = None) -> str:
    """A timezone setting for the sandbox (an IANA zone name like
    "America/Los_Angeles", or whatever ``TZ`` value is configured) so ``date`` /
    file timestamps reflect LOCAL time, not the container's default UTC — without it
    the `time` skill answers "what's today" in UTC (wrong for an evening PT user).
    Order: ASSIST_TIMEZONE override (operator config; the context-rider milestone
    will set per-user later), else ``TZ``, else the host's zone, else UTC.

    ``override`` is the per-turn context-rider timezone (the user's actual zone);
    when given it wins, so this turn's ``date`` runs in the user's local time."""
    if override:
        return override
    tz = os.environ.get("ASSIST_TIMEZONE") or os.environ.get("TZ")
    if tz:
        return tz
    try:  # Debian-likes: /etc/timezone is a plain zone name
        with open("/etc/timezone") as f:
            name = f.read().strip()
        if name:
            return name
    except OSError:
        pass
    try:  # most distros: /etc/localtime symlinks into zoneinfo
        link = os.readlink("/etc/localtime")
        if "zoneinfo/" in link:
            return link.split("zoneinfo/", 1)[1]
    except OSError:
        pass
    return "UTC"


SANDBOX_IMAGE = "assist-sandbox"

# Egress allowlist layer.  See docs/2026-05-08-sandbox-network-allowlist.org
# for the threat model and design.
EGRESS_NETWORK = "assist-egress-network"
EGRESS_PROXY_NAME = "assist-egress-proxy"
EGRESS_PROXY_IMAGE = "assist-egress-proxy"
EGRESS_PROXY_PORT = 8888
EGRESS_ALLOWLIST_FILE = os.path.join(
    os.path.dirname(__file__), "..", "dockerfiles", "egress-allowlist.conf"
)


def _load_egress_allowlist() -> list[str]:
    """Read the sandbox egress allowlist from the committed conf file.

    Single source of truth: ``dockerfiles/egress-allowlist.conf``.
    Returns a sorted list (deterministic hash for change detection).
    Raises FileNotFoundError if the file is missing — fail-closed,
    no baked-in fallback that could silently disagree with the repo.
    """
    with open(EGRESS_ALLOWLIST_FILE) as f:
        return sorted({
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        })


def _egress_proxy_config_hash(allowlist_csv: str, approvals_dir: str | None) -> str:
    """The proxy's config fingerprint (a container label): allowlist content
    plus the approvals-mount schema/path and proxy-policy schema, so a change
    to any of them recreates the proxy on the next sandbox start. The version
    marker makes containers with an earlier approval or throttle policy
    recreate once and gain the current behavior."""
    return hashlib.sha256(
        (allowlist_csv + "|v4-host-throttle-guidance-approvals:" + (approvals_dir or "")).encode()
    ).hexdigest()[:16]


class SandboxManager:
    """Manages Docker sandbox container lifecycle.

    Class-level state: one Docker client shared across all callers,
    with a container registry keyed by work_dir for cleanup.
    """

    _docker_client = None
    _containers: dict[str, "docker.models.containers.Container"] = {}  # type: ignore[name-defined]
    # Serialize bring-up of the shared egress proxy.  Without this,
    # two threads simultaneously starting sandboxes after an allowlist
    # change race in the get/remove/recreate flow and the loser fails
    # with "container name already in use" — which the broad
    # `except Exception` in get_sandbox_backend then swallows as
    # "Docker unavailable", silently degrading one of the two threads.
    _egress_lock = threading.Lock()

    @classmethod
    def _get_docker_client(cls):
        """Lazily create and cache a Docker client."""
        if cls._docker_client is None:
            import docker
            cls._docker_client = docker.from_env()
        return cls._docker_client

    @classmethod
    def _ensure_egress_proxy_running(cls, client) -> str:
        """Idempotent: bring up the egress allowlist proxy + isolated network.

        - Network ``assist-egress-network`` is internal=True (no host
          gateway, no NAT) — the only way out for a sandbox attached to
          this network is the proxy container.
        - Proxy container ``assist-egress-proxy`` is dual-homed: it
          starts on the default bridge (so it has internet access) and
          is then connected to the egress network (so sandboxes can
          reach it as ``assist-egress-proxy:8888``).
        - The current allowlist (from ``egress-allowlist.conf``) is
          hashed and stamped on the container as a label.  When the
          file changes, the proxy is recreated on the next call.

        Fails closed: if the proxy can't be brought up, callers see the
        exception and the sandbox start fails — which is the correct
        behavior (no fallback to direct egress).
        """
        from docker.errors import APIError, NotFound

        allowlist = _load_egress_allowlist()
        allowlist_csv = ",".join(allowlist)
        # The approvals mount (user-approved egress grants — docs/
        # 2026-07-21-egress-approval-hitl.org): a read-only mount of the
        # approvals SUBDIR only (projection + client map; proposal state and
        # agent free text stay host-side). Unset env ⇒ no mount ⇒ the
        # feature is dormant and the proxy runs exactly as before. The
        # schema marker rides the hash so an already-running proxy from
        # before this feature is recreated once and gains the mount.
        approvals_dir = os.environ.get("ASSIST_EGRESS_APPROVALS_DIR") or None
        if approvals_dir:
            from assist.egress.store import APPROVALS_SUBDIR, approvals_dir_is_safe
            if not approvals_dir_is_safe(approvals_dir):
                # The SAME self-approval guard the web wiring applies: an
                # approvals dir under the thread root sits inside a sandbox's
                # rw /workspace mount — mounting it would let the sandbox
                # write its own grants. Refuse the mount too, not just the
                # tools (a one-sided guard is theater).
                logger.error(
                    "ASSIST_EGRESS_APPROVALS_DIR=%s is under the thread root "
                    "— refusing the proxy mount; egress approvals dormant",
                    approvals_dir)
                approvals_dir = None
            else:
                approvals_dir = os.path.join(approvals_dir, APPROVALS_SUBDIR)
                os.makedirs(approvals_dir, exist_ok=True)
        allowlist_hash = _egress_proxy_config_hash(allowlist_csv, approvals_dir)

        with cls._egress_lock:
            try:
                egress_net = client.networks.get(EGRESS_NETWORK)
            except NotFound:
                egress_net = client.networks.create(
                    EGRESS_NETWORK, driver="bridge", internal=True,
                )
                logger.info("Created egress network %s (internal)", EGRESS_NETWORK)
            else:
                # An attacker (or a hand-rolled docker network create that
                # forgot --internal) could leave a same-named network that
                # has a default gateway, re-opening unrestricted egress.
                # Fail closed — refuse to attach a sandbox to a non-internal
                # network of this name.  Operator fix: `docker network rm
                # assist-egress-network`; SandboxManager recreates it
                # correctly on the next sandbox start.
                if not egress_net.attrs.get("Internal", False):
                    raise RuntimeError(
                        f"Egress network {EGRESS_NETWORK!r} exists but is "
                        "not internal=True.  Refusing to attach the "
                        "sandbox — that would bypass the allowlist layer.  "
                        f"Fix: `docker network rm {EGRESS_NETWORK}` and "
                        "the next sandbox start will recreate it."
                    )

            existing = None
            try:
                existing = client.containers.get(EGRESS_PROXY_NAME)
                existing.reload()
            except NotFound:
                pass

            needs_recreate = (
                existing is None
                or existing.status != "running"
                or existing.labels.get("assist.egress-allowlist-hash") != allowlist_hash
            )
            if not needs_recreate:
                return EGRESS_PROXY_NAME

            if existing is not None:
                try:
                    existing.remove(force=True)
                    logger.info("Removed stale egress proxy %s", existing.id[:12])
                except APIError as e:
                    logger.warning("Could not remove existing egress proxy: %s", e)

            from assist.egress.guidance import EGRESS_DENY_BODY, EGRESS_THROTTLE_BODY
            proxy = client.containers.run(
                EGRESS_PROXY_IMAGE,
                name=EGRESS_PROXY_NAME,
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                extra_hosts={"host.docker.internal": "host-gateway"},
                # EGRESS_DENY_BODY / EGRESS_THROTTLE_BODY: the proxy response
                # text, centralized in
                # assist/egress/guidance.py and delivered here so no
                # guidance prose lives in proxy code (Pierre, PR #200).
                environment={"EGRESS_ALLOWLIST": allowlist_csv,
                             "EGRESS_DENY_BODY": EGRESS_DENY_BODY,
                             "EGRESS_THROTTLE_BODY": EGRESS_THROTTLE_BODY},
                labels={
                    "assist.egress-proxy": "true",
                    "assist.egress-allowlist-hash": allowlist_hash,
                },
                **({"volumes": {approvals_dir: {"bind": "/approvals",
                                                "mode": "ro"}}}
                   if approvals_dir else {}),
            )
            try:
                egress_net.connect(proxy)
            except APIError as e:
                if "already exists" not in str(e).lower():
                    raise
            cls._wait_for_egress_proxy_ready(proxy)
            logger.info(
                "Started egress proxy %s with %d allowlist entries (hash=%s)",
                proxy.id[:12], len(allowlist), allowlist_hash,
            )
            return EGRESS_PROXY_NAME

    @classmethod
    def _wait_for_egress_proxy_ready(cls, proxy, timeout: float = 10.0) -> None:
        """Block until the proxy logs 'listening on'.

        The proxy's TCP listener is bound only after Python startup +
        allowlist parse + ``socket.bind`` — typically <100ms but not
        instant.  Without this, the very first sandbox launched
        immediately after a proxy recreate can hit connection-refused
        on its first outbound call.  Polls container logs every 100ms.
        Raises RuntimeError if the proxy never reports ready within
        ``timeout`` (fail-closed).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                logs = proxy.logs().decode("utf-8", errors="replace")
            except Exception:
                logs = ""
            if "listening on" in logs:
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Egress proxy {proxy.id[:12]} did not report 'listening on' "
            f"within {timeout}s.  Last logs: {logs[-500:]!r}"
        )

    @classmethod
    def _get_sandbox_backend(cls, work_dir: str, tz: str | None,
                             agent_dir: str | None, include_assist_env: bool,
                             include_egress_approvals: bool):
        """Create one per-turn sandbox from a named authority profile.

        ``include_assist_env`` is the line between ordinary Deep Agents work and
        Pi preview work.  A Pi sandbox retains Docker's workspace and egress
        containment but receives no generic application environment or private
        agent mount.
        """
        # Per-turn lifecycle: never reuse a container across turns.  The web
        # layer tears each container down at the end of its turn
        # (manage/web/threads.py), so a registry entry surviving to here means
        # a prior turn's teardown didn't run (the worker died mid-turn).  Reap
        # that stale container before creating a fresh one — the registry is
        # keyed by work_dir, so creating without reaping would overwrite the
        # reference and orphan it (the 3h backstop TTL would eventually catch
        # it, but reaping now is immediate).
        if work_dir in cls._containers:
            cls.cleanup(work_dir)

        # Run the container as the host bind-mount's owner, not as
        # root.  Two reasons:
        #   1. Files the agent writes into /workspace land owned by
        #      the deploying user on the host, so the host process
        #      can clean them up via shutil.rmtree without the
        #      alpine-rm fallback in thread.py.
        #   2. As a non-root uid, the agent cannot read
        #      /usr/bin/git-real (locked at mode 0700 root:root by
        #      Dockerfile.sandbox).  This is the privilege-separation
        #      layer that closes the cp+exec-a bypass left over from
        #      PR #97.  See docs/2026-05-08-restrict-git-real-via-non-root-sandbox.org.
        # Read the bind-mount uid:gid here, not at module import,
        # so a misconfigured workspace surfaces as a thread-level
        # error rather than a manager-import crash.
        try:
            st = os.stat(work_dir)
        except OSError as e:
            raise RuntimeError(
                f"Cannot start sandbox for {work_dir}: stat failed ({e}). "
                "The workspace directory must exist and be readable by the web process."
            ) from e

        # Refuse to run the container as root.  A root-owned workspace
        # would mean `containers.run(user="0:0")` — which silently
        # restores the bypass this whole layer exists to close (the
        # agent inside the sandbox could read mode-0700 git-real and
        # copy it).  Pre-migration thread workspaces created before
        # this layer shipped *are* root-owned, so this check is what
        # catches them; the operator runs the documented chown and
        # the thread comes back online.
        if st.st_uid == 0:
            raise RuntimeError(
                f"Workspace {work_dir} is owned by root.  Refusing to "
                "start the sandbox because that would defeat the "
                "privilege-separation layer that prevents the agent "
                "from bypassing the git push refusal.  Migrate with: "
                f"sudo chown -R $USER:$USER {work_dir}  (or, for the "
                "whole threads dir at once, $ASSIST_THREADS_DIR).  "
                "See docs/2026-05-08-restrict-git-real-via-non-root-sandbox.org."
            )
        # SINGLE SOURCE OF TRUTH for the sandbox run user.  This one value
        # (the workspace owner) is applied via `containers.run(user=...)`
        # below and governs everything downstream: every `exec_run` inherits
        # it (no call passes `user=`), and `DockerSandboxBackend.upload_files`
        # reads it back off the container (`_run_uid_gid`) to stamp uploaded
        # files with the same ownership — otherwise put_archive's root default
        # leaves write_file output un-editable.  New features must NOT pass a
        # different `user=` to `exec_run` or hardcode a uid; derive from the
        # container's run user so ownership and execution stay aligned.
        user_arg = f"{st.st_uid}:{st.st_gid}"

        # Egress proxy bring-up runs OUTSIDE the broad-except below.
        # If the allowlist file is missing, or the egress network
        # exists but isn't internal=True, or the proxy image is
        # broken, we want a loud RuntimeError — silently returning
        # None here would mean threads keep working WITHOUT the
        # egress gate, defeating the layer entirely.  DockerException
        # (daemon down, transient API error) still degrades to None
        # via the explicit catch below.
        from docker.errors import DockerException
        try:
            client = cls._get_docker_client()
            cls._ensure_egress_proxy_running(client)
        except DockerException as e:
            logger.warning("Docker unavailable for egress setup: %s", e)
            return None
        # Anything else (RuntimeError from policy checks, etc) raises.

        try:
            proxy_url = f"http://{EGRESS_PROXY_NAME}:{EGRESS_PROXY_PORT}"
            # The sandbox is on an internal Docker network — no host-gateway
            # route, no NAT.  The only reachable name on this network is
            # ``assist-egress-proxy``, which terminates allowlisted CONNECT
            # tunnels and forwards allowlisted HTTP requests.
            #
            # NO_PROXY is *not* set: with internal=True there is no direct
            # path to bypass even for localhost references — every byte
            # must traverse the proxy.  ASSIST_MODEL_URL (rewritten to
            # host.docker.internal) reaches the host via the proxy's
            # bridge-side connection; ``host.docker.internal`` is in the
            # default allowlist for that reason.
            sandbox_env = {
                "HTTPS_PROXY": proxy_url,
                "HTTP_PROXY": proxy_url,
                "https_proxy": proxy_url,
                "http_proxy": proxy_url,
                "TZ": _sandbox_timezone(tz),  # local time (rider tz > host), not UTC
            }
            if include_assist_env:
                sandbox_env.update({
                    k: _rewrite_localhost(v)
                    for k, v in os.environ.items()
                    if k.startswith("ASSIST_") and k != "ASSIST_CAPTURES_DIR"
                })
            # Persistent /tmp: a host dir SIBLING of work_dir, bind-mounted at the
            # container's /tmp so it survives the per-turn container teardown (the
            # container's own /tmp would be lost with it).  Lets the agent stash a
            # file under /tmp (e.g. a renderable .md copy) that outlives the turn and
            # is reachable by the web renderer.  Created here (idempotent) + aligned to
            # the SAME uid:gid the container runs as (work_dir's owner) so the run-as
            # uid can write it even if the web process's own uid differs from work_dir's
            # owner (best-effort chown: a no-op when they already match, the common case).
            tmp_dir = os.path.join(os.path.dirname(work_dir), "tmp")
            if not os.path.isdir(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=True)
                try:
                    os.chown(tmp_dir, st.st_uid, st.st_gid)
                except OSError:
                    pass  # not permitted (web non-root, uids differ) — mount still
                          # works when web uid == work_dir owner (the deployment case)

            volumes = {work_dir: {"bind": "/workspace", "mode": "rw"},
                       tmp_dir: {"bind": "/tmp", "mode": "rw"}}
            if agent_dir is not None:
                os.makedirs(agent_dir, exist_ok=True)
                try:
                    os.chown(agent_dir, st.st_uid, st.st_gid)
                except OSError:
                    pass
                volumes[agent_dir] = {
                    "bind": "/agent",
                    "mode": "rw",
                }

            container = client.containers.run(
                SANDBOX_IMAGE,
                detach=True,
                remove=True,
                user=user_arg,
                volumes=volumes,
                working_dir="/workspace",
                stdin_open=True,
                tty=False,
                labels={"assist.sandbox": "true"},
                network=EGRESS_NETWORK,
                environment=sandbox_env,
            )
            logger.info("Started sandbox container %s for %s", container.id[:12], work_dir)
            cls._containers[work_dir] = container
            if include_egress_approvals:
                cls._record_egress_client(container, work_dir)
            from assist.sandbox import DockerSandboxBackend
            return DockerSandboxBackend(
                container, native_agent_dir=agent_dir is not None)
        except DockerException as e:
            logger.warning("Docker sandbox unavailable: %s", e)
            return None

    @classmethod
    def get_sandbox_backend(cls, work_dir: str, tz: str | None = None,
                            agent_dir: str | None = None):
        """Return the ordinary Docker sandbox, including its established app env."""
        return cls._get_sandbox_backend(
            work_dir, tz, agent_dir, include_assist_env=True, include_egress_approvals=True)

    @classmethod
    def get_pi_sandbox_backend(cls, work_dir: str, tz: str | None = None):
        """Return Pi's workspace-only Docker sandbox, without app secrets or `/agent`."""
        return cls._get_sandbox_backend(
            work_dir, tz, None, include_assist_env=False, include_egress_approvals=False)

    # work_dir -> egress-network IP for the client-attribution map (thread-
    # scoped egress grants; docs/2026-07-21-egress-approval-hitl.org).
    _egress_client_ips: dict[str, str] = {}

    @classmethod
    def _record_egress_client(cls, container, work_dir: str) -> None:
        """Map this sandbox's egress-network IP to its thread in the proxy's
        client-map. Best-effort by contract: a failure only costs this turn
        its thread-scoped grants (the proxy denies fail-closed on an unknown
        IP) — it must never fail the sandbox start. Dormant when the
        approvals dir env is unset. The thread id is work_dir's parent dir
        name (work_dir is ``<root>/<tid>/<subdir>`` — the
        ``thread_default_working_dir`` layout)."""
        egress_dir = os.environ.get("ASSIST_EGRESS_APPROVALS_DIR")
        if not egress_dir:
            return
        from assist.egress.store import approvals_dir_is_safe
        if not approvals_dir_is_safe(egress_dir):
            return   # same refusal as the mount — never write under a sandbox-reachable dir
        try:
            container.reload()
            ip = (container.attrs["NetworkSettings"]["Networks"]
                  [EGRESS_NETWORK]["IPAddress"])
            tid = os.path.basename(os.path.dirname(work_dir))
            if ip and tid:
                from assist.egress.client_map import record_client
                record_client(egress_dir, ip, tid)
                cls._egress_client_ips[work_dir] = ip
        except Exception:
            logger.warning("egress client-map record failed for %s (thread-"
                           "scoped grants deny this turn)", work_dir,
                           exc_info=True)

    @classmethod
    def _forget_egress_client(cls, work_dir: str) -> None:
        egress_dir = os.environ.get("ASSIST_EGRESS_APPROVALS_DIR")
        ip = cls._egress_client_ips.pop(work_dir, None)
        if egress_dir and ip:
            from assist.egress.client_map import forget_client
            forget_client(egress_dir, ip)

    @classmethod
    def current_container(cls, work_dir: str):
        """Return the registered container generation, if any."""
        return cls._containers.get(work_dir)

    @classmethod
    def cleanup(cls, work_dir: str, expected_container=_ANY_CONTAINER) -> None:
        """Tear down the container for a work_dir. Removal is automatic (--rm).

        With an ``expected_container``, cleanup is generation-safe: a stale turn
        cannot remove a replacement registered for the same workspace.

        SIGKILL, not a graceful ``stop()``.  A sandbox has nothing to shut
        down gracefully — it is ``--rm`` and all durable work is already
        flushed to the host bind mount — and its PID 1 is a bare ``sleep``
        with no SIGTERM handler (PID 1 gets no default handlers), so a
        ``stop()`` would just block the full 5s timeout before the daemon
        SIGKILLs anyway.  Killing keeps the per-turn teardown off that 5s
        path (and the same applies to thread-delete and shutdown).
        """
        container = cls._containers.get(work_dir)
        if (expected_container is not _ANY_CONTAINER
                and container is not expected_container):
            return
        container = cls._containers.pop(work_dir, None)
        cls._forget_egress_client(work_dir)
        if container:
            try:
                container.kill()
                logger.info("Cleaned up container for %s", work_dir)
            except Exception as e:
                logger.warning("Container cleanup failed: %s", e)

    @classmethod
    def cleanup_all(cls) -> None:
        """Kill all tracked sandbox containers. Removal is automatic (--rm).

        SIGKILL for the same reason as ``cleanup`` (nothing to flush; PID 1
        ignores SIGTERM) — and at lifespan shutdown a fast teardown is
        strictly better than burning 5s per container on the event loop.
        """
        for path, container in list(cls._containers.items()):
            cls._forget_egress_client(path)
            try:
                container.kill()
                logger.info("Cleaned up container for %s", path)
            except Exception as e:
                logger.warning("Container cleanup failed for %s: %s", path, e)
        cls._containers.clear()

    @classmethod
    def reap_orphans(cls, root_dir: str) -> None:
        """Kill THIS INSTANCE's surviving sandbox containers — by docker label
        plus workspace mount, not the in-memory registry. After a web-process
        crash ``_containers`` is empty, but the containers survive — and a
        ``docker exec``'d tool command keeps running inside one, mutating the
        host-bind-mounted /workspace that a recovery resume's FRESH container
        mounts too, for up to the 3h backstop TTL. Startup recovery calls this
        before dispatching any resume so a zombie writer can never share a
        workspace with a resumed turn.

        Scoped to containers whose /workspace bind-mount lives under
        ``root_dir`` (this deployment's threads root): the label alone is
        host-global, and prod + eval/dev sandboxes share ONE docker daemon on
        this box — a bare-label reap would kill a concurrently running eval's
        LIVE containers mid-turn. The mount filter needs no new create-time
        labeling, so it also covers orphans from pre-existing code. Best-effort:
        docker being down must not block recovery (turns fail fast on their own
        if it stays down)."""
        root = os.path.realpath(root_dir) + os.sep
        try:
            client = cls._get_docker_client()
            candidates = client.containers.list(
                filters={"label": "assist.sandbox=true"})
        except Exception as e:
            logger.warning("orphan-sandbox reap skipped (docker unavailable): %s", e)
            return
        for container in candidates:
            mounts = (container.attrs or {}).get("Mounts", [])
            ours = any(m.get("Destination") == "/workspace"
                       and os.path.realpath(m.get("Source", "")).startswith(root)
                       for m in mounts)
            if not ours:
                continue
            try:
                container.kill()
                logger.info("Reaped orphaned sandbox %s", container.id[:12])
            except Exception as e:
                logger.warning("orphan reap failed for %s: %s", container.id[:12], e)
        cls._containers.clear()
