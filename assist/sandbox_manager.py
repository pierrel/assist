import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)


def _rewrite_localhost(value: str) -> str:
    """Replace localhost/127.0.0.1 references with host.docker.internal."""
    return re.sub(r'localhost|127\.0\.0\.1', 'host.docker.internal', value)

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
    """Read the sandbox egress allowlist.

    ASSIST_SANDBOX_EGRESS_ALLOWLIST env (comma-separated) takes
    precedence; falls back to dockerfiles/egress-allowlist.conf.
    Returns a sorted list (deterministic hash for change detection).
    """
    env = os.environ.get("ASSIST_SANDBOX_EGRESS_ALLOWLIST", "").strip()
    if env:
        return sorted({h.strip() for h in env.split(",") if h.strip()})
    try:
        with open(EGRESS_ALLOWLIST_FILE) as f:
            return sorted({
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            })
    except FileNotFoundError:
        logger.warning(
            "Egress allowlist file %s missing; falling back to baked-in defaults",
            EGRESS_ALLOWLIST_FILE,
        )
        return sorted({"pypi.org", "files.pythonhosted.org", "pip.pypa.io",
                       "host.docker.internal"})


class SandboxManager:
    """Manages Docker sandbox container lifecycle.

    Class-level state: one Docker client shared across all callers,
    with a container registry keyed by work_dir for cleanup.
    """

    _docker_client = None
    _containers: dict[str, "docker.models.containers.Container"] = {}  # type: ignore[name-defined]

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
        - The current allowlist is hashed and stamped on the container
          as a label.  When the allowlist changes (env or file), the
          proxy is recreated on the next call.

        Fails closed: if the proxy can't be brought up, callers see the
        exception and the sandbox start fails — which is the correct
        behavior (no fallback to direct egress).
        """
        from docker.errors import APIError, NotFound

        allowlist = _load_egress_allowlist()
        allowlist_csv = ",".join(allowlist)
        allowlist_hash = hashlib.sha256(allowlist_csv.encode()).hexdigest()[:16]

        try:
            egress_net = client.networks.get(EGRESS_NETWORK)
        except NotFound:
            egress_net = client.networks.create(
                EGRESS_NETWORK, driver="bridge", internal=True,
            )
            logger.info("Created egress network %s (internal)", EGRESS_NETWORK)

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

        proxy = client.containers.run(
            EGRESS_PROXY_IMAGE,
            name=EGRESS_PROXY_NAME,
            detach=True,
            restart_policy={"Name": "unless-stopped"},
            extra_hosts={"host.docker.internal": "host-gateway"},
            environment={"EGRESS_ALLOWLIST": allowlist_csv},
            labels={
                "assist.egress-proxy": "true",
                "assist.egress-allowlist-hash": allowlist_hash,
            },
        )
        try:
            egress_net.connect(proxy)
        except APIError as e:
            if "already exists" not in str(e).lower():
                raise
        logger.info(
            "Started egress proxy %s with %d allowlist entries (hash=%s)",
            proxy.id[:12], len(allowlist), allowlist_hash,
        )
        return EGRESS_PROXY_NAME

    @classmethod
    def get_sandbox_backend(cls, work_dir: str):
        """Return a DockerSandboxBackend for work_dir, creating a container if needed.

        Returns None if Docker is not available.
        """
        if work_dir in cls._containers:
            container = cls._containers[work_dir]
            try:
                container.reload()
                if container.status == "running":
                    from assist.sandbox import DockerSandboxBackend
                    return DockerSandboxBackend(container)
            except Exception:
                cls._containers.pop(work_dir, None)

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
        user_arg = f"{st.st_uid}:{st.st_gid}"

        try:
            client = cls._get_docker_client()
            cls._ensure_egress_proxy_running(client)
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
            }
            sandbox_env.update({
                k: _rewrite_localhost(v)
                for k, v in os.environ.items()
                if k.startswith("ASSIST_")
            })
            container = client.containers.run(
                SANDBOX_IMAGE,
                detach=True,
                remove=True,
                user=user_arg,
                volumes={work_dir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                stdin_open=True,
                tty=False,
                labels={"assist.sandbox": "true"},
                network=EGRESS_NETWORK,
                environment=sandbox_env,
            )
            cls._containers[work_dir] = container
            logger.info("Started sandbox container %s for %s", container.id[:12], work_dir)
            from assist.sandbox import DockerSandboxBackend
            return DockerSandboxBackend(container)
        except Exception as e:
            logger.warning("Docker sandbox unavailable: %s", e)
            return None

    @classmethod
    def cleanup(cls, work_dir: str) -> None:
        """Stop the container for a given work_dir. Removal is automatic (--rm)."""
        container = cls._containers.pop(work_dir, None)
        if container:
            try:
                container.stop(timeout=5)
                logger.info("Cleaned up container for %s", work_dir)
            except Exception as e:
                logger.warning("Container cleanup failed: %s", e)

    @classmethod
    def cleanup_all(cls) -> None:
        """Stop all tracked sandbox containers. Removal is automatic (--rm)."""
        for path, container in list(cls._containers.items()):
            try:
                container.stop(timeout=5)
                logger.info("Cleaned up container for %s", path)
            except Exception as e:
                logger.warning("Container cleanup failed for %s: %s", path, e)
        cls._containers.clear()
