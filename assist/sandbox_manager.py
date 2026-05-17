import hashlib
import logging
import os
import re
import threading
import time

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
        allowlist_hash = hashlib.sha256(allowlist_csv.encode()).hexdigest()[:16]

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
            logger.info("Started sandbox container %s for %s", container.id[:12], work_dir)
            # Pre-create /workspace/references inside the container so
            # the research-subagent's DockerSandboxBackend (which uses
            # work_dir="/workspace/references" — see assist/agent.py:315-322)
            # can chdir into it on its first exec call.  Without this,
            # every research-subagent tool call fails with `OCI runtime
            # exec failed: chdir to cwd ("/workspace/references")`.
            # ReferencesCleanupRunnable._ensure_dir() exists as a lazy
            # fallback but didn't fire in the 2026-05-16 winged-horse-flag
            # thread for reasons not yet understood (suspected
            # deepagents 0.6.1 wrapper bypass); this layer is the
            # load-bearing guarantee.  Fail-closed if mkdir fails — a
            # sandbox that can't pre-create this dir is broken for the
            # whole research-flow path and we'd rather surface that here
            # than at minute 13 of a research thread.
            #
            # exec_run inherits the container's --user (host uid:gid) when
            # `user=` is unset — docker SDK posts User='' which the engine
            # interprets as "use Config.User from containers.run".  So the
            # dir lands on the host bind mount owned consistently with the
            # rest of /workspace.  (The SDK docstring "Default: root" is
            # misleading; it only applies when --user was not set on
            # containers.run, which is never the case for us — see line
            # 290-302.)  Bind mount is guaranteed ready: container start
            # blocks on mount-namespace setup before PID 1 is exec'd.
            try:
                exit_code, output = container.exec_run(
                    ["mkdir", "-p", "/workspace/references"]
                )
            except Exception:
                # exec_run itself failed before returning a tuple — stop
                # the container and re-raise so the caller sees a clean
                # error instead of a half-initialised backend.
                try:
                    container.stop(timeout=5)
                except Exception:
                    pass  # best-effort; remove=True will GC if/when it exits
                raise
            if exit_code != 0:
                output_str = output.decode("utf-8", errors="replace") if output else ""
                # Stop the container before raising — otherwise it lingers
                # as an orphan (`remove=True` only fires on process exit).
                # Stopping triggers the auto-remove.  Also prevents the
                # _containers registry from a poisoned entry — we haven't
                # written to it yet (intentionally — see the post-mkdir
                # registry write below), but defense in depth.
                try:
                    container.stop(timeout=5)
                except Exception:
                    pass  # best-effort
                raise RuntimeError(
                    f"Failed to pre-create /workspace/references in sandbox "
                    f"container {container.id[:12]}: exit_code={exit_code} "
                    f"output={output_str!r}"
                )
            # Only register the container AFTER mkdir succeeded.  Writing
            # to _containers before mkdir would leave a poisoned entry on
            # mkdir failure — the next get_sandbox_backend call would
            # short-circuit on the "running" early-return (line 197-203)
            # and hand out a backend wrapping the broken container.
            cls._containers[work_dir] = container
            from assist.sandbox import DockerSandboxBackend
            return DockerSandboxBackend(container)
        except DockerException as e:
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
