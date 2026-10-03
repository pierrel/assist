import hashlib
import ipaddress
import logging
import os
import re
import subprocess
import threading
import time
import fcntl
from contextlib import contextmanager
from uuid import uuid4

from assist.egress import runtime_state

logger = logging.getLogger(__name__)
_ANY_CONTAINER = object()


def confirm_generation_stopped(container_id: str) -> None:
    """Prove one exact sandbox generation is gone and its name is reusable."""
    try:
        subprocess.run(["docker", "kill", container_id],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("sandbox generation teardown is unconfirmed") from error
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2,
                check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("sandbox generation teardown is unconfirmed") from error
        if inspected.returncode == 0 and inspected.stdout.strip() == b"false":
            try:
                subprocess.run(["docker", "rm", "-f", container_id],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=2,
                               check=False)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise RuntimeError("sandbox generation removal is unconfirmed") from error
        if (inspected.returncode != 0 and
                (b"no such object" in inspected.stderr.lower()
                 or b"no such container" in inspected.stderr.lower())):
            return
        time.sleep(0.05)
    raise RuntimeError("sandbox generation teardown is unconfirmed")


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
BROWSER_NETWORK = "assist-browser-network"
EGRESS_PROXY_NAME = "assist-egress-proxy"
EGRESS_PROXY_IMAGE = "assist-egress-proxy"
EGRESS_PROXY_PORT = 8888
BROWSER_PROXY_PORT = 8889
BROWSER_SECCOMP_PROFILE = os.path.join(
    os.path.dirname(__file__), "..", "dockerfiles", "browser-seccomp.json")
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


def _egress_proxy_config_hash(allowlist_csv: str, approvals_dir: str | None,
                              network_ref: str = "", map_dir: str | None = None) -> str:
    """The proxy's config fingerprint (a container label): allowlist content
    plus the approvals-mount schema/path and proxy-policy schema, so a change
    to any of them recreates the proxy on the next sandbox start. The version
    marker makes containers with an earlier approval, throttle or browser
    destination policy recreate once and gain the current behavior. The
    service UID/GID binds proxy read access to host-only map mounts."""
    return hashlib.sha256(
        (allowlist_csv + "|v10-read-only-proxy-mounts:"
         + f"{os.getuid()}:{os.getgid()}|" + (approvals_dir or "")
         + "|" + (map_dir or "") + "|" + network_ref).encode()
    ).hexdigest()[:16]


def _network_identity(network, *, isolated: bool) -> tuple[str, str]:
    """Verify Docker's live bridge settings before admitting a client."""
    network.reload()
    attrs = network.attrs
    options = attrs.get("Options") or {}
    configs = (attrs.get("IPAM") or {}).get("Config") or []
    if attrs.get("Internal") is not True:
        raise RuntimeError("egress network is not internal")
    if (attrs.get("Driver") != "bridge" or attrs.get("EnableIPv6") is not False
            or len(configs) != 1):
        raise RuntimeError("egress network has unsupported routing settings")
    if isolated and (options.get("com.docker.network.bridge.gateway_mode_ipv4") != "isolated"
                     or configs[0].get("Gateway")):
        raise RuntimeError("browser network lacks isolated gateway mode")
    subnet = str(ipaddress.ip_network(configs[0]["Subnet"], strict=True))
    if not isinstance(attrs.get("Id"), str) or not attrs["Id"]:
        raise RuntimeError("egress network lacks identity")
    return attrs["Id"], subnet


def _settle_rejected_create(error, kind: str, name: str, token: str) -> None:
    """A daemon 400/404 rejects create before an object exists; timeouts do not."""
    if error.status_code in (400, 404):
        runtime_state.settle_creation(kind, name, token)


@contextmanager
def _proxy_setup_lock():
    """Serialize shared Docker egress setup independently of optional grants/map."""
    with open(os.path.join(runtime_state.runtime_directory(),
                           ".proxy-setup.lock"), "a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _bounded_egress_worker(argv: list[str], timeout: float = 20):
    """A timeout kills the process that owns SDK calls and setup locks."""
    import subprocess
    try:
        return subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("egress Docker setup timed out; generation retired") from error


def _egress_proxy_mounts(approvals_dir: str | None, map_dir: str | None = None):
    """Keep both read-only targets when approvals and map share one source."""
    from docker.types import Mount

    return [
        Mount(target, source, type="bind", read_only=True)
        for target, source in (("/approvals", approvals_dir),
                               ("/client-map", map_dir))
        if source
    ]


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
            cls._docker_client = docker.from_env(timeout=5)
        return cls._docker_client

    @classmethod
    def _ensure_egress_proxy_running(cls, client) -> str:
        """Verify live network identities and serve both isolated client kinds."""
        import docker
        if isinstance(client, docker.DockerClient):
            return cls._ensure_egress_proxy_bounded()
        return cls._ensure_egress_proxy_running_direct(client)

    @classmethod
    def _ensure_egress_proxy_bounded(cls) -> str:
        """Kill a stalled SDK worker without leaving its host locks held."""
        import sys
        result = _bounded_egress_worker(
            [sys.executable, "-m", "assist.egress.runtime_worker", "proxy"])
        if result.returncode or result.stdout.strip() != EGRESS_PROXY_NAME.encode():
            raise RuntimeError(
                "egress Docker setup failed: "
                + result.stderr.decode("utf-8", errors="replace")[-1000:])
        return EGRESS_PROXY_NAME

    @classmethod
    def _ensure_egress_proxy_running_direct(cls, client) -> str:
        """Run inside the killable worker, or with a deterministic test double."""
        from docker.errors import APIError, NotFound
        from assist.egress.client_map import configured_directory

        allowlist = _load_egress_allowlist()
        allowlist_csv = ",".join(allowlist)
        # Approved-host grants and client attribution use distinct read-only
        # proxy mounts. Browser attribution remains available when the
        # optional grants projection is disabled.
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
        try:
            map_dir = configured_directory()
        except (OSError, RuntimeError) as error:
            # Client attribution is optional for an ordinary shell sandbox.
            # A bad client map must not remove its base-list egress.
            logger.warning("browser client map unavailable: %s", error)
            map_dir = None

        with cls._egress_lock, _proxy_setup_lock():
            browser_armed = False
            if map_dir:
                from assist.egress.client_map import sandbox_browser_records
                browser_armed = bool(sandbox_browser_records(map_dir))
            if browser_armed and any(runtime_state.pending_creation(kind, name)
                                     for kind, name in (
                                         ("network", EGRESS_NETWORK),
                                         ("network", BROWSER_NETWORK),
                                         ("proxy", EGRESS_PROXY_NAME))):
                raise RuntimeError(
                    "browser turn still pins the egress proxy endpoint")
            ordinary_creation_token = None
            browser_creation_token = None
            for name in ((EGRESS_NETWORK, BROWSER_NETWORK)
                         if map_dir is not None else (EGRESS_NETWORK,)):
                pending = runtime_state.pending_creation("network", name)
                if pending:
                    try:
                        candidate = client.networks.get(name)
                    except NotFound:
                        raise RuntimeError(
                            f"{name} creation outcome is uncertain; retry after recovery")
                    candidate.reload()
                    label = (candidate.attrs.get("Labels") or {}).get(
                        "assist.egress-generation")
                    if label == pending:
                        runtime_state.retire("network", candidate.id,
                                             "incomplete create")
                        client.networks.get(candidate.id).remove()
                        try:
                            client.networks.get(candidate.id)
                        except NotFound:
                            runtime_state.settle_creation("network", name, pending)
                        else:
                            raise RuntimeError(f"{name} removal is unconfirmed")
                    elif label:
                        # A distinct labelled successor occupies the immutable
                        # name; the old late create cannot replace it.
                        runtime_state.settle_creation("network", name, pending)
                    else:
                        raise RuntimeError(f"{name} creation outcome is uncertain")
            pending_proxy = runtime_state.pending_creation("proxy", EGRESS_PROXY_NAME)
            if pending_proxy:
                try:
                    candidate = client.containers.get(EGRESS_PROXY_NAME)
                    candidate.reload()
                except NotFound:
                    raise RuntimeError("egress proxy creation outcome is uncertain")
                label = candidate.labels.get("assist.egress-generation")
                if label == pending_proxy:
                    runtime_state.retire("proxy", candidate.id,
                                         "incomplete create")
                    client.containers.get(candidate.id).remove(force=True)
                    try:
                        client.containers.get(candidate.id)
                    except NotFound:
                        runtime_state.settle_creation("proxy", EGRESS_PROXY_NAME,
                                                      pending_proxy)
                    else:
                        raise RuntimeError("egress proxy removal is unconfirmed")
                elif label:
                    runtime_state.settle_creation("proxy", EGRESS_PROXY_NAME,
                                                  pending_proxy)
                else:
                    raise RuntimeError("egress proxy creation outcome is uncertain")
            try:
                egress_net = client.networks.get(EGRESS_NETWORK)
            except NotFound:
                if browser_armed:
                    raise RuntimeError(
                        "browser turn still pins the egress proxy endpoint")
                ordinary_creation_token = uuid4().hex
                runtime_state.begin_creation("network", EGRESS_NETWORK,
                                             ordinary_creation_token)
                try:
                    response = client.api.create_network(
                        EGRESS_NETWORK, driver="bridge", internal=True,
                        labels={"assist.egress-generation": ordinary_creation_token},
                    )
                except APIError as error:
                    _settle_rejected_create(
                        error, "network", EGRESS_NETWORK, ordinary_creation_token)
                    raise
                egress_net = client.networks.get(response["Id"])
                logger.info("Created egress network %s (internal)", EGRESS_NETWORK)
            ordinary_id, ordinary_cidr = _network_identity(
                egress_net, isolated=False)
            if ordinary_creation_token:
                if (egress_net.attrs.get("Labels") or {}).get(
                        "assist.egress-generation") != ordinary_creation_token:
                    raise RuntimeError("new egress network generation changed")
                runtime_state.settle_creation("network", EGRESS_NETWORK,
                                              ordinary_creation_token)
            runtime_state.assert_admissible("network", ordinary_id)
            browser_net = None
            browser_id = browser_cidr = ""
            if map_dir is not None:
                try:
                    browser_net = client.networks.get(BROWSER_NETWORK)
                except NotFound:
                    if browser_armed:
                        raise RuntimeError(
                            "browser turn still pins the egress proxy endpoint")
                    browser_creation_token = uuid4().hex
                    runtime_state.begin_creation("network", BROWSER_NETWORK,
                                                 browser_creation_token)
                    try:
                        response = client.api.create_network(
                            BROWSER_NETWORK, driver="bridge", internal=True,
                            enable_ipv6=False,
                            options={"com.docker.network.bridge.gateway_mode_ipv4":
                                     "isolated"},
                            labels={"assist.egress-generation": browser_creation_token})
                    except APIError as error:
                        _settle_rejected_create(
                            error, "network", BROWSER_NETWORK, browser_creation_token)
                        raise
                    browser_net = client.networks.get(response["Id"])
                browser_id, browser_cidr = _network_identity(
                    browser_net, isolated=True)
                if browser_creation_token:
                    if (browser_net.attrs.get("Labels") or {}).get(
                            "assist.egress-generation") != browser_creation_token:
                        raise RuntimeError("new browser network generation changed")
                    runtime_state.settle_creation("network", BROWSER_NETWORK,
                                                  browser_creation_token)
                runtime_state.assert_admissible("network", browser_id)
                if ipaddress.ip_network(ordinary_cidr).overlaps(
                        ipaddress.ip_network(browser_cidr)):
                    raise RuntimeError("browser and sandbox networks overlap")
            network_ref = f"{ordinary_id}:{ordinary_cidr}|{browser_id}:{browser_cidr}"
            allowlist_hash = _egress_proxy_config_hash(
                allowlist_csv, approvals_dir, network_ref, map_dir)

            existing = None
            try:
                existing = client.containers.get(EGRESS_PROXY_NAME)
                existing.reload()
            except NotFound:
                pass
            prior_retirement = (runtime_state.retirement("proxy", existing.id)
                                if existing is not None else None)
            if prior_retirement not in (None, "remove"):
                raise RuntimeError("egress proxy generation is retired")

            needs_recreate = (
                existing is None
                or prior_retirement is not None
                or existing.status != "running"
                or existing.labels.get("assist.egress-allowlist-hash") != allowlist_hash
            )
            if not needs_recreate:
                attached = existing.attrs.get("NetworkSettings", {}).get("Networks", {})
                needs_recreate = (
                    attached.get(EGRESS_NETWORK, {}).get("NetworkID") != ordinary_id
                    or (browser_net is not None and
                        attached.get(BROWSER_NETWORK, {}).get("NetworkID") != browser_id)
                )
            if not needs_recreate:
                return EGRESS_PROXY_NAME

            if browser_armed:
                raise RuntimeError(
                    "browser turn still pins the egress proxy endpoint")

            if os.getuid() == 0:
                raise RuntimeError("egress proxy cannot run as root")
            # A missing image is a definitive prerequisite failure. Check it
            # before removing the old proxy or journaling an uncertain create.
            client.images.get(EGRESS_PROXY_IMAGE)
            old_attached = (existing.attrs.get("NetworkSettings", {}).get("Networks", {})
                            if existing is not None else {})
            old_mounts = (existing.attrs.get("Mounts")
                          if existing is not None else None)
            old_map = ([mount.get("Source") for mount in old_mounts
                        if mount.get("Destination") == "/client-map"]
                       if isinstance(old_mounts, list) else [])
            preserve_map = bool(
                map_dir and existing is not None
                and old_map == [map_dir]
                and existing.attrs.get("Config", {}).get("User")
                == f"{os.getuid()}:{os.getgid()}"
                and old_attached.get(EGRESS_NETWORK, {}).get("NetworkID") == ordinary_id
                and (browser_net is None or
                     old_attached.get(BROWSER_NETWORK, {}).get("NetworkID") == browser_id))
            if existing is not None:
                try:
                    runtime_state.retire("proxy", existing.id, "remove")
                    client.containers.get(existing.id).remove(force=True)
                    logger.info("Removed stale egress proxy %s", existing.id[:12])
                except APIError as e:
                    raise RuntimeError("old egress proxy removal is unconfirmed") from e
            if map_dir is not None and not preserve_map:
                from assist.egress.client_map import clear_clients
                # A changed map/network may hold stale attribution. Revoke
                # only after old-proxy removal is confirmed; a same-map,
                # same-network replacement keeps active clients usable.
                clear_clients(map_dir)

            from assist.egress.guidance import EGRESS_DENY_BODY, EGRESS_THROTTLE_BODY
            token = uuid4().hex
            from docker.models.containers import _create_container_args
            create_kwargs = _create_container_args({
                "image": EGRESS_PROXY_IMAGE,
                "command": None,
                "version": client.api._version,
                "name": EGRESS_PROXY_NAME,
                "restart_policy": {"Name": "unless-stopped"},
                "extra_hosts": {"host.docker.internal": "host-gateway"},
                "environment": {"EGRESS_ALLOWLIST": allowlist_csv,
                                "EGRESS_SANDBOX_CIDR": ordinary_cidr,
                                "EGRESS_BROWSER_CIDR": browser_cidr,
                                "EGRESS_DENY_BODY": EGRESS_DENY_BODY,
                                "EGRESS_THROTTLE_BODY": EGRESS_THROTTLE_BODY},
                "labels": {
                    "assist.egress-proxy": "true",
                    "assist.egress-allowlist-hash": allowlist_hash,
                    "assist.egress-generation": token,
                },
                "user": f"{os.getuid()}:{os.getgid()}",
                "mounts": _egress_proxy_mounts(approvals_dir, map_dir),
            })
            runtime_state.begin_creation("proxy", EGRESS_PROXY_NAME, token)
            try:
                response = client.api.create_container(**create_kwargs)
            except APIError as error:
                _settle_rejected_create(error, "proxy", EGRESS_PROXY_NAME, token)
                raise
            proxy = client.containers.get(response["Id"])
            proxy.start()
            runtime_state.retire("network", ordinary_id, "connect:" + proxy.id)
            try:
                egress_net.connect(proxy)
            except APIError as e:
                if "already exists" not in str(e).lower():
                    raise
            proxy.reload()
            attached = proxy.attrs.get("NetworkSettings", {}).get("Networks", {})
            if attached.get(EGRESS_NETWORK, {}).get("NetworkID") != ordinary_id:
                raise RuntimeError("egress proxy ordinary attachment differs from policy")
            runtime_state.settle_retirement("network", ordinary_id,
                                            "connect:" + proxy.id)
            if browser_net is not None:
                runtime_state.retire("network", browser_id, "connect:" + proxy.id)
                browser_net.connect(proxy)
            proxy.reload()
            attached = proxy.attrs.get("NetworkSettings", {}).get("Networks", {})
            if (attached.get(EGRESS_NETWORK, {}).get("NetworkID") != ordinary_id
                    or (browser_net is not None and
                        attached.get(BROWSER_NETWORK, {}).get("NetworkID") != browser_id)):
                raise RuntimeError("egress proxy network attachment differs from policy")
            if browser_net is not None:
                runtime_state.settle_retirement("network", browser_id,
                                                "connect:" + proxy.id)
            cls._wait_for_egress_proxy_ready(proxy)
            runtime_state.settle_creation("proxy", EGRESS_PROXY_NAME, token)
            logger.info(
                "Started egress proxy %s with %d allowlist entries (hash=%s)",
                proxy.id[:12], len(allowlist), allowlist_hash,
            )
            return EGRESS_PROXY_NAME

    @classmethod
    def _wait_for_egress_proxy_ready(cls, proxy, timeout: float = 10.0) -> None:
        """Block until both ordinary and browser proxy listeners are bound.

        The proxy's TCP listeners are bound only after Python startup +
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
            if (f"listening on 0.0.0.0:{EGRESS_PROXY_PORT}" in logs
                    and f"listening on 0.0.0.0:{BROWSER_PROXY_PORT}" in logs):
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Egress proxy {proxy.id[:12]} did not report both listeners "
            f"within {timeout}s.  Last logs: {logs[-500:]!r}"
        )

    @classmethod
    def _get_sandbox_backend(cls, work_dir: str, tz: str | None,
                             agent_dir: str | None, include_assist_env: bool,
                             include_egress_approvals: bool,
                             before_start=None, readonly_workspace=False,
                             browser_capable: bool = False,
                             thread_scope: tuple[str, str] | None = None,
                             owner_run_id: str | None = None):
        if thread_scope is None:
            return cls._create_sandbox_backend(
                work_dir, tz, agent_dir, include_assist_env,
                include_egress_approvals, before_start=before_start,
                readonly_workspace=readonly_workspace,
                browser_capable=browser_capable, thread_scope=thread_scope,
                owner_run_id=owner_run_id)
        from assist.browser import authority
        from assist.browser.manager import BrowserManager
        threads_root, thread_id = thread_scope
        authority._thread_dir(threads_root, thread_id)
        expected = os.path.join(os.path.realpath(threads_root), thread_id, "domain")
        if os.path.realpath(work_dir) != expected:
            raise RuntimeError("sandbox workspace does not match thread authority")
        if not BrowserManager.reconcile_startup(threads_root):
            raise RuntimeError("sandbox startup reconciliation is incomplete")
        with authority.generation_fence(threads_root, thread_id):
            return cls._create_sandbox_backend(
                work_dir, tz, agent_dir, include_assist_env,
                include_egress_approvals, before_start=before_start,
                readonly_workspace=readonly_workspace,
                browser_capable=browser_capable, thread_scope=thread_scope,
                owner_run_id=owner_run_id)

    @classmethod
    def _create_sandbox_backend(cls, work_dir: str, tz: str | None,
                                agent_dir: str | None, include_assist_env: bool,
                                include_egress_approvals: bool,
                                before_start=None, readonly_workspace=False,
                                browser_capable: bool = False,
                                thread_scope: tuple[str, str] | None = None,
                                owner_run_id: str | None = None):
        """Create one per-turn sandbox from a named authority profile.

        ``include_assist_env`` is the line between ordinary Deep Agents work and
        Pi preview work.  A Pi sandbox retains Docker's workspace and egress
        containment but receives no generic application environment or private
        agent mount. ``before_start`` records a Git recovery fence immediately
        before Docker create; earlier policy/setup failures create no new fence.
        A preceding Git generation's retained fence is cleared only after verification.
        Read-only Git verification omits persistent scratch/private mounts, so
        configured filters cannot mutate the checked worktree through an alias.
        ``thread_scope`` binds managed web/Pi callers to their
        exact thread workspace; generic callers never infer authority from a path.
        """
        # Never reuse container generations. The web layer tears each down at
        # its phase boundary (manage/web/threads.py), so a registry entry
        # surviving to here means teardown did not run (the worker died). Reap
        # that stale container before creating a fresh one — the registry is
        # keyed by work_dir, so creating without reaping would overwrite the
        # reference and orphan it (the 3h backstop TTL would eventually catch
        # it, but reaping now is immediate).
        from assist.browser import authority
        from assist.browser.manager import BrowserManager

        if browser_capable and thread_scope is None:
            raise RuntimeError("browser sandbox requires explicit thread authority")
        if thread_scope is not None:
            threads_root, thread_id = thread_scope
            expected = os.path.join(os.path.realpath(threads_root), thread_id, "domain")
            if os.path.realpath(work_dir) != expected:
                raise RuntimeError("sandbox workspace does not match thread authority")
            # The scoped Docker scan also covers pre-journal generations whose
            # host registry disappeared with a former web worker.
            BrowserManager.confirm_owner_stopped(
                threads_root, thread_id, None, before_replacement=True)
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
        if browser_capable and st.st_uid == 10001:
            raise RuntimeError(
                "Workspace owner conflicts with the reserved browser identity")
        # SINGLE SOURCE OF TRUTH for the sandbox run user.  This one value
        # (the workspace owner) is applied via `containers.run(user=...)`
        # below and governs agent tool execution; DockerSandboxBackend.upload_files
        # reads it back off the container (`_run_uid_gid`) to stamp uploaded
        # files with the same ownership. The host-controlled browser launcher
        # is the deliberate exception: it uses a separate fixed UID and a
        # private runtime tmpfs so Chromium cannot inherit the agent identity.
        user_arg = f"{st.st_uid}:{st.st_gid}"

        runtime_state.assert_outside_mounts(
            work_dir, os.path.join(os.path.dirname(work_dir), "tmp"), agent_dir)

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
            # This ordinary shell sandbox is on an internal Docker network;
            # its configured HTTP(S) clients use the shared egress proxy.
            # A separate browser UID uses a kernel OUTPUT fence and its own
            # proxy listener; shell traffic keeps this ordinary route.
            #
            # NO_PROXY is not set: configured HTTP(S) client traffic uses
            # the proxy. ASSIST_MODEL_URL (rewritten to host.docker.internal)
            # reaches the host through that proxy when allowlisted.
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
            if not readonly_workspace and not os.path.isdir(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=True)
                try:
                    os.chown(tmp_dir, st.st_uid, st.st_gid)
                except OSError:
                    pass  # not permitted (web non-root, uids differ) — mount still
                          # works when web uid == work_dir owner (the deployment case)

            volumes = {work_dir: {"bind": "/workspace",
                                  "mode": "ro" if readonly_workspace else "rw"}}
            if not readonly_workspace:
                volumes[tmp_dir] = {"bind": "/tmp", "mode": "rw"}
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

            browser_options = {}
            if browser_capable:
                with open(BROWSER_SECCOMP_PROFILE, encoding="utf-8") as profile:
                    browser_options = {
                        "tmpfs": {"/run/assist-browser":
                                  "rw,nosuid,nodev,size=134217728,uid=10001,gid=10001,mode=0700"},
                        "security_opt": ["seccomp=" + profile.read()],
                    }
            owner = owner_run_id or uuid4().hex
            if thread_scope is not None:
                from assist.run_service import RunService
                runs = RunService(threads_root).list(thread_id)
                sequence = max((run.admission_sequence for run in runs
                                if run.user_event_id == run.id), default=0)
                with authority.fence(threads_root, thread_id) as state:
                    state.begin(owner, sequence)
            try:
                if before_start is not None:
                    before_start()
            except Exception:
                # No Docker generation exists yet. Do not strand a browser
                # authority lease when the Git pre-create fence refuses.
                if thread_scope is not None:
                    with authority.fence(threads_root, thread_id) as state:
                        state.clear(owner)
                raise
            container = client.containers.run(
                SANDBOX_IMAGE,
                detach=True,
                remove=True,
                **({"name": "assist-turn-" + hashlib.sha256(
                    (os.path.realpath(threads_root) + "\0" + thread_id).encode()
                ).hexdigest()[:32]} if thread_scope is not None else {}),
                user=user_arg,
                volumes=volumes,
                working_dir="/workspace",
                stdin_open=True,
                tty=False,
                labels={"assist.sandbox": "true"},
                network=EGRESS_NETWORK,
                environment=sandbox_env,
                **browser_options,
            )
            if thread_scope is not None:
                with authority.fence(threads_root, thread_id) as state:
                    state.add_generation(owner, container.id)
                cls._generation_owners[container.id] = (threads_root, thread_id, owner)
            logger.info("Started sandbox container %s for %s", container.id[:12], work_dir)
            cls._containers[work_dir] = container
            try:
                cls._record_egress_client(
                    client, container, work_dir,
                    kind="sandbox" if include_egress_approvals else "pi")
            except Exception:
                cls.cleanup(work_dir, container)
                raise
            from assist.sandbox import DockerSandboxBackend
            return DockerSandboxBackend(
                container, native_agent_dir=agent_dir is not None)
        except DockerException as e:
            logger.warning("Docker sandbox unavailable: %s", e)
            return None

    @classmethod
    def get_sandbox_backend(cls, work_dir: str, tz: str | None = None,
                            agent_dir: str | None = None,
                            before_start=None, browser_capable: bool = False,
                            thread_scope: tuple[str, str] | None = None,
                            owner_run_id: str | None = None):
        """Return the ordinary sandbox; managed callers bind its thread authority."""
        return cls._get_sandbox_backend(
            work_dir, tz, agent_dir, include_assist_env=True,
            include_egress_approvals=True, before_start=before_start,
            browser_capable=browser_capable,
            thread_scope=thread_scope, owner_run_id=owner_run_id)

    @classmethod
    def get_pi_sandbox_backend(cls, work_dir: str, tz: str | None = None, *,
                               before_start=None,
                               thread_scope: tuple[str, str] | None = None,
                               owner_run_id: str | None = None):
        """Return Pi's workspace-only Docker sandbox, without app secrets or `/agent`."""
        return cls._get_sandbox_backend(
            work_dir, tz, None, include_assist_env=False,
            include_egress_approvals=False, before_start=before_start,
            thread_scope=thread_scope,
            owner_run_id=owner_run_id)

    @classmethod
    def get_git_verification_backend(cls, work_dir: str, tz: str | None = None,
                                     before_start=None):
        """Credential-free read-only worktree, with ephemeral scratch and no `/agent`."""
        return cls._get_sandbox_backend(
            work_dir, tz, None, include_assist_env=False,
            include_egress_approvals=False, before_start=before_start,
            readonly_workspace=True)

    # work_dir -> (map directory, egress-network IP, container generation)
    # for shell attribution or an explicit Pi no-grant marker.
    _egress_client_ips: dict[str, tuple[str, str, str]] = {}
    _generation_owners: dict[str, tuple[str, str, str]] = {}

    @classmethod
    def _record_egress_client(cls, client, container, work_dir: str,
                              *, kind: str) -> None:
        """Use a bounded worker for production attribution, or verify base-only mode."""
        import docker
        if isinstance(client, docker.DockerClient):
            import json
            import sys
            result = _bounded_egress_worker([
                sys.executable, "-m", "assist.egress.runtime_worker", "record",
                container.id, work_dir, kind])
            if result.returncode:
                raise RuntimeError(
                    "egress client attribution failed: "
                    + result.stderr.decode("utf-8", errors="replace")[-1000:])
            identity = json.loads(result.stdout)
        else:
            identity = cls._record_egress_client_direct(
                client, container, work_dir, kind=kind)
        if identity is not None:
            cls._egress_client_ips[work_dir] = tuple(identity)

    @classmethod
    def _record_egress_client_direct(cls, client, container, work_dir: str,
                                     *, kind: str) -> tuple[str, str, str] | None:
        """Publish under the setup fence, or prove a base-only no-map proxy."""
        from assist.egress.client_map import ClientRecord, configured_directory, record_client
        try:
            directory = configured_directory()
        except (OSError, RuntimeError):
            directory = None
        # The same interprocess lock protects proxy replacement. Without it,
        # A could be inspected, then replaced by B before attribution writes
        # to A's map, exposing a recycled IP under B's stale grant.
        with _proxy_setup_lock():
            proxy = client.containers.get(EGRESS_PROXY_NAME)
            proxy.reload()
            mounts = proxy.attrs.get("Mounts")
            if proxy.status != "running" or not isinstance(mounts, list):
                raise RuntimeError("proxy client-map mode is unconfirmed")
            map_mounts = [mount.get("Source") for mount in mounts
                          if mount.get("Destination") == "/client-map"]
            if directory is None:
                # A missing optional map is safe only with a base-only proxy.
                if map_mounts:
                    raise RuntimeError("proxy client-map mode is unconfirmed")
                return
            if map_mounts != [directory]:
                raise RuntimeError("proxy client-map mode is unconfirmed")
            container.reload()
            if (container.status != "running"
                    or (container.attrs["NetworkSettings"]["Networks"]
                        [EGRESS_NETWORK]["NetworkID"])
                    != (proxy.attrs["NetworkSettings"]["Networks"]
                        [EGRESS_NETWORK]["NetworkID"])):
                raise RuntimeError("sandbox proxy network identity changed")
            runtime_state.assert_admissible("proxy", proxy.id)
            runtime_state.assert_admissible(
                "network", proxy.attrs["NetworkSettings"]["Networks"]
                [EGRESS_NETWORK]["NetworkID"])
            ip = (container.attrs["NetworkSettings"]["Networks"]
                  [EGRESS_NETWORK]["IPAddress"])
            tid = os.path.basename(os.path.dirname(work_dir))
            if not ip or not tid:
                raise RuntimeError("sandbox has no proxy client identity")
            record_client(directory, ip, ClientRecord(tid, container.id, kind))
            return directory, ip, container.id

    @classmethod
    def _forget_egress_client(cls, work_dir: str, generation: str | None = None) -> None:
        identity = cls._egress_client_ips.get(work_dir)
        if identity and (generation is None or identity[2] == generation):
            from assist.egress.client_map import forget_client
            forget_client(*identity)
            if cls._egress_client_ips.get(work_dir) == identity:
                cls._egress_client_ips.pop(work_dir, None)

    @classmethod
    def current_container(cls, work_dir: str):
        """Return the registered container generation, if any."""
        return cls._containers.get(work_dir)

    @classmethod
    def cleanup_verified(cls, work_dir: str, expected_container) -> None:
        """Confirm this generation exited before allowing host Git object reads.

        Failed teardown retains the registry entry. The Git owner also persists
        a quarantine so a process restart cannot silently permit another writer.
        """
        from docker.errors import NotFound
        if expected_container is None or cls._containers.get(work_dir) is not expected_container:
            raise RuntimeError("Git sandbox generation is unavailable")
        try:
            expected_container.kill()
            expected_container.wait(timeout=10)
        except NotFound:
            pass  # Auto-removal is also proof that this exact generation exited.
        if cls._containers.get(work_dir) is not expected_container:
            raise RuntimeError("Git sandbox generation changed during teardown")
        scoped_owner = cls._generation_owners.get(expected_container.id)
        if scoped_owner is not None:
            from assist.browser import authority
            threads_root, thread_id, owner = scoped_owner
            if os.path.isdir(os.path.join(threads_root, thread_id)):
                with authority.fence(threads_root, thread_id) as state:
                    if (state.lease is not None
                            and state.lease["owner_run_id"] == owner):
                        state.remove_generation(owner, expected_container.id)
                        state.clear(owner)
        cls._containers.pop(work_dir)
        cls._forget_egress_client(work_dir)
        cls._generation_owners.pop(expected_container.id, None)


    @classmethod
    def cleanup(cls, work_dir: str, expected_container=_ANY_CONTAINER) -> None:
        """Prove the captured generation stopped before releasing its identity.

        A stale caller may still stop its own captured container, but cannot
        remove a newer registry entry or proxy attribution for the workspace.
        """
        container = (cls._containers.get(work_dir)
                     if expected_container is _ANY_CONTAINER else expected_container)
        if container is None:
            return
        confirm_generation_stopped(container.id)
        scoped_owner = cls._generation_owners.get(container.id)
        if scoped_owner is not None:
            from assist.browser import authority
            threads_root, thread_id, owner = scoped_owner
            if os.path.isdir(os.path.join(threads_root, thread_id)):
                with authority.fence(threads_root, thread_id) as state:
                    if (state.lease is not None
                            and state.lease["owner_run_id"] == owner):
                        state.remove_generation(owner, container.id)
                        state.clear(owner)
        if cls._containers.get(work_dir) is container:
            cls._forget_egress_client(work_dir, container.id)
            cls._containers.pop(work_dir, None)
        cls._generation_owners.pop(container.id, None)
        logger.info("Cleaned up container %s for %s", container.id[:12], work_dir)

    @classmethod
    def cleanup_all(cls) -> None:
        """Confirm tracked generations stopped; retain failed obligations."""
        for path, container in list(cls._containers.items()):
            try:
                cls.cleanup(path, container)
            except Exception as e:
                logger.error("Container cleanup unconfirmed for %s: %s", path, e)
