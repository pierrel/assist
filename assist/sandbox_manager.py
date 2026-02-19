import logging

logger = logging.getLogger(__name__)

SANDBOX_IMAGE = "assist-sandbox"


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

        try:
            client = cls._get_docker_client()
            container = client.containers.run(
                SANDBOX_IMAGE,
                detach=True,
                remove=True,
                volumes={work_dir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                stdin_open=True,
                tty=False,
                labels={"assist.sandbox": "true"},
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
