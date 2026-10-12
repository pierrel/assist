# Reusable real Assist fixture

Run from an Assist checkout with Docker, Git, SSH-keygen and OpenSSL installed:

```sh
python tests/e2e/start.py --logs /tmp/assist-fixture-new --seconds 600
```

The foreground launcher prints a private bridge HTTPS URL and writes a private
`fixture.json`. In another terminal, without Emacs:

```sh
python tests/e2e/smoke.py --fixture /tmp/assist-fixture-new/fixture.json
```

The fixture runs the real web server, ThreadManager/Thread, Run stores,
checkpointing, Git lifecycle, browser authority/reset workers, SandboxManager and
DockerSandboxBackend. Only model readiness, agent graph inference and title
inference are deterministic. `Write`, `Read` and bounded `Hold` messages provide
small, repeatable actions through the real backend. Authenticated test routes
inspect actual state, release inference holds and dispatch a hidden child through
the actual protocol service. They never dispatch or replace reset workers.

A separate Docker daemon inside the server uses fixed production resource names
only within its own disposable Docker-managed volume. Both the SDK and Docker
CLI target its socket. The outer network remains internal with a host-addressed
bridge. The host connects directly to its fixed private server IP over HTTPS; no public host port
is exposed. Server and Git addresses come from the fresh Docker-allocated subnet
and remain fixed across restart; the server address is included in its TLS SAN.
Runtime sandbox/proxy images must already exist under the usual tags.
Transfer is limited to those two exact image IDs, at most 3GiB and 180 seconds
per export/import; the launcher requires 8GiB free on temp and Docker filesystems.
Build is bounded to 600 seconds, startup to 1200 seconds, server lifetime to the
selected 1..1800 seconds. No model/GPU or production state is used.

**The server container is privileged.** It is not a host-security boundary.
Use only reviewed code and trusted synthetic fixture assets. This provides
protocol/lifecycle coverage, not proof of sandbox security. Never attach a host
Docker socket, checkout, credentials or production directory. The fresh volume
contains only this fixture daemon's state.

The descriptor includes host and container URLs, the exact internal network and
paths to temporary synthetic credentials. CLI/browser clients on the Docker host
use the private host URL; container drivers join the recorded network and use `container_url`.
Credentials disappear when the launcher exits. SIGINT/SIGTERM and failures
trigger bounded removal of exact labelled containers (including anonymous
volumes), the daemon volume and network. No prune or production restart occurs.
Source hashes, native image import receipts, build/container logs and smoke
receipts survive in the log directory; source-only build images remain cached. `--assist-source` selects
an existing runtime checkout for a readonly snapshot, allowing a separately
reviewed Root composition; the fixture files come from this launcher checkout.
