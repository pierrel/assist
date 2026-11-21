"""Check ordinary/native model attribution on a disposable internal Docker network.

No host ports, real model, production proxy, or external destinations are used.
Run with --proxy-image naming the exact candidate egress image. The fake model
is another container, mapped to host.docker.internal only inside this proxy.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import tempfile
import uuid

import docker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-image", required=True)
    parser.add_argument("--sandbox-image", default="assist-sandbox")
    args = parser.parse_args()
    client = docker.from_env(timeout=15)
    containers = []
    network = None
    results = {}
    signal.signal(signal.SIGTERM, lambda *_args: sys.exit(124))
    with tempfile.TemporaryDirectory(prefix="native-egress-check-") as temporary:
        directory = Path(temporary)
        directory.chmod(0o755)
        try:
            network = client.networks.create("native-egress-check-" + uuid.uuid4().hex[:12], internal=True)
            network.reload()
            cidr = network.attrs["IPAM"]["Config"][0]["Subnet"]

            def start(image, **kwargs):
                container = client.containers.run(image, detach=True, network=network.name, **kwargs)
                containers.append(container)
                container.reload()
                return container

            provider = start(args.proxy_image, entrypoint="python3", command=["-u", "-c", """
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'fixture')
HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
"""])
            provider_ip = provider.attrs["NetworkSettings"]["Networks"][network.name]["IPAddress"]
            box = start(args.sandbox_image, command=["sleep", "120"])
            box_ip = box.attrs["NetworkSettings"]["Networks"][network.name]["IPAddress"]

            def proxy(mapped):
                return start(args.proxy_image,
                             environment={"ASSIST_PORT": "5050",
                                          "EGRESS_ALLOWLIST": "host.docker.internal,fixture-external.test",
                                          "EGRESS_SANDBOX_CIDR": cidr},
                             extra_hosts={"host.docker.internal": provider_ip,
                                          "fixture-external.test": provider_ip},
                             volumes={str(directory): {"bind": "/client-map", "mode": "ro"}} if mapped else {})

            mapped = proxy(True)

            def status(proxy_container, host):
                result = box.exec_run([
                    "curl", "--noproxy", "", "--proxy", f"http://{proxy_container.name}:8888",
                    "--retry", "3", "--retry-connrefused", "--retry-delay", "1", "--max-time", "3",
                    "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://{host}:8000/v1/models"], demux=True)
                stdout, stderr = result.output
                assert result.exit_code == 0, (stderr or b"").decode(errors="replace")
                return int(stdout)

            path = directory / "client-map.json"
            for kind, expected in (("sandbox", 200), ("hermes", 403), ("pi", 403)):
                path.write_text(json.dumps({box_ip: {"thread_id": "fixture", "generation": box.id, "kind": kind}}))
                path.chmod(0o644)
                results[kind] = status(mapped, "host.docker.internal")
                assert results[kind] == expected, results
            path.unlink()
            results["missing"] = status(mapped, "host.docker.internal")
            assert results["missing"] == 403
            path.write_text("not-json")
            path.chmod(0o644)
            results["malformed"] = status(mapped, "host.docker.internal")
            results["external_allowlist"] = status(mapped, "fixture-external.test")
            assert results["malformed"] == 403 and results["external_allowlist"] == 200
            legacy = proxy(False)
            results["legacy_no_map"] = status(legacy, "host.docker.internal")
            results["legacy_external_allowlist"] = status(legacy, "fixture-external.test")
            assert results["legacy_no_map"] == 403 and results["legacy_external_allowlist"] == 200
            print(json.dumps({"status": "PASS", "proxy_image": mapped.image.id,
                              "results": results, "limits": "Private fake upstream; no live/model access."}, indent=2))
        finally:
            for container in reversed(containers):
                container.remove(force=True)
            if network is not None:
                network.remove()
            client.close()


if __name__ == "__main__":
    main()
