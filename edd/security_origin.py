"""Disposable, address-bound synthetic origin; only the fixture runs this file."""
import argparse
import base64
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import sys
import threading


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.host)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= 65536 or self.headers.get("Transfer-Encoding"):
                raise ValueError("invalid body")
            self.connection.settimeout(5)
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("short body")
            request = {"method": self.command, "path": self.path, "peer_ip": self.client_address[0],
                       "headers": dict(self.headers), "body": base64.b64encode(body).decode()}
            connection = UnixConnection(self.server.control_socket, timeout=15)
            connection.request("POST", "/" + self.server.role,
                               json.dumps(request).encode(), {"Content-Type": "application/json"})
            response = connection.getresponse()
            result = response.read(256 * 1024 + 1)
            if len(result) > 256 * 1024:
                raise ValueError("oversized fixture response")
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "text/plain"))
            self.send_header("Content-Length", str(len(result)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(result)
            connection.close()
        except (OSError, ValueError, http.client.HTTPException):
            self.send_error(502, "Fixture transport failed")

    def log_message(self, *_args):
        pass


class ObservedServer(ThreadingHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        if self.role == 'control':
            observer = UnixConnection(self.control_socket, timeout=5)
            try:
                observer.request('POST', '/control', json.dumps({
                    'method': 'TCP', 'path': '<accepted connection>', 'peer_ip': address[0]}).encode(),
                    {'Content-Type': 'application/json'})
                response = observer.getresponse()
                if response.status != 403:
                    raise RuntimeError('Control accept observation failed')
                response.read(1024)
            except Exception as error:
                connection.close()
                raise RuntimeError("Control accept observation failed") from error
            finally:
                observer.close()
        return connection, address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-ip", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--probe-path")
    args = parser.parse_args()
    if args.probe_path:
        connection = http.client.HTTPConnection(args.private_ip, 9090, timeout=5)
        try:
            connection.request("GET", args.probe_path)
            response = connection.getresponse()
            body = response.read(256 * 1024 + 1)
            if len(body) > 256 * 1024:
                raise RuntimeError("Private UI response exceeded bound")
            sys.stdout.buffer.write(body + b"\n" + str(response.status).encode())
        finally:
            connection.close()
        return
    server = ObservedServer((args.private_ip, 9090), Handler)
    server.role, server.control_socket = "control", args.socket
    print("SECURITY_ORIGIN_READY", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
