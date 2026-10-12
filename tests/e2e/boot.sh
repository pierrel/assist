#!/bin/sh
set -eu
export DOCKER_HOST=unix:///run/docker.sock
mkdir -p /run/fixture-docker /var/lib/docker
unset DOCKER_TLS_VERIFY DOCKER_CERT_PATH DOCKER_CONTEXT
dockerd-entrypoint.sh dockerd --storage-driver=overlay2 --host="$DOCKER_HOST" \
  --data-root=/var/lib/docker --exec-root=/run/fixture-docker \
  --pidfile=/run/fixture-docker/docker.pid > /runtime/docker.log 2>&1 &
daemon_pid=$!
server_pid=
cleanup() {
  if [ -n "$server_pid" ]; then kill -TERM "$server_pid" 2>/dev/null || true; fi
  kill -TERM "$daemon_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
startup_end=$(( $(date +%s) + 1200 ))
until docker info > /dev/null 2>&1 && [ -f /runtime/images-ready ]; do
  if [ "$(date +%s)" -ge "$startup_end" ]; then exit 1; fi
  sleep 1
done
chown 1000:1000 /run/docker.sock
chown -R 1000:1000 /secrets /home/test/.ssh
HOME=/home/test setpriv --reuid=1000 --regid=1000 --init-groups \
  python tests/e2e/server.py &
server_pid=$!
printf '%s\n' "$server_pid" > /runtime/server.pid
wait "$server_pid"
