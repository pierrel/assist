FROM docker:29.7.2-dind AS docker
FROM python:3.13-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    iptables iproute2 libseccomp2 git openssh-client openssh-server openssl util-linux ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 test \
    && mkdir /runtime /state /state/threads /repos /secrets /home/test/.ssh \
    && chown -R test:test /runtime /state /repos /secrets /home/test/.ssh
COPY --from=docker /usr/local/bin/ /usr/local/bin/
COPY pyproject.toml requirements.txt /deps/
COPY tests/e2e/requirements.txt /deps/e2e-constraints.txt
RUN python -c 'import tomllib; print("\n".join(tomllib.load(open("/deps/pyproject.toml", "rb"))["project"]["dependencies"]))' > /deps/runtime.txt \
    && pip install --no-cache-dir -r /deps/runtime.txt -c /deps/requirements.txt -c /deps/e2e-constraints.txt
COPY . /src/assist/
RUN mkdir -p /src/assist/logs && chown test:test /src/assist/logs \
    && pip install --no-deps -e /src/assist
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/src/assist DOCKER_HOST=unix:///run/docker.sock
WORKDIR /src/assist
USER test
CMD ["sleep", "600"]
