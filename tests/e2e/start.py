#!/usr/bin/env python3
"""Start a reusable real Assist server with synthetic Git and deterministic inference."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import resource
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
import ssl
import uuid

SOURCE = 'ssh://test@git:2222/repos/domain.git'
LABEL = 'assist-e2e.owner'
FILES = ('Dockerfile', 'server.py', 'start.py', 'smoke.py', 'boot.sh', 'git.py',
         'requirements.txt', '__init__.py', 'README.md')
IMAGES = ('assist-sandbox', 'assist-egress-proxy')
ARCHIVE_LIMIT = 3 * 1024**3


class Fixture:
    def __init__(self, args):
        self.args = args
        self.name = 'assist-e2e-' + uuid.uuid4().hex[:12]
        self.image = self.name + ':server'
        self.network = self.name
        self.volume = self.name + '-docker'
        self.containers = []
        self.network_attempted = self.volume_attempted = False
        self.token = secrets.token_urlsafe(32)
        self.deadline = time.monotonic() + 1200
        args.logs.mkdir(parents=True, exist_ok=False)
        args.logs.chmod(0o700)

    def command(self, *args, timeout=45, **kwargs):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Fixture operation deadline exceeded')
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=min(timeout, remaining), **kwargs)
        if result.returncode:
            raise RuntimeError(f'{args[0]} failed ({result.returncode}): '
                               + self.scrub(result.stdout + result.stderr)[-6000:])
        return result.stdout

    def docker(self, *args, **kwargs):
        return self.command('docker', *args, **kwargs)

    def scrub(self, text):
        return text.replace(self.token, '[test-token]')

    def stage(self, repo, target):
        for relative in self.command('git', '-C', str(repo), 'ls-files', '-z').split('\0'):
            if not relative:
                continue
            path = Path(relative)
            if any(part == '.git' or part.endswith('.env') or part.startswith('.env')
                   for part in path.parts):
                continue
            source = repo / path
            if any(parent.is_symlink() for parent in (source, *source.parents)
                   if parent == repo or repo in parent.parents):
                raise RuntimeError('Source staging refuses symlinks: ' + relative)
            if source.is_file():
                destination = target / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    def create(self, service, *command, extra=()):
        name = self.name + '-' + service
        self.containers.append(name)  # Reconcile interrupted create, including timeouts.
        self.docker('create', '--name', name, '--network', self.network,
                    '--network-alias', service, '--user', '0:0',
                    '--label', LABEL + '=' + self.name, '--memory', '3g',
                    '--pids-limit', '512', '--cpus', '2', *extra, self.image, *command)

    def copy(self, source, service, destination):
        self.docker('cp', '-a', str(source), self.name + '-' + service + ':' + destination,
                    timeout=180)

    def wait(self, description, predicate, seconds=180):
        deadline = min(self.deadline, time.monotonic() + seconds)
        last = None
        while time.monotonic() < deadline:
            try:
                value = predicate()
                if value:
                    return value
            except (RuntimeError, OSError) as error:
                last = error
            time.sleep(.2)
        raise TimeoutError(description + ': ' + str(last or 'condition not reached'))

    def bootstrap(self, temporary):
        context = temporary / 'source'
        self.stage(self.args.assist_source, context)
        for file in FILES:
            source = Path(__file__).parent / file
            if source.is_symlink():
                raise RuntimeError('Fixture source refuses symlinks: ' + file)
            destination = context / 'tests/e2e' / file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        identity = {'assist_head': self.command('git', '-C', str(self.args.assist_source),
                    'rev-parse', 'HEAD').strip(), 'files': {
                    str(p.relative_to(context)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(context.rglob('*')) if p.is_file()}}
        (self.args.logs / 'source.json').write_text(json.dumps(identity, indent=2))
        print('Building standalone fixture; logs: ' + str(self.args.logs), flush=True)
        with (self.args.logs / 'build.log').open('w') as log:
            build = subprocess.run(['docker', 'build', '-t', self.image, '-f',
                str(context / 'tests/e2e/Dockerfile'), str(context)],
                stdout=log, stderr=log, timeout=600)
        if build.returncode:
            raise RuntimeError('Fixture image build failed; see build.log')
        images = json.loads(self.docker('image', 'inspect', *IMAGES))
        if sum(image['Size'] for image in images) > ARCHIVE_LIMIT:
            raise RuntimeError('Runtime images exceed 3GiB transfer bound')
        docker_root = Path(json.loads(self.docker('info', '--format', '{{json .DockerRootDir}}')))
        for location in (temporary, docker_root):
            if shutil.disk_usage(location).free < 8 * 1024**3:
                raise RuntimeError('Fixture requires 8GiB free at ' + str(location))
        (self.args.logs / 'runtime-images.json').write_text(json.dumps([
            {'id': image['Id'], 'tags': image['RepoTags'], 'size': image['Size']}
            for image in images], indent=2))
        archives = []
        for name, image in zip(IMAGES, images):
            remaining_bytes = ARCHIVE_LIMIT - sum(p.stat().st_size for p in archives)
            def limit_archive():
                resource.setrlimit(resource.RLIMIT_FSIZE, (remaining_bytes, remaining_bytes))
            archive = temporary / (name + '.tar')
            self.docker('save', '-o', str(archive), image['Id'],
                        timeout=180, preexec_fn=limit_archive)
            archives.append(archive)
        self.network_attempted = True
        self.docker('network', 'create', '--internal', '--opt',
                    'com.docker.network.bridge.gateway_mode_ipv4=nat',
                    '--label', LABEL + '=' + self.name, self.network)
        config = json.loads(self.docker('network', 'inspect', self.network))[0]['IPAM']['Config'][0]
        addresses = (str(address) for address in ipaddress.ip_network(config['Subnet']).hosts()
                     if str(address) != config['Gateway'])
        server_ip, git_ip = next(addresses), next(addresses)
        self.volume_attempted = True
        self.docker('volume', 'create', '--label', LABEL + '=' + self.name, self.volume)
        private = temporary / 'secrets'
        private.mkdir(mode=0o700)
        for key in ('client_key', 'server_ssh', 'ssh_host'):
            self.command('ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(private / key))
        (private / 'authorized_keys').write_text(
            (private / 'client_key.pub').read_text() + (private / 'server_ssh.pub').read_text())
        host = (private / 'ssh_host.pub').read_text().split()
        (private / 'known_hosts').write_text('[git]:2222 ' + ' '.join(host[:2]) + '\n')
        self.command('openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(private / 'ca.key'), '-out', str(private / 'ca.crt'), '-days', '1',
            '-subj', '/CN=Disposable E2E CA', '-addext', 'basicConstraints=critical,CA:TRUE',
            '-addext', 'keyUsage=critical,keyCertSign,cRLSign')
        self.command('openssl', 'req', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(private / 'server.key'), '-out', str(private / 'server.csr'), '-subj', '/CN=assist')
        (private / 'extensions').write_text('subjectAltName=DNS:assist,DNS:localhost,IP:127.0.0.1,IP:' + server_ip + '\n'
            'extendedKeyUsage=serverAuth\nkeyUsage=critical,digitalSignature,keyEncipherment\n'
            'basicConstraints=critical,CA:FALSE\nauthorityKeyIdentifier=keyid,issuer\n')
        self.command('openssl', 'x509', '-req', '-in', str(private / 'server.csr'),
            '-CA', str(private / 'ca.crt'), '-CAkey', str(private / 'ca.key'), '-CAcreateserial',
            '-out', str(private / 'server.crt'), '-days', '1', '-extfile', str(private / 'extensions'))
        (private / 'token').write_text(self.token)
        (private / 'ssh_config').write_text('Host git\n HostName git\n Port 2222\n User test\n'
            ' IdentityFile /secrets/server_ssh\n UserKnownHostsFile /secrets/known_hosts\n'
            ' GlobalKnownHostsFile /dev/null\n StrictHostKeyChecking yes\n IdentitiesOnly yes\n BatchMode yes\n')
        key = hashlib.sha256(SOURCE.encode()).hexdigest()[:20]
        (private / 'remotes.json').write_text(json.dumps({key: SOURCE}))
        for file in private.iterdir():
            file.chmod(0o600)
        self.create('git', 'python', 'tests/e2e/git.py', extra=('--ip', git_ip))
        self.create('assist', 'sh', 'tests/e2e/boot.sh', extra=(
            '--privileged', '--mount', 'type=volume,source=' + self.volume + ',target=/var/lib/docker',
            '--ip', server_ip, '-e', 'ASSIST_THREADS_DIR=/state/threads',
            '-e', 'ASSIST_DOMAINS=' + SOURCE, '-e', 'ASSIST_MODEL_URL=http://model.invalid:1/v1'))
        for file, service, destination in (
            ('authorized_keys', 'git', '/home/test/.ssh/authorized_keys'),
            ('ssh_host', 'git', '/secrets/ssh_host'),
            ('server_ssh', 'assist', '/secrets/server_ssh'),
            ('known_hosts', 'assist', '/secrets/known_hosts'),
            ('ssh_config', 'assist', '/home/test/.ssh/config'),
            ('server.crt', 'assist', '/secrets/server.crt'),
            ('server.key', 'assist', '/secrets/server.key'),
            ('token', 'assist', '/secrets/token')):
            self.copy(private / file, service, destination)
        for service in ('git', 'assist'):
            self.docker('start', self.name + '-' + service)
        self.wait('private Docker daemon', lambda: self.docker('exec', self.name + '-assist', 'docker', 'info'))
        for name, image, archive in zip(IMAGES, images, archives):
            self.copy(archive, 'assist', '/runtime/images.tar')
            loaded = self.docker('exec', self.name + '-assist', 'docker', 'load',
                                 '-i', '/runtime/images.tar', timeout=180)
            (self.args.logs / (name + '-load.log')).write_text(loaded)
            identities = [line.removeprefix('Loaded image ID: ').strip()
                          for line in loaded.splitlines() if line.startswith('Loaded image ID: ')]
            if len(identities) != 1:
                raise RuntimeError('Single-image import did not return exactly one image identity')
            imported = json.loads(self.docker('exec', self.name + '-assist',
                                  'docker', 'image', 'inspect', identities[0]))[0]
            compared = ('RootFS', 'Config', 'Architecture', 'Os')
            receipt = {'exported_id': image['Id'], 'loaded_id': identities[0],
                'imported_id': imported['Id'], 'rootfs': imported['RootFS'],
                'config_sha256': hashlib.sha256(json.dumps(imported['Config'], sort_keys=True).encode()).hexdigest(),
                'matching': {field: imported[field] == image[field] for field in compared}}
            (self.args.logs / (name + '-import.json')).write_text(json.dumps(receipt, indent=2))
            if not all(receipt['matching'].values()):
                raise RuntimeError('Imported runtime image differs from the exact exported image')
            self.docker('exec', self.name + '-assist', 'docker', 'tag', identities[0], name)
            self.docker('exec', self.name + '-assist', 'rm', '/runtime/images.tar')
            archive.unlink()
        self.docker('exec', self.name + '-assist', 'touch', '/runtime/images-ready')
        url = 'https://' + server_ip + ':8443'
        context_tls = ssl.create_default_context(cafile=str(private / 'ca.crt'))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                             urllib.request.HTTPSHandler(context=context_tls))
        def ready():
            request = urllib.request.Request(url + '/api/v1/phone/threads',
                headers={'Authorization': 'Bearer ' + self.token})
            with opener.open(request, timeout=5) as response:
                return json.load(response)
        self.wait('real Assist HTTP readiness', ready)
        descriptor = {'owner': self.name, 'label': LABEL, 'url': url,
            'container_url': 'https://assist:8443', 'network': self.network,
            'server_container': self.name + '-assist', 'repo_source': SOURCE,
            'credentials': {name: str(private / name) for name in
                ('token', 'ca.crt', 'client_key', 'known_hosts', 'remotes.json')}}
        path = self.args.logs / 'fixture.json'
        path.write_text(json.dumps(descriptor, indent=2))
        path.chmod(0o600)
        print('Ready: ' + url + ' descriptor=' + str(path), flush=True)
        return path

    def cleanup(self):
        failures = []
        resources = [('container', name) for name in reversed(self.containers)]
        if self.volume_attempted:
            resources.append(('volume', self.volume))
        if self.network_attempted:
            resources.append(('network', self.network))
        for kind, name in resources:
            try:
                inspected = subprocess.run(['docker', kind, 'inspect', name],
                                           capture_output=True, text=True, timeout=5)
                if inspected.returncode:
                    if 'No such' in inspected.stderr:
                        continue
                    raise RuntimeError(inspected.stderr)
                record = json.loads(inspected.stdout)[0]
                labels = (record['Config'] if kind == 'container' else record).get('Labels', {})
                if (labels or {}).get(LABEL) != self.name:
                    continue
            except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError, KeyError) as error:
                failures.append('ownership inspection: ' + str(error))
                continue
            if kind == 'container':
                try:
                    logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True, timeout=5)
                    (self.args.logs / (name.rsplit('-', 1)[-1] + '.log')).write_text(
                        self.scrub(logs.stdout + logs.stderr))
                except (OSError, subprocess.TimeoutExpired) as error:
                    failures.append('log collection: ' + str(error))
            command = ['docker', 'rm', '-f', '-v', name] if kind == 'container' else ['docker', kind, 'rm', name]
            try:
                removal = subprocess.run(command, capture_output=True, text=True, timeout=20)
                if removal.returncode:
                    failures.append('resource removal: ' + removal.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                failures.append('resource removal: ' + str(error))
        if failures:
            (self.args.logs / 'cleanup-errors.log').write_text(self.scrub('\n'.join(failures)))
            raise RuntimeError('Fixture cleanup had failures; see cleanup-errors.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assist-source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=600, choices=range(1, 1801), metavar='1..1800')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Fixture interrupted; cleaning up owned resources')
    signal.signal(signal.SIGTERM, interrupted)
    fixture = Fixture(args)
    try:
        with tempfile.TemporaryDirectory(prefix=fixture.name + '-') as name:
            fixture.bootstrap(Path(name))
            time.sleep(args.seconds)
    finally:
        fixture.cleanup()


if __name__ == '__main__':
    main()
