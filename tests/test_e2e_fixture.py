"""Fixture error/receipt, source privacy and cleanup regressions; no Docker required."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from tests.e2e.start import Fixture, LABEL


class FixtureTests(unittest.TestCase):
    def test_smoke_rejects_incorrect_idempotent_replay_flags(self):
        from io import BytesIO
        from unittest.mock import Mock
        from tests.e2e import smoke
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'token').write_text('synthetic-token')
            fixture = root / 'fixture.json'
            fixture.write_text(json.dumps({'url': 'https://fixture', 'credentials': {
                'token': str(root / 'token'), 'ca.crt': str(root / 'ca.crt')}}))
            for flags in [(True, True), (False, False)]:
                with self.subTest(flags=flags):
                    creates = iter(flags)
                    def respond(request, **kwargs):
                        if request.get_method() == 'POST':
                            result = {'thread_id': 'thread', 'run_id': 'run', 'replayed': next(creates)}
                        elif request.full_url.endswith('/threads'):
                            result = {'repositories': [{'repo_key': 'repo'}]}
                        elif request.full_url.endswith('/events'):
                            return BytesIO(b'event: terminal\ndata: {"status":"success"}\n\n')
                        elif request.full_url.endswith('/runs/run'):
                            result = {'status': 'success'}
                        else:
                            result = {'messages': [{'text': 'Saved http-smoke.txt'}]}
                        return BytesIO(json.dumps(result).encode())
                    opener = Mock()
                    opener.open.side_effect = respond
                    with (patch('sys.argv', ['smoke', '--fixture', str(fixture)]),
                          patch('tests.e2e.smoke.ssl.create_default_context', return_value=None),
                          patch('tests.e2e.smoke.urllib.request.build_opener', return_value=opener)):
                        with self.assertRaisesRegex(RuntimeError, 'incorrect replay flags'):
                            smoke.main()

    def test_failed_backend_read_is_not_a_successful_model_result(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from langchain_core.messages import HumanMessage
        from tests.e2e import server
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=1, output='cat: missing-file')
        graph = server.deterministic_agent(None, '/fixture', sandbox_backend=backend)
        with self.assertRaisesRegex(RuntimeError, 'missing-file'):
            graph.invoke({'messages': [HumanMessage(content='Read missing-file')]})

    def test_snapshot_excludes_environment_and_untracked_credentials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "source.py").write_text("public source")
            (repo / ".dev.env").write_text("synthetic private environment")
            (repo / "untracked-key").write_text("synthetic private credential")
            subprocess.run(["git", "-C", str(repo), "add", "source.py", ".dev.env"], check=True)
            harness = Fixture(argparse.Namespace(logs=root / "logs"))
            harness.stage(repo, root / "snapshot")
            self.assertEqual([p.name for p in (root / "snapshot").iterdir()], ["source.py"])
            (repo / "escape").symlink_to(repo / "untracked-key")
            subprocess.run(["git", "-C", str(repo), "add", "escape"], check=True)
            with self.assertRaisesRegex(RuntimeError, "refuses symlinks"):
                harness.stage(repo, root / "second")

    def test_log_timeout_still_removes_all_owned_runtime_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            harness = Fixture(argparse.Namespace(logs=Path(temporary) / "logs"))
            harness.network_attempted = harness.volume_attempted = True
            harness.containers = [harness.name + "-client", harness.name + "-assist"]
            calls = []

            def command(args, **kwargs):
                calls.append(args)
                if args[2] == "inspect":
                    labels = {"Labels": {LABEL: harness.name}}
                    record = {"Config": labels} if args[1] == "container" else labels
                    return subprocess.CompletedProcess(args, 0, json.dumps([record]), "")
                if args[1] == "logs":
                    raise subprocess.TimeoutExpired(args, 5)
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch("tests.e2e.start.subprocess.run", command):
                with self.assertRaisesRegex(RuntimeError, "cleanup had failures"):
                    harness.cleanup()
            self.assertEqual([c for c in calls if c[1] == "rm"], [
                ["docker", "rm", "-f", "-v", harness.containers[1]],
                ["docker", "rm", "-f", "-v", harness.containers[0]]])
            self.assertIn(["docker", "volume", "rm", harness.volume], calls)
            self.assertIn(["docker", "network", "rm", harness.network], calls)

    def test_interrupted_create_is_reconciled_without_removing_foreign_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            harness = Fixture(argparse.Namespace(logs=Path(temporary) / "logs"))
            with patch.object(harness, "docker", side_effect=subprocess.TimeoutExpired("docker create", 45)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    harness.create("assist", "sleep", "300")
            owned = harness.name + "-assist"
            foreign = harness.name + "-client"
            harness.containers.append(foreign)
            calls = []

            def command(args, **kwargs):
                calls.append(args)
                if args[2] == "inspect":
                    owner = harness.name if args[3] == owned else "different-harness"
                    return subprocess.CompletedProcess(args, 0, json.dumps([
                        {"Config": {"Labels": {LABEL: owner}}}]), "")
                return subprocess.CompletedProcess(args, 0, "", "")

            with patch("tests.e2e.start.subprocess.run", command):
                harness.cleanup()
            self.assertIn(["docker", "rm", "-f", "-v", owned], calls)
            self.assertNotIn(["docker", "rm", "-f", "-v", foreign], calls)

    def test_snapshot_refuses_symlink_parent_of_tracked_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            (repo / "nested").mkdir(parents=True)
            subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
            (repo / "nested/file").write_text("tracked")
            subprocess.run(["git", "-C", str(repo), "add", "nested/file"], check=True)
            (repo / "nested/file").unlink()
            (repo / "nested").rmdir()
            (root / "private").mkdir()
            (root / "private/file").write_text("private")
            (repo / "nested").symlink_to(root / "private")
            harness = Fixture(argparse.Namespace(logs=root / "logs"))
            with self.assertRaisesRegex(RuntimeError, "refuses symlinks"):
                harness.stage(repo, root / "snapshot")

    def test_import_keeps_test_routes_out_of_shared_app(self):
        import sys
        result = subprocess.run([sys.executable, "-c", """
from manage.web import app
before = [route.path for route in app.routes]
from tests.e2e import server
assert [route.path for route in app.routes] == before
server.install()
assert sorted(route.path for route in app.routes if route.path.startswith('/__e2e/')) == [
        '/__e2e/approval/{tid}', '/__e2e/child/{tid}', '/__e2e/release', '/__e2e/state/{tid}']
"""], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_state_requires_positive_native_running_generation(self):
        from unittest.mock import Mock
        from docker.errors import NotFound
        from tests.e2e import server
        container = Mock(id='generation')
        with (patch.object(server.state.MANAGER, 'root_dir', '/fixture/threads'),
              patch.object(server, '_authenticate'),
              patch.object(server.authority, 'fence') as fence,
              patch.object(server.threads, '_runs') as runs,
              patch.object(server.SandboxManager, 'current_container', return_value=container) as current):
            fence.return_value.__enter__.return_value.lease = {
                'owner_run_id': 'child', 'generations': ['generation']}
            runs.return_value.list.return_value = []
            for running, error, expected in ((True, None, 'generation'),
                                             (False, None, None),
                                             (True, NotFound('removed'), None)):
                container.attrs = {'State': {'Running': running}}
                container.reload.side_effect = error
                result = server.inspect_state('parent', None)
                self.assertEqual(result['live_registered_generation'], expected)
            current.assert_called_with('/fixture/threads/parent/domain')
            self.assertEqual(container.reload.call_count, 3)

    def test_released_inference_hold_can_be_used_again(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        from langchain_core.messages import HumanMessage
        from tests.e2e import server
        graph = server.deterministic_agent(None, '/fixture')
        with ThreadPoolExecutor(max_workers=1) as worker:
            for _ in range(2):
                result = worker.submit(graph.invoke, {'messages': [HumanMessage(content='Hold repeat')]})
                deadline = time.monotonic() + 2
                while 'Hold repeat' not in server.HOLDS and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertIn('Hold repeat', server.HOLDS)
                self.assertFalse(result.done())
                server.HOLDS['Hold repeat'].set()
                self.assertEqual(result.result(timeout=2)['messages'][-1].content,
                                 'Completed: Hold repeat')
                self.assertNotIn('Hold repeat', server.HOLDS)
