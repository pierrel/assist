"""Check native bundled-skill startup inside the pinned runtime, without a model.

Run with /opt/hermes/.venv/bin/python in a network:none container. Each turn
uses the actual worker and AIAgent constructor in a fresh process, with a
scripted conversation and stubbed host coordination. Constructor metadata is
served locally; inference and provider forwarding are denied. No model,
integration activation or behavioral acceptance is exercised.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from unittest.mock import patch


def write_skill(home, relative, name, body, extra=""):
    path = home / "skills" / relative / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: Synthetic startup fixture\n{extra}---\n{body}\n")
    return path


def turn(home, scenario):
    os.environ["HERMES_HOME"] = str(home)
    os.environ["TERMINAL_CWD"] = "/workspace"
    os.environ["HERMES_SESSION_PLATFORM"] = "api_server"
    sys.path.insert(0, "/opt/hermes")
    import hermes_yaml as yaml
    from agent.skill_utils import ESSENTIAL_SKILLS, get_untrusted_project_skills_root
    from hermes_constants import get_bundled_skills_dir
    from run_agent import AIAgent
    from tools.skills_tool import find_all_skills, skills_list, skill_view

    spec = importlib.util.spec_from_file_location("worker", "/opt/assist/hermes_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    results = []
    worker.control = lambda path, value: results.append((path, value))
    worker._register_boundaries = lambda *_args, **_kwargs: None
    unexpected_provider_requests = []

    def metadata_only(self):
        # Native construction probes server type and context length. Answer
        # metadata locally, without invoking the worker's Unix relay.
        if self.command == "GET" and self.path in {"/v1/models", "/models"}:
            status, value = 200, {"data": [{"id": "fixture", "context_length": 131072}]}
        elif self.command == "GET" and self.path in {
                "/api/v1/models", "/api/tags", "/v1/props", "/props", "/version"}:
            status, value = 404, {}
        else:
            ollama_metadata = False
            if self.command == "POST" and self.path == "/api/show":
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    ollama_metadata = 0 < size <= 256 and json.loads(self.rfile.read(size)) == {"name": "fixture"}
                except ValueError:
                    pass
            if not ollama_metadata:
                unexpected_provider_requests.append((self.command, self.path))
                self.send_error(500, "This proof permits metadata only, with no inference provider")
                return
            status, value = 404, {}
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    worker._ProviderBridge._forward = metadata_only

    def conversation(self, *_args, **_kwargs):
        names = {s["name"] for s in find_all_skills()}
        listed = json.loads(skills_list())
        assert listed["success"] and names <= {s["name"] for s in listed["skills"]}
        viewed = json.loads(skill_view("hermes-agent", preprocess=False))
        assert viewed["success"], viewed
        if scenario != "legacy":
            assert ("custom-guide" if scenario == "alias" else "hermes-agent") in names
        assert get_untrusted_project_skills_root() == (Path("/workspace"), 1)
        assert "untrusted-project" not in names
        if scenario == "clean":
            assert "# Hermes Agent" in viewed["content"]
            assert get_bundled_skills_dir(Path("/opt/hermes/skills")) == Path("/opt/hermes/skills")
            linked = json.loads(skill_view("hermes-agent", "references/configuration.md", preprocess=False))
            assert linked["success"] and linked["content"]
        else:
            assert "LOCAL CUSTOMIZATION" in viewed["content"]
            assert "local-note" in names
            assert "pdf" not in names and "codex" not in names
            assert not json.loads(skill_view("pdf", preprocess=False))["success"]
            assert not (home / "skills/autonomous-ai-agents/codex").exists()
            assert not {"wrong-platform", "wrong-environment", "missing-app", "untrusted-project"} & names
        if scenario == "optout":
            bundle = Path("/opt/hermes/skills")
            seeded = {p.parent.name for p in bundle.rglob("SKILL.md")
                      if (home / "skills" / p.relative_to(bundle)).exists()}
            assert seeded <= ESSENTIAL_SKILLS
        if scenario in {"alternate", "alias", "legacy"}:
            assert not (home / "skills/autonomous-ai-agents/hermes-agent/SKILL.md").exists()
            manifest = (home / "skills/.bundled_manifest").read_text()
            assert not any(line.startswith("hermes-agent:") for line in manifest.splitlines())
        assert yaml.safe_load((home / "config.yaml").read_text()).get("skills", {}) == preferences
        return {"final_response": "Native skill startup checked."}

    config_path = home / "config.yaml"
    preferences = yaml.safe_load(config_path.read_text()).get("skills", {}) if config_path.exists() else {}
    request = home / "request.json"
    request.write_text(json.dumps({"session_id": "skills-proof", "model": "fixture",
                                   "toolsets": ["file", "terminal", "skills", "memory", "delegation"],
                                   "prompt": "List your available skills."}))
    with patch.object(AIAgent, "run_conversation", conversation):
        worker.main(str(request))
    assert not unexpected_provider_requests, unexpected_provider_requests
    assert sum(path == "/result" for path, _value in results) == 1


def check():
    # The actual worker fixes the agent's cwd to /workspace. This disposable
    # container has no host workspace mounted; make a real untrusted project.
    (Path("/workspace") / ".git").mkdir()
    write_skill(Path("/workspace/.agents"), "untrusted-project", "untrusted-project", "Untrusted")
    with tempfile.TemporaryDirectory(prefix="hermes-skills-") as directory:
        root = Path(directory)
        clean = root / "clean"
        clean.mkdir()
        run_turn(clean, "clean")
        # Simulate native edits and a deliberate deletion after first seeding.
        write_skill(clean, "autonomous-ai-agents/hermes-agent", "hermes-agent", "LOCAL CUSTOMIZATION")
        shutil.rmtree(clean / "skills/autonomous-ai-agents/codex")
        for home, scenario in ((clean, "populated"), (root / "populated", "populated"),
                               (root / "optout", "optout"), (root / "alternate", "alternate"),
                               (root / "alias", "alias"), (root / "legacy", "legacy")):
            home.mkdir(exist_ok=True)
            relative = "custom/hermes-agent" if scenario in {"alternate", "alias"} else "autonomous-ai-agents/hermes-agent"
            name = "custom-guide" if scenario == "alias" else "hermes-agent"
            if scenario == "legacy":
                (home / "skills").mkdir()
                (home / "skills/hermes-agent.md").write_text(
                    "---\nname: hermes-agent\ndescription: Synthetic legacy fixture\n---\nLOCAL CUSTOMIZATION\n")
            else:
                write_skill(home, relative, name, "LOCAL CUSTOMIZATION")
            write_skill(home, "local-note", "local-note", "A custom profile skill.")
            write_skill(home, "wrong-platform", "wrong-platform", "Hidden", "platforms: [windows]\n")
            write_skill(home, "wrong-environment", "wrong-environment", "Hidden", "environments: [kanban]\n")
            write_skill(home, "missing-app", "missing-app", "Hidden", "requires_apps: [fixture-app-unavailable]\n")
            # Native manifest deletion state and disabled preferences are distinct.
            manifest = home / "skills/.bundled_manifest"
            if not manifest.exists():
                manifest.write_text("codex:deleted-fixture\n")
            (home / "config.yaml").write_text("skills:\n  disabled: [pdf]\n  external_dirs: []\n")
            if scenario == "optout":
                (home / ".no-bundled-skills").touch()
            run_turn(home, scenario)
            run_turn(home, scenario)
    print(json.dumps({"status": "PASS", "worker_turns": 13,
                      "coverage": "clean discovery/view/linked file; repeated initialization; custom skills; "
                                  "alternate-path names/aliases/legacy files; deletion manifest; disabled preferences; "
                                  "opt-out; native filters/project trust",
                      "limits": "Native startup and tools with scripted conversation; no model or live acceptance."}))


def run_turn(home, scenario):
    subprocess.run([sys.executable, __file__, str(home), scenario], check=True, timeout=90)


if __name__ == "__main__":
    signal.alarm(180)
    if len(sys.argv) == 3:
        turn(Path(sys.argv[1]), sys.argv[2])
    else:
        check()
