# Assist

Assist is an extensible, local-focused, LLM-based assistant. The principles guiding this project include:
- **Privacy** - The user has full control and no information leaves their computer
- **Extensibility** - All agentic behavior is easy to modify or add to
- **Generic** - Assist can help with anything

It is not just a coding assistant, but should be able to help with any project or goal.

See [README.org](README.org) for detailed concepts and architecture.

## Current Capabilities

Assist combines several pieces that are uncommon in off-the-shelf agent
frameworks, tuned specifically to run well against small local models
where reliability is harder than with frontier APIs.

- **Endpoint auto-discovery.** Model name and runtime context length
  are discovered from your local LLM endpoint on first request — probes
  `GET /v1/models` first (vLLM convention) and falls back to llama.cpp's
  `/props` for engines that don't expose `max_model_len`. No config drift
  across `.dev.env`, `.deploy.env`, and the systemd unit; swap the model
  on the server and the cache busts itself on the next call.

- **Small-LLM-tuned memory.** Persistent facts and preferences live in
  the workspace's `AGENTS.md` and auto-load on every session. The
  standard deepagents memory mechanism asks the model to call
  `edit_file` against a path it has to construct; small models routinely
  pick the wrong path. Assist replaces that with a dedicated
  `save_memory(content=...)` tool — the path is baked in, the model
  only chooses what to remember. A pre-action prompt step makes it
  capture user preferences automatically without being asked.

- **Small-LLM-tuned skills.** Skills are progressively-disclosed
  capability bundles under `assist/skills/<name>/SKILL.md`. Where
  upstream expects the model to read the SKILL path itself, assist
  exposes a single `load_skill(name=...)` tool — the model never sees
  paths. A new skill is a single-file add: no system-prompt edits, no
  registration. Description-driven matching means the right skill loads
  even when the user hasn't named it (e.g. mentioning a `.org` file
  triggers the `org-format` skill).

- **Built-in sandboxing.** Every tool call that touches the filesystem
  or runs code happens inside a per-thread Docker container with the
  workspace bind-mounted at `/workspace`. Each shell command is bounded
  by a wall-clock cap (default 600s) and on timeout returns concrete
  adjustment guidance to the agent — so a runaway tool call gets a
  recoverable error the model can act on, not an indefinite hang. If
  Docker isn't available the agent transparently falls back to
  unsandboxed execution.

- **Specialized agents and skills out of the box.**
  - **Research agent** with internet search (`search_internet`) and
    URL fetch (`read_url`), plus critique and fact-check sub-agents for
    multi-pass refinement of long-form output.
  - **Context agent** for read-only filesystem exploration — finds the
    right file, surfaces evidence, never modifies. Runs as a sub-agent
    of the general agent so any task can ground itself in local context
    before acting.
  - **Dev agent + dev skill** for code work — TDD-style plan/test/implement
    flow, designed to keep small models from skipping ahead.
  - **Calculate skill** for arithmetic, statistics, simulations, and
    financial projections — forces the agent to verify its answer by
    running real Python rather than guessing a number.
  - **Org-format skill** scopes editing rules to `.org` files
    (heading-body discipline, no orphaned content) without affecting
    other formats.

- **Resilient context handling.** `ContextAwareToolEvictionMiddleware`
  evicts large tool results to a stateful filesystem before they
  overflow context. `BadRequestRetryMiddleware` catches context-overflow
  errors from the server, sanitises and truncates, and retries.
  `RollbackRunnable` snapshots each turn and restores on hard errors so
  a transient failure doesn't end the thread.

- **Loop and stall guards.** `LoopDetectionMiddleware` catches repeating
  tool calls. `EmptyResponseRecoveryMiddleware` handles the case where a
  small model returns a blank message. Together they keep agents from
  getting stuck in degenerate states without human intervention.

- **Git domain integration.** Each thread gets its own git branch in a
  configured "domain" repo (your life repo, your work repo, etc.).
  Agent edits stay isolated until you decide to merge from the web UI.
  Multiple domains coexist — choose which one a new thread belongs to.

- **Multiple frontends, one agent core.** Web UI (FastAPI), CLI
  (`manage.cli`), and an Emacs integration (`assist/emacs/`) all share
  the same agent runtime, the same `AGENTS.md` memory, and the same
  domain repos.

---

## Quick Start

### Prerequisites

- Python 3.13+ (3.14 recommended)
- Git
- Docker (for the sandbox — the agent executes commands inside a container)
- An OpenAI-compatible model endpoint (local or remote)

Your user must be in the `docker` group so the web process can start containers:
```bash
sudo usermod -aG docker $USER
# Log out and back in for the group change to take effect
```

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd assist
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

4. **Configure local environment**
   ```bash
   # Copy the development environment template
   cp .dev.env.example .dev.env

   # Edit .dev.env with your local settings
   nano .dev.env
   ```

   **Required settings in `.dev.env`:**
   ```bash
   # Local Model Configuration
   # Model name and context length are auto-discovered from
   # ${ASSIST_MODEL_URL}/models on first request — no need to set them.
   ASSIST_MODEL_URL=http://localhost:8000/v1

   # Optional: API key (defaults to "EMPTY", which vLLM accepts)
   # ASSIST_API_KEY=your-api-key

   # Optional: Git repositories for domain integration (comma-separated)
   ASSIST_DOMAINS=user@localhost:/path/to/repo.git

   # Local settings
   ASSIST_THREADS_DIR=/tmp/assist_threads
   ASSIST_PORT=8000
   ```

5. **Run the web interface**
   ```bash
   make web
   # Or manually: .venv/bin/python -m manage.web
   ```

   This builds the sandbox Docker image and starts the server.

   The server will start on the port specified in `.dev.env` (default: 8000).

### Running Tests and Evaluations

Assist includes two types of testing — and **the distinction is by what the test calls, not how long it runs**:

| Where | What | Touches the LLM? |
|-------|------|------------------|
| `tests/` | Unit + integration tests. Pure-Python, mocked tool calls, mocked model responses. Run on every change. | **No** |
| `edd/eval/` | Agent evaluations. Drive the real model end-to-end and assert on observed agent behaviour. Network- and model-bound. | **Yes** |

Where to put a new test:
- Does it construct a `MagicMock` handler / synthetic `ToolMessage` / `ModelRequest` and assert on a transformation? → `tests/` (or `tests/middleware/` for middleware).
- Does it call `select_chat_model`, `Thread`, `AgentHarness`, `create_agent`, `agent.invoke()`, or otherwise reach a model server? → `edd/eval/`.

Mis-classification was an issue early on; the table above is the rule.

**Unit/Integration Tests** (`tests/`):
```bash
# Run all tests
make test

# Run specific test file
.venv/bin/pytest tests/test_domain_manager.py -v
.venv/bin/pytest tests/middleware/test_loop_detection.py -v
```

**Agent Evaluations** (`edd/`):

The `edd/` directory contains evaluations that test agent behavior under realistic conditions. These are longer-running tests that validate the agent's ability to handle complex, multi-turn interactions.

```bash
# Run all evaluations
make eval

# Run specific evaluation
.venv/bin/pytest edd/eval/test_agent.py -v

# Run individual eval scripts
.venv/bin/python edd/eval/eval_multi_turn_research.py
.venv/bin/python edd/eval/eval_large_tool_results.py
```

**Available evaluations**:

Per-agent:
- `test_agent.py` — general agent: planning, routing, task management, file operations
- `test_context_agent.py` — read-only filesystem discovery
- `test_research_agent.py` — research delegation and external knowledge handling
- `test_dev_agent.py`, `test_dev_agent_planning_flow.py`, `test_dev_agent_runs_eval.py` — legacy harness for the dev-agent (the dev skill now runs on the general agent; these still validate the skill content)

Skill-specific:
- `test_org_format_skill.py` — org-format skill: heading-body insertion rule
- `test_dev_skill_multi_turn.py` — dev skill on the general agent: multi-turn TDD with approvals + mid-task clarification

Cross-cutting:
- `test_domain_integration.py` — git integration and domain management
- `test_memory.py`, `test_various_failures.py` — memory + failure modes
- `test_thread_e2e.py` — thread/conversation persistence
- `eval_multi_turn_research.py` — long multi-turn research (10+ turns)
- `eval_large_tool_results.py` — context overflow handling
- `eval_summarization_long_context.py` — summarization in long contexts

Results are saved to `edd/history/results-YYYYMMDD-HHMM.xml` in JUnit format.  Snapshot baselines for diffing live in `docs/baselines/`.

See [edd/eval/README.md](edd/eval/README.md) for detailed documentation on evaluations.

---

## Production Deployment

### One-Time Server Setup

1. **Configure SSH access**

   Add your production server to `~/.ssh/config`:
   ```bash
   # ~/.ssh/config
   Host assist-prod
       HostName your.server.ip
       User your_username
       IdentityFile ~/.ssh/id_rsa
       Port 22
   ```

   Test connection:
   ```bash
   ssh assist-prod 'echo "SSH working"'
   ```

2. **Create deployment configuration**
   ```bash
   # Copy the deployment template
   cp .deploy.env.example .deploy.env

   # Edit with your production values
   nano .deploy.env
   ```

   **Required settings in `.deploy.env`:**
   ```bash
   # SSH Configuration
   DEPLOY_HOST=assist-prod
   DEPLOY_PATH=/opt/assist
   SERVICE_NAME=assist-web
   PYTHON=python3.14

   # Production Model Configuration
   # Model name and context length are auto-discovered from
   # ${ASSIST_MODEL_URL}/models on first request.
   ASSIST_MODEL_URL=http://production-server:8000/v1

   # Optional: API key (defaults to "EMPTY", which vLLM accepts)
   # ASSIST_API_KEY=production-api-key

   # Production Settings
   ASSIST_THREADS_DIR=/var/lib/assist/threads
   ASSIST_PORT=5051

   # Production Domains (optional, comma-separated git URLs)
   ASSIST_DOMAINS=/home/user/git/repo1.git,/home/user/git/repo2.git
   ```

3. **Prepare the production server**
   ```bash
   # Create deployment directory
   ssh assist-prod 'sudo mkdir -p /opt/assist && sudo chown $USER:$USER /opt/assist'

   # Install git, rsync, and docker (if not already installed)
   ssh assist-prod 'sudo pacman -S git rsync docker'  # Arch Linux
   # ssh assist-prod 'sudo apt install git rsync docker.io'  # Ubuntu/Debian

   # Enable Docker and add user to docker group
   ssh assist-prod 'sudo systemctl enable --now docker && sudo usermod -aG docker $USER'
   ```

4. **Set up passwordless sudo (optional)**
   ```bash
   # This allows deployment without password prompts
   make setup-sudo
   ```

   Enter your password when prompted. This is a one-time setup.

5. **Initial deployment**
   ```bash
   # Deploy code
   make deploy-code

   # Install Python dependencies on remote
   make install-prod

   # Install and start the systemd service
   make deploy-service
   make restart
   ```

### Subsequent Deployments

After initial setup, deploy with a single command:

```bash
make deploy
```

This will:
1. Sync code to the production server
2. Update the systemd service configuration
3. Restart the service
4. Show service logs (press Ctrl+C when satisfied)

### Deployment Commands

```bash
make deploy         # Full deployment (code + service + restart)
make deploy-code    # Deploy code only (no restart)
make deploy-service # Install/update systemd service
make install-prod   # Install dependencies on remote
make restart        # Restart the service (then follow logs)
make status         # Check service status
make logs           # View service logs (live tail)
make help           # Show all available commands
```

---

## Environment Variables

All configuration is done via environment variables. Different files are used for different environments:

| File                  | Purpose               | Gitignored?       | When Used     |
|-----------------------|-----------------------|-------------------|---------------|
| `.dev.env`            | Local development     | ✅ Yes            | `make web`    |
| `.dev.env.example`    | Development template  | ❌ No (committed) | Documentation |
| `.deploy.env`         | Production deployment | ✅ Yes            | `make deploy` |
| `.deploy.env.example` | Production template   | ❌ No (committed) | Documentation |

### Required Variables

| Variable           | Description                    | Example                    |
|--------------------|--------------------------------|----------------------------|
| `ASSIST_MODEL_URL` | OpenAI-compatible API endpoint | `http://localhost:8000/v1` |

The model identifier and context window are discovered automatically by
calling `${ASSIST_MODEL_URL}/models` on the first request. The cache is
busted automatically when the upstream returns `model_not_found` (e.g.
after the operator swaps the model on the serving host).

### Optional Variables

| Variable             | Description                                                                                              | Default               |
|----------------------|----------------------------------------------------------------------------------------------------------|-----------------------|
| `ASSIST_API_KEY`     | API key. Falls back to `OPENAI_API_KEY`, then to `"EMPTY"` (which vLLM accepts and real OpenAI rejects). | `"EMPTY"`             |
| `ASSIST_DOMAINS`     | Git repositories, comma-separated                                                                        | None                  |
| `ASSIST_THREADS_DIR` | Data storage location                                                                                    | `/tmp/assist_threads` |
| `ASSIST_PORT`        | Server port                                                                                              | `8000`                |

See [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for complete reference.

---

## Architecture

### Local Development
- Configuration: `.dev.env` (gitignored)
- Data: `ASSIST_THREADS_DIR` (default: `/tmp/assist_threads`)
- Model: Your local model endpoint
- Domains: Your local git repositories (optional)
- Sandbox: Docker container per thread for isolated command execution

### Production Deployment
- Configuration: `.deploy.env` (gitignored)
- Data: `/var/lib/assist/threads` (persists across deployments)
- Model: Production model endpoint
- Domains: Production git repositories (optional, local to server)
- Sandbox: Docker container per thread
- Service: systemd manages the process
- Logs: `journalctl -u assist-web`

---

## Skills

The skills system is built on the `SkillsMiddleware` from
[deepagents](https://github.com/langchain-ai/deepagents) but with a few
modifications tuned for the small models we run locally (e.g.
Qwen3-Coder-30B). Upstream's mechanism is **read the SKILL.md path with
`read_file`** — the small model is unreliable at constructing that path
from a description. Our variant in `assist/middleware/skills_middleware.py`
keeps progressive disclosure but changes how the model interacts with it:

- A dedicated **`load_skill(name=...)`** tool, registered by the
  middleware, replaces the upstream "use `read_file` with this path"
  mechanism. The model never sees skill paths.
- A short, imperative **system-prompt template** that ends with a
  mandatory pre-action check — scan the latest user message against
  every skill description before any other tool call.
- A **name-only skill listing** (`- **name**: description`) — upstream's
  `-> Read \`{path}\` for full instructions` line is stripped so there
  are no path strings for the model to copy.

### The four layers of guidance

We treat the prompt as four nested concentric rings, each more specific
than the last. Keeping each layer focused on its own concern is what
makes the system extensible — a new skill should land without editing
the outer rings.

1. **System prompt** (`assist/templates/deepagents/general_instructions.md.j2`)
   — what agents exist, what general process to follow, and the
   project-wide rules. Should never name a specific skill or describe
   skill rules. May reference the **Skills** section abstractly.
2. **Skills middleware prompt** (`SMALL_MODEL_SKILLS_PROMPT` in
   `skills_middleware.py`) — how skills work in general: the
   description-then-load contract, the `load_skill` tool, the pre-action
   check. Generic — never names a specific skill.
3. **Skill description** (YAML frontmatter `description:` field) — a
   single sentence-block answering *when should the agent load this
   skill*. Not what the rules are.
4. **Skill body** (everything after the frontmatter in `SKILL.md`) —
   the actual rules the agent applies once it has loaded the skill.

A new skill is added by creating `assist/skills/<name>/SKILL.md` —
nothing else needs to change. If you find yourself editing layer 1 or
layer 2 to make a skill match, the description (layer 3) is wrong.

### Writing a skill description

The description is the only thing the model has to decide whether to
load. It is matched against the user's latest message by the
pre-action check (and by the model's general attention). Keep these
constraints in mind:

- **Front-load trigger keywords.** Empirically, the small model only
  reliably matches a description when high-signal tokens appear early.
  Putting natural prose first and a `TRIGGER WORDS — ...` list later
  drops pass rate from ~95% to under 30%.
- **Use a `TRIGGER WORDS — <comma-separated tokens>` segment** for
  domain anchors: file extensions (`.org`), example filenames
  (`projects.org`), and concrete topic words the user might say
  (`asterisk heading`, `orphan`). These are the tokens that make
  matching robust.
- **Include a `MUST load before <conditions>` clause.** Without an
  imperative-shaped condition, the model treats the description as
  informational and skips loading. This is one of the few places
  imperative wording earns its keep.
- **Do not describe the rules.** Anything the body explains belongs in
  the body, not the description. The agent only reads the description
  to decide *whether* to load — once it loads, the body is what it
  applies.
- **Do not mention `read_file` or any tool name.** The middleware
  exposes `load_skill`; the description is content, not mechanism.
- A short friendly opening (≤ 8 words, e.g. `Guidance for org-mode
  (\`.org\`) files.`) before the trigger list reads naturally and does
  not measurably hurt match rate.

Example (the `org-format` skill):

```yaml
---
name: org-format
description: Guidance for org-mode (`.org`) files. TRIGGER WORDS — `.org`, org-mode, org file, headings, heading body, asterisk heading, orphan, projects.org. MUST load before any tool call that reads, edits, writes, or mentions a `.org` file.
---
```

### Writing a skill body

The body is what the agent applies after `load_skill` returns. By the
time the body is read the agent has already committed to applying the
skill, so the body's job is to be a clear reference for the rules — not
to re-justify loading.

- Skip "When to apply" / "When to use" sections. The description (layer
  3) already covered the trigger conditions, and the body is read only
  once the skill matches.
- Lead with a one-line H1 anchor (`# Org-mode format guide`) — this
  small heading measurably improves the model's adherence to the rules
  that follow.
- Structure as concrete rules, examples, and procedures. Mark wrong vs.
  right patterns side-by-side where helpful.

### Adding a new skill

1. Create `assist/skills/<skill-name>/SKILL.md` with frontmatter
   (`name:` matching the directory, `description:` following the rules
   above) and a body of rules.
2. That's it. `SkillsMiddleware` discovers the skill via the
   `/skills/` source path on next agent construction; the system prompt
   automatically lists it; `load_skill(name="<skill-name>")` works.
3. Add an eval under `edd/eval/` that exercises the trigger conditions
   and the rules, similar to `test_org_format_skill.py` and
   `test_skill_loading.py`.

### Why the layered structure matters

The four-layer split is what lets us add the eleventh skill without
rewriting the outer prompts. If the skills prompt named "org-mode" or
"markdown" as examples, every new file-format skill would tempt an
edit. If the description summarized the rules, the body would drift
out of sync. If the body re-justified loading, the description would
get longer for no reason. Each layer answering exactly one question
keeps the surface area constant.

---

## Memory

The memory system is built on `MemoryMiddleware` from
[deepagents](https://github.com/langchain-ai/deepagents) but with the
same small-model-tuning treatment as Skills. Upstream's mechanism is
**read happens automatically; write happens through `edit_file` with
guidance prose**. Empirically the small model invents file names
(`/user_info.md`) instead of writing to the configured `AGENTS.md`,
or it acknowledges the fact in conversation and never calls a tool at
all. Our variant in `assist/middleware/memory_middleware.py` keeps
the auto-load read pathway unchanged and replaces the write pathway:

- A dedicated **`save_memory(content=...)`** tool, registered by the
  middleware, replaces "use `edit_file` against the memory path". The
  model only chooses the content; the path and the append-vs-replace
  strategy are baked into the tool. The tool returns a `Command` that
  also updates `memory_contents` in agent state, so a subsequent turn
  on the same thread sees the freshly-saved fact (the upstream
  `before_agent` only loads once per session).
- A short, imperative **system-prompt template** that ends with a
  mandatory pre-action check — scan the latest user message for a
  fact about the user that's not already in `<agent_memory>`, and if
  found, call `save_memory` before any work tool.
- A **fail-closed read-then-write**: if the existing memory file
  can't be read (transient backend error, decode failure), the tool
  refuses to write rather than clobbering with empty content.

### The two layers of guidance

Memory has fewer layers than Skills because there is only one memory
file (no per-skill descriptions to write):

1. **System prompt** (`assist/templates/deepagents/general_instructions.md.j2`)
   — Step 0 names `save_memory` as a pre-action tool alongside
   `load_skill`. Project-wide rules; never describes when to save
   specifically.
2. **Memory middleware prompt** (`SMALL_MODEL_MEMORY_PROMPT` in
   `memory_middleware.py`) — how memory works in general: the
   `<agent_memory>` frame, the `save_memory` tool, when to save vs.
   not, the pre-action check. Generic — never names a specific user
   fact or memory-file format.

The agent never sees the memory path; it only ever sees the loaded
content (inside `<agent_memory>` tags) and the `save_memory` tool.
Adding a second source (e.g. a global `~/AGENTS.md`) would mean
extending `sources=` in `agent.py` and updating the `_format_agent_memory`
override — no prompt edits.

### Concurrency note

`save_memory` does a read-then-write of the entire memory file. The
deepagents tool loop runs sequentially within a thread, so a single
`save_memory` call per turn is safe. If the model ever emits two
parallel `save_memory` tool calls in one assistant message, the
second can clobber the first. We have not seen this in practice and
do not lock; the constraint is documented at the tool-factory call
site instead.

### Adding a new memory source

The current `SmallModelMemoryMiddleware.__init__` takes a single
`memories_path: str`. Multi-source support would require:

1. Widen the constructor to `memories_paths: list[str]` and pass the
   list through to `super().__init__(sources=memories_paths)` — the
   upstream load pathway already iterates `self.sources`.
2. Decide the write strategy. The `save_memory` tool currently writes
   to a single fixed path. Options for multi-source writes: add a
   `target` enum argument the model picks, or register one tool per
   source (e.g. `save_user_memory`, `save_project_memory`).

---

## Project Structure

```
assist/
├── assist/                  # Core application code
│   ├── agent.py             # Agent factories (general, context, research, dev)
│   ├── promptable.py        # Jinja prompt rendering + skill body loading
│   ├── model_manager.py     # Model selection and configuration
│   ├── domain_manager.py    # Git repository and sandbox management
│   ├── sandbox.py           # Docker sandbox backend
│   ├── sandbox_manager.py   # Sandbox lifecycle and per-thread containers
│   ├── thread.py            # Conversation thread state
│   ├── backends.py          # CompositeBackend wiring (state + filesystem + skills)
│   ├── checkpoint_rollback.py
│   ├── env.py
│   ├── git.py
│   ├── tools.py             # Tool functions (read_url, search_internet, …)
│   ├── middleware/          # Custom AgentMiddleware classes
│   │   ├── skills_middleware.py        # SmallModelSkillsMiddleware
│   │   ├── memory_middleware.py        # SmallModelMemoryMiddleware
│   │   ├── read_only_enforcer.py       # ReadOnlyEnforcerMiddleware
│   │   ├── loop_detection.py           # LoopDetectionMiddleware
│   │   ├── context_aware_tool_eviction.py
│   │   └── …                           # other middleware
│   ├── skills/              # Agent skills loaded via SkillsMiddleware
│   │   ├── dev/SKILL.md     # TDD workflow + code-task routing
│   │   └── org-format/SKILL.md
│   └── templates/           # Jinja prompt templates
│       ├── deepagents/      # Per-agent system prompts
│       └── reference/       # Inline references (legacy; being moved into skills)
├── dockerfiles/             # Docker images
│   └── Dockerfile.sandbox   # Sandbox container (Arch-based, with git/python/emacs)
├── edd/                     # Agent evaluations (LLM-driven, network-bound)
│   ├── eval/                # Evaluation test suite — anything that calls the real model
│   └── history/             # Test results history (JUnit XML)
├── manage/                  # Management interfaces
│   ├── web.py               # Web UI (FastAPI)
│   └── cli.py               # CLI interface
├── docs/                    # Per-improvement design records — see Documentation below
│   └── baselines/           # Snapshot eval-suite results for diffing
├── tests/                   # Unit/integration tests (no LLM — mocked model + tools)
│   └── middleware/          # Per-middleware unit tests (test_<module>.py)
├── scripts/                 # Deployment and setup scripts
├── roadmap.org              # Open work, organized by theme
├── .dev.env.example         # Development config template
├── .deploy.env.example      # Production config template
├── Makefile                 # Build and deployment commands
└── README.md                # This file
```

## Documentation

This repo keeps two kinds of design material side-by-side:

- **`roadmap.org`** at the repo root — an opinionated, theme-organized list of open work. Items are written as `** TODO` with a short rationale and links to the underlying `docs/` proposal where one exists. This is the place to look (or to add) when you want to know *what's next*.
- **`docs/*.org` (and a few `*.md`)** — one document per improvement, proposal, investigation, or diagnosis. Each `.org` proposal carries a `State:` header (`Not started` / `Done` / etc.) at the top, followed by `* Problem` and `* Solution` sections. `.md` files in `docs/` tend to be specific incident write-ups or external-source notes (e.g. paste from a conversation).

Conventions:
- New proposals: name them `YYYY-MM-DD-<short-slug>.org` and start with `State: Not started`. Add a corresponding `** TODO` entry under the appropriate section in `roadmap.org` linking back.
- When a proposal lands, change the `State:` header to `Done (YYYY-MM-DD)` (and link the docs that describe the resulting code if helpful). The `roadmap.org` entry can be marked `** DONE`.
- Keep proposals concise. Look at `docs/2026-04-26-skills-system.org` or `docs/2026-04-26-read-only-enforcer.org` for the expected shape; the long-running migration record at `docs/2026-04-25-skills-rearchitecture.org` is the exception, not the rule.

Investigations and diagnoses (e.g. `docs/4822cf50-diagnosis.md`, `docs/2026-04-26-token-max-mismatch-investigation.md`) don't carry a `State:` header — they're snapshots, not commitments.

---

## Domain Integration (Optional)

Assist can integrate with one or more git repositories. Each thread works with a single domain. When multiple domains are configured, the web UI shows a dropdown to choose which repository a new thread should use. The first domain is the default.

### Local Development
```bash
# .dev.env — single repo
ASSIST_DOMAINS=user@localhost:/path/to/repo.git

# .dev.env — multiple repos (comma-separated)
ASSIST_DOMAINS=user@localhost:/path/to/life.git,user@localhost:/path/to/work.git
```

### Production
```bash
# Create local bare repositories on production server
ssh assist-prod 'git init --bare /path/to/repo1.git'
ssh assist-prod 'git init --bare /path/to/repo2.git'

# .deploy.env
ASSIST_DOMAINS=/path/to/repo1.git,/path/to/repo2.git
```

When enabled, each thread creates a git branch and can merge changes back to main.

## Docker Sandbox

The agent executes shell commands inside a Docker container rather than on the host. Each thread gets its own container with the domain repository bind-mounted at `/workspace`.

The sandbox image is built automatically by `make web` (and `make deploy`). To build it manually:
```bash
make sandbox-build
```

If Docker is unavailable, the agent falls back to running without a sandbox.

---

## Troubleshooting

### Local Development

**Server won't start:**
```bash
# Check if .dev.env exists
cat .dev.env

# Verify required variables
grep -E "ASSIST_MODEL_URL" .dev.env

# Test manually
export $(grep -v '^#' .dev.env | xargs)
.venv/bin/python -m manage.web
```

**Model connection fails:**
```bash
# Test model endpoint
curl $ASSIST_MODEL_URL/models
```

**Docker sandbox not working:**
```bash
# Verify Docker is running
docker info

# Verify your user is in the docker group
groups | grep docker

# Build the sandbox image manually
make sandbox-build

# Test the sandbox image
make sandbox-shell
```

### Production Deployment

**Deployment fails:**
```bash
# Test SSH
ssh assist-prod 'echo "Connected"'

# Check deployment config
cat .deploy.env

# Deploy with verbose output
make deploy-code
```

**Service won't start:**
```bash
# Check service status
make status

# View logs
make logs

# Check environment variables
ssh assist-prod 'systemctl show assist-web --property=Environment'
```

**Port already in use:**
```bash
# Change port in .deploy.env
ASSIST_PORT=5052

# Redeploy service
make deploy-service
make restart
```

---

## Security

**Never commit these files:**
- `.dev.env` - Contains your local API keys
- `.deploy.env` - Contains production credentials

These files are gitignored. Use the `.example` templates to see what values are needed.

---

## Getting Help

```bash
# Show all make commands
make help

# Check service status
make status

# View service logs
make logs
```
