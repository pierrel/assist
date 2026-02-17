# Assist

Assist is an extensible, local-focused, LLM-based assistant. The principles guiding this project include:
- **Privacy** - The user has full control and no information leaves their computer
- **Extensibility** - All agentic behavior is easy to modify or add to
- **Generic** - Assist can help with anything

It is not just a coding assistant, but should be able to help with any project or goal.

See [README.org](README.org) for detailed concepts and architecture.

---

## Quick Start

### Prerequisites

- Python 3.13+ (3.14 recommended)
- Git
- An OpenAI-compatible model endpoint (local or remote)

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
   ASSIST_MODEL_URL=http://localhost:8000/v1
   ASSIST_MODEL_NAME=your-model-name
   ASSIST_API_KEY=your-api-key
   ASSIST_CONTEXT_LEN=78000
   ASSIST_TEST_URL_PATH=/models

   # Optional: Local git repository for domain integration
   ASSIST_DOMAIN=user@localhost:/path/to/repo.git

   # Local settings
   ASSIST_THREADS_DIR=/tmp/assist_threads
   ASSIST_PORT=8000
   ```

5. **Run the web interface**
   ```bash
   make web
   # Or manually: .venv/bin/python -m manage.web
   ```

   The server will start on the port specified in `.dev.env` (default: 8000).

### Running Tests and Evaluations

Assist includes two types of testing:

**Unit/Integration Tests** (`tests/`):
```bash
# Run all tests
make test

# Run specific test file
.venv/bin/pytest tests/test_domain_manager.py -v
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
- `test_agent.py` - Core agent behavior (planning, task management, file operations)
- `test_research_agent.py` - Research delegation and external knowledge handling
- `test_context_agent.py` - Filesystem discovery and context gathering
- `test_domain_integration.py` - Git integration and domain management
- `eval_multi_turn_research.py` - Multi-turn conversation handling (10+ turns)
- `eval_large_tool_results.py` - Context overflow handling with large tool results

Results are saved to `edd/history/results-YYYYMMDD-HHMM.xml` in JUnit format.

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
   ASSIST_MODEL_URL=http://production-server:8000/v1
   ASSIST_MODEL_NAME=production-model
   ASSIST_API_KEY=production-api-key
   ASSIST_CONTEXT_LEN=78000
   ASSIST_TEST_URL_PATH=/models

   # Production Settings
   ASSIST_THREADS_DIR=/var/lib/assist/threads
   ASSIST_PORT=5051

   # Production Domain (optional - local repo on production server)
   ASSIST_DOMAIN=/home/user/git/repo.git
   ```

3. **Prepare the production server**
   ```bash
   # Create deployment directory
   ssh assist-prod 'sudo mkdir -p /opt/assist && sudo chown $USER:$USER /opt/assist'

   # Install git and rsync (if not already installed)
   ssh assist-prod 'sudo pacman -S git rsync'  # Arch Linux
   # ssh assist-prod 'sudo apt install git rsync'  # Ubuntu/Debian
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

| Variable            | Description                    | Example                                  |
|---------------------|--------------------------------|------------------------------------------|
| `ASSIST_MODEL_URL`  | OpenAI-compatible API endpoint | `http://localhost:8000/v1`               |
| `ASSIST_MODEL_NAME` | Model identifier               | `mistralai/Ministral-3-8B-Instruct-2512` |
| `ASSIST_API_KEY`    | API authentication key         | `sk-your-api-key`                        |

### Optional Variables

| Variable               | Description                           | Default               |
|------------------------|---------------------------------------|-----------------------|
| `ASSIST_CONTEXT_LEN`   | Context window size (chars)           | `32768`               |
| `ASSIST_TEST_URL_PATH` | Endpoint to test API availability     | None                  |
| `ASSIST_DOMAIN`        | Git repository for domain integration | None                  |
| `ASSIST_THREADS_DIR`   | Data storage location                 | `/tmp/assist_threads` |
| `ASSIST_PORT`          | Server port                           | `8000`                |

See [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for complete reference.

---

## Architecture

### Local Development
- Configuration: `.dev.env` (gitignored)
- Data: `ASSIST_THREADS_DIR` (default: `/tmp/assist_threads`)
- Model: Your local model endpoint
- Domain: Your local git repository (optional)

### Production Deployment
- Configuration: `.deploy.env` (gitignored)
- Data: `/var/lib/assist/threads` (persists across deployments)
- Model: Production model endpoint
- Domain: Production git repository (optional, local to server)
- Service: systemd manages the process
- Logs: `journalctl -u assist-web`

---

## Project Structure

```
assist/
├── assist/                 # Core application code
│   ├── agent.py           # Agent implementation
│   ├── model_manager.py   # Model selection and configuration
│   ├── domain_manager.py  # Git repository management
│   └── tools/             # Agent tools
├── edd/                   # Agent evaluations
│   ├── eval/              # Evaluation test suite
│   └── history/           # Test results history
├── manage/                # Management interfaces
│   ├── web.py            # Web UI (FastAPI)
│   └── cli.py            # CLI interface
├── tests/                 # Unit/integration tests
├── scripts/               # Deployment and setup scripts
├── .dev.env.example      # Development config template
├── .deploy.env.example   # Production config template
├── Makefile              # Build and deployment commands
└── README.md             # This file
```

---

## Domain Integration (Optional)

Assist can integrate with git repositories to manage changes:

### Local Development
```bash
# .dev.env
ASSIST_DOMAIN=user@localhost:/path/to/repo.git
```

### Production
```bash
# Create a local bare repository on production server
ssh assist-prod 'git init --bare /path/to/repo.git'

# .deploy.env
ASSIST_DOMAIN=/path/to/repo.git  # Local path on production server
```

When enabled, each thread creates a git branch and can merge changes back to main.

---

## Troubleshooting

### Local Development

**Server won't start:**
```bash
# Check if .dev.env exists
cat .dev.env

# Verify required variables
grep -E "ASSIST_MODEL_URL|ASSIST_MODEL_NAME|ASSIST_API_KEY" .dev.env

# Test manually
export $(grep -v '^#' .dev.env | xargs)
.venv/bin/python -m manage.web
```

**Model connection fails:**
```bash
# Test model endpoint
curl $ASSIST_MODEL_URL/models
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
