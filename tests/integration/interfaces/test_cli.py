import os
import subprocess
import sys
import time
import pytest

@pytest.mark.integration
def test_cli_interaction():
    assert os.getenv("TAVILY_API_KEY") is not None, "TAVILY_API_KEY must be set for integration"
    proc = subprocess.Popen([sys.executable, "-m", "manage.cli"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Send a simple prompt and then quit
        out = proc.stdout.readline()  # Working directory line
        proc.stdin.write("Say hi in one sentence.\n")
        proc.stdin.flush()
        # Read response line(s)
        # Allow some time for the model/agent
        time.sleep(2)
        proc.stdin.write("/quit\n")
        proc.stdin.flush()
        stdout, stderr = proc.communicate(timeout=30)
        combined = out + stdout
        assert "Working directory:" in combined
        # Expect some assistant output
        assert "hi" in combined.lower() or "hello" in combined.lower()
    finally:
        proc.kill()