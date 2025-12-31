import os
import subprocess
import time
import socket
import pytest

from playwright.sync_api import sync_playwright

PORT = 5051
URL = f"http://127.0.0.1:{PORT}"


def wait_port(port: int, timeout: float = 20.0):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.2)
    raise TimeoutError("Server did not start in time")


@pytest.mark.integration
def test_web_ui_flow():
    assert os.getenv("TAVILY_API_KEY") is not None, "TAVILY_API_KEY must be set for integration"
    # Start server
    srv = subprocess.Popen(["python", "-m", "uvicorn", "assist.web:app", "--host", "127.0.0.1", "--port", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_port(PORT, timeout=30)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(URL)
            page.wait_for_selector("text=Assist Web")
            # Create a new thread
            page.click("text=New thread")
            page.wait_for_selector("text=Thread ")
            # Send a message
            page.fill("textarea#text", "Say hello in one short sentence.")
            page.click("button:has-text('Send')")
            # Wait for assistant reply to appear
            page.wait_for_selector(".msg.assistant .content", timeout=30000)
            content = page.inner_text(".msg.assistant .content")
            assert content.strip() != ""
            assert "hello" in content.lower() or "hi" in content.lower()
            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()