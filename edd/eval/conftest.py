"""Pytest configuration for evals.

Sets up logging so that model call progress (assist.model) is visible
live in the terminal and written to edd/history/last_eval.log, while
noisy HTTP/transport loggers are suppressed.
"""
import logging


def pytest_configure(config):
    # Show model call progress at INFO (the [general-agent] Model Call #N lines)
    logging.getLogger("assist.model").setLevel(logging.INFO)
    # Show our middleware and agent logs at INFO
    logging.getLogger("assist").setLevel(logging.INFO)
    # Suppress high-frequency HTTP transport noise — only show warnings+
    for noisy in ("httpcore", "httpx", "urllib3", "openai._base_client",
                  "docker", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
