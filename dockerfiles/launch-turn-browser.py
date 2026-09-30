"""Reset the turn container environment before executing the browser worker."""
import os


def main():
    if os.getpid() != 1:
        raise SystemExit("browser worker must own its PID namespace")
    browser_env = {name: os.environ[name] for name in (
        "BROWSER_SESSION_TOKEN", "BROWSER_BOOT_ID", "BROWSER_DEADLINE_NS",
        "BROWSER_GENERATION", "BROWSER_PROXY_HOST", "BROWSER_PROXY_PORT")}
    browser_env.update({
        "HOME": "/run/assist-browser",
        "TMPDIR": "/run/assist-browser",
        "BROWSER_RUNTIME_DIR": "/run/assist-browser",
        "BROWSER_DOWNLOAD_DIR": "/run/assist-browser/downloads",
        "PLAYWRIGHT_BROWSERS_PATH": "/opt/ms-playwright",
        "PATH": "/usr/bin:/bin",
    })
    os.execve("/opt/assist-browser/bin/python",
              ["/opt/assist-browser/bin/python",
               "/opt/assist/browser_runner.py", "serve"], browser_env)


if __name__ == "__main__":
    main()
