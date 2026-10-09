#!/usr/bin/env bash
set -euo pipefail

runtime_name="assist-browser-runtime-smoke-$$"
host_boot_id=$(< /proc/sys/kernel/random/boot_id)
host_clock_ns=$(python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_BOOTTIME))')
cleanup() { timeout 5s docker rm -f "$runtime_name" >/dev/null 2>&1 || true; }
trap cleanup EXIT

timeout --kill-after=5s 30s docker run --rm --name "$runtime_name" \
  --env "BROWSER_TEST_BOOT_ID=$host_boot_id" \
  --env "BROWSER_TEST_CLOCK_NS=$host_clock_ns" \
  --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges \
  --security-opt "seccomp=$(pwd)/dockerfiles/browser-seccomp.json" \
  --memory 1g --pids-limit 128 --cpus 1 --shm-size 64m \
  --tmpfs /run:rw,nosuid,size=1048576,uid=10001,gid=10001 \
  --tmpfs /tmp:rw,nosuid,size=33554432,uid=10001,gid=10001 \
  --tmpfs /home/browser:rw,nosuid,size=134217728,uid=10001,gid=10001 \
  --entrypoint python assist-browser -c \
  'import os, time
assert open("/proc/sys/kernel/random/boot_id").read().strip() == os.environ["BROWSER_TEST_BOOT_ID"]
elapsed_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME) - int(os.environ["BROWSER_TEST_CLOCK_NS"])
assert 0 <= elapsed_ns < 30_000_000_000, elapsed_ns
from playwright.sync_api import sync_playwright
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, chromium_sandbox=True)
    page = browser.new_page()
    page.set_content("<h1>Chromium sandbox ready</h1>")
    assert page.locator("h1").inner_text() == "Chromium sandbox ready"
    browser.close()
print("Chromium runtime smoke passed")'
