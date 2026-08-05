import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { SessionManager } from "@earendil-works/pi-coding-agent";

const cases = [
  { phase: "before_custom_append", state: "not_appended" },
  { phase: "after_custom_append", state: "appended_not_started" },
  { phase: "after_custom_fsync", state: "appended_not_started" },
  { phase: "after_user_append", state: "completed_input" },
  { phase: "after_user_fsync", state: "completed_input" },
] as const;

for (const crash of cases) {
  test(`input identity classifies recovery at ${crash.phase}`, async () => {
    const directory = await mkdtemp(join(tmpdir(), "assist-pi-input-crash-"));
    const sessionPath = join(directory, "session.jsonl");
    const markerPath = join(directory, "halt.log");
    closeSync(openSync(sessionPath, "wx", 0o600));
    try {
      const child = spawn(process.execPath, [
        new URL("../src/input-crash-probe.js", import.meta.url).pathname,
        sessionPath,
        crash.phase,
        markerPath,
      ], { stdio: ["ignore", "pipe", "pipe"] });
      const outcome = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
        child.once("exit", (code, signal) => resolve({ code, signal }));
      });
      assert.equal(outcome.code, null);
      assert.equal(outcome.signal, "SIGKILL");
      assert.equal((await readFile(markerPath, "utf8")).trim(), crash.phase);

      const manager = SessionManager.open(sessionPath, undefined, "/workspace");
      const branch = manager.getBranch();
      const identities = branch.filter(
        (entry) => entry.type === "custom" && entry.customType === "assist-input-identity",
      );
      const inputs = branch.filter(
        (entry) => entry.type === "message" && entry.message.role === "user",
      );
      assert.ok(identities.length <= 1);
      assert.ok(inputs.length <= 1);
      const state = identities.length === 0
        ? "not_appended"
        : inputs.length === 0
          ? "appended_not_started"
          : "completed_input";
      assert.equal(state, crash.state);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
}
