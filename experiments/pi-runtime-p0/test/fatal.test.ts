import assert from "node:assert/strict";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { test } from "node:test";

import { startScriptedProvider, toolCall } from "../src/sdk-fixture.js";

const phases = [
  "session_start",
  "message_end:user",
  "before_agent_start",
  "before_provider_request",
  "message_end:assistant",
  "tool_call",
  "custom_tool_return",
  "tool_result",
  "turn_end",
  "session_shutdown",
] as const;

for (const phase of phases) {
  test(`mustHandle terminates the process at ${phase}`, async () => {
    const toolPhase = ["tool_call", "custom_tool_return", "tool_result"].includes(phase);
    const provider = await startScriptedProvider([
      toolPhase ? toolCall(0, "call-fatal", "load_skill", { name: "fixture" }) : "fatal text",
    ]);
    const directory = await mkdtemp(join(tmpdir(), "assist-pi-fatal-"));
    const sessionPath = join(directory, "session.jsonl");
    const markerPath = join(directory, "halt.log");
    const effectMarkerPath = join(directory, "effects.log");
    closeSync(openSync(sessionPath, "wx", 0o600));
    try {
      const child = spawn(
        process.execPath,
        [
          new URL("../src/fatal-probe.js", import.meta.url).pathname,
          provider.baseUrl,
          sessionPath,
          phase,
          markerPath,
          effectMarkerPath,
        ],
        { stdio: ["ignore", "pipe", "pipe"] },
      );
      const outcome = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
        child.once("exit", (code, signal) => resolve({ code, signal }));
      });
      assert.equal(outcome.code, null);
      assert.equal(outcome.signal, "SIGKILL");
      assert.match(await readFile(markerPath, "utf8"), new RegExp(phase.replace(":", "\\:")));
      assert.ok(provider.requests.length <= 1, `later provider request escaped ${phase}`);
      const effects = await readFile(effectMarkerPath, "utf8").catch(() => "");
      if (["session_start", "message_end:user", "before_agent_start", "before_provider_request", "message_end:assistant", "tool_call"].includes(phase)) {
        assert.equal(effects, "", `effect escaped ${phase}`);
      } else if (toolPhase) {
        assert.equal(effects, "load_skill\n", `expected one completed fixture effect at ${phase}`);
      }
    } finally {
      await provider.close();
      await rm(directory, { recursive: true, force: true });
    }
  });
}
