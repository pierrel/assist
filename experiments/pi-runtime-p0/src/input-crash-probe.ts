import { appendFileSync } from "node:fs";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import { fsyncFile } from "./sdk-fixture.js";

const [sessionPath, phase, markerPath] = process.argv.slice(2);
if (!sessionPath || !phase || !markerPath) {
  throw new Error("usage: input-crash-probe SESSION_PATH PHASE MARKER_PATH");
}

const halt = (): never => {
  appendFileSync(markerPath, `${phase}\n`, { encoding: "utf8", mode: 0o600 });
  process.kill(process.pid, "SIGKILL");
  throw new Error("SIGKILL returned");
};

const manager = SessionManager.open(sessionPath, undefined, "/workspace");
if (phase === "before_custom_append") halt();
manager.appendCustomEntry("assist-input-identity", {
  engineSessionId: "engine-fixture",
  inputSha256: "0".repeat(64),
  runId: "run-fixture",
  workId: "work-fixture",
});
if (phase === "after_custom_append") halt();
fsyncFile(sessionPath);
if (phase === "after_custom_fsync") halt();
manager.appendMessage({
  role: "user",
  content: [{ type: "text", text: "durable fixture input" }],
  timestamp: Date.now(),
});
if (phase === "after_user_append") halt();
fsyncFile(sessionPath);
if (phase === "after_user_fsync") halt();
throw new Error(`unknown crash phase: ${phase}`);
