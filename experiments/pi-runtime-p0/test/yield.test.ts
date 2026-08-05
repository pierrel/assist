import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { startScriptedProvider, toolCall } from "../src/sdk-fixture.js";

test("yield completes the current tool boundary, emits paused, exits, and resumes fresh", async () => {
  const directory = await mkdtemp(join(tmpdir(), "assist-pi-yield-"));
  const sessionPath = join(directory, "session.jsonl");
  const markerPath = join(directory, "paused.json");
  const decisionsPath = join(directory, "decisions.json");
  closeSync(openSync(sessionPath, "wx", 0o600));
  await writeFile(decisionsPath, "[]\n", { mode: 0o600 });
  const firstProvider = await startScriptedProvider([
    toolCall(0, "call-load", "load_skill", { name: "fixture" }),
    "must not cross the yield boundary",
  ]);
  try {
    const paused = await runNode("yield-probe.js", [firstProvider.baseUrl, sessionPath, markerPath]);
    assert.equal(paused.code, 75, paused.stderr);
    assert.equal(paused.signal, null);
    assert.equal(firstProvider.requests.length, 1);
    const marker = JSON.parse(await readFile(markerPath, "utf8")) as Record<string, unknown>;
    assert.equal(marker.state, "paused");
    assert.match(String(marker.reason), /before_provider_request:next/);
    assert.ok(Number(marker.activeNs) > 0);
    const records = parseSession(await readFile(sessionPath, "utf8"));
    assert.equal(records.some(
      (entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "toolResult",
    ), true);
  } finally {
    await firstProvider.close();
  }

  const continuationProvider = await startScriptedProvider(["resumed once"]);
  try {
    const resumed = await runNode("resume-probe.js", [
      continuationProvider.baseUrl,
      sessionPath,
      decisionsPath,
    ]);
    assert.equal(resumed.code, 0, resumed.stderr);
    assert.equal(continuationProvider.requests.length, 1);
    const result = JSON.parse(resumed.stdout) as Record<string, unknown>;
    assert.deepEqual(
      result.activeTools,
      ["load_skill", "fixture_workspace_probe", "fixture_secret"],
    );
    assert.deepEqual(result.effects, []);
  } finally {
    await continuationProvider.close();
    await rm(directory, { recursive: true, force: true });
  }
});

function parseSession(text: string): Array<Record<string, unknown>> {
  return text.trimEnd().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
}

async function runNode(script: string, args: readonly string[]): Promise<{
  readonly code: number | null;
  readonly signal: NodeJS.Signals | null;
  readonly stdout: string;
  readonly stderr: string;
}> {
  const child = spawn(process.execPath, [new URL(`../src/${script}`, import.meta.url).pathname, ...args], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  return await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`child timed out: ${script}`));
    }, 10_000);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      resolve({
        code,
        signal,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8"),
      });
    });
  });
}
