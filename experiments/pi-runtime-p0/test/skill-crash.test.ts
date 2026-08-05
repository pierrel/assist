import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { startScriptedProvider, toolCall } from "../src/sdk-fixture.js";

const cases = [
  { phase: "custom_tool_return", activation: false, result: false },
  { phase: "activation_after_append", activation: true, result: false },
  { phase: "activation_after_fsync", activation: true, result: false },
  { phase: "activation_after_active_set", activation: true, result: false },
  { phase: "tool_result", activation: true, result: false },
  { phase: "before_provider_request:next", activation: true, result: true },
] as const;

for (const crash of cases) {
  test(`skill activation recovers exactly at ${crash.phase}`, async () => {
    const directory = await mkdtemp(join(tmpdir(), "assist-pi-skill-crash-"));
    const sessionPath = join(directory, "session.jsonl");
    const haltPath = join(directory, "halt.log");
    const effectPath = join(directory, "effects.log");
    const decisionsPath = join(directory, "decisions.json");
    closeSync(openSync(sessionPath, "wx", 0o600));
    const provider = await startScriptedProvider([
      toolCall(0, "call-load", "load_skill", { name: "fixture" }),
      "must not complete after crash",
    ]);
    try {
      const killed = await runNode("fatal-probe.js", [
        provider.baseUrl,
        sessionPath,
        crash.phase,
        haltPath,
        effectPath,
      ]);
      assert.equal(killed.signal, "SIGKILL", killed.stderr);
      assert.equal(await readFile(effectPath, "utf8"), "load_skill\n");
      assert.equal(provider.requests.length, 1);
    } finally {
      await provider.close();
    }

    const before = parseSession(await readFile(sessionPath, "utf8"));
    assert.equal(hasActivation(before), crash.activation);
    assert.equal(toolResults(before).length > 0, crash.result);
    const decisions = crash.result ? [] : [{
      id: "call-load",
      name: "load_skill",
      isError: !crash.activation,
      text: crash.activation ? "recovered committed activation" : "load interrupted before commit",
    }];
    await writeFile(decisionsPath, `${JSON.stringify(decisions)}\n`, { mode: 0o600 });
    const recoveryProvider = await startScriptedProvider(["recovered"]);
    try {
      const recovered = await runNode("resume-probe.js", [
        recoveryProvider.baseUrl,
        sessionPath,
        decisionsPath,
      ]);
      assert.equal(recovered.code, 0, recovered.stderr);
      const result = JSON.parse(recovered.stdout) as Record<string, unknown>;
      assert.deepEqual(
        result.activeTools,
        crash.activation
          ? ["load_skill", "fixture_workspace_probe", "fixture_secret"]
          : ["load_skill", "fixture_workspace_probe"],
      );
      assert.deepEqual(result.effects, []);
      assert.equal(recoveryProvider.requests.length, 1);
      assert.equal(await readFile(effectPath, "utf8"), "load_skill\n");
      const after = parseSession(await readFile(sessionPath, "utf8"));
      assert.equal(toolResults(after).length, 1);
    } finally {
      await recoveryProvider.close();
      await rm(directory, { recursive: true, force: true });
    }
  });
}

function parseSession(text: string): Array<Record<string, unknown>> {
  return text.trimEnd().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
}

function hasActivation(records: readonly Record<string, unknown>[]): boolean {
  return records.some((entry) => entry.type === "custom" && entry.customType === "assist-skill-activation");
}

function toolResults(records: readonly Record<string, unknown>[]): Array<Record<string, unknown>> {
  return records.filter(
    (entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "toolResult",
  );
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
