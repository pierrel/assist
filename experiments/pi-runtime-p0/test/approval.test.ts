import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import { fsyncFile, startScriptedProvider, toolCall } from "../src/sdk-fixture.js";

interface ChildResult {
  readonly code: number | null;
  readonly signal: NodeJS.Signals | null;
  readonly stdout: string;
  readonly stderr: string;
}

test("actual Pi multi-tool approval appends ordered rejections and continues once", async () => {
  await runApprovalScenario({
    calls: [
      toolCall(0, "call-gated-1", "load_skill", { name: "fixture" }),
      toolCall(1, "call-gated-2", "load_skill", { name: "fixture" }),
    ],
    decisions: [
      { id: "call-gated-1", name: "load_skill", isError: true, text: "approval required" },
      { id: "call-gated-2", name: "load_skill", isError: true, text: "approval required" },
    ],
    expectedTools: ["load_skill", "fixture_workspace_probe"],
  });
});

test("a mixed gated and permitted batch is deferred whole without effects", async () => {
  await runApprovalScenario({
    preactivate: true,
    calls: [
      toolCall(0, "call-gated", "load_skill", { name: "fixture" }),
      toolCall(1, "call-permitted", "fixture_secret", { value: "must-not-run" }),
    ],
    decisions: [
      { id: "call-gated", name: "load_skill", isError: true, text: "whole batch deferred" },
      { id: "call-permitted", name: "fixture_secret", isError: true, text: "whole batch deferred" },
    ],
    expectedTools: ["load_skill", "fixture_workspace_probe", "fixture_secret"],
  });
});

test("an approved load_skill decision durably activates its schema before one continuation", async () => {
  await runApprovalScenario({
    calls: [toolCall(0, "call-approved", "load_skill", { name: "fixture" })],
    decisions: [
      {
        id: "call-approved",
        name: "load_skill",
        isError: false,
        text: "approved fixture activation",
        activateFixture: true,
      },
    ],
    expectedTools: ["load_skill", "fixture_workspace_probe", "fixture_secret"],
  });
});

async function runApprovalScenario(options: {
  readonly calls: readonly object[];
  readonly decisions: readonly Record<string, unknown>[];
  readonly expectedTools: readonly string[];
  readonly preactivate?: boolean;
}): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "assist-pi-approval-"));
  const sessionPath = join(directory, "session.jsonl");
  const markerPath = join(directory, "halt.log");
  const effectMarkerPath = join(directory, "effects.log");
  const decisionsPath = join(directory, "decisions.json");
  closeSync(openSync(sessionPath, "wx", 0o600));
  if (options.preactivate) {
    const writer = SessionManager.open(sessionPath, undefined, "/workspace");
    writer.appendCustomEntry("assist-skill-activation", {
      name: "fixture",
      tools: ["fixture_secret"],
      version: 1,
    });
    fsyncFile(sessionPath);
  }
  await writeFile(decisionsPath, `${JSON.stringify(options.decisions)}\n`, { mode: 0o600 });
  const proposalProvider = await startScriptedProvider([options.calls]);
  try {
    const proposed = await runNode("fatal-probe.js", [
      proposalProvider.baseUrl,
      sessionPath,
      "tool_call",
      markerPath,
      effectMarkerPath,
    ]);
    assert.equal(proposed.code, null, proposed.stderr);
    assert.equal(proposed.signal, "SIGKILL");
    assert.equal(proposalProvider.requests.length, 1);
    assert.equal(await readFile(effectMarkerPath, "utf8").catch(() => ""), "");
  } finally {
    await proposalProvider.close();
  }

  const continuationProvider = await startScriptedProvider(["approval handled"]);
  try {
    const resumed = await runNode("resume-probe.js", [
      continuationProvider.baseUrl,
      sessionPath,
      decisionsPath,
    ]);
    assert.equal(resumed.code, 0, resumed.stderr);
    assert.equal(resumed.signal, null);
    const result = JSON.parse(resumed.stdout) as Record<string, unknown>;
    assert.deepEqual(result.activeTools, options.expectedTools);
    assert.deepEqual(result.effects, []);
    assert.equal(continuationProvider.requests.length, 1);
    const payload = continuationProvider.requests[0] as Record<string, unknown>;
    const messages = payload.messages as Array<Record<string, unknown>>;
    const results = messages.filter((message) => message.role === "tool");
    assert.deepEqual(
      results.map((message) => message.tool_call_id),
      options.decisions.map((decision) => decision.id),
    );
    const tools = ((payload.tools as Array<Record<string, unknown>> | undefined) ?? [])
      .map((tool) => (tool.function as Record<string, unknown>).name);
    assert.deepEqual(tools, options.expectedTools);
    const records = (await readFile(sessionPath, "utf8")).trimEnd().split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    const persistedResults = records.filter(
      (entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "toolResult",
    );
    assert.deepEqual(
      persistedResults.map((entry) => (entry.message as Record<string, unknown>).toolCallId),
      options.decisions.map((decision) => decision.id),
    );
  } finally {
    await continuationProvider.close();
    await rm(directory, { recursive: true, force: true });
  }
}

async function runNode(script: string, args: readonly string[]): Promise<ChildResult> {
  const child = spawn(process.execPath, [
    new URL(`../src/${script}`, import.meta.url).pathname,
    ...args,
  ], { stdio: ["ignore", "pipe", "pipe"] });
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  return await new Promise<ChildResult>((resolve, reject) => {
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
