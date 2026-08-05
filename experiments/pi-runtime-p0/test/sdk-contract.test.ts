import assert from "node:assert/strict";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  createFixture,
  OWNED_PROMPT,
  startScriptedProvider,
  toolCall,
} from "../src/sdk-fixture.js";

test("real Pi SDK preserves prompt ownership and activates a skill for the next iteration", async () => {
  const provider = await startScriptedProvider([
    toolCall(0, "call-load", "load_skill", { name: "fixture" }),
    toolCall(0, "call-secret", "fixture_secret", { value: "next-iteration" }),
    "complete",
  ]);
  const directory = await mkdtemp(join(tmpdir(), "assist-pi-sdk-"));
  const sessionPath = join(directory, "session.jsonl");
  closeSync(openSync(sessionPath, "wx", 0o600));
  const fixture = await createFixture({ baseUrl: provider.baseUrl, sessionPath });
  try {
    assert.deepEqual(fixture.session.getActiveToolNames(), ["load_skill", "fixture_workspace_probe"]);
    await fixture.session.prompt("Run the fixture sequence.", { expandPromptTemplates: false });
    assert.equal(provider.requests.length, 3);
    const payloads = provider.requests as Array<Record<string, unknown>>;
    assertOwnedPayload(payloads[0]!, ["load_skill", "fixture_workspace_probe"]);
    assertOwnedPayload(payloads[1]!, ["load_skill", "fixture_workspace_probe", "fixture_secret"]);
    assertOwnedPayload(payloads[2]!, ["load_skill", "fixture_workspace_probe", "fixture_secret"]);
    assert.deepEqual(fixture.trace.effects, ["load_skill", "fixture_secret:next-iteration"]);

    const assistantMessage = fixture.trace.events.find(
      (event) => event.phase === "message_end" && event.role === "assistant",
    );
    assert.equal(assistantMessage?.leafRole, "user");
    const firstTool = fixture.trace.events.find((event) => event.phase === "tool_call");
    assert.equal(firstTool?.leafRole, "assistant");
    assert.ok(fixture.trace.events.some((event) => event.phase === "activation_committed"));

    const records = (await readFile(sessionPath, "utf8"))
      .trimEnd()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    assert.ok(records.some((entry) => entry.type === "custom" && entry.customType === "assist-skill-activation"));
    assert.equal(records.at(-1)?.type, "message");
  } finally {
    await fixture.session.dispose();
    await provider.close();
    await rm(directory, { recursive: true, force: true });
  }
});

test("fresh-process public Agent continuation preserves prompt ownership and session persistence", async () => {
  const directory = await mkdtemp(join(tmpdir(), "assist-pi-continue-"));
  const sessionPath = join(directory, "session.jsonl");
  closeSync(openSync(sessionPath, "wx", 0o600));
  const unusedProvider = await startScriptedProvider([]);
  const first = await createFixture({ baseUrl: unusedProvider.baseUrl, sessionPath });
  first.sessionManager.appendMessage({
    role: "user",
    content: [{ type: "text", text: "continue this durable input" }],
    timestamp: Date.now(),
  });
  await first.session.dispose();
  await unusedProvider.close();

  const provider = await startScriptedProvider(["continued"]);
  const resumed = await createFixture({ baseUrl: provider.baseUrl, sessionPath });
  try {
    await resumed.session.agent.continue();
    assert.equal(provider.requests.length, 1);
    assertOwnedPayload(
      provider.requests[0] as Record<string, unknown>,
      ["load_skill", "fixture_workspace_probe"],
    );
    const records = (await readFile(sessionPath, "utf8"))
      .trimEnd()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    assert.equal((records.at(-1)?.message as Record<string, unknown>).role, "assistant");
    assert.ok(resumed.trace.events.some((event) => event.phase === "turn_end"));
    assert.equal(resumed.trace.events.some((event) => event.phase === "before_agent_start"), false);
  } finally {
    await resumed.session.dispose();
    await provider.close();
    await rm(directory, { recursive: true, force: true });
  }
});

for (const invalid of [
  { name: "fixture_read", args: {}, label: "malformed arguments" },
  { name: "fixture_missing", args: {}, label: "unknown tool" },
] as const) {
  test(`stock Pi provider converts ${invalid.label} into a recoverable tool result`, async () => {
    const directory = await mkdtemp(join(tmpdir(), "assist-pi-invalid-tool-"));
    const sessionPath = join(directory, "session.jsonl");
    closeSync(openSync(sessionPath, "wx", 0o600));
    const provider = await startScriptedProvider([
      toolCall(0, "call-invalid", invalid.name, invalid.args),
      "recovered",
    ]);
    const fixture = await createFixture({
      baseUrl: provider.baseUrl,
      sessionPath,
      liveTools: true,
    });
    try {
      await fixture.session.prompt("Exercise the invalid tool response.", {
        expandPromptTemplates: false,
      });
      assert.equal(provider.requests.length, 2);
      const records = (await readFile(sessionPath, "utf8")).trimEnd().split("\n")
        .map((line) => JSON.parse(line) as Record<string, unknown>);
      const result = records.find(
        (entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "toolResult",
      )?.message as Record<string, unknown> | undefined;
      assert.equal(result?.isError, true);
      assert.equal(result?.toolCallId, "call-invalid");
    } finally {
      await fixture.session.dispose();
      await provider.close();
      await rm(directory, { recursive: true, force: true });
    }
  });
}

function assertOwnedPayload(payload: Record<string, unknown>, expectedTools: string[]): void {
  const messages = payload.messages as Array<Record<string, unknown>>;
  assert.equal(messages[0]?.role, "system");
  assert.equal(messages[0]?.content, OWNED_PROMPT);
  const tools = (payload.tools as Array<Record<string, unknown>> | undefined) ?? [];
  const names = tools.map((tool) => (tool.function as Record<string, unknown>).name);
  assert.deepEqual(names, expectedTools);
  const serialized = JSON.stringify(payload);
  assert.equal(serialized.includes("Current working directory"), false);
  assert.equal(serialized.includes("expert coding assistant operating inside pi"), false);
}
