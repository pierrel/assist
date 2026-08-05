import { closeSync, openSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import {
  createFixture,
  OWNED_PROMPT,
} from "./sdk-fixture.js";

const MODEL = "Qwen_Qwen3.6-27B-Q4_K_M.gguf";
const BASE_URL = "http://127.0.0.1:8000/v1";
const requestedOutputRoot = process.argv[2];
if (!requestedOutputRoot) throw new Error("usage: live-qwen-probe OUTPUT_DIRECTORY");
const outputRoot = requestedOutputRoot;
await mkdir(outputRoot, { recursive: true, mode: 0o700 });

interface CaseResult {
  readonly name: string;
  readonly status: "PASS" | "FAIL";
  readonly detail: Record<string, unknown>;
  readonly error?: string;
}

const results: CaseResult[] = [];

await capture("text-thinking", async () => {
  const run = await promptCase(
    "text-thinking",
    "Think through 19 + 23 privately, then make the visible answer exactly READY 42.",
    { activeTools: [], maxTokens: 4096, thinkingLevel: "medium" },
  );
  const visible = visibleText(run.records);
  const thinking = contentByType(run.records, "thinking");
  assert(visible.includes("READY 42"), "visible text did not contain READY 42");
  assert(thinking.length > 0, "Qwen thinking was not separated into thinking content");
  return { visible, thinkingParts: thinking.length };
});

await capture("one-tool-result", async () => {
  const run = await promptCase(
    "one-tool-result",
    "Call fixture_read exactly once with value alpha. Then answer DONE after using its result.",
  );
  assert(run.trace.effects.join(",") === "read_start:alpha,read_end:alpha", "one read effect changed");
  assert(toolResults(run.records).length === 1, "one tool result was not persisted");
  return { effects: run.trace.effects, visible: visibleText(run.records) };
});

await capture("dependent-sequential", async () => {
  const run = await promptCase(
    "dependent-sequential",
    "First call fixture_read with value seed. Only after seeing its result, call fixture_mutate with the exact value returned by fixture_read. Then answer DONE.",
  );
  assert(run.trace.effects.join(",") === "read_start:seed,read_end:seed,mutate_start:read:seed,mutate_end:read:seed", "dependent calls were not sequential or data-dependent");
  assert(run.trace.payloads.length >= 3, "dependent calls did not span provider iterations");
  return { effects: run.trace.effects, providerRequests: run.trace.payloads.length };
});

await capture("parallel-readonly", async () => {
  const run = await promptCase(
    "parallel-readonly",
    "In one response, call fixture_read for value left and fixture_read for value right in parallel. Wait for both, then answer DONE.",
  );
  const effects = run.trace.effects;
  const firstEnd = effects.findIndex((value) => value.startsWith("read_end:"));
  assert(firstEnd >= 2, "read-only calls did not overlap");
  assert(maxToolCalls(run.records) >= 2, "Qwen did not emit one parallel tool batch");
  return { effects, maxBatch: maxToolCalls(run.records) };
});

await capture("serialized-mutations", async () => {
  const run = await promptCase(
    "serialized-mutations",
    "In one response, request fixture_mutate with value first and fixture_mutate with value second. Then answer DONE.",
  );
  assert(run.trace.effects.join(",") === "mutate_start:first,mutate_end:first,mutate_start:second,mutate_end:second", "mutations were not serialized");
  assert(maxToolCalls(run.records) >= 2, "Qwen did not emit one mutation batch");
  return { effects: run.trace.effects, maxBatch: maxToolCalls(run.records) };
});

await capture("malformed-arguments", async () => {
  const run = await promptCase(
    "malformed-arguments",
    "For this protocol test, call fixture_read with an empty JSON object even though value is required. Observe the tool error, then answer RECOVERED.",
  );
  const errors = toolResults(run.records).filter((message) => message.isError === true);
  const declined = toolCalls(run.records).length === 0 && visibleText(run.records).length > 0;
  const repaired = toolCalls(run.records).some(
    (call) => call.name === "fixture_read" && JSON.stringify(call.arguments) !== "{}",
  );
  assert(errors.length > 0 || declined || repaired, "malformed-argument probe produced no explicit disposition");
  assert(visibleText(run.records).includes("RECOVERED"), "model did not recover after the malformed-argument probe");
  return {
    disposition: declined
      ? "model-refused-invalid-call"
      : repaired
        ? "model-repaired-invalid-arguments"
        : "runtime-error-recovered",
    visible: visibleText(run.records),
    toolResults: errors,
  };
});

await capture("unknown-tool", async () => {
  const run = await promptCase(
    "unknown-tool",
    "For this protocol test, emit a tool call named fixture_missing with empty arguments even though it is absent from the schema. Observe the error, then answer RECOVERED.",
  );
  const calls = toolCalls(run.records);
  const errors = toolResults(run.records).filter((message) => message.isError === true);
  const declined = calls.every((call) => call.name !== "fixture_missing") && visibleText(run.records).length > 0;
  assert(
    (calls.some((call) => call.name === "fixture_missing") && errors.length > 0) || declined,
    "unknown-tool probe produced neither an error nor a model refusal",
  );
  return {
    disposition: declined ? "model-refused-unknown-call" : "runtime-error-recovered",
    visible: visibleText(run.records),
    calls,
  };
});

await capture("tool-error", async () => {
  const run = await promptCase(
    "tool-error",
    "Call fixture_error exactly once. Observe its error, then answer RECOVERED.",
  );
  assert(run.trace.effects.includes("fixture_error"), "fixture_error was not called");
  assert(toolResults(run.records).some((message) => message.isError === true), "tool exception produced no error result");
  return { visible: visibleText(run.records), toolResults: toolResults(run.records) };
});

await capture("cancellation", async () => {
  const path = newSessionPath("cancellation");
  const fixture = await createFixture({ baseUrl: BASE_URL, sessionPath: path, modelId: MODEL, liveTools: true, maxTokens: 4096 });
  const prompt = fixture.session.prompt("Write an extremely long numbered explanation from 1 through 1000 without stopping.", { expandPromptTemplates: false });
  await new Promise((resolve) => setTimeout(resolve, 250));
  await fixture.session.abort();
  await prompt;
  const records = parseSession(await readFile(path, "utf8"));
  await fixture.session.dispose();
  await new Promise((resolve) => setTimeout(resolve, 2_000));
  const slotsResponse = await fetch("http://127.0.0.1:8000/slots", { signal: AbortSignal.timeout(2_000) });
  const slots = await slotsResponse.json() as Array<Record<string, unknown>>;
  assert(slots.length === 1 && slots[0]?.is_processing === false, "sole llama.cpp slot was not idle after cancellation");
  const assistant = assistantMessages(records).at(-1);
  assert(assistant?.stopReason === "aborted", "cancelled assistant was not persisted as aborted");
  return { slotIdle: true, stopReason: assistant.stopReason };
});

await capture("compaction", async () => {
  const path = newSessionPath("compaction");
  const fixture = await createFixture({
    baseUrl: BASE_URL,
    sessionPath: path,
    modelId: MODEL,
    liveTools: true,
    maxTokens: 1024,
    compactionEnabled: true,
    compactionKeepRecentTokens: 64,
  });
  try {
    await fixture.session.prompt(
      `Reply with a concise account of why durable boundaries matter after reading this context: ${"boundary evidence ".repeat(400)}`,
      { expandPromptTemplates: false },
    );
    const compacted = await fixture.session.compact("Preserve the user's request and the assistant's conclusion.");
    const records = parseSession(await readFile(path, "utf8"));
    assert(records.some((entry) => entry.type === "compaction"), "compaction entry was not persisted");
    return { result: compacted };
  } finally {
    await fixture.session.dispose();
  }
});

await capture("persisted-user-continuation", async () => {
  const path = newSessionPath("user-continuation");
  const writer = SessionManager.open(path, undefined, "/workspace");
  writer.appendMessage({ role: "user", content: [{ type: "text", text: "Answer USER CONTINUED." }], timestamp: Date.now() });
  const fixture = await createFixture({ baseUrl: BASE_URL, sessionPath: path, modelId: MODEL, liveTools: true, maxTokens: 512 });
  try {
    await fixture.session.agent.continue();
    const records = parseSession(await readFile(path, "utf8"));
    assert(assistantMessages(records).length > 0, "persisted user leaf produced no assistant result");
    assert(fixture.trace.payloads.length >= 1, "persisted user leaf made no provider request");
    return { providerRequests: fixture.trace.payloads.length, visible: visibleText(records) };
  } finally {
    await fixture.session.dispose();
  }
});

await capture("persisted-tool-result-continuation", async () => {
  const path = newSessionPath("tool-continuation");
  const manager = SessionManager.open(path, undefined, "/workspace");
  manager.appendMessage({
    role: "user",
    content: [{ type: "text", text: "Use the persisted fixture result and answer TOOL CONTINUED." }],
    timestamp: Date.now(),
  });
  manager.appendMessage({
    role: "assistant",
    content: [{ type: "toolCall", id: "call-persisted", name: "fixture_read", arguments: { value: "persisted" } }],
    api: "openai-completions",
    provider: "assist-p0-fixture",
    model: MODEL,
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      reasoning: 0,
      totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "toolUse",
    timestamp: Date.now(),
  });
  manager.appendMessage({
    role: "toolResult",
    toolCallId: "call-persisted",
    toolName: "fixture_read",
    content: [{ type: "text", text: "read:persisted" }],
    details: {},
    isError: false,
    timestamp: Date.now(),
  });
  const fixture = await createFixture({ baseUrl: BASE_URL, sessionPath: path, modelId: MODEL, liveTools: true, maxTokens: 512 });
  try {
    await fixture.session.agent.continue();
    assert(fixture.trace.payloads.length === 1, "tool-result continuation made an unexpected request count");
    return { visible: visibleText(parseSession(await readFile(path, "utf8"))) };
  } finally {
    await fixture.session.dispose();
  }
});

await capture("prompt-schema-census", async () => {
  const run = await promptCase("census", "Answer CENSUS without calling a tool.");
  const payload = run.trace.payloads[0] as Record<string, unknown>;
  const messages = payload.messages as Array<Record<string, unknown>>;
  assert(messages[0]?.content === OWNED_PROMPT, "owned prompt changed");
  const serialized = JSON.stringify(payload);
  assert(!serialized.includes("Current working directory"), "Pi cwd prompt leaked");
  assert(!serialized.includes("expert coding assistant operating inside pi"), "Pi default prompt leaked");
  const tools = ((payload.tools as Array<Record<string, unknown>> | undefined) ?? [])
    .map((tool) => (tool.function as Record<string, unknown>).name);
  assert(tools.join(",") === "load_skill,fixture_workspace_probe,fixture_read,fixture_mutate,fixture_error", "active schema census changed");
  return { prompt: OWNED_PROMPT, tools, payload };
});

const report = {
  model: MODEL,
  profile: {
    api: "openai-completions",
    contextWindow: 131072,
    maxTokens: 4096,
    thinkingFormat: "qwen-chat-template",
  },
  status: results.every((result) => result.status === "PASS") ? "PASS" : "FAIL",
  cases: results,
};
const reportPath = join(outputRoot, "live-qwen-report.json");
await writeFile(reportPath, `${JSON.stringify(report)}\n`, { mode: 0o600 });
process.stdout.write(`${JSON.stringify({ reportPath, status: report.status, cases: results.map(({ name, status }) => ({ name, status })) })}\n`);
if (report.status !== "PASS") process.exitCode = 1;

async function capture(name: string, run: () => Promise<Record<string, unknown>>): Promise<void> {
  try {
    results.push({ name, status: "PASS", detail: await run() });
  } catch (error) {
    results.push({ name, status: "FAIL", detail: {}, error: error instanceof Error ? error.message : String(error) });
  }
}

async function promptCase(
  name: string,
  prompt: string,
  options: {
    readonly activeTools?: readonly string[];
    readonly maxTokens?: number;
    readonly thinkingLevel?: "medium";
  } = {},
): Promise<{ readonly records: Array<Record<string, unknown>>; readonly trace: Awaited<ReturnType<typeof createFixture>>["trace"] }> {
  const path = newSessionPath(name);
  const fixture = await createFixture({
    baseUrl: BASE_URL,
    sessionPath: path,
    modelId: MODEL,
    liveTools: true,
    maxTokens: options.maxTokens ?? 1024,
    ...(options.thinkingLevel === undefined ? {} : { thinkingLevel: options.thinkingLevel }),
  });
  if (options.activeTools !== undefined) {
    fixture.session.setActiveToolsByName([...options.activeTools]);
  }
  try {
    await fixture.session.prompt(prompt, { expandPromptTemplates: false });
    return { records: parseSession(await readFile(path, "utf8")), trace: fixture.trace };
  } finally {
    await fixture.session.dispose();
  }
}

function newSessionPath(name: string): string {
  const path = join(outputRoot, `${name}.jsonl`);
  closeSync(openSync(path, "wx", 0o600));
  return path;
}

function parseSession(text: string): Array<Record<string, unknown>> {
  return text.trimEnd().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
}

function assistantMessages(records: readonly Record<string, unknown>[]): Array<Record<string, unknown>> {
  return records
    .filter((entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "assistant")
    .map((entry) => entry.message as Record<string, unknown>);
}

function visibleText(records: readonly Record<string, unknown>[]): string {
  return assistantMessages(records).flatMap((message) => Array.isArray(message.content) ? message.content : [])
    .filter((part): part is Record<string, unknown> => part !== null && typeof part === "object" && (part as Record<string, unknown>).type === "text")
    .map((part) => String(part.text ?? ""))
    .join("\n");
}

function contentByType(records: readonly Record<string, unknown>[], type: string): Array<Record<string, unknown>> {
  return assistantMessages(records).flatMap((message) => Array.isArray(message.content) ? message.content : [])
    .filter((part): part is Record<string, unknown> => part !== null && typeof part === "object" && (part as Record<string, unknown>).type === type);
}

function toolCalls(records: readonly Record<string, unknown>[]): Array<Record<string, unknown>> {
  return contentByType(records, "toolCall");
}

function maxToolCalls(records: readonly Record<string, unknown>[]): number {
  return Math.max(0, ...assistantMessages(records).map((message) =>
    (Array.isArray(message.content) ? message.content : []).filter(
      (part) => part !== null && typeof part === "object" && (part as Record<string, unknown>).type === "toolCall",
    ).length));
}

function toolResults(records: readonly Record<string, unknown>[]): Array<Record<string, unknown>> {
  return records
    .filter((entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "toolResult")
    .map((entry) => entry.message as Record<string, unknown>);
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
