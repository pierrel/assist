import { readFileSync } from "node:fs";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import { createFixture, fsyncFile } from "./sdk-fixture.js";

interface Decision {
  readonly id: string;
  readonly name: string;
  readonly isError: boolean;
  readonly text: string;
  readonly activateFixture?: boolean;
}

const [baseUrl, sessionPath, decisionsPath] = process.argv.slice(2);
if (!baseUrl || !sessionPath || !decisionsPath) {
  throw new Error("usage: resume-probe BASE_URL SESSION_PATH DECISIONS_PATH");
}

const parsed: unknown = JSON.parse(readFileSync(decisionsPath, "utf8"));
if (!Array.isArray(parsed)) {
  throw new Error("decisions must be an array");
}
const decisions = parsed.map(validateDecision);

if (decisions.length > 0) {
  const writer = SessionManager.open(sessionPath, undefined, "/workspace");
  const branch = writer.getBranch();
  let assistant: (typeof branch)[number] | undefined;
  for (let index = branch.length - 1; index >= 0; index -= 1) {
    const entry = branch[index];
    if (entry?.type === "message" && entry.message.role === "assistant") {
      assistant = entry;
      break;
    }
  }
  if (assistant?.type !== "message" || assistant.message.role !== "assistant") {
    throw new Error("approval recovery requires an assistant proposal");
  }
  const proposals = assistant.message.content.filter(
    (part): part is Extract<(typeof assistant.message.content)[number], { type: "toolCall" }> =>
      part.type === "toolCall",
  );
  if (proposals.length !== decisions.length) {
    throw new Error("decision count does not match the assistant tool batch");
  }
  for (const [index, decision] of decisions.entries()) {
    const proposal = proposals[index];
    if (proposal === undefined || proposal.id !== decision.id || proposal.name !== decision.name) {
      throw new Error(`decision ${index} does not match the ordered tool proposal`);
    }
    if (decision.activateFixture) {
      if (decision.name !== "load_skill" || decision.isError) {
        throw new Error("fixture activation requires an approved load_skill decision");
      }
      writer.appendCustomEntry("assist-skill-activation", {
        name: "fixture",
        tools: ["fixture_secret"],
        version: 1,
      });
      fsyncFile(sessionPath);
    }
    writer.appendMessage({
      role: "toolResult",
      toolCallId: decision.id,
      toolName: decision.name,
      content: [{ type: "text", text: decision.text }],
      details: { approvalDecision: true },
      isError: decision.isError,
      timestamp: Date.now(),
    });
    fsyncFile(sessionPath);
  }
}

const fixture = await createFixture({ baseUrl, sessionPath });
try {
  await fixture.session.agent.continue();
  process.stdout.write(`${JSON.stringify({
    activeTools: fixture.session.getActiveToolNames(),
    effects: fixture.trace.effects,
    events: fixture.trace.events,
    payloads: fixture.trace.payloads,
  })}\n`);
} finally {
  await fixture.session.dispose();
}

function validateDecision(value: unknown, index: number): Decision {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`decision ${index} is not an object`);
  }
  const record = value as Record<string, unknown>;
  const allowed = new Set(["id", "name", "isError", "text", "activateFixture"]);
  if (Object.keys(record).some((key) => !allowed.has(key))) {
    throw new Error(`decision ${index} has unknown fields`);
  }
  if (
    typeof record.id !== "string"
    || typeof record.name !== "string"
    || typeof record.isError !== "boolean"
    || typeof record.text !== "string"
    || (record.activateFixture !== undefined && typeof record.activateFixture !== "boolean")
  ) {
    throw new Error(`decision ${index} is malformed`);
  }
  return {
    id: record.id,
    name: record.name,
    isError: record.isError,
    text: record.text,
    ...(record.activateFixture === undefined ? {} : { activateFixture: record.activateFixture }),
  };
}
