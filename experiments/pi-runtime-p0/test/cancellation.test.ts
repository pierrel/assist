import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo, Socket } from "node:net";
import { closeSync, openSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { createFixture, startScriptedProvider } from "../src/sdk-fixture.js";

test("cancel during streaming closes the old provider generation before a fresh process resumes", async () => {
  const directory = await mkdtemp(join(tmpdir(), "assist-pi-cancel-"));
  const sessionPath = join(directory, "session.jsonl");
  closeSync(openSync(sessionPath, "wx", 0o600));
  let requestStarted!: () => void;
  const started = new Promise<void>((resolve) => { requestStarted = resolve; });
  let providerClosed!: () => void;
  const closed = new Promise<void>((resolve) => { providerClosed = resolve; });
  const sockets = new Set<Socket>();
  const server = createServer(async (request, response) => {
    for await (const _chunk of request) { /* drain the exact request */ }
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write(`data: ${JSON.stringify({
      id: "chatcmpl-cancel",
      object: "chat.completion.chunk",
      created: 1,
      model: "fixture-model",
      choices: [{ index: 0, delta: { role: "assistant", content: "partial" }, finish_reason: null }],
    })}\n\n`);
    requestStarted();
    response.once("close", providerClosed);
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address() as AddressInfo;
  const first = await createFixture({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    sessionPath,
  });
  try {
    const prompt = first.session.prompt("Begin the cancellable stream.", {
      expandPromptTemplates: false,
    });
    await started;
    await first.session.abort();
    await prompt;
    await Promise.race([
      closed,
      new Promise<never>((_resolve, reject) => {
        setTimeout(() => reject(new Error("provider socket did not close after cancellation")), 2_000);
      }),
    ]);
    const records = parseSession(await readFile(sessionPath, "utf8"));
    const assistants = records.filter(
      (entry) => entry.type === "message" && (entry.message as Record<string, unknown>).role === "assistant",
    );
    assert.equal((assistants.at(-1)?.message as Record<string, unknown>).stopReason, "aborted");
  } finally {
    await first.session.dispose();
    for (const socket of sockets) socket.destroy();
    await new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    });
  }

  const resumedProvider = await startScriptedProvider(["fresh generation"]);
  const resumed = await createFixture({ baseUrl: resumedProvider.baseUrl, sessionPath });
  try {
    await resumed.session.prompt("Continue after the cancelled generation.", {
      expandPromptTemplates: false,
    });
    assert.equal(resumedProvider.requests.length, 1);
  } finally {
    await resumed.session.dispose();
    await resumedProvider.close();
    await rm(directory, { recursive: true, force: true });
  }
});

function parseSession(text: string): Array<Record<string, unknown>> {
  return text.trimEnd().split("\n").map((line) => JSON.parse(line) as Record<string, unknown>);
}
