import { spawn } from "node:child_process";
import { openSync, writeSync } from "node:fs";
import { createConnection, createServer, type Server, type Socket } from "node:net";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  type ExtensionAPI,
  type ExtensionFactory,
} from "@earendil-works/pi-coding-agent";
const CONTROL = "/run/assist-p0/control.sock";
const FIFO = "/run/assist-p0/liveness.fifo";
const SESSION = "/session/session.jsonl";
const PROMPT = "ASSIST_P0_PROMPT_OWNERSHIP_CANARY_v1";
const USER = "Return the fixed deterministic response.";
const MAX_FRAME = 512 * 1024;
const OPERATION_MS = 12_000;
type Frame = Record<string, unknown>;
function within<T>(promise: Promise<T>, label: string, timeout = OPERATION_MS): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out`)), timeout);
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (error: unknown) => { clearTimeout(timer); reject(error); },
    );
  });
}
function exact(frame: Frame, expected: Frame, label: string): void {
  const actualKeys = Object.keys(frame).sort();
  const expectedKeys = Object.keys(expected).sort();
  if (JSON.stringify(actualKeys) !== JSON.stringify(expectedKeys)) throw new Error(`${label} fields changed`);
  for (const [key, value] of Object.entries(expected)) {
    if (frame[key] !== value) throw new Error(`${label}.${key} changed`);
  }
}
class Channel {
  readonly socket: Socket;
  #buffer = Buffer.alloc(0);
  #frames: Frame[] = [];
  #waiters: Array<{ resolve(value: Frame): void; reject(error: Error): void }> = [];
  constructor(socket: Socket) {
    this.socket = socket;
    socket.on("data", (chunk: Buffer) => this.#receive(chunk));
    socket.on("error", (error) => this.#fail(error));
    socket.on("close", () => this.#fail(new Error("control socket closed")));
  }
  async write(frame: Frame): Promise<void> {
    const data = Buffer.from(`${JSON.stringify(frame)}\n`);
    if (data.length > MAX_FRAME) throw new Error("outbound frame too large");
    await within(new Promise<void>((resolve, reject) => {
      this.socket.write(data, (error) => error ? reject(error) : resolve());
    }), "control write");
  }
  read(): Promise<Frame> {
    const value = this.#frames.shift();
    if (value !== undefined) return Promise.resolve(value);
    return within(new Promise((resolve, reject) => this.#waiters.push({ resolve, reject })), "control read");
  }
  fatal(): never {
    const frame = Buffer.alloc(256, 0x20);
    frame.write('{"reason":"before_provider_request","type":"fatal"}');
    frame[255] = 0x0a;
    try { this.socket.write(frame); } finally { process.kill(process.pid, "SIGKILL"); }
    throw new Error("SIGKILL returned");
  }
  #receive(chunk: Buffer): void {
    this.#buffer = Buffer.concat([this.#buffer, chunk]);
    if (this.#buffer.length > MAX_FRAME) return this.#fail(new Error("inbound frame too large"));
    for (;;) {
      const newline = this.#buffer.indexOf(0x0a);
      if (newline < 0) return;
      const raw = this.#buffer.subarray(0, newline);
      this.#buffer = this.#buffer.subarray(newline + 1);
      let value: unknown;
      try { value = JSON.parse(raw.toString("utf8")); } catch { return this.#fail(new Error("invalid frame")); }
      if (value === null || typeof value !== "object" || Array.isArray(value)) {
        return this.#fail(new Error("frame is not an object"));
      }
      const waiter = this.#waiters.shift();
      if (waiter === undefined) this.#frames.push(value as Frame); else waiter.resolve(value as Frame);
    }
  }
  #fail(error: Error): void {
    for (const waiter of this.#waiters.splice(0)) waiter.reject(error);
  }
}
async function connect(): Promise<Channel> {
  const socket = createConnection(CONTROL);
  await within(new Promise<void>((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  }), "control connect");
  return new Channel(socket);
}
async function startBridge(
  channel: Channel,
  abort: () => Promise<void>,
): Promise<{ baseUrl: string; server: Server; completion: Promise<void>; counts(): [number, number] }> {
  let connections = 0;
  let rejected = 0;
  let resolveCompletion!: () => void;
  let rejectCompletion!: (error: Error) => void;
  const completion = new Promise<void>((resolve, reject) => { resolveCompletion = resolve; rejectCompletion = reject; });
  const server = createServer((client) => {
    client.on("error", () => {});
    connections += 1;
    if (connections !== 1) {
      rejected += 1;
      client.end("HTTP/1.1 409 Conflict\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
      return;
    }
    void serveProvider(client, channel, abort).then(resolveCompletion, (error: unknown) => {
      client.destroy(error as Error); rejectCompletion(error as Error);
    });
  });
  await within(new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  }), "bridge listen");
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("bridge has no port");
  return { baseUrl: `http://127.0.0.1:${address.port}/v1`, server, completion, counts: () => [connections, rejected] };
}
async function serveProvider(client: Socket, channel: Channel, abort: () => Promise<void>): Promise<void> {
  const raw = await readRawRequest(client);
  await channel.write({ request: 1, type: "provider_request", raw: raw.toString("base64") });
  for (;;) {
    const frame = await channel.read();
    if (frame.type === "cancel") {
      exact(frame, { request: 1, type: "cancel" }, "cancel");
      const closed = within(new Promise<void>((resolve) => client.once("close", resolve)), "provider close");
      await within(abort(), "Pi abort");
      await closed;
      await channel.write({ request: 1, type: "client_closed" });
      exact(await channel.read(), { request: 1, type: "cancel_ack" }, "cancel acknowledgement");
      return;
    }
    if (frame.type === "response_start") {
      exact(frame, { request: 1, type: "response_start" }, "response start");
      client.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
      continue;
    }
    if (frame.type === "response_chunk" && typeof frame.data === "string") {
      exact(frame, { data: frame.data, request: 1, type: "response_chunk" }, "response chunk");
      await within(new Promise<void>((resolve, reject) => {
        client.write(Buffer.from(frame.data as string, "base64"), (error) => error ? reject(error) : resolve());
      }), "provider response write");
      continue;
    }
    if (frame.type === "response_end") {
      exact(frame, { request: 1, type: "response_end" }, "response end");
      await within(new Promise<void>((resolve) => client.end(resolve)), "provider response end");
      return;
    }
    throw new Error("unexpected provider frame");
  }
}
async function readRawRequest(client: Socket): Promise<Buffer> {
  return within((async () => {
    let data = Buffer.alloc(0);
    for await (const chunk of client.iterator({ destroyOnReturn: false })) {
      data = Buffer.concat([data, Buffer.from(chunk)]);
      if (data.length > MAX_FRAME) throw new Error("provider request too large");
      const end = data.indexOf("\r\n\r\n");
      if (end < 0) continue;
      const match = /^content-length:\s*(\d+)$/im.exec(data.subarray(0, end).toString("ascii"));
      if (match?.[1] === undefined) throw new Error("missing content length");
      const total = end + 4 + Number(match[1]);
      if (data.length === total) return data;
      if (data.length > total) throw new Error("provider sent trailing bytes");
    }
    throw new Error("provider request closed early");
  })(), "provider request read");
}
function extension(channel: Channel, fail: boolean): ExtensionFactory {
  return (pi: ExtensionAPI) => {
    pi.on("before_agent_start", async () => ({ systemPrompt: PROMPT }));
    pi.on("before_provider_request", async (event) => {
      try {
        if (fail) throw new Error("injected failure");
        const payload = structuredClone(event.payload) as Record<string, unknown>;
        const messages = payload.messages;
        if (!Array.isArray(messages) || messages.length < 2) throw new Error("missing messages");
        messages[0] = { role: "system", content: PROMPT };
        return payload;
      } catch {
        return channel.fatal();
      }
    });
  };
}
async function runPi(channel: Channel, scenario: string): Promise<void> {
  let abort = async (): Promise<void> => {};
  const bridge = await startBridge(channel, async () => abort());
  const runtime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(), modelsPath: null, allowModelNetwork: false,
  });
  runtime.registerProvider("assist-p0", {
    name: "Assist P0", api: "openai-completions", apiKey: "local", authHeader: false,
    models: [{
      id: "fixture-model", name: "Fixture", reasoning: true, input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 131072, maxTokens: 256, baseUrl: bridge.baseUrl,
      compat: {
        supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false,
        supportsUsageInStreaming: true, supportsStrictMode: false,
        maxTokensField: "max_tokens", thinkingFormat: "qwen-chat-template",
      },
    }],
  });
  await runtime.setRuntimeApiKey("assist-p0", "local");
  const model = runtime.getModel("assist-p0", "fixture-model");
  if (model === undefined) throw new Error("model registration failed");
  const settings = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false, maxRetries: 0 } });
  const loader = new DefaultResourceLoader({
    cwd: "/workspace", agentDir: "/agent", settingsManager: settings,
    extensionFactories: [extension(channel, scenario.startsWith("fail-"))], noExtensions: true,
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    systemPrompt: PROMPT, extensionsOverride: (base) => base,
    skillsOverride: () => ({ skills: [], diagnostics: [] }),
    promptsOverride: () => ({ prompts: [], diagnostics: [] }),
    themesOverride: () => ({ themes: [], diagnostics: [] }),
    agentsFilesOverride: () => ({ agentsFiles: [] }),
    systemPromptOverride: () => PROMPT, appendSystemPromptOverride: () => [],
  });
  await within(loader.reload(), "resource reload");
  const sessionManager = SessionManager.open(SESSION, undefined, "/workspace");
  const created = await within(createAgentSession({
    cwd: "/workspace", agentDir: "/agent", model, thinkingLevel: "off", modelRuntime: runtime,
    noTools: "all", resourceLoader: loader, sessionManager, settingsManager: settings,
  }), "session creation");
  abort = () => created.session.abort();
  await within(created.session.bindExtensions({ mode: "json" }), "extension binding");
  await channel.write({ type: "runner_ready" });
  exact(await channel.read(), { type: "begin" }, "begin");
  if (scenario === "fail-closed") {
    const closed = within(new Promise<void>((resolve) => channel.socket.once("close", resolve)), "intentional close");
    channel.socket.destroy();
    await closed;
  } else if (scenario === "fail-backpressure") {
    const padding = Buffer.alloc(16 * 1024, 0x20);
    while (channel.socket.write(padding)) { /* fill Node and kernel send buffers */ }
    const descriptor = openSync(FIFO, "w");
    writeSync(descriptor, Buffer.from("B"));
  }
  const prompt = created.session.prompt(USER, { expandPromptTemplates: false });
  await within(Promise.all([prompt, bridge.completion]), "Pi prompt");
  const [connections, rejected] = bridge.counts();
  created.session.dispose();
  await within(new Promise<void>((resolve, reject) => bridge.server.close((error) => error ? reject(error) : resolve())), "bridge close");
  await channel.write({ connections, rejected, request: 1, type: "scenario_done" });
}
async function startHostile(): Promise<void> {
  process.on("SIGTERM", () => {});
  const script = `
    const fs=require('node:fs');
    process.on('SIGTERM',()=>{});fs.openSync('${FIFO}','w');
    process.send('ready');process.disconnect();setInterval(()=>{},1000);
  `;
  const child = spawn(process.execPath, ["--library-path", "/runtime/lib", "/runtime/node", "--eval", script],
    { detached: true, stdio: ["ignore", "ignore", "ignore", "ipc"] });
  await within(new Promise<void>((resolve, reject) => {
    child.once("message", (value) => value === "ready" ? resolve() : reject(new Error("invalid child readiness")));
    child.once("error", reject);
    child.once("exit", (code, signal) => reject(new Error(`hostile child exited: code=${code} signal=${signal}`)));
  }), "hostile child start");
  child.unref();
  if (child.pid === undefined) throw new Error("hostile child has no pid");
}
async function main(): Promise<void> {
  const scenario = process.argv[2];
  if (process.version !== "v22.23.1") throw new Error("unexpected Node version");
  if (!scenario) throw new Error("missing scenario");
  const channel = await connect();
  await channel.write({ scenario, type: "hello" });
  exact(await channel.read(), { type: "accepted" }, "acceptance");
  if (scenario === "hostile") {
    await startHostile();
    await channel.write({ type: "hostile_ready" });
    await new Promise<never>(() => {});
  }
  if (scenario === "authority-death") await startHostile();
  await runPi(channel, scenario);
}
await main();
