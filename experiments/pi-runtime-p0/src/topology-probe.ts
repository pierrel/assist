import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { readdir, readFile, readlink } from "node:fs/promises";
import { createConnection } from "node:net";

import { startLocalHttpBridge } from "./local-http-bridge.js";
import { createFixture, startScriptedProvider, toolCall } from "./sdk-fixture.js";
import {
  assertExactKeys,
  FrameChannel,
  MAX_CONTROL_FRAME_BYTES,
  requireInteger,
  requireString,
} from "./protocol.js";

const CONTROL_SOCKET = "/run/assist-p0/control.sock";
const PROVIDER_SOCKET = "/run/assist-p0/provider.sock";
const RELEASE_FILE = "/runtime/release.json";
const SESSION_FILE = "/session/session.jsonl";
const FORBIDDEN_PATHS = [
  "/forbidden/workspace-canary",
  "/forbidden/home-canary",
  "/forbidden/session-canary",
  "/run/docker.sock",
] as const;

async function main(): Promise<void> {
  assertRuntimeEnvironment();
  const release = await readRelease();
  const socket = createConnection(CONTROL_SOCKET);
  await new Promise<void>((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  });
  const control = new FrameChannel(socket, MAX_CONTROL_FRAME_BYTES);
  await control.write({
    type: "hello",
    seq: 0,
    pid: process.pid,
    release: release.digest,
  });
  const challenge = await control.read();
  assertExactKeys(challenge, [
    "type", "seq", "gateway_generation", "lease", "model", "nonce",
    "mode", "profile_digest", "request_budget_bytes", "response_budget_bytes", "run_id",
    "server_generation",
  ]);
  if (challenge.type !== "challenge" || requireInteger(challenge.seq, "seq") !== 0) {
    throw new Error("invalid control challenge");
  }
  const nonce = requireString(challenge.nonce, "nonce");
  const runId = requireString(challenge.run_id, "run_id");
  const mode = requireString(challenge.mode, "mode");

  if (mode === "teardown-race") {
    await waitForTeardown(control, release.digest, challenge, nonce, runId);
    return;
  }
  if (mode !== "topology") throw new Error(`unknown topology mode: ${mode}`);

  const bridge = await startLocalHttpBridge(PROVIDER_SOCKET, {
    gatewayGeneration: requireInteger(challenge.gateway_generation, "gateway_generation"),
    lease: requireString(challenge.lease, "lease"),
    model: requireString(challenge.model, "model"),
    nonce,
    profileDigest: requireString(challenge.profile_digest, "profile_digest"),
    release: release.digest,
    requestBudgetBytes: requireInteger(challenge.request_budget_bytes, "request_budget_bytes"),
    responseBudgetBytes: requireInteger(challenge.response_budget_bytes, "response_budget_bytes"),
    runId,
    serverGeneration: requireInteger(challenge.server_generation, "server_generation"),
  });
  try {
    const gatewayResponse = await fetch(`${bridge.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "topology-fixture", stream: true, messages: [] }),
      redirect: "error",
      signal: AbortSignal.timeout(5_000),
    });
    const gatewayBody = await gatewayResponse.text();
    const replay = await fetch(`${bridge.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "topology-fixture", stream: true, messages: [] }),
      redirect: "error",
      signal: AbortSignal.timeout(5_000),
    });
    const fixtureProvider = await startScriptedProvider([
      toolCall(0, "call-workspace", "fixture_workspace_probe", {}),
      "contained fixture complete",
    ]);
    const fixture = await createFixture({
      baseUrl: fixtureProvider.baseUrl,
      sessionPath: SESSION_FILE,
      workspaceProbePath: "/workspace/host-canary",
    });
    try {
      await fixture.session.prompt("Probe the fixed workspace canary.", {
        expandPromptTemplates: false,
      });
    } finally {
      await fixture.session.dispose();
      await fixtureProvider.close();
    }
    const sessionHandle = await import("node:fs/promises").then(({ open }) => open(SESSION_FILE, "r+"));
    try {
      await sessionHandle.sync();
    } finally {
      await sessionHandle.close();
    }

    const census = await collectCensus();
    await control.write({
      type: "topology_result",
      seq: 1,
      nonce,
      run_id: runId,
      release: release.digest,
      gateway_status: gatewayResponse.status,
      gateway_body_sha256: createHash("sha256").update(gatewayBody).digest("hex"),
      replay_denied: replay.status === 409,
      workspace_model_tool_denied:
        fixture.trace.effects.length === 1 && fixture.trace.effects[0] === "workspace_visible:false",
      sdk_census: {
        active_tools: fixture.session.getActiveToolNames(),
        events: fixture.trace.events,
        payloads: fixture.trace.payloads,
      },
      direct_model_denied: await connectionDenied("http://127.0.0.1:8000/health"),
      public_egress_denied: await connectionDenied("http://1.1.1.1/"),
      forbidden_paths: await forbiddenPathCensus(),
      session_sha256: createHash("sha256").update(await readFile(SESSION_FILE)).digest("hex"),
      census,
    });
    const accepted = await control.read();
    assertExactKeys(accepted, ["type", "seq"]);
    if (accepted.type !== "accepted" || requireInteger(accepted.seq, "seq") !== 1) {
      throw new Error("parent did not accept topology result");
    }
  } finally {
    await bridge.close();
    control.close();
  }
}

async function waitForTeardown(
  control: FrameChannel,
  release: string,
  challenge: Record<string, unknown>,
  nonce: string,
  runId: string,
): Promise<never> {
  const providerSocket = createConnection(PROVIDER_SOCKET);
  await new Promise<void>((resolve, reject) => {
    providerSocket.once("connect", resolve);
    providerSocket.once("error", reject);
  });
  const provider = new FrameChannel(providerSocket, MAX_CONTROL_FRAME_BYTES);
  await provider.write({
    type: "provider_hello",
    seq: 0,
    gateway_generation: challenge.gateway_generation,
    lease: challenge.lease,
    model: challenge.model,
    nonce,
    profile_digest: challenge.profile_digest,
    release,
    request_budget_bytes: challenge.request_budget_bytes,
    response_budget_bytes: challenge.response_budget_bytes,
    run_id: runId,
    server_generation: challenge.server_generation,
  });
  const descendant = spawn(
    "/runtime/node",
    ["--eval", "setInterval(() => {}, 1000)"],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  await control.write({
    type: "race_ready",
    seq: 1,
    nonce,
    run_id: runId,
    descendant_pid: descendant.pid,
  });
  return await new Promise<never>(() => {});
}

function assertRuntimeEnvironment(): void {
  if (process.version !== "v22.23.1") {
    throw new Error(`unexpected Node version: ${process.version}`);
  }
  const allowed = new Set(["HOME", "PATH", "PWD"]);
  for (const key of Object.keys(process.env)) {
    if (!allowed.has(key)) {
      throw new Error(`unexpected environment variable: ${key}`);
    }
  }
  if (process.env.PATH !== "/runtime") {
    throw new Error("PATH is not the closed runtime path");
  }
  if (process.env.PWD !== "/tmp") {
    throw new Error("PWD is not the private temporary directory");
  }
  if (process.env.HOME !== "/tmp/home") {
    throw new Error("HOME is not the private empty home directory");
  }
}

async function readRelease(): Promise<{ readonly digest: string }> {
  const value: unknown = JSON.parse(await readFile(RELEASE_FILE, "utf8"));
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("release metadata is not an object");
  }
  const record = value as Record<string, unknown>;
  assertExactKeys(record, ["digest", "node", "profile"]);
  return { digest: requireString(record.digest, "digest") };
}

async function collectCensus(): Promise<Record<string, unknown>> {
  const fds: Record<string, string> = {};
  for (const name of await readdir("/proc/self/fd")) {
    try {
      fds[name] = await readlink(`/proc/self/fd/${name}`);
    } catch {
      fds[name] = "<closed>";
    }
  }
  return {
    uid: process.getuid?.(),
    gid: process.getgid?.(),
    cwd: process.cwd(),
    env: process.env,
    cgroup: await readFile("/proc/self/cgroup", "utf8"),
    limits: await readFile("/proc/self/limits", "utf8"),
    mounts: await readFile("/proc/self/mountinfo", "utf8"),
    fds,
  };
}

async function forbiddenPathCensus(): Promise<Record<string, boolean>> {
  const result: Record<string, boolean> = {};
  for (const path of FORBIDDEN_PATHS) {
    try {
      await readFile(path);
      result[path] = false;
    } catch {
      result[path] = true;
    }
  }
  return result;
}

async function connectionDenied(url: string): Promise<boolean> {
  try {
    await fetch(url, { redirect: "error", signal: AbortSignal.timeout(500) });
    return false;
  } catch {
    return true;
  }
}

await main();
