import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createConnection } from "node:net";

import {
  assertExactKeys,
  FrameChannel,
  MAX_PROVIDER_FRAME_BYTES,
  requireInteger,
  requireString,
} from "./protocol.js";

const MAX_REQUEST_BYTES = 4 * 1024 * 1024;
const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;

export interface BridgeIdentity {
  readonly gatewayGeneration: number;
  readonly lease: string;
  readonly model: string;
  readonly nonce: string;
  readonly profileDigest: string;
  readonly release: string;
  readonly requestBudgetBytes: number;
  readonly responseBudgetBytes: number;
  readonly runId: string;
  readonly serverGeneration: number;
}

export async function startLocalHttpBridge(
  socketPath: string,
  identity: BridgeIdentity,
): Promise<{ readonly baseUrl: string; close(): Promise<void> }> {
  const socket = createConnection(socketPath);
  await new Promise<void>((resolve, reject) => {
    socket.once("connect", resolve);
    socket.once("error", reject);
  });
  const channel = new FrameChannel(socket, MAX_PROVIDER_FRAME_BYTES);
  await channel.write({
    type: "provider_hello",
    seq: 0,
    gateway_generation: identity.gatewayGeneration,
    lease: identity.lease,
    model: identity.model,
    nonce: identity.nonce,
    profile_digest: identity.profileDigest,
    release: identity.release,
    request_budget_bytes: identity.requestBudgetBytes,
    response_budget_bytes: identity.responseBudgetBytes,
    run_id: identity.runId,
    server_generation: identity.serverGeneration,
  });
  const hello = await channel.read();
  assertExactKeys(hello, ["type", "seq"]);
  if (hello.type !== "provider_ready" || requireInteger(hello.seq, "seq") !== 0) {
    throw new Error("provider gateway rejected hello");
  }

  let seq = 1;
  let active = false;
  let consumed = false;
  const server = createServer(async (request, response) => {
    try {
      if (active || consumed) {
        rejectHttp(response, 409, "provider lease is already active or consumed");
        return;
      }
      active = true;
      consumed = true;
      await forwardRequest(channel, request, response, identity, seq++);
    } catch (error) {
      if (!response.headersSent) {
        rejectHttp(response, 502, error instanceof Error ? error.message : String(error));
      } else {
        response.destroy(error instanceof Error ? error : new Error(String(error)));
      }
    } finally {
      active = false;
    }
  });
  server.maxHeadersCount = 32;
  server.headersTimeout = 5_000;
  server.requestTimeout = 30_000;
  server.keepAliveTimeout = 1_000;
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("loopback bridge did not receive a TCP port");
  }
  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    close: async () => {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      });
      channel.close();
    },
  };
}

async function forwardRequest(
  channel: FrameChannel,
  request: IncomingMessage,
  response: ServerResponse,
  identity: BridgeIdentity,
  seq: number,
): Promise<void> {
  if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
    rejectHttp(response, 404, "only POST /v1/chat/completions is allowed");
    return;
  }
  if (request.headers["transfer-encoding"] !== undefined) {
    rejectHttp(response, 400, "transfer encoding is forbidden");
    return;
  }
  if (request.headers["content-encoding"] !== undefined) {
    rejectHttp(response, 400, "content encoding is forbidden");
    return;
  }
  const lengthText = request.headers["content-length"];
  if (typeof lengthText !== "string" || !/^\d+$/.test(lengthText)) {
    rejectHttp(response, 411, "one content length is required");
    return;
  }
  const length = Number(lengthText);
  if (
    !Number.isSafeInteger(length)
    || length < 1
    || length > MAX_REQUEST_BYTES
    || length > identity.requestBudgetBytes
  ) {
    rejectHttp(response, 413, "request body is out of bounds");
    return;
  }
  const body = await readExactBody(request, length);
  const rawDigest = createHash("sha256").update(body).digest("hex");
  await channel.write({
    type: "provider_request",
    seq,
    gateway_generation: identity.gatewayGeneration,
    lease: identity.lease,
    model: identity.model,
    nonce: identity.nonce,
    profile_digest: identity.profileDigest,
    release: identity.release,
    request_budget_bytes: identity.requestBudgetBytes,
    response_budget_bytes: identity.responseBudgetBytes,
    run_id: identity.runId,
    server_generation: identity.serverGeneration,
    method: "POST",
    path: "/v1/chat/completions",
    content_type: request.headers["content-type"] ?? "",
    body_sha256: rawDigest,
    body_base64: body.toString("base64"),
  });
  const start = await channel.read();
  assertExactKeys(start, ["type", "seq", "status", "content_type"]);
  if (start.type !== "provider_response_start" || requireInteger(start.seq, "seq") !== seq) {
    throw new Error("invalid provider response start");
  }
  const status = requireInteger(start.status, "status");
  const contentType = requireString(start.content_type, "content_type");
  response.writeHead(status, {
    "cache-control": "no-store",
    "content-encoding": "identity",
    "content-type": contentType,
  });
  let total = 0;
  while (true) {
    const frame = await channel.read();
    if (frame.type === "provider_response_end") {
      assertExactKeys(frame, ["type", "seq"]);
      if (requireInteger(frame.seq, "seq") !== seq) {
        throw new Error("response end sequence mismatch");
      }
      response.end();
      return;
    }
    assertExactKeys(frame, ["type", "seq", "chunk_base64"]);
    if (frame.type !== "provider_response_chunk" || requireInteger(frame.seq, "seq") !== seq) {
      throw new Error("invalid provider response chunk");
    }
    const chunk = Buffer.from(requireString(frame.chunk_base64, "chunk_base64"), "base64");
    total += chunk.length;
    if (total > MAX_RESPONSE_BYTES || total > identity.responseBudgetBytes) {
      throw new Error("provider response exceeds bound");
    }
    if (!response.write(chunk)) {
      await new Promise<void>((resolve) => response.once("drain", resolve));
    }
  }
}

async function readExactBody(request: IncomingMessage, expected: number): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const value of request) {
    const chunk = Buffer.isBuffer(value) ? value : Buffer.from(value as Uint8Array);
    total += chunk.length;
    if (total > expected || total > MAX_REQUEST_BYTES) {
      throw new Error("request body exceeds declared length");
    }
    chunks.push(chunk);
  }
  if (total !== expected) {
    throw new Error("request body length mismatch");
  }
  return Buffer.concat(chunks, total);
}

function rejectHttp(response: ServerResponse, status: number, message: string): void {
  response.writeHead(status, { "content-type": "text/plain; charset=utf-8" });
  response.end(message);
}
