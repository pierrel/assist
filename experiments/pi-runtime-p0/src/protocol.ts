import { once } from "node:events";
import type { Socket } from "node:net";

export const MAX_CONTROL_FRAME_BYTES = 64 * 1024;
export const MAX_PROVIDER_FRAME_BYTES = 8 * 1024 * 1024;

export type JsonObject = { readonly [key: string]: unknown };

export class FrameChannel {
  readonly #socket: Socket;
  readonly #maxBytes: number;
  #buffer = Buffer.alloc(0);

  constructor(socket: Socket, maxBytes: number) {
    this.#socket = socket;
    this.#maxBytes = maxBytes;
  }

  async write(frame: JsonObject): Promise<void> {
    const payload = Buffer.from(`${JSON.stringify(frame)}\n`, "utf8");
    if (payload.length > this.#maxBytes) {
      throw new Error(`outbound frame exceeds ${this.#maxBytes} bytes`);
    }
    if (!this.#socket.write(payload)) {
      await once(this.#socket, "drain");
    }
  }

  async read(): Promise<JsonObject> {
    while (true) {
      const newline = this.#buffer.indexOf(0x0a);
      if (newline >= 0) {
        const line = this.#buffer.subarray(0, newline);
        this.#buffer = this.#buffer.subarray(newline + 1);
        if (line.length === 0) {
          throw new Error("empty frame");
        }
        const value: unknown = JSON.parse(line.toString("utf8"));
        if (value === null || typeof value !== "object" || Array.isArray(value)) {
          throw new Error("frame must be a JSON object");
        }
        return value as JsonObject;
      }
      if (this.#buffer.length >= this.#maxBytes) {
        throw new Error(`inbound frame exceeds ${this.#maxBytes} bytes`);
      }
      const chunk = await this.#readChunk();
      this.#buffer = Buffer.concat([this.#buffer, chunk]);
    }
  }

  close(): void {
    this.#socket.end();
  }

  #readChunk(): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const onData = (chunk: Buffer): void => {
        cleanup();
        resolve(chunk);
      };
      const onEnd = (): void => {
        cleanup();
        reject(new Error("protocol socket closed"));
      };
      const onError = (error: Error): void => {
        cleanup();
        reject(error);
      };
      const cleanup = (): void => {
        this.#socket.off("data", onData);
        this.#socket.off("end", onEnd);
        this.#socket.off("error", onError);
      };
      this.#socket.once("data", onData);
      this.#socket.once("end", onEnd);
      this.#socket.once("error", onError);
    });
  }
}

export function assertExactKeys(
  value: JsonObject,
  required: readonly string[],
  optional: readonly string[] = [],
): void {
  const allowed = new Set([...required, ...optional]);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`unknown frame field: ${key}`);
    }
  }
  for (const key of required) {
    if (!(key in value)) {
      throw new Error(`missing frame field: ${key}`);
    }
  }
}

export function requireString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${field} must be a non-empty string`);
  }
  return value;
}

export function requireInteger(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${field} must be a non-negative safe integer`);
  }
  return value as number;
}
