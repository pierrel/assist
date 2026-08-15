import { createConnection } from "node:net";
import {
  createBashToolDefinition,
  createEditToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";

const SOCKET_PATH = "/run/pi/broker.sock";
const MAX_FRAME_BYTES = 512 * 1024;
const CALL_TIMEOUT_MS = 125_000;
const WORKSPACE = "/workspace";

type BrokerOperation = "access" | "bash" | "mkdir" | "read" | "write" | "load_skill" | "map_data";
type BrokerRequest = {
  version: 1;
  id: number;
  capability: string;
  operation: BrokerOperation;
  path?: string;
  mode?: "read" | "write";
  content?: string;
  command?: string;
  cwd?: string;
  timeout?: number;
  tool_call_id?: string;
  name?: string;
  arguments?: { name: string };
  places?: string;
  routes?: string;
};

type BrokerResponse = {
  version: 1;
  id: number;
  ok: boolean;
  value?: unknown;
  error?: string;
};

function responseError(): Error {
  return new Error("Pi workspace operation failed");
}

function validateResponse(value: unknown, id: number): BrokerResponse {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw responseError();
  }
  const response = value as Record<string, unknown>;
  if (response.version !== 1 || response.id !== id || typeof response.ok !== "boolean") {
    throw responseError();
  }
  if (response.ok && Object.hasOwn(response, "error")) throw responseError();
  if (!response.ok && typeof response.error !== "string") throw responseError();
  return response as BrokerResponse;
}

/** Pi's closed worker-to-host client for coding, loading, and map operations. */
export class BrokerClient {
  #nextId = 0;

  constructor(private readonly capability: string) {}

  async call(request: Omit<BrokerRequest, "capability" | "id" | "version">): Promise<unknown> {
    const id = ++this.#nextId;
    const frame: BrokerRequest = { version: 1, id, capability: this.capability, ...request };
    const raw = Buffer.from(`${JSON.stringify(frame)}\n`, "utf8");
    if (raw.length > MAX_FRAME_BYTES) throw responseError();
    const response = await new Promise<BrokerResponse>((resolve, reject) => {
      const socket = createConnection(SOCKET_PATH);
      const chunks: Buffer[] = [];
      let size = 0;
      const fail = () => {
        socket.destroy();
        reject(responseError());
      };
      socket.setTimeout(CALL_TIMEOUT_MS, fail);
      socket.once("error", fail);
      socket.once("connect", () => socket.end(raw));
      socket.on("data", (chunk: Buffer) => {
        size += chunk.length;
        if (size > MAX_FRAME_BYTES) {
          fail();
          return;
        }
        chunks.push(chunk);
      });
      socket.once("end", () => {
        try {
          const text = Buffer.concat(chunks).toString("utf8");
          if (!text.endsWith("\n") || text.indexOf("\n") !== text.length - 1) throw responseError();
          resolve(validateResponse(JSON.parse(text), id));
        } catch {
          fail();
        }
      });
    });
    if (!response.ok) throw responseError();
    return response.value;
  }

  async loadSkill(toolCallId: string, name: string): Promise<string> {
    const value = await this.call({ operation: "load_skill", tool_call_id: toolCallId, name, arguments: { name } });
    if (typeof value !== "string") throw responseError();
    return value;
  }

  async mapData(places: string, routes: string): Promise<string> {
    const value = await this.call({ operation: "map_data", places, routes });
    if (typeof value !== "string") throw responseError();
    return value;
  }

  async readFile(path: string): Promise<Buffer> {
    const value = await this.call({ operation: "read", path });
    if (typeof value !== "string") throw responseError();
    return Buffer.from(value, "base64");
  }

  async access(path: string, mode: "read" | "write"): Promise<void> {
    await this.call({ operation: "access", path, mode });
  }

  async writeFile(path: string, content: string): Promise<void> {
    await this.call({ operation: "write", path, content });
  }

  async mkdir(path: string): Promise<void> {
    await this.call({ operation: "mkdir", path });
  }

  async bash(command: string, cwd: string, timeout?: number): Promise<{ exitCode: number; output: string }> {
    const value = await this.call({
      operation: "bash", command, cwd, timeout: timeout ?? 120,
    });
    if (value === null || typeof value !== "object" || Array.isArray(value)) throw responseError();
    const result = value as Record<string, unknown>;
    const { exitCode, output } = result;
    if (typeof exitCode !== "number" || !Number.isInteger(exitCode) || typeof output !== "string") {
      throw responseError();
    }
    return { exitCode, output };
  }
}

export function createBrokerTools(capability: string): ToolDefinition<any, any, any>[] {
  const broker = new BrokerClient(capability);
  return [
    createReadToolDefinition(WORKSPACE, {
      autoResizeImages: false,
      operations: {
        readFile: (path) => broker.readFile(path),
        access: (path) => broker.access(path, "read"),
      },
    }),
    createWriteToolDefinition(WORKSPACE, {
      operations: {
        writeFile: (path, content) => broker.writeFile(path, content),
        mkdir: (path) => broker.mkdir(path),
      },
    }),
    createEditToolDefinition(WORKSPACE, {
      operations: {
        readFile: (path) => broker.readFile(path),
        writeFile: (path, content) => broker.writeFile(path, content),
        access: (path) => broker.access(path, "write"),
      },
    }),
    createBashToolDefinition(WORKSPACE, {
      exposeSessionEnvironment: false,
      operations: {
        async exec(command, cwd, options) {
          const result = await broker.bash(command, cwd, options.timeout);
          options.onData(Buffer.from(result.output, "base64"));
          return { exitCode: result.exitCode };
        },
      },
    }),
  ];
}
