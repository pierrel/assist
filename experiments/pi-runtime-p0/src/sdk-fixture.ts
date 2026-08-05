import { appendFileSync, closeSync, fsyncSync, openSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";

import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
  defineTool,
  type ExtensionAPI,
  type ExtensionFactory,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
  extensionFactories,
  type CapabilityContribution,
  type SuiteRuntime,
} from "./registry.js";

export const OWNED_PROMPT = "ASSIST_P0_PROMPT_OWNERSHIP_CANARY_v1";

export interface FixtureTrace {
  readonly events: Array<Record<string, unknown>>;
  readonly payloads: unknown[];
  readonly effects: string[];
}

export interface FixtureOptions {
  readonly baseUrl: string;
  readonly sessionPath: string;
  readonly failAt?: string;
  readonly halt?: (reason: string) => never;
  readonly effectMarkerPath?: string;
  readonly workspaceProbePath?: string;
  readonly liveTools?: boolean;
  readonly modelId?: string;
  readonly maxTokens?: number;
  readonly compactionEnabled?: boolean;
  readonly compactionKeepRecentTokens?: number;
  readonly thinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
}

export async function createFixture(options: FixtureOptions) {
  const trace: FixtureTrace = { events: [], payloads: [], effects: [] };
  const modelId = options.modelId ?? "fixture-model";
  const baseTools = [
    "load_skill",
    "fixture_workspace_probe",
    ...(options.liveTools ? ["fixture_read", "fixture_mutate", "fixture_error"] : []),
  ];
  const runtime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsPath: null,
    allowModelNetwork: false,
  });
  runtime.registerProvider("assist-p0-fixture", {
    name: "Assist P0 fixture",
    api: "openai-completions",
    apiKey: "local",
    authHeader: false,
    models: [
      {
        id: modelId,
        name: "Fixture model",
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 131072,
        maxTokens: options.maxTokens ?? 4096,
        baseUrl: options.baseUrl,
        compat: {
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsUsageInStreaming: true,
          supportsStrictMode: false,
          maxTokensField: "max_tokens",
          thinkingFormat: "qwen-chat-template",
        },
      },
    ],
  });
  await runtime.setRuntimeApiKey("assist-p0-fixture", "local");
  const model = runtime.getModel("assist-p0-fixture", modelId);
  if (model === undefined) throw new Error("fixture model was not registered");

  const settings = SettingsManager.inMemory({
    compaction: {
      enabled: options.compactionEnabled ?? false,
      ...(options.compactionKeepRecentTokens === undefined
        ? {}
        : { keepRecentTokens: options.compactionKeepRecentTokens }),
    },
    retry: { enabled: false, maxRetries: 0 },
  });
  const halt = options.halt ?? ((reason: string): never => { throw new Error(`HALT:${reason}`); });
  const suiteRuntime: SuiteRuntime = {
    kernelFactory: kernelFactory(trace, halt, options.failAt, baseTools),
    skillsFactory: skillsFactory(trace, options.effectMarkerPath, options.workspaceProbePath),
    registerCapabilityTools: (pi, tools) => {
      for (const tool of tools) {
        if (tool.name !== "load_skill") {
          pi.registerTool(tool);
          continue;
        }
        pi.registerTool({
          ...tool,
          execute: async (...args) => {
            const result = await tool.execute(...args);
            if (options.failAt === "custom_tool_return") {
              return halt("custom_tool_return:injected custom tool return");
            }
            pi.appendEntry("assist-skill-activation", {
              name: "fixture",
              tools: ["fixture_secret"],
              version: 1,
            });
            if (options.failAt === "activation_after_append") {
              return halt("activation_after_append:injected after activation append");
            }
            const sessionPath = args[4].sessionManager.getSessionFile();
            if (sessionPath === undefined) throw new Error("fixture session is not persisted");
            fsyncFile(sessionPath);
            if (options.failAt === "activation_after_fsync") {
              return halt("activation_after_fsync:injected after activation fsync");
            }
            pi.setActiveTools([...baseTools, "fixture_secret"]);
            if (options.failAt === "activation_after_active_set") {
              return halt("activation_after_active_set:injected after active tool update");
            }
            trace.events.push({ phase: "activation_committed" });
            return result;
          },
        });
      }
    },
  };
  const loader = new DefaultResourceLoader({
    cwd: "/workspace",
    agentDir: "/agent",
    settingsManager: settings,
    extensionFactories: extensionFactories(suiteRuntime),
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
    systemPrompt: OWNED_PROMPT,
    extensionsOverride: (base) => base,
    skillsOverride: () => ({ skills: [], diagnostics: [] }),
    promptsOverride: () => ({ prompts: [], diagnostics: [] }),
    themesOverride: () => ({ themes: [], diagnostics: [] }),
    agentsFilesOverride: () => ({ agentsFiles: [] }),
    systemPromptOverride: () => OWNED_PROMPT,
    appendSystemPromptOverride: () => [],
  });
  await loader.reload();
  const sessionManager = SessionManager.open(options.sessionPath, undefined, "/workspace");
  const created = await createAgentSession({
    cwd: "/workspace",
    agentDir: "/agent",
    model,
    thinkingLevel: options.thinkingLevel ?? "off",
    modelRuntime: runtime,
    noTools: "builtin",
    resourceLoader: loader,
    sessionManager,
    settingsManager: settings,
  });
  await created.session.bindExtensions({ mode: "json" });
  return { ...created, modelRuntime: runtime, sessionManager, trace };
}

function kernelFactory(
  trace: FixtureTrace,
  halt: (reason: string) => never,
  failAt?: string,
  baseTools: readonly string[] = ["load_skill", "fixture_workspace_probe"],
): ExtensionFactory {
  return (pi: ExtensionAPI) => {
    const mustHandle = <F extends (...args: never[]) => unknown>(phase: string, handler: F): F => {
      return (async (...args: never[]) => {
        try {
          if (failAt === phase) throw new Error(`injected ${phase}`);
          return await handler(...args);
        } catch (error) {
          return halt(`${phase}:${error instanceof Error ? error.message : String(error)}`);
        }
      }) as F;
    };
    pi.on("session_start", mustHandle("session_start", async (_event, context) => {
      trace.events.push({ phase: "session_start" });
      const activated = context.sessionManager.getEntries().some(
        (entry) => entry.type === "custom" && entry.customType === "assist-skill-activation",
      );
      pi.setActiveTools(
        activated ? [...baseTools, "fixture_secret"] : [...baseTools],
      );
    }));
    pi.on("before_agent_start", mustHandle("before_agent_start", async (event) => {
      trace.events.push({ phase: "before_agent_start", incoming: event.systemPrompt });
      return { systemPrompt: OWNED_PROMPT };
    }));
    pi.on("before_provider_request", mustHandle("before_provider_request", async (event) => {
      if (failAt === "before_provider_request:next" && trace.payloads.length === 1) {
        throw new Error("injected before_provider_request:next");
      }
      const payload = enforceOwnedProviderPrompt(event.payload);
      trace.payloads.push(structuredClone(payload));
      trace.events.push({ phase: "before_provider_request", tools: pi.getActiveTools() });
      return payload;
    }));
    pi.on("message_end", mustHandle("message_end", async (event, context) => {
      if (failAt === `message_end:${event.message.role}`) {
        throw new Error(`injected message_end:${event.message.role}`);
      }
      const leaf = context.sessionManager.getLeafEntry();
      trace.events.push({
        phase: "message_end",
        role: event.message.role,
        leafRole: leaf?.type === "message" ? leaf.message.role : undefined,
      });
    }));
    pi.on("tool_call", mustHandle("tool_call", async (event, context) => {
      const leaf = context.sessionManager.getLeafEntry();
      trace.events.push({
        phase: "tool_call",
        tool: event.toolName,
        leafRole: leaf?.type === "message" ? leaf.message.role : undefined,
      });
    }));
    pi.on("tool_result", mustHandle("tool_result", async (event) => {
      trace.events.push({ phase: "tool_result", tool: event.toolName });
    }));
    pi.on("turn_end", mustHandle("turn_end", async (event) => {
      trace.events.push({ phase: "turn_end", role: event.message.role });
    }));
    pi.on("session_shutdown", mustHandle("session_shutdown", async () => {
      trace.events.push({ phase: "session_shutdown" });
    }));
  };
}

function enforceOwnedProviderPrompt(payload: unknown): unknown {
  if (payload === null || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("provider payload is not an object");
  }
  const copy = structuredClone(payload) as Record<string, unknown>;
  const messages = copy.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error("provider payload has no messages");
  }
  const first = messages[0];
  if (first === null || typeof first !== "object" || Array.isArray(first)) {
    throw new Error("provider system message is malformed");
  }
  const system = first as Record<string, unknown>;
  if (system.role !== "system") {
    throw new Error("provider payload does not start with system message");
  }
  system.content = OWNED_PROMPT;
  return copy;
}

function skillsFactory(
  trace: FixtureTrace,
  effectMarkerPath?: string,
  workspaceProbePath?: string,
): (contribution: CapabilityContribution) => void {
  return (contribution) => {
    contribution.addTool(defineTool({
      name: "load_skill",
      label: "Load fixture skill",
      description: "Load the named fixture skill.",
      parameters: Type.Object({ name: Type.Literal("fixture") }),
      executionMode: "sequential",
      execute: async (_id, _params, _signal, _update, context) => {
        markEffect(effectMarkerPath, "load_skill");
        trace.effects.push("load_skill");
        return {
          content: [{ type: "text", text: "fixture skill loaded" }],
          details: { activated: true },
          addedToolNames: ["fixture_secret"],
        };
      },
    }));
    contribution.addTool(defineTool({
      name: "fixture_workspace_probe",
      label: "Probe host workspace isolation",
      description: "Report whether the fixed host-workspace canary is visible.",
      parameters: Type.Object({}),
      executionMode: "sequential",
      execute: async () => {
        let visible = false;
        if (workspaceProbePath !== undefined) {
          try {
            readFileSync(workspaceProbePath);
            visible = true;
          } catch {
            visible = false;
          }
        }
        trace.effects.push(`workspace_visible:${visible}`);
        return {
          content: [{ type: "text", text: visible ? "host workspace visible" : "host workspace denied" }],
          details: { visible },
        };
      },
    }));
    contribution.addTool(defineTool({
      name: "fixture_read",
      label: "Read fixture value",
      description: "Return the supplied fixture value without mutation.",
      parameters: Type.Object({ value: Type.String() }),
      executionMode: "parallel",
      execute: async (_id, params) => {
        trace.effects.push(`read_start:${params.value}`);
        await new Promise((resolve) => setTimeout(resolve, 25));
        trace.effects.push(`read_end:${params.value}`);
        return {
          content: [{ type: "text", text: `read:${params.value}` }],
          details: {},
        };
      },
    }));
    contribution.addTool(defineTool({
      name: "fixture_mutate",
      label: "Mutate fixture value",
      description: "Record the supplied fixture mutation in serialized order.",
      parameters: Type.Object({ value: Type.String() }),
      executionMode: "sequential",
      execute: async (_id, params) => {
        trace.effects.push(`mutate_start:${params.value}`);
        await new Promise((resolve) => setTimeout(resolve, 25));
        trace.effects.push(`mutate_end:${params.value}`);
        return {
          content: [{ type: "text", text: `mutated:${params.value}` }],
          details: {},
        };
      },
    }));
    contribution.addTool(defineTool({
      name: "fixture_error",
      label: "Return fixture error",
      description: "Raise the fixed fixture error for provider recovery testing.",
      parameters: Type.Object({}),
      executionMode: "sequential",
      execute: async () => {
        trace.effects.push("fixture_error");
        throw new Error("intentional fixture error");
      },
    }));
    contribution.addTool(defineTool({
      name: "fixture_secret",
      label: "Fixture secret tool",
      description: "Return the supplied fixture value.",
      parameters: Type.Object({ value: Type.String() }),
      executionMode: "sequential",
      execute: async (_id, params) => {
        markEffect(effectMarkerPath, `fixture_secret:${params.value}`);
        trace.effects.push(`fixture_secret:${params.value}`);
        return {
          content: [{ type: "text", text: `secret:${params.value}` }],
          details: {},
        };
      },
    }));
  };
}

function markEffect(path: string | undefined, value: string): void {
  if (path !== undefined) appendFileSync(path, `${value}\n`, { encoding: "utf8", mode: 0o600 });
}

export interface ScriptedProvider {
  readonly baseUrl: string;
  readonly requests: unknown[];
  close(): Promise<void>;
}

export async function startScriptedProvider(scripts: readonly unknown[]): Promise<ScriptedProvider> {
  const requests: unknown[] = [];
  let index = 0;
  const server = createServer(async (request, response) => {
    try {
      const body = await readRequest(request);
      requests.push(JSON.parse(body));
      const script = scripts[index++];
      if (script === undefined) throw new Error("provider received an extra request");
      sendSse(response, script);
    } catch (error) {
      response.writeHead(500, { "content-type": "text/plain" });
      response.end(error instanceof Error ? error.message : String(error));
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    requests,
    close: () => new Promise<void>((resolve, reject) => {
      server.close((error) => (error ? reject(error) : resolve()));
    }),
  };
}

async function readRequest(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString("utf8");
}

function sendSse(response: ServerResponse, message: unknown): void {
  response.writeHead(200, { "content-type": "text/event-stream" });
  const id = "chatcmpl-fixture";
  const base = { id, object: "chat.completion.chunk", created: 1, model: "fixture-model" };
  if (typeof message === "string") {
    response.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: "assistant", content: message }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } })}\n\n`);
  } else {
    const toolCalls = Array.isArray(message) ? message : [message];
    response.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: "assistant", tool_calls: toolCalls }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } })}\n\n`);
  }
  response.end("data: [DONE]\n\n");
}

export function toolCall(index: number, id: string, name: string, args: object): object {
  return { index, id, type: "function", function: { name, arguments: JSON.stringify(args) } };
}

export function fsyncFile(path: string): void {
  const descriptor = openSync(path, "r");
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}
