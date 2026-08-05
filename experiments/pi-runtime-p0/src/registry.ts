import type {
  ExtensionAPI,
  ExtensionFactory,
  ToolDefinition,
} from "@earendil-works/pi-coding-agent";

export type ExtensionKind = "kernel" | "capability";

export interface ExtensionDescriptor {
  readonly id: string;
  readonly version: string;
  readonly kind: ExtensionKind;
  readonly dependsOn: readonly string[];
  readonly hooks: readonly string[];
  readonly tools: readonly string[];
  readonly schemaVersion: number;
}

export interface CapabilityContribution {
  addTool(tool: ToolDefinition): void;
}

export interface SuiteRuntime {
  readonly kernelFactory: ExtensionFactory;
  readonly skillsFactory: (contribution: CapabilityContribution) => void;
  readonly registerCapabilityTools: (pi: ExtensionAPI, tools: readonly ToolDefinition[]) => void;
}

interface SuiteEntry {
  readonly descriptor: ExtensionDescriptor;
  readonly factory: (runtime: SuiteRuntime) => ExtensionFactory;
}

export const SUITE: readonly SuiteEntry[] = [
  {
    descriptor: {
      id: "assist-kernel",
      version: "0.0.1",
      kind: "kernel",
      dependsOn: [],
      hooks: [
        "session_start",
        "before_agent_start",
        "before_provider_request",
        "message_end",
        "tool_call",
        "tool_result",
        "turn_end",
        "session_shutdown",
      ],
      tools: [],
      schemaVersion: 1,
    },
    factory: (runtime) => runtime.kernelFactory,
  },
  {
    descriptor: {
      id: "assist-skills",
      version: "0.0.1",
      kind: "capability",
      dependsOn: ["assist-kernel"],
      hooks: [],
      tools: [
        "load_skill",
        "fixture_workspace_probe",
        "fixture_secret",
        "fixture_read",
        "fixture_mutate",
        "fixture_error",
      ],
      schemaVersion: 1,
    },
    factory: (runtime) => (pi: ExtensionAPI) => {
      const tools: ToolDefinition[] = [];
      runtime.skillsFactory({ addTool: (tool) => tools.push(tool) });
      runtime.registerCapabilityTools(pi, tools);
    },
  },
] as const;

export function validateSuite(descriptors: readonly ExtensionDescriptor[]): void {
  const ids = new Set<string>();
  const tools = new Set<string>();
  const idVersions = new Map<string, string>();
  for (const descriptor of descriptors) {
    if (!/^[a-z][a-z0-9-]*$/.test(descriptor.id)) {
      throw new Error(`invalid extension id: ${descriptor.id}`);
    }
    const priorVersion = idVersions.get(descriptor.id);
    if (priorVersion !== undefined && priorVersion !== descriptor.version) {
      throw new Error(`version collision: ${descriptor.id}`);
    }
    if (ids.has(descriptor.id)) {
      throw new Error(`duplicate extension id: ${descriptor.id}`);
    }
    ids.add(descriptor.id);
    idVersions.set(descriptor.id, descriptor.version);
    if (!/^\d+\.\d+\.\d+$/.test(descriptor.version)) {
      throw new Error(`invalid extension version: ${descriptor.id}`);
    }
    if (!Number.isSafeInteger(descriptor.schemaVersion) || descriptor.schemaVersion < 1) {
      throw new Error(`invalid schema version: ${descriptor.id}`);
    }
    if (descriptor.kind === "capability" && descriptor.hooks.length > 0) {
      throw new Error(`capability extension registers hooks: ${descriptor.id}`);
    }
    for (const tool of descriptor.tools) {
      if (tools.has(tool)) {
        throw new Error(`duplicate tool: ${tool}`);
      }
      tools.add(tool);
    }
  }
  for (const descriptor of descriptors) {
    for (const dependency of descriptor.dependsOn) {
      if (!ids.has(dependency)) {
        throw new Error(`missing dependency: ${descriptor.id} -> ${dependency}`);
      }
    }
  }
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const byId = new Map(descriptors.map((descriptor) => [descriptor.id, descriptor]));
  const visit = (id: string): void => {
    if (visiting.has(id)) {
      throw new Error(`extension dependency cycle at ${id}`);
    }
    if (visited.has(id)) return;
    visiting.add(id);
    for (const dependency of byId.get(id)?.dependsOn ?? []) visit(dependency);
    visiting.delete(id);
    visited.add(id);
  };
  for (const id of ids) visit(id);
}

export function extensionFactories(runtime: SuiteRuntime): ExtensionFactory[] {
  validateSuite(SUITE.map((entry) => entry.descriptor));
  return SUITE.map((entry) => entry.factory(runtime));
}

export function emittedManifest(): readonly ExtensionDescriptor[] {
  validateSuite(SUITE.map((entry) => entry.descriptor));
  return SUITE.map((entry) => entry.descriptor);
}
