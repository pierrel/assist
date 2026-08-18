import type { ExtensionAPI, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { BrokerClient } from "./broker_client.js";

const KERNEL_TOOLS = ["read", "write", "edit", "bash", "load_skill"];

/** Add only declared tools to Pi's visible set after the loader has succeeded. */
export function activateDisclosure(pi: ExtensionAPI, names: string[]): void {
  const active = pi.getActiveTools();
  pi.setActiveTools([...new Set([...active, ...names])]);
}

/** Register application tools and own their per-turn visibility lifecycle. */
export function toolDisclosureExtension(capability: string, retainedTools: string[]): ExtensionFactory {
  const broker = new BrokerClient(capability);
  return (pi) => {
    let initialized = false;
    pi.registerTool({
      name: "map_data",
      label: "Map data",
      description: "Look up exact coordinates and walking-route polylines for a map.",
      parameters: Type.Object({
        places: Type.Optional(Type.String({ maxLength: 4096 })),
        routes: Type.Optional(Type.String({ maxLength: 4096 })),
      }, { additionalProperties: false }),
      execute: async (_toolCallId, params) => ({
        content: [{ type: "text", text: await broker.mapData(params.places ?? "", params.routes ?? "") }],
        details: {},
      }),
    });
    pi.on("before_agent_start", () => {
      if (!initialized) {
        // Resource registration follows session_start in Pi. Set the initial
        // kernel plus retained-skill menu here so registration cannot expose
        // any tool outside the host-approved per-turn authority.
        pi.setActiveTools([...new Set([...KERNEL_TOOLS, ...retainedTools])]);
        initialized = true;
      }
    });
  };
}
