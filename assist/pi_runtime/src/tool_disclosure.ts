import type { ExtensionAPI, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { BrokerClient } from "./broker_client.js";

const INITIAL_TOOLS = ["read", "write", "edit", "bash", "load_skill"];

/** Add only declared tools to Pi's visible set after the loader has succeeded. */
export function activateDisclosure(pi: ExtensionAPI, names: string[]): void {
  const active = pi.getActiveTools();
  pi.setActiveTools([...new Set([...active, ...names])]);
}

/** Register inactive application tools and own their visibility lifecycle. */
export function toolDisclosureExtension(capability: string): ExtensionFactory {
  const broker = new BrokerClient(capability);
  return (pi) => {
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
    pi.on("session_start", () => {
      // The registered map tool is deliberately absent until a host-verified
      // loader result reaches the next provider request.
      pi.setActiveTools(INITIAL_TOOLS);
    });
  };
}
