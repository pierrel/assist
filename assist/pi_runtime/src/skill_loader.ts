import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { BrokerClient } from "./broker_client.js";
import { activateDisclosure } from "./tool_disclosure.js";

export type SkillManifest = {
  name: string;
  description: string;
  declaredTools: string[];
};

export type RetainedSkill = {
  name: string;
  body: string;
  bodySha256: string;
  declaredTools: string[];
};

/** Register the host-brokered Pi skill loader for one worker. */
export function skillLoaderExtension(capability: string, catalog: SkillManifest[]): ExtensionFactory {
  const byName = new Map(catalog.map((skill) => [skill.name, skill]));
  const broker = new BrokerClient(capability);
  return (pi) => {
    pi.registerTool({
      name: "load_skill",
      label: "Load skill",
      description: "Load the complete rules for one skill from the available catalog.",
      parameters: Type.Object({
        name: Type.String({ minLength: 1, maxLength: 128 }),
      }, { additionalProperties: false }),
      execute: async (toolCallId, params) => {
        if (!byName.has(params.name)) throw new Error("Pi skill is unavailable");
        const text = await broker.loadSkill(toolCallId, params.name);
        const skill = byName.get(params.name);
        activateDisclosure(pi, skill?.declaredTools ?? []);
        return { content: [{ type: "text", text }], details: {} };
      },
    });
  };
}
