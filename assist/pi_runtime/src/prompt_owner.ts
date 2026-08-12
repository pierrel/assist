import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";

/** Own the host prompt and model-use capability for one fresh Pi turn. */
export function promptOwner(systemPrompt: string, providerCapability: string): ExtensionFactory {
  return (pi) => {
    pi.on("before_agent_start", () => ({ systemPrompt }));
    pi.on("before_provider_headers", ({ headers }) => {
      headers["x-assist-pi-capability"] = providerCapability;
    });
  };
}
