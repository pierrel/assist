import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";

/** Own the system prompt supplied by the host for this one fresh Pi turn. */
export function promptOwner(systemPrompt: string): ExtensionFactory {
  return (pi) => {
    pi.on("before_agent_start", () => ({ systemPrompt }));
  };
}
