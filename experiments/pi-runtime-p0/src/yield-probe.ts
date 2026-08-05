import { writeFileSync } from "node:fs";

import { createFixture, fsyncFile } from "./sdk-fixture.js";

const [baseUrl, sessionPath, markerPath] = process.argv.slice(2);
if (!baseUrl || !sessionPath || !markerPath) {
  throw new Error("usage: yield-probe BASE_URL SESSION_PATH MARKER_PATH");
}

const started = process.hrtime.bigint();
const halt = (reason: string): never => {
  writeFileSync(markerPath, `${JSON.stringify({
    activeNs: Number(process.hrtime.bigint() - started),
    reason,
    state: "paused",
  })}\n`, { encoding: "utf8", mode: 0o600 });
  fsyncFile(markerPath);
  process.exit(75);
};

const fixture = await createFixture({
  baseUrl,
  sessionPath,
  failAt: "before_provider_request:next",
  halt,
});
await fixture.session.prompt("Load the fixture and pause at the complete boundary.", {
  expandPromptTemplates: false,
});
throw new Error("yield probe reached another provider boundary");
