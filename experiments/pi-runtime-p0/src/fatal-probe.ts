import { appendFileSync } from "node:fs";

import { createFixture } from "./sdk-fixture.js";

const [baseUrl, sessionPath, failAt, markerPath, effectMarkerPath] = process.argv.slice(2);
if (!baseUrl || !sessionPath || !failAt || !markerPath || !effectMarkerPath) {
  throw new Error("usage: fatal-probe BASE_URL SESSION_PATH FAIL_AT MARKER_PATH EFFECT_MARKER_PATH");
}

const halt = (reason: string): never => {
  appendFileSync(markerPath, `${reason}\n`, { encoding: "utf8", mode: 0o600 });
  process.kill(process.pid, "SIGKILL");
  throw new Error("SIGKILL returned");
};

const fixture = await createFixture({ baseUrl, sessionPath, failAt, halt, effectMarkerPath });
await fixture.session.prompt("Exercise the fatal phase.", { expandPromptTemplates: false });
if (failAt === "session_shutdown") await fixture.session.reload();
throw new Error(`fatal phase did not terminate: ${failAt}`);
