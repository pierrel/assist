import { emittedManifest } from "./registry.js";

process.stdout.write(`${JSON.stringify(emittedManifest())}\n`);
