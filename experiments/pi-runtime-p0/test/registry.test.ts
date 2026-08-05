import assert from "node:assert/strict";
import { test } from "node:test";

import { emittedManifest, validateSuite, type ExtensionDescriptor } from "../src/registry.js";

const manifest = (): ExtensionDescriptor[] => [...structuredClone(emittedManifest())];

test("static suite is valid and contains only the two P0 extensions", () => {
  const value = manifest();
  assert.doesNotThrow(() => validateSuite(value));
  assert.deepEqual(value.map((entry) => entry.id), ["assist-kernel", "assist-skills"]);
});

test("manifest rejects duplicate ids and tools", () => {
  const duplicateId = manifest();
  duplicateId.push(structuredClone(duplicateId[0]!));
  assert.throws(() => validateSuite(duplicateId), /duplicate extension id/);

  const duplicateTool = manifest();
  duplicateTool[0] = { ...duplicateTool[0]!, tools: ["load_skill"] };
  assert.throws(() => validateSuite(duplicateTool), /duplicate tool/);

  const collision = manifest();
  collision.push({ ...structuredClone(collision[0]!), version: "0.0.2" });
  assert.throws(() => validateSuite(collision), /version collision/);

  const invalidVersion = manifest();
  invalidVersion[0] = { ...invalidVersion[0]!, version: "latest" };
  assert.throws(() => validateSuite(invalidVersion), /invalid extension version/);
});

test("manifest rejects missing dependencies, cycles, schemas, and capability hooks", () => {
  const missing = manifest();
  missing[1] = { ...missing[1]!, dependsOn: ["missing"] };
  assert.throws(() => validateSuite(missing), /missing dependency/);

  const cycle = manifest();
  cycle[0] = { ...cycle[0]!, dependsOn: ["assist-skills"] };
  assert.throws(() => validateSuite(cycle), /dependency cycle/);

  const schema = manifest();
  schema[0] = { ...schema[0]!, schemaVersion: 0 };
  assert.throws(() => validateSuite(schema), /invalid schema version/);

  const hooks = manifest();
  hooks[1] = { ...hooks[1]!, hooks: ["before_provider_request"] };
  assert.throws(() => validateSuite(hooks), /capability extension registers hooks/);
});
