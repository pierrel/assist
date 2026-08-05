import assert from "node:assert/strict";
import { test } from "node:test";

import { assertExactKeys, requireInteger, requireString } from "../src/protocol.js";

test("closed frame schemas reject missing and unknown fields", () => {
  assert.doesNotThrow(() => assertExactKeys({ type: "x", seq: 0 }, ["type", "seq"]));
  assert.throws(() => assertExactKeys({ type: "x", seq: 0, extra: true }, ["type", "seq"]));
  assert.throws(() => assertExactKeys({ type: "x" }, ["type", "seq"]));
});

test("scalar validators reject ambiguous protocol values", () => {
  assert.equal(requireString("value", "field"), "value");
  assert.throws(() => requireString("", "field"));
  assert.equal(requireInteger(3, "field"), 3);
  assert.throws(() => requireInteger(-1, "field"));
  assert.throws(() => requireInteger(1.5, "field"));
});
