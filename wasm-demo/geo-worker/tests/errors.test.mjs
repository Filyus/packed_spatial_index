import assert from "node:assert/strict";
import test from "node:test";

import { HttpError, errorBody } from "../src/errors.ts";

test("renders the server's envelope", () => {
  const { status, body } = errorBody(
    new HttpError(422, "query_too_large", "narrow the bbox"),
  );
  assert.equal(status, 422);
  // Nested under `error`, exactly as `server/src/error.rs` emits it: a client
  // written against one server must read the other without special cases.
  assert.deepEqual(body, {
    error: { code: "query_too_large", message: "narrow the bbox" },
  });
  assert.deepEqual(Object.keys(body), ["error"]);
  assert.deepEqual(Object.keys(body.error), ["code", "message"]);
});

test("anything that is not an HttpError is an internal error", () => {
  const { status, body } = errorBody(new TypeError("boom"));
  assert.equal(status, 500);
  assert.equal(body.error.code, "internal_error");
  assert.match(body.error.message, /boom/);

  // A bare string — what a rejected wasm promise carries — must not lose its
  // text on the way out.
  const bare = errorBody("wasm said no");
  assert.equal(bare.status, 500);
  assert.equal(bare.body.error.message, "wasm said no");
});
