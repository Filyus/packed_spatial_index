import assert from "node:assert/strict";
import test from "node:test";

import { HttpError } from "../src/artifact.ts";
import { parseBbox } from "../src/query.ts";

function bbox(raw) {
  return parseBbox(new URL(`https://example.test/search?bbox=${raw}`));
}

function expectHttpError(run, status, code, message) {
  assert.throws(run, (error) => {
    assert.ok(error instanceof HttpError);
    assert.equal(error.status, status);
    assert.equal(error.code, code);
    assert.match(error.message, message);
    return true;
  });
}

test("accepts both the 2D and the 3D arity", () => {
  assert.deepEqual(bbox("64,23,71,29"), [64, 23, 71, 29]);
  assert.deepEqual(bbox("64,23,0,71,29,100"), [64, 23, 0, 71, 29, 100]);
  assert.deepEqual(bbox(" 64 , 23 , 71 , 29 "), [64, 23, 71, 29]);
});

test("rejects any other length", () => {
  for (const raw of ["64", "64,23", "64,23,71", "64,23,0,71,29", "64,23,0,71,29,100,7"]) {
    expectHttpError(() => bbox(raw), 400, "invalid_bbox", /4 numbers \(2D\) or 6 numbers \(3D\)/);
  }
});

test("rejects values that are not finite numbers", () => {
  expectHttpError(() => bbox("64,23,north,29"), 400, "invalid_bbox", /4 numbers/);
  expectHttpError(() => bbox("64,23,Infinity,29"), 400, "invalid_bbox", /4 numbers/);
});

test("requires a bbox at all", () => {
  assert.throws(
    () => parseBbox(new URL("https://example.test/search")),
    (error) => error instanceof HttpError && /required/.test(error.message),
  );
});

test("checks min <= max on every axis, including z", () => {
  expectHttpError(() => bbox("71,23,64,29"), 400, "invalid_bbox", /min values/);
  expectHttpError(() => bbox("64,29,71,23"), 400, "invalid_bbox", /min values/);
  // The z pair is the one an xy-only check would wave through.
  expectHttpError(() => bbox("64,23,100,71,29,0"), 400, "invalid_bbox", /min values/);
});
