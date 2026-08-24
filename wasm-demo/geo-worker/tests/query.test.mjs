import assert from "node:assert/strict";
import test from "node:test";

import { HttpError } from "../src/artifact.ts";
import {
  parseBbox,
  parseEnum,
  parseFrustum,
  parseIntParam,
  parsePolygon,
} from "../src/query.ts";

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

test("names the parameter in the error code, as the server does", () => {
  const bad = (query) =>
    assert.throws(
      () => {
        const url = new URL(`https://example.test/search?${query}`);
        parseEnum(url, "payload", "summary", ["none", "summary", "full"]);
        parseEnum(url, "level", "feature", ["entry", "feature"]);
        parseEnum(url, "identity", "ref", ["ref", "full"]);
        parseEnum(url, "count", "records", ["records", "only"]);
        parseIntParam(url, "limit", 100, 1, 1000);
        parseIntParam(url, "offset", 0, 0, 1e6);
      },
      (error) => error instanceof HttpError && error.status === 400 && error.code === expected,
    );

  let expected = "invalid_payload";
  bad("payload=lots");
  expected = "invalid_level";
  bad("level=deep");
  expected = "invalid_identity";
  bad("identity=maybe");
  expected = "invalid_count";
  bad("count=all");
  expected = "invalid_limit";
  bad("limit=0");
  expected = "invalid_offset";
  bad("offset=-1");
  // A parameter with no dedicated code keeps the generic one.
  expected = "invalid_query";
  assert.throws(
    () =>
      parseIntParam(
        new URL("https://example.test/search?maxReads=nope"),
        "maxReads",
        0,
        0,
        10,
      ),
    (error) => error instanceof HttpError && error.code === "invalid_query",
  );
});

function frustum(query) {
  return parseFrustum(new URL(`https://example.test/search?${query}`));
}

const SIX_PLANES = [
  [1, 0, 0, 0],
  [-1, 0, 0, 40],
  [0, 1, 0, 0],
  [0, -1, 0, 40],
  [0, 0, 1, 0],
  [0, 0, -1, 40],
]
  .flat()
  .join(",");

test("a frustum is six planes of four numbers, or nothing at all", () => {
  assert.deepEqual(frustum("bbox=0,0,1,1"), []);
  assert.equal(frustum(`frustum=${SIX_PLANES}`).length, 24);
});

test("a malformed frustum is named, not folded into invalid_query", () => {
  const bad = (raw) =>
    expectHttpError(() => frustum(`frustum=${raw}`), 400, "invalid_frustum", /frustum/);

  bad("1,2,3");
  bad(Array(23).fill(1).join(","));
  bad(Array(24).fill("x").join(","));
  bad(["inf", ...Array(23).fill(1)].join(","));
  // A plane with a zero normal constrains nothing, so it would silently widen
  // the frustum rather than fail.
  bad(
    [
      [0, 0, 0, 1],
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, -1, 0, 40],
      [0, 0, 1, 0],
      [0, 0, -1, 40],
    ]
      .flat()
      .join(","),
  );
});

test("one shape per query, as the native server requires", () => {
  expectHttpError(
    () => frustum(`bbox=0,0,0,1,1,1&frustum=${SIX_PLANES}`),
    400,
    "invalid_query",
    /mutually exclusive/,
  );
});

function polygon(query) {
  return parsePolygon(new URL(`https://example.test/search?${query}`));
}

test("a polygon travels as text, and only alone", () => {
  assert.equal(polygon("bbox=0,0,1,1"), "");
  assert.equal(polygon("polygon=[[[[0,0],[1,0],[1,1],[0,0]]]]"), "[[[[0,0],[1,0],[1,1],[0,0]]]]");

  // The coordinates themselves are checked on the wasm side, by the same code
  // a POST body would go through; the query string only owns this rule.
  expectHttpError(
    () => polygon("bbox=0,0,1,1&polygon=[[[[0,0],[1,0],[1,1],[0,0]]]]"),
    400,
    "invalid_query",
    /mutually exclusive/,
  );
});
