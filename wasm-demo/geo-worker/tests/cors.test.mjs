import assert from "node:assert/strict";
import test from "node:test";

import { corsHeaders } from "../src/cors.ts";

test("caches preflight responses instead of repeating them per request", () => {
  const headers = corsHeaders();
  assert.equal(headers.get("access-control-max-age"), "86400");
});

test("still exposes the read/byte counters and allows Range-free JSON GETs", () => {
  const headers = corsHeaders();
  assert.equal(headers.get("access-control-allow-origin"), "*");
  assert.equal(headers.get("access-control-allow-methods"), "GET, OPTIONS");
  assert.equal(headers.get("access-control-allow-headers"), "content-type");
  assert.equal(
    headers.get("access-control-expose-headers"),
    "X-PSI-Reads, X-PSI-Bytes, X-PSI-R2-Operations",
  );
});
