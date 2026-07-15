import assert from "node:assert/strict";
import test from "node:test";

import { HttpError, withArtifact } from "../src/artifact.ts";

const OBJECT_KEY = "artifact.psindex";
const HEAD = { etag: "etag-1", size: 1024 };

function body(length, read = async () => new ArrayBuffer(length)) {
  return { body: {}, arrayBuffer: read };
}

async function expectHttpError(run, status, code, message) {
  await assert.rejects(run, (error) => {
    assert.ok(error instanceof HttpError);
    assert.equal(error.status, status);
    assert.equal(error.code, code);
    assert.match(error.message, message);
    return true;
  });
}

test("reads an ETag-bound range and reports exact counters", async () => {
  const bucket = {
    async head(key) {
      assert.equal(key, OBJECT_KEY);
      return HEAD;
    },
    async get(key, options) {
      assert.equal(key, OBJECT_KEY);
      assert.deepEqual(options, {
        onlyIf: { etagMatches: HEAD.etag },
        range: { offset: 8, length: 4 },
      });
      return body(4);
    },
  };

  const result = await withArtifact(bucket, OBJECT_KEY, async (artifact) => {
    assert.equal(artifact.fileLen, HEAD.size);
    assert.equal(artifact.objectEtag, HEAD.etag);
    return Array.from(await artifact.readRange(8, 4));
  });

  assert.deepEqual(result.body, [0, 0, 0, 0]);
  assert.deepEqual(
    { ...result.metrics, ms: 0 },
    { reads: 1, bytes: 4, r2Operations: 2, ms: 0 },
  );
  assert.ok(result.metrics.ms >= 0);
});

test("classifies missing artifacts and HEAD failures", async (t) => {
  await t.test("missing", async () => {
    const bucket = { head: async () => null };
    await expectHttpError(
      () => withArtifact(bucket, OBJECT_KEY, async () => null),
      404,
      "artifact_not_found",
      /missing R2 object/,
    );
  });

  await t.test("HEAD failure", async () => {
    const bucket = {
      head: async () => {
        throw new Error("network down");
      },
    };
    await expectHttpError(
      () => withArtifact(bucket, OBJECT_KEY, async () => null),
      502,
      "artifact_io_error",
      /R2 HEAD failed: network down/,
    );
  });
});

test("classifies replacement during a conditional range read", async () => {
  const bucket = {
    head: async () => HEAD,
    get: async () => ({ etag: "etag-2", size: HEAD.size }),
  };
  await expectHttpError(
    () =>
      withArtifact(bucket, OBJECT_KEY, async (artifact) => {
        await artifact.readRange(0, 4);
      }),
    409,
    "artifact_changed",
    /object changed during the request/,
  );
});

test("classifies range transport, body, and length failures", async (t) => {
  await t.test("GET failure", async () => {
    const bucket = {
      head: async () => HEAD,
      get: async () => {
        throw new Error("timeout");
      },
    };
    await expectHttpError(
      () =>
        withArtifact(bucket, OBJECT_KEY, async (artifact) => {
          await artifact.readRange(0, 4);
        }),
      502,
      "artifact_io_error",
      /R2 range GET failed: timeout/,
    );
  });

  await t.test("body failure", async () => {
    const bucket = {
      head: async () => HEAD,
      get: async () =>
        body(4, async () => {
          throw new Error("body reset");
        }),
    };
    await expectHttpError(
      () =>
        withArtifact(bucket, OBJECT_KEY, async (artifact) => {
          await artifact.readRange(0, 4);
        }),
      502,
      "artifact_io_error",
      /R2 range body failed: body reset/,
    );
  });

  await t.test("short range", async () => {
    const bucket = {
      head: async () => HEAD,
      get: async () => body(3),
    };
    await expectHttpError(
      () =>
        withArtifact(bucket, OBJECT_KEY, async (artifact) => {
          await artifact.readRange(0, 4);
        }),
      502,
      "artifact_io_error",
      /returned 3 bytes; expected 4/,
    );
  });
});

test("keeps unmarked WASM/query failures separate from R2 failures", async () => {
  const bucket = { head: async () => HEAD };
  await expectHttpError(
    () =>
      withArtifact(bucket, OBJECT_KEY, async () => {
        throw new Error("invalid bbox");
      }),
    422,
    "query_error",
    /invalid bbox/,
  );
});
