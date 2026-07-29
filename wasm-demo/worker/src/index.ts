// Cloudflare Worker: answer a box query by streaming an index from R2.
//
// The Worker owns the R2 binding, so it passes a `readRange` callback into the
// wasm module. The callback does the R2 range `get` and tallies reads/bytes —
// the headline signal — which we echo back in the response and headers.
import initSync, { query } from "../pkg/psi_worker.js";
import wasmModule from "../pkg/psi_worker_bg.wasm";

export interface Env {
  BUCKET: R2Bucket;
}

const KEY = "index.psi";
let ready = false;

// Same envelope and code vocabulary as the native server and the geo Worker,
// so a client reading one reads all three.
function errorResponse(status: number, code: string, message: string): Response {
  return Response.json({ error: { code, message } }, { status });
}

// An R2 failure raised inside `readRange` comes back as text: the wasm layer
// extracts the message and wraps it, so the JS class identity is gone by then.
// A marker survives that trip, which is the same trick the geo Worker uses.
const ARTIFACT_IO_MARKER = "PSI_ARTIFACT_IO:";

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!ready) {
      initSync(wasmModule);
      ready = true;
    }

    if (req.method !== "GET") {
      return errorResponse(405, "method_not_allowed", "only GET is supported");
    }

    const url = new URL(req.url);
    const num = (k: string, d: number) => Number(url.searchParams.get(k) ?? d);
    const minx = num("minx", 0);
    const miny = num("miny", 0);
    const maxx = num("maxx", 50);
    const maxy = num("maxy", 50);
    // Cap reads to the Worker subrequest budget; 0 = unbounded.
    const maxReads = num("maxReads", 0);

    const head = await env.BUCKET.head(KEY);
    if (!head) {
      return errorResponse(
        404,
        "artifact_not_found",
        `missing R2 object "${KEY}" — run the seed + upload`,
      );
    }

    // The wasm module caches the parsed directory across requests (crate's
    // StreamDirectory), so on a warm isolate these reads cover only the query's
    // own leaves/payload — the directory rounds are not re-issued.
    let reads = 0; // R2 round-trips actually issued
    let bytes = 0; // bytes fetched from R2
    const readRange = async (
      offset: number,
      length: number,
    ): Promise<Uint8Array> => {
      reads++;
      bytes += length;
      const obj = await env.BUCKET.get(KEY, { range: { offset, length } });
      if (!obj) {
        throw new Error(`${ARTIFACT_IO_MARKER}R2 range get returned null`);
      }
      return new Uint8Array(await obj.arrayBuffer());
    };

    const t0 = Date.now();
    let result: {
      hits: number;
      payloadBytes: number;
      ids: number[];
      geometries: string[];
    };
    try {
      result = (await query(
        readRange,
        head.size,
        minx,
        miny,
        maxx,
        maxy,
        maxReads,
      )) as typeof result;
    } catch (e) {
      // An R2 failure is the storage's fault (502); anything else the wasm
      // module rejects with is about the query (422). Reporting both as 502 --
      // as this demo used to -- makes a too-wide query look like a broken
      // object.
      const message = e instanceof Error ? e.message : String(e);
      const marker = message.indexOf(ARTIFACT_IO_MARKER);
      return marker === -1
        ? errorResponse(422, "query_error", message)
        : errorResponse(
            502,
            "artifact_io_error",
            message.slice(marker + ARTIFACT_IO_MARKER.length).trim(),
          );
    }
    const ms = Date.now() - t0;

    return Response.json(
      { ...result, reads, bytes, ms, query: { minx, miny, maxx, maxy } },
      {
        headers: {
          "X-PSI-Reads": String(reads),
          "X-PSI-Bytes": String(bytes),
        },
      },
    );
  },
};
