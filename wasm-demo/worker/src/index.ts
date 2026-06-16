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

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!ready) {
      initSync(wasmModule);
      ready = true;
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
      return new Response(`missing R2 object "${KEY}" — run the seed + upload`, {
        status: 404,
      });
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
      if (!obj) throw new Error("R2 range get returned null");
      return new Uint8Array(await obj.arrayBuffer());
    };

    const t0 = Date.now();
    let result: { hits: number; payloadBytes: number; ids: number[] };
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
      return new Response(`query error: ${e}`, { status: 502 });
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
