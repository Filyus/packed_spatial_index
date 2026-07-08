// Cloudflare Worker: OGC-ish feature/search API over a GeoPSINDEX object in R2.
//
// The Worker owns the R2 binding, passes range reads into the wasm module, and
// exposes the read/byte counters that make the object-storage story visible.
import initSync, {
  collection as wasmCollection,
  items as wasmItems,
  search as wasmSearch,
} from "../pkg/psi_geo_worker.js";
import wasmModule from "../pkg/psi_geo_worker_bg.wasm";

export interface Env {
  BUCKET: R2Bucket;
}

const COLLECTION_ID = "cities";
const OBJECT_KEY = "cities.psindex";
const DEFAULT_LIMIT = 100;
const MAX_LIMIT = 1000;

let ready = false;

type Metrics = {
  reads: number;
  bytes: number;
  ms: number;
};

type ArtifactContext = {
  readRange: (offset: number, length: number) => Promise<Uint8Array>;
  fileLen: number;
  metrics: Omit<Metrics, "ms">;
};

class HttpError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!ready) {
      initSync(wasmModule);
      ready = true;
    }

    if (req.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }
    if (req.method !== "GET") {
      return jsonResponse(
        { code: "method_not_allowed", message: "only GET is supported" },
        { status: 405 },
      );
    }

    try {
      return await route(req, env);
    } catch (error) {
      if (error instanceof HttpError) {
        return jsonResponse(
          { code: error.code, message: error.message },
          { status: error.status },
        );
      }
      return jsonResponse(
        { code: "internal_error", message: String(error) },
        { status: 500 },
      );
    }
  },
};

async function route(req: Request, env: Env): Promise<Response> {
  const url = new URL(req.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";

  if (path === "/health") {
    return jsonResponse({ status: "ok", objectKey: OBJECT_KEY });
  }

  if (path === "/collections") {
    const { body, metrics } = await withArtifact(env, async (artifact) => {
      const json = await wasmCollection(
        artifact.readRange,
        artifact.fileLen,
        maxReads(url),
        false,
      );
      return [JSON.parse(json)];
    });
    return jsonResponse(body, { metrics });
  }

  const collectionPrefix = `/collections/${COLLECTION_ID}`;
  if (path === collectionPrefix) {
    const { body, metrics } = await withArtifact(env, async (artifact) => {
      const json = await wasmCollection(
        artifact.readRange,
        artifact.fileLen,
        maxReads(url),
        true,
      );
      return JSON.parse(json);
    });
    return jsonResponse(body, { metrics });
  }

  if (path.startsWith("/collections/") && !path.startsWith(collectionPrefix)) {
    throw new HttpError(404, "collection_not_found", "unknown collection");
  }

  if (path === `${collectionPrefix}/search`) {
    const bbox = parseBbox(url);
    const limit = parseIntParam(url, "limit", DEFAULT_LIMIT, 0, MAX_LIMIT);
    const offset = parseIntParam(url, "offset", 0, 0, Number.MAX_SAFE_INTEGER);
    const payload = parseEnum(url, "payload", "summary", ["none", "summary", "full"]);
    const level = parseEnum(url, "level", "feature", ["entry", "feature"]);
    rejectUnsupportedSearchParams(url, ["bbox", "limit", "offset", "payload", "level", "maxReads"]);

    const { body, metrics } = await withArtifact(env, async (artifact) => {
      const json = await wasmSearch(
        artifact.readRange,
        artifact.fileLen,
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        limit,
        offset,
        payload,
        level,
        maxReads(url),
      );
      return JSON.parse(json);
    });
    return jsonResponse({ ...body, ...metrics }, { metrics });
  }

  if (path === `${collectionPrefix}/items`) {
    const bbox = parseBbox(url);
    const limit = parseIntParam(url, "limit", DEFAULT_LIMIT, 0, MAX_LIMIT);
    const offset = parseIntParam(url, "offset", 0, 0, Number.MAX_SAFE_INTEGER);
    rejectUnsupportedSearchParams(url, ["bbox", "limit", "offset", "maxReads"]);

    const { body, metrics } = await withArtifact(env, async (artifact) => {
      const json = await wasmItems(
        artifact.readRange,
        artifact.fileLen,
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        limit,
        offset,
        maxReads(url),
      );
      return JSON.parse(json);
    });
    return jsonResponse({ ...body, ...metrics }, { metrics });
  }

  throw new HttpError(404, "not_found", "unknown endpoint");
}

async function withArtifact<T>(
  env: Env,
  run: (artifact: ArtifactContext) => Promise<T>,
): Promise<{ body: T; metrics: Metrics }> {
  const head = await env.BUCKET.head(OBJECT_KEY);
  if (!head) {
    throw new HttpError(
      404,
      "artifact_not_found",
      `missing R2 object "${OBJECT_KEY}"; run npm run seed:geo && npm run upload`,
    );
  }

  const counters = { reads: 0, bytes: 0 };
  const readRange = async (
    offset: number,
    length: number,
  ): Promise<Uint8Array> => {
    counters.reads++;
    counters.bytes += length;
    const obj = await env.BUCKET.get(OBJECT_KEY, { range: { offset, length } });
    if (!obj) throw new Error("R2 range get returned null");
    return new Uint8Array(await obj.arrayBuffer());
  };

  const t0 = Date.now();
  try {
    const body = await run({ readRange, fileLen: head.size, metrics: counters });
    return { body, metrics: { ...counters, ms: Date.now() - t0 } };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new HttpError(422, "query_error", message);
  }
}

function parseBbox(url: URL): [number, number, number, number] {
  const raw = url.searchParams.get("bbox");
  if (!raw) {
    throw new HttpError(400, "invalid_bbox", "bbox is required");
  }
  const values = raw.split(",").map((part) => Number(part.trim()));
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) {
    throw new HttpError(
      400,
      "invalid_bbox",
      "bbox must be four comma-separated numbers",
    );
  }
  const [minX, minY, maxX, maxY] = values;
  if (minX > maxX || minY > maxY) {
    throw new HttpError(400, "invalid_bbox", "bbox min values must be <= max values");
  }
  return [minX, minY, maxX, maxY];
}

function parseIntParam(
  url: URL,
  key: string,
  fallback: number,
  min: number,
  max: number,
): number {
  const raw = url.searchParams.get(key);
  if (raw === null || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new HttpError(400, "invalid_query", `${key} must be an integer in [${min}, ${max}]`);
  }
  return value;
}

function parseEnum<T extends string>(
  url: URL,
  key: string,
  fallback: T,
  allowed: readonly T[],
): T {
  const raw = url.searchParams.get(key);
  if (raw === null || raw === "") return fallback;
  if (allowed.includes(raw as T)) return raw as T;
  throw new HttpError(400, "invalid_query", `${key} must be one of ${allowed.join(", ")}`);
}

function maxReads(url: URL): number {
  return parseIntParam(url, "maxReads", 0, 0, 10_000);
}

function rejectUnsupportedSearchParams(url: URL, allowed: string[]): void {
  for (const key of url.searchParams.keys()) {
    if (!allowed.includes(key)) {
      throw new HttpError(422, "unsupported_query", `${key} is not supported by this endpoint`);
    }
  }
}

function jsonResponse(
  body: unknown,
  init: ResponseInit & { metrics?: Metrics } = {},
): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json; charset=utf-8");
  for (const [key, value] of corsHeaders()) {
    headers.set(key, value);
  }
  if (init.metrics) {
    headers.set("X-PSI-Reads", String(init.metrics.reads));
    headers.set("X-PSI-Bytes", String(init.metrics.bytes));
  }
  return new Response(JSON.stringify(body, null, 2), {
    ...init,
    headers,
  });
}

function corsHeaders(): Headers {
  return new Headers({
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Expose-Headers": "X-PSI-Reads, X-PSI-Bytes",
  });
}
