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
import { withArtifact, type Metrics } from "./artifact";
import { errorBody, HttpError } from "./errors.ts";
import {
  maxReads,
  parseBbox,
  parseEnum,
  parseIntParam,
  rejectUnsupportedSearchParams,
} from "./query";

export interface Env {
  BUCKET: R2Bucket;
}

const COLLECTION_ID = "synthetic-points";
const OBJECT_KEY = "synthetic-points.psindex";
const DEFAULT_LIMIT = 100;
const MAX_LIMIT = 1000;

let ready = false;

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
      return errorResponse(
        new HttpError(405, "method_not_allowed", "only GET is supported"),
      );
    }

    try {
      return await route(req, env);
    } catch (error) {
      return errorResponse(error);
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
    const { body, metrics } = await withArtifact(
      env.BUCKET,
      OBJECT_KEY,
      async (artifact) => {
        const json = await wasmCollection(
          artifact.readRange,
          artifact.fileLen,
          artifact.objectEtag,
          maxReads(url),
          false,
        );
        return [JSON.parse(json)];
      },
    );
    return jsonResponse(body, { metrics });
  }

  const collectionPrefix = `/collections/${COLLECTION_ID}`;
  if (path === collectionPrefix) {
    const { body, metrics } = await withArtifact(
      env.BUCKET,
      OBJECT_KEY,
      async (artifact) => {
        const json = await wasmCollection(
          artifact.readRange,
          artifact.fileLen,
          artifact.objectEtag,
          maxReads(url),
          true,
        );
        return JSON.parse(json);
      },
    );
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
    const identity = parseEnum(url, "identity", "ref", ["ref", "full"]);
    // `/items` deliberately keeps its own shorter list, so `identity` there is a
    // 422 rather than a silently ignored parameter.
    rejectUnsupportedSearchParams(url, [
      "bbox",
      "limit",
      "offset",
      "payload",
      "level",
      "identity",
      "maxReads",
    ]);

    const { body, metrics } = await withArtifact(
      env.BUCKET,
      OBJECT_KEY,
      async (artifact) => {
        const json = await wasmSearch(
          artifact.readRange,
          artifact.fileLen,
          artifact.objectEtag,
          Float64Array.from(bbox),
          limit,
          offset,
          payload,
          level,
          identity,
          maxReads(url),
        );
        return JSON.parse(json);
      },
    );
    return jsonResponse({ ...body, ...metrics }, { metrics });
  }

  if (path === `${collectionPrefix}/items`) {
    const bbox = parseBbox(url);
    const limit = parseIntParam(url, "limit", DEFAULT_LIMIT, 0, MAX_LIMIT);
    const offset = parseIntParam(url, "offset", 0, 0, Number.MAX_SAFE_INTEGER);
    rejectUnsupportedSearchParams(url, ["bbox", "limit", "offset", "maxReads"]);

    const { body, metrics } = await withArtifact(
      env.BUCKET,
      OBJECT_KEY,
      async (artifact) => {
        const json = await wasmItems(
          artifact.readRange,
          artifact.fileLen,
          artifact.objectEtag,
          Float64Array.from(bbox),
          limit,
          offset,
          maxReads(url),
        );
        return JSON.parse(json);
      },
    );
    return jsonResponse({ ...body, ...metrics }, { metrics });
  }

  throw new HttpError(404, "not_found", "unknown endpoint");
}

// Render any failure in the shared envelope. Every error exit goes through
// here, so the wire shape has exactly one definition.
function errorResponse(error: unknown): Response {
  const { status, body } = errorBody(error);
  return jsonResponse(body, { status });
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
    headers.set("X-PSI-R2-Operations", String(init.metrics.r2Operations));
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
    "Access-Control-Expose-Headers":
      "X-PSI-Reads, X-PSI-Bytes, X-PSI-R2-Operations",
  });
}
