// Query-string parsing for the feature/search API.
//
// Kept apart from `index.ts` so it can be exercised without the wasm module:
// `index.ts` imports the compiled `.wasm` as a Cloudflare module, which only
// resolves inside the Worker runtime.
import { HttpError } from "./errors.ts";

// Either arity is accepted here and the artifact decides which one is right:
// the Worker cannot know whether the object is 2D or 3D until it is open, so
// a 4-number bbox against a 3D artifact comes back from wasm as a 422 naming
// the length that artifact wants.
export function parseBbox(url: URL): number[] {
  const raw = url.searchParams.get("bbox");
  if (!raw) {
    throw new HttpError(400, "invalid_bbox", "bbox is required");
  }
  const values = raw.split(",").map((part) => Number(part.trim()));
  if (
    (values.length !== 4 && values.length !== 6) ||
    values.some((value) => !Number.isFinite(value))
  ) {
    throw new HttpError(
      400,
      "invalid_bbox",
      "bbox must contain either 4 numbers (2D) or 6 numbers (3D)",
    );
  }
  const axes = values.length / 2;
  for (let axis = 0; axis < axes; axis += 1) {
    if (values[axis] > values[axis + axes]) {
      throw new HttpError(400, "invalid_bbox", "bbox min values must be <= max values");
    }
  }
  return values;
}

/**
 * The error code for a bad value of `key`.
 *
 * The server names the parameter in the code — `invalid_limit`, not a blanket
 * `invalid_query` — so a client can branch on which one it got wrong without
 * parsing prose. Anything without a dedicated code keeps the generic one.
 */
export function invalidCode(key: string): string {
  switch (key) {
    case "bbox":
      return "invalid_bbox";
    case "limit":
      return "invalid_limit";
    case "offset":
      return "invalid_offset";
    case "payload":
      return "invalid_payload";
    case "level":
      return "invalid_level";
    case "identity":
      return "invalid_identity";
    default:
      return "invalid_query";
  }
}

export function parseIntParam(
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
    throw new HttpError(400, invalidCode(key), `${key} must be an integer in [${min}, ${max}]`);
  }
  return value;
}

export function parseEnum<T extends string>(
  url: URL,
  key: string,
  fallback: T,
  allowed: readonly T[],
): T {
  const raw = url.searchParams.get(key);
  if (raw === null || raw === "") return fallback;
  if (allowed.includes(raw as T)) return raw as T;
  throw new HttpError(400, invalidCode(key), `${key} must be one of ${allowed.join(", ")}`);
}

export function maxReads(url: URL): number {
  return parseIntParam(url, "maxReads", 0, 0, 10_000);
}

export function rejectUnsupportedSearchParams(url: URL, allowed: string[]): void {
  for (const key of url.searchParams.keys()) {
    if (!allowed.includes(key)) {
      throw new HttpError(422, "unsupported_query", `${key} is not supported by this endpoint`);
    }
  }
}
