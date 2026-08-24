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
 * Six inward-pointing planes as 24 numbers, or `[]` when no frustum was asked
 * for.
 *
 * Planes rather than a view-projection matrix on purpose: a matrix carries a
 * clip-space depth convention (`ClipSpaceZ`, which the library refuses to
 * default silently) and a row/column-major convention, neither recoverable
 * from the numbers, and either one wrong moves the near plane without failing.
 * A client resolves both locally -- `Frustum3D::from_view_projection` is right
 * there -- and sends the result.
 */
export function parseFrustum(url: URL): number[] {
  const raw = url.searchParams.get("frustum");
  if (raw === null || raw === "") return [];
  if (url.searchParams.get("bbox") !== null) {
    throw new HttpError(400, "invalid_query", "bbox and frustum are mutually exclusive");
  }
  const values = raw.split(",").map((part) => Number(part.trim()));
  if (values.length !== 24 || values.some((value) => !Number.isFinite(value))) {
    throw new HttpError(
      400,
      "invalid_frustum",
      "frustum must contain 24 finite numbers (six planes of a,b,c,d)",
    );
  }
  for (let plane = 0; plane < 6; plane += 1) {
    const [a, b, c] = values.slice(plane * 4, plane * 4 + 3);
    if (a === 0 && b === 0 && c === 0) {
      throw new HttpError(
        400,
        "invalid_frustum",
        `frustum plane ${plane} has a zero normal, so it constrains nothing`,
      );
    }
  }
  return values;
}

/**
 * The polygon parameter, passed through as text.
 *
 * The coordinates are parsed on the wasm side, by the same code that would
 * read them out of a POST body -- keeping one parser rather than two that can
 * disagree. This checks only what belongs to the query string: that a polygon
 * is not asked for alongside another shape.
 */
export function parsePolygon(url: URL): string {
  const raw = url.searchParams.get("polygon");
  if (raw === null || raw === "") return "";
  for (const other of ["bbox", "frustum"]) {
    if (url.searchParams.get(other) !== null) {
      throw new HttpError(400, "invalid_query", `polygon and ${other} are mutually exclusive`);
    }
  }
  return raw;
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
    case "count":
      return "invalid_count";
    case "frustum":
      return "invalid_frustum";
    case "polygon":
      return "invalid_polygon";
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
