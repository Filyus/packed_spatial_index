import { HttpError } from "./errors.ts";

export type Metrics = {
  reads: number;
  bytes: number;
  r2Operations: number;
  ms: number;
};

export type ArtifactContext = {
  readRange: (offset: number, length: number) => Promise<Uint8Array>;
  fileLen: number;
  objectEtag: string;
  metrics: Omit<Metrics, "ms">;
};

export { HttpError };

const ARTIFACT_IO_MARKER = "PSI_ARTIFACT_IO:";
const ARTIFACT_CHANGED_MARKER = "PSI_ARTIFACT_CHANGED:";

export async function withArtifact<T>(
  bucket: R2Bucket,
  objectKey: string,
  run: (artifact: ArtifactContext) => Promise<T>,
): Promise<{ body: T; metrics: Metrics }> {
  const t0 = Date.now();
  const counters = { reads: 0, bytes: 0, r2Operations: 1 };

  let head: R2Object | null;
  try {
    head = await bucket.head(objectKey);
  } catch (error) {
    throw new HttpError(
      502,
      "artifact_io_error",
      `R2 HEAD failed: ${errorMessage(error)}`,
    );
  }
  if (!head) {
    throw new HttpError(
      404,
      "artifact_not_found",
      `missing R2 object "${objectKey}"; run npm run seed:geo && npm run upload`,
    );
  }

  const readRange = async (
    offset: number,
    length: number,
  ): Promise<Uint8Array> => {
    counters.reads++;
    counters.r2Operations++;

    let obj: R2ObjectBody | R2Object | null;
    try {
      obj = await bucket.get(objectKey, {
        onlyIf: { etagMatches: head.etag },
        range: { offset, length },
      });
    } catch (error) {
      throw markedError(
        ARTIFACT_IO_MARKER,
        `R2 range GET failed: ${errorMessage(error)}`,
      );
    }
    if (!obj) {
      throw markedError(
        ARTIFACT_CHANGED_MARKER,
        "R2 object disappeared during the request",
      );
    }
    if (!("body" in obj)) {
      throw markedError(
        ARTIFACT_CHANGED_MARKER,
        "R2 object changed during the request",
      );
    }

    let buffer: ArrayBuffer;
    try {
      buffer = await obj.arrayBuffer();
    } catch (error) {
      throw markedError(
        ARTIFACT_IO_MARKER,
        `R2 range body failed: ${errorMessage(error)}`,
      );
    }
    if (buffer.byteLength !== length) {
      throw markedError(
        ARTIFACT_IO_MARKER,
        `R2 range GET returned ${buffer.byteLength} bytes; expected ${length}`,
      );
    }
    counters.bytes += buffer.byteLength;
    return new Uint8Array(buffer);
  };

  try {
    const body = await run({
      readRange,
      fileLen: head.size,
      objectEtag: head.etag,
      metrics: counters,
    });
    return { body, metrics: { ...counters, ms: Date.now() - t0 } };
  } catch (error) {
    if (error instanceof HttpError) {
      throw error;
    }
    const message = errorMessage(error);
    // Markers first, and deliberately so: an R2 failure raised inside
    // `readRange` comes back wrapped by the wasm layer, so its marker is
    // embedded in whatever the classifier there decided to call it. The
    // original cause wins over the wrapper's guess.
    const changed = markedMessage(message, ARTIFACT_CHANGED_MARKER);
    if (changed !== null) {
      throw new HttpError(409, "artifact_changed", changed);
    }
    const ioFailure = markedMessage(message, ARTIFACT_IO_MARKER);
    if (ioFailure !== null) {
      throw new HttpError(502, "artifact_io_error", ioFailure);
    }
    throw classifiedError(message);
  }
}

/**
 * Recover the status and code the wasm layer chose.
 *
 * `wasm_bindgen` can only reject with a `JsValue`, so the Rust side encodes one
 * as a JSON object; without it every wasm failure — a wrong-length bbox, an
 * exhausted read budget, a corrupt artifact — arrives as the same opaque
 * string. Anything that is not one of those objects keeps the old catch-all.
 */
function classifiedError(message: string): HttpError {
  let parsed: unknown;
  try {
    parsed = JSON.parse(message);
  } catch {
    return new HttpError(422, "query_error", message);
  }
  if (
    typeof parsed === "object" &&
    parsed !== null &&
    typeof (parsed as WasmError).status === "number" &&
    typeof (parsed as WasmError).code === "string" &&
    typeof (parsed as WasmError).message === "string"
  ) {
    const { status, code, message: detail } = parsed as WasmError;
    return new HttpError(status, code, detail);
  }
  return new HttpError(422, "query_error", message);
}

type WasmError = { status: number; code: string; message: string };

function markedError(marker: string, message: string): Error {
  return new Error(`${marker}${message}`);
}

function markedMessage(message: string, marker: string): string | null {
  const index = message.indexOf(marker);
  return index === -1 ? null : message.slice(index + marker.length).trim();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
