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

export class HttpError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

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
    const message = errorMessage(error);
    const changed = markedMessage(message, ARTIFACT_CHANGED_MARKER);
    if (changed !== null) {
      throw new HttpError(409, "artifact_changed", changed);
    }
    const ioFailure = markedMessage(message, ARTIFACT_IO_MARKER);
    if (ioFailure !== null) {
      throw new HttpError(502, "artifact_io_error", ioFailure);
    }
    throw new HttpError(422, "query_error", message);
  }
}

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
