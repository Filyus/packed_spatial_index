// The error contract, shared by every route.
//
// One envelope and one code vocabulary, both matching the native server in
// `server/src/error.rs` — a client written against one should read the other
// without special cases. Kept out of `index.ts` so it can be tested from Node:
// `index.ts` imports the compiled `.wasm` as a Cloudflare module, which only
// resolves inside the Worker runtime.

export class HttpError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

/**
 * The server's shape: the payload is nested under `error`, so a success body
 * and a failure body can never be told apart by accident.
 */
export type ErrorBody = {
  error: { code: string; message: string };
};

export function errorBody(error: unknown): { status: number; body: ErrorBody } {
  if (error instanceof HttpError) {
    return {
      status: error.status,
      body: { error: { code: error.code, message: error.message } },
    };
  }
  return {
    status: 500,
    body: { error: { code: "internal_error", message: String(error) } },
  };
}
