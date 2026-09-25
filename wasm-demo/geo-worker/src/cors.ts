// The CORS headers this Worker answers with. Kept out of `index.ts` so it can
// be tested from Node: `index.ts` imports the compiled `.wasm` as a
// Cloudflare module, which only resolves inside the Worker runtime.

// A browser preflights each distinct query-string shape this API takes; a
// day-long Access-Control-Max-Age means it does that once per shape instead
// of before every request. Browsers cap what they honor well under a day
// regardless (Firefox 24h, Chromium 2h), so asking for more costs nothing.
export function corsHeaders(): Headers {
  return new Headers({
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Expose-Headers":
      "X-PSI-Reads, X-PSI-Bytes, X-PSI-R2-Operations",
    "Access-Control-Max-Age": "86400",
  });
}
