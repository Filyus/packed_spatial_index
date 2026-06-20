# Cloudflare Worker + R2 streaming demo

Answers an arbitrary 2D box query by **streaming** a serialized
`packed_spatial_index` index out of an R2 object — fetching only the few range
reads the traversal needs, never the whole file. This is the "cloud store over
arbitrary AABBs" target the streaming/payload/async stack was built for.

```
GET /?minx=100&miny=100&maxx=140&maxy=140
  -> { "hits": 98, "payloadBytes": 6272, "ids": [...], "reads": 6, "bytes": 25000, "ms": 41 }
```

`reads` / `bytes` (also in `X-PSI-Reads` / `X-PSI-Bytes` headers) are the headline
signal: how cheap a spatial query over a remote object actually is.

## How it works

- `src/index.ts` (the Worker) owns the R2 binding. It passes a
  `readRange(offset, length) => Promise<Uint8Array>` callback into the wasm
  module and tallies reads/bytes around it.
- `src/lib.rs` wraps that callback as an [`AsyncRangeReader`], opens
  `StreamIndex2D` over it, and runs `search_payloads_async`. The crate's async
  futures are `!Send` — a perfect fit for the single-threaded isolate.
- The parsed directory is cached across requests via the crate's
  `StreamDirectory` (`into_directory` / `from_directory`): the first request
  opens the index, later ones reattach a fresh R2 reader with no directory I/O,
  so the directory round-trips (the bulk of per-query latency) are paid once per
  warm isolate.
- The directory budget is raised (`StreamLimits::directory_budget_bytes`) to
  cache all internal tree levels, so a warm query streams little more than its
  own payload. Live: a warm point query is ~1-2 R2 reads (down from ~13 cold).
  Result memory is capped (`max_read_bytes` / `max_items`) well under the
  isolate's 128 MB so a broad query can't OOM and evict the warm directory;
  peak ~32 MB. No concurrency tracking — Cloudflare schedules isolates.
- The index is **interleaved + fixed-width records** (the layout the local
  simulator showed issues the fewest reads/bytes).

## Local validation (no cloud)

The reads/bytes counts come from the crate's real coalescing/traversal, so they
match what R2 would do. Measure them with no account:

```sh
cargo run --release --manifest-path ../r2-sim/Cargo.toml -- 200000
```

Typical per-query cost (200k items, 64 B payloads, interleaved + fixed-width):

| window | hits  | reads | bytes/query |
|-------:|------:|------:|------------:|
|   0.5% |     9 |   4.0 |       5.7 KB |
|   2%   |    98 |   5.9 |        25 KB |
|   8%   |  1346 |  13.2 |       195 KB |
|  25%   | 12718 |  37.6 |       1.5 MB |

## Deploy (needs your Cloudflare account)

```sh
# 0. one-time auth (interactive OAuth) — or set CLOUDFLARE_API_TOKEN
wrangler login

# 1. build the wasm module
npm run build:wasm            # wasm-pack build --target web --out-dir pkg --release

# 2. generate the index file (interleaved + fixed-width) for upload
npm run seed                  # writes ./index.psi (200k items)

# 3. create the bucket and upload the index at key "index.psi"
wrangler r2 bucket create psi-demo
npm run upload                # wrangler r2 object put psi-demo/index.psi --file index.psi

# 4. deploy, then query
npm run deploy
curl "https://psi-r2-demo.<your-subdomain>.workers.dev/?minx=100&miny=100&maxx=140&maxy=140"
```

`wrangler dev` runs it locally against R2 first if you prefer.

## Real GeoParquet variant

Instead of the synthetic index, seed from a **real GeoParquet** file through the
[`packed_spatial_index_geo`](../../geo) converter — the "GeoParquet → cloud-served
geometry" story end to end:

```sh
npm run seed:geo   # geo-seed -> cities.parquet -> gp2psindex -> ./index.psi
npm run upload && npm run deploy
```

`seed:geo` generates a realistic clustered point GeoParquet (`../geo-seed`), then
runs the `gp2psindex` CLI to convert it. The Worker is unchanged — it streams any
PSINDEX. With a geo index the response also carries the matching features'
geometry as base64 WKB:

```
GET /?minx=0&miny=40&maxx=30&maxy=60
  -> { "hits": 1474, "geometries": ["AQE…"], "ids": [...], "reads": 8, "bytes": 4.3MB, "ms": 241 }
```

## Query params

| param      | default | meaning                                  |
|------------|---------|------------------------------------------|
| `minx/miny/maxx/maxy` | `0,0,50,50` | query box                  |
| `maxReads` | `0` (off) | abort with `LimitExceeded` past N reads (Worker subrequest guard) |

[`AsyncRangeReader`]: https://docs.rs/packed_spatial_index/latest/packed_spatial_index/trait.AsyncRangeReader.html
