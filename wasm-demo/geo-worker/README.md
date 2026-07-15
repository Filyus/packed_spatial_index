# GeoPSINDEX Cloudflare Worker + R2 feature API

End-to-end demo for the main static serving story:

```text
GeoParquet -> gp2psindex -> immutable .psindex in R2 -> HTTP feature/search API
```

The Worker never reads back the source GeoParquet and never talks to a database.
It streams byte ranges from one `synthetic-points.psindex` R2 object, caches the
parsed GeoPSINDEX directory in the warm isolate, and serves GeoJSON features
directly from the artifact's embedded `feature-json` payload.

The directory cache is tied to the object's R2 ETag and byte length. Range reads
also carry an ETag precondition, so replacing the fixed demo key cannot mix a
cached directory from one artifact with bytes from another artifact.

The seed dataset is deliberately synthetic: deterministic clustered WKB points
with a `bbox` covering column and GeoParquet metadata. It is shaped to exercise
realistic spatial access patterns, but it is not a city database and contains no
real place names or population attributes.

This example is geo-first and intentionally separate from
[`../worker`](../worker), which remains the low-level core `PSINDEX` range-read
demo.

## Endpoints

- `GET /health`
- `GET /collections`
- `GET /collections/synthetic-points`
- `GET /collections/synthetic-points/search?bbox=minx,miny,maxx,maxy&limit=&offset=&payload=none|summary|full&level=entry|feature`
- `GET /collections/synthetic-points/items?bbox=minx,miny,maxx,maxy&limit=&offset=`

`/search` returns an artifact-native envelope with `numberMatched`,
`numberReturned`, `query`, `payloadKind`, and `matches`. `/items` returns a
GeoJSON `FeatureCollection`. Search is bbox-only in this milestone; exact
predicate/source read-back is deliberately left to the native server.
For artifacts whose manifest says entries cannot duplicate rows, summary search
uses an entry/rank header path and omits `featureRef`; request `payload=full` or
use `/items` when the client needs the returned GeoJSON features.
That path uses `search_payload_headers_page_async`, so a broad bbox counts every
match but retains only `offset + limit` headers before fetching the requested
payload bodies. For artifacts with split/duplicate source rows, `level=entry`
uses the same bounded strategy through `search_match_headers_page_async`.
`level=feature` and `/items` keep the full-header fallback because exact
feature-level deduplication needs global identity state.

Every R2-backed response includes `X-PSI-Reads`, `X-PSI-Bytes`, and
`X-PSI-R2-Operations`. Search and items responses expose the same counters in
the JSON body together with `ms`:

- `reads` counts range GETs issued for artifact bytes.
- `bytes` counts the range response body bytes received by the Worker.
- `r2Operations` counts the initial HEAD plus all range GETs.
- `ms` covers the full R2-backed request, starting before HEAD.

If the object changes between HEAD and a conditional range GET, the Worker
returns `409 artifact_changed`; R2 transport failures return
`502 artifact_io_error`. Artifact/query validation failures remain
`422 query_error`.

## Local build

```sh
npm install
npm run build:wasm
npm run seed:geo
npm run typecheck
npm test
```

The Node tests mock R2 to cover ETag replacement, missing objects, transport
and body failures, short ranges, query errors, and read/byte operation counters.

`seed:geo` writes:

- `synthetic-points.parquet`: deterministic synthetic clustered GeoParquet from `../geo-seed`
- `synthetic-points.psindex`: `gp2psindex build --payload feature-json --properties all`

The wasm module depends on `packed_spatial_index_geo` with
`default-features = false, features = ["async"]`, so it keeps Arrow/Parquet out
of the Worker. The conversion CLI still uses the full geo crate locally.

## Deploy

```sh
# one-time auth, or set CLOUDFLARE_API_TOKEN
wrangler login

npm run bucket:create      # ok if the bucket already exists
npm run upload             # uploads synthetic-points.psindex to psi-geo-demo/synthetic-points.psindex
npm run deploy
```

Defaults:

- Worker: `psi-geo-r2-demo`
- R2 bucket: `psi-geo-demo`
- Object key: `synthetic-points.psindex`
- Collection id: `synthetic-points`

## Live smoke

```sh
npm run smoke:live -- https://psi-geo-r2-demo.<your-subdomain>.workers.dev
```

The smoke script checks `/health`, `/collections`, `/search`, and `/items` with
a deterministic bbox around one synthetic seed-data cluster:

```text
bbox=64,23,71,29
```

Representative response using the counters measured on a deployed Worker (the
`r2Operations` field is derived as HEAD plus range reads because it was added
after that deployment):

```json
{
  "collectionId": "synthetic-points",
  "query": {
    "bbox": [64, 23, 71, 29],
    "predicate": "bbox",
    "level": "feature",
    "payload": "summary",
    "limit": 3,
    "offset": 0
  },
  "payloadKind": "feature_json",
  "numberMatched": 553,
  "numberReturned": 3,
  "matches": [
    {
      "entryId": 169,
      "payload": {
        "kind": "feature_json"
      }
    }
  ],
  "reads": 1,
  "bytes": 7360,
  "r2Operations": 2,
  "ms": 57
}
```

The same deployed Worker also handles a world-sized bbox without reading all
GeoJSON bodies:

| query | matched | returned | range reads | R2 ops | bytes | ms |
|---|---:|---:|---:|---:|---:|---:|
| `/search?bbox=-180,-90,180,90&limit=3&payload=summary&level=entry` | 100000 | 3 | 1 | 2 | 800008 | 72 |
| `/items?bbox=-180,-90,180,90&limit=3` | 100000 | 3 | 6 | 7 | 972629 | 220 |

The copied timings predate the explicit `r2Operations` field; its values above
are the corresponding HEAD plus range-read counts. Exact counters and timings
vary with the query and cold/warm isolate state, but the important proof is
stable: a public HTTP API can serve feature results from a single immutable R2
object with bounded range reads and no database.
