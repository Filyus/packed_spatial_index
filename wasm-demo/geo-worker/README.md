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
- `GET /collections/synthetic-points/search?bbox=&limit=&offset=&payload=none|summary|full&level=entry|feature&identity=ref|full&count=records|only`
- `GET /collections/synthetic-points/items?bbox=&limit=&offset=`

`bbox` is `minx,miny,maxx,maxy` for a 2D artifact and
`minx,miny,minz,maxx,maxy,maxz` for a 3D one. The Worker takes the dimensions
from the artifact manifest rather than from configuration, so whichever object
is uploaded is the one it serves; any other length is a `400 invalid_bbox`,
whether the request or the artifact is the odd one out.

`/search` returns an artifact-native envelope with `numberMatched`,
`numberReturned`, `query`, `payloadKind`, and `matches`. `/items` returns a
GeoJSON `FeatureCollection`. Search is bbox-only in this milestone; exact
predicate/source read-back is deliberately left to the native server.
Every match carries `entryId` and `featureRef`, whichever way the query was
answered. `payload` picks how much of the stored value comes back;
`identity=ref|full` picks how much of the source identity does. They are
separate because a source `featureId` lives inside a `feature_json` body, so
`full` reads bodies even at `payload=summary`.

Like `level`, `identity` is resolved against the collection and the rest of the
request rather than taken literally, and the echoed value is the mode that
applied. A collection storing no source id resolves to `ref` whatever was
asked, so `full` never buys a page of reads for an identical answer -- only
`feature_json` bodies have room for an id, and only GeoJSON sources supply one,
so an artifact built from Parquet or FlatGeobuf lands here too. `payload=full`
resolves to `full`, because the bodies are read regardless and the returned
GeoJSON feature carries its own `id` anyway. `full` stays accepted everywhere,
and `capabilities.identityModes` lists it only where it can add something.

`count=only` answers `numberMatched` with an empty `matches` array, counting
the matches in the artifact instead of materializing them -- the cheapest form
of "how many are in this bbox", and the one that shows up directly in the
`reads` counter below. It is refused with `422 unsupported_query` at
`level=feature` on an artifact whose entries can duplicate a source row, since
collapsing them to features means reading the matches this mode exists to
skip; `level=entry` counts there. The native server takes the same parameter
with the same rules.

Paging happens inside the artifact whenever entry order is already answer
order -- at `level=entry`, or when the manifest says entries cannot duplicate
rows. Otherwise the full header set is collected first, because deduplicating
split rows into features needs to see every match before a page can be cut.

Every R2-backed response includes `X-PSI-Reads`, `X-PSI-Bytes`, and
`X-PSI-R2-Operations`. Search and items responses expose the same counters in
the JSON body together with `ms`:

- `reads` counts range GETs issued for artifact bytes.
- `bytes` counts the range response body bytes received by the Worker.
- `r2Operations` counts the initial HEAD plus all range GETs.
- `ms` covers the full R2-backed request, starting before HEAD.

## Errors

One envelope, matching the native server, so a client written against one reads
the other without special cases:

```json
{
  "error": {
    "code": "query_too_large",
    "message": "query exceeded its configured limits"
  }
}
```

| status | code | when |
| --- | --- | --- |
| 400 | `invalid_bbox` | missing, malformed, inverted, or the wrong length for this artifact |
| 400 | `invalid_limit` / `invalid_offset` | not an integer in range (`limit` is 1-1000 here) |
| 400 | `invalid_payload` / `invalid_level` / `invalid_identity` / `invalid_count` | not one of the accepted values |
| 400 | `invalid_query` | any other parameter out of range |
| 404 | `artifact_not_found` | the R2 object is missing; seed and upload it |
| 404 | `collection_not_found` / `not_found` | unknown collection or route |
| 405 | `method_not_allowed` | only GET is served |
| 409 | `artifact_changed` | the object changed between HEAD and a conditional range GET |
| 422 | `unsupported_query` | a parameter this endpoint does not take, such as `identity` on `/items`, or `count=only` at `level=feature` where entries can duplicate a row |
| 422 | `unsupported_payload` | `/items` against an artifact without `feature-json` payloads |
| 422 | `unsupported_level` | `level=feature` on an artifact that stores no feature references |
| 422 | `query_too_large` | the query exceeded `maxReads` or the built-in budgets |
| 500 | `artifact_error` | the artifact itself is unreadable or inconsistent |
| 502 | `artifact_io_error` | R2 transport failure |
| 500 | `internal_error` | anything unclassified |

The distinction that matters: 4xx means the request can be fixed, 5xx means the
object cannot. An R2 failure keeps its own classification even when it surfaces
through the wasm layer, which only sees an opaque I/O error.

## Local build

```sh
npm install
npm run build:wasm
npm run seed:geo
npm run typecheck
npm test
```

The Node tests mock R2 to cover ETag replacement, missing objects, transport
and body failures, short ranges, query errors, and read/byte operation counters,
and check the bbox parser against both arities. They need no artifact: the
request parsers live in `src/query.ts` precisely so they can be tested without
the wasm module, which only resolves inside the Worker runtime.

`seed:geo` writes:

- `synthetic-points.parquet`: deterministic synthetic clustered GeoParquet from `../geo-seed`
- `synthetic-points.psindex`: `gp2psindex build --payload feature-json --properties all`

`seed:geo:3d` writes the same two file names from `Point Z` geometries, so the
Worker serves a 3D collection instead. What makes the artifact 3D is the
six-field covering the seed emits alongside them: the covering decides the
dimensions of the built index, so a `Point Z` column whose covering carries only
`xmin`/`ymin`/`xmax`/`ymax` still converts to a 2D artifact.

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

Pass a different bbox as a second argument (or in `WORKER_BBOX`) when the
deployed object is 3D:

```sh
npm run smoke:live -- https://psi-geo-r2-demo.<your-subdomain>.workers.dev 64,23,0,71,29,4000
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
